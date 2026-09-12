# datasets/Flickr8k.py
# 面向 Attention/caption/models/ImageCaptioningModel 的 Flickr8k 图像描述数据集。
#   __getitem__ 返回 (image_tensor, caption_id_tensor)，其中 caption 定长 CAPTION_MAX_LEN，
#   满足 train.py 里 cap[:, :-1] / cap[:, 1:] 的 teacher-forcing 切法；
#   PAD 固定在 0，与 train.py 的 CrossEntropyLoss(ignore_index=PAD_ID) 一致。
#
# 词表只在 train 划分上统计，避免测试集词表泄漏；每张图随机取 5 句标注之一。
# 数据文件自动定位到 Attention/data/（图片目录 zip 里的拼写是 Flicker8k_Dataset）。
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---- 超参（train.py 里模型的 max_seq_len 必须等于 CAPTION_MAX_LEN）----
CAPTION_MAX_LEN = 32      # 文本最长长度（含 <sos>/<eos>）
IMAGE_SIZE = 96           # 图片 resize 边长(patch=4 → 24x24=576 个 patch)
MIN_WORD_FREQ = 2         # 词频低于该值 → <unk>（train 唯一词 7705，>=2 约 4521）
TRAIN_AUG = False         # 默认不增广（caption 有空间/颜色语义）；如开则只做轻微颜色抖动

# 特殊 token id：<pad> 必须是 0（train.py 的 PAD_ID）
_SPECIAL = ['<pad>', '<sos>', '<eos>', '<unk>']

# ---- 数据文件定位 ----
_BASE = Path(__file__).resolve().parents[2]              # .../Attention（caption/datasets 往上三层）
DATA_DIR = _BASE / 'data'
IMAGE_DIR = DATA_DIR / 'Flicker8k_Dataset'
CAPTION_FILE = DATA_DIR / 'Flickr8k.token.txt'
_SPLIT_FILE = {
    'train': DATA_DIR / 'Flickr_8k.trainImages.txt',
    'dev': DATA_DIR / 'Flickr_8k.devImages.txt',
    'test': DATA_DIR / 'Flickr_8k.testImages.txt',
}

# ---- 全局缓存：解析一次，进程内复用 ----
_CAPS_BY_IMG = None       # {img_name: [[token,...]*5]}
_IMG_IDS = {}             # {split: [img_name,...]}
_TOKEN2ID = None
_ID2TOKEN = None


def _ensure_captions():
    """Flickr8k.token.txt -> {图片名: [5 句已小写分词]}. 句内单词已按空格分开。"""
    global _CAPS_BY_IMG
    if _CAPS_BY_IMG is None:#缓存切分
        caps = {}
        for line in CAPTION_FILE.open(encoding='utf-8'):
            name, _, text = line.rstrip('\n').partition('\t')
            img = name.split('#', 1)[0]
            caps.setdefault(img, []).append([w.lower() for w in text.split()])
        _CAPS_BY_IMG = caps
    return _CAPS_BY_IMG


def _ensure_img_ids(split):
    if split not in _IMG_IDS:
        f = _SPLIT_FILE[split]
        _IMG_IDS[split] = [l.strip() for l in f.open(encoding='utf-8') if l.strip()]
    return _IMG_IDS[split]


def _ensure_vocab():
    """只在 train 划分建词表。token2id 从 0 开始：<pad>=0,<sos>=1,<eos>=2,<unk>=3。"""
    global _TOKEN2ID, _ID2TOKEN
    if _TOKEN2ID is not None:
        return _TOKEN2ID
    caps = _ensure_captions()
    counter = Counter()
    for img in _ensure_img_ids('train'):
        for cap in caps.get(img, []):
            counter.update(cap)
    kept = sorted(w for w, c in counter.items() if c >= MIN_WORD_FREQ)
    _TOKEN2ID = {t: i for i, t in enumerate(_SPECIAL + kept)}
    _ID2TOKEN = {i: t for t, i in _TOKEN2ID.items()}
    return _TOKEN2ID


def get_vocab_size():
    """训练前调用，得到与数据集一致的词表大小，用于建模型。"""
    return len(_ensure_vocab())


def decode(ids):
    """把 token id 序列还原为字符串（用于日后推理/看样本）。"""
    tok2id, id2tok = _ensure_vocab(), _ID2TOKEN
    eos = tok2id['<eos>']
    words = []
    for i in ids.tolist() if torch.is_tensor(ids) else ids:
        if i == eos:
            break
        w = id2tok[i]
        if w not in ('<pad>', '<sos>', '<unk>'):
            words.append(w)
    return ' '.join(words)


def _default_transform():
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if TRAIN_AUG:
        t.append(transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15))
    t += [transforms.ToTensor(), transforms.Normalize((.5,) * 3, (.5,) * 3)]
    return transforms.Compose(t)

def get_test_transformer():
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5,) * 3, (.5,) * 3)
    ])
    return test_transforms

def get_train_transformer():
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((.5,) * 3, (.5,) * 3)
    ])
    return train_transforms

class Flickr8kCaptions(Dataset):
    def __init__(self, split='train', transform=None):
        assert split in _SPLIT_FILE, f"split 只能是 {list(_SPLIT_FILE)}"
        self.split = split
        self.caps = _ensure_captions()
        self.vocab = _ensure_vocab()
        self.ids = [img for img in _ensure_img_ids(split) if img in self.caps]
        if not self.ids:
            raise RuntimeError(f"[{split}] 没有任何图片同时存在于 {CAPTION_FILE} 与 {IMAGE_DIR}")
        self.transform = transform if transform is not None else _default_transform()
        self._eos = self.vocab['<eos>']
        self._unk = self.vocab['<unk>']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.ids[i]
        # 每张图有 5 句标注，随机取 1 句作为训练目标
        tokens = random.choice(self.caps[img])
        words = [self.vocab.get(w, self._unk) for w in tokens]
        # 保留 <eos>：句子超长则只截断中间单词
        seq = [self.vocab['<sos>']] + words[:(CAPTION_MAX_LEN - 2)] + [self._eos]
        pad = [0] * (CAPTION_MAX_LEN - len(seq))
        caption = torch.tensor(seq + pad, dtype=torch.long)

        im = Image.open(IMAGE_DIR / img).convert('RGB')
        image = self.transform(im)
        return image, caption


def get_data_loaders(batch_size=64, num_workers=0):
    """返回 (train_loader, test_loader)，供 train.py 迭代 (image, caption)。"""
    train_ds = Flickr8kCaptions('train', transform=get_train_transformer())
    test_ds = Flickr8kCaptions('test', transform=get_test_transformer())
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


if __name__ == '__main__':
    print('vocab_size =', get_vocab_size())
    train_loader, test_loader = get_data_loaders(batch_size=8)
    print('train batches =', len(train_loader), ' test batches =', len(test_loader))
    images, captions = next(iter(train_loader))
    print('image   :', tuple(images.shape), images.dtype, ' range', images.min().item(), images.max().item())
    print('caption :', tuple(captions.shape), captions.dtype)
    print('sample  :', captions[0].tolist())
    print('decode  :', decode(captions[0]))
