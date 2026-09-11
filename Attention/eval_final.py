# eval_final.py —— 独立可信评估（不吃训练脚本的内联打印）：
#   BLEU-1..4(200图, seed=0, 确定性) + 固定句 image_gain + 8 张 real/zero greedy 样例。
#   关键修复: 每次遍历 test loader 前 random.seed 全局 RNG -> real 与 zero 用同一批目标句,
#   把 image_gain 从"图信号+标注抽样噪声"里剥离出纯图像信号(确定性)。
import random
import torch
import train4caption_resnet as tr
from Attention.datasets.Flickr8k import get_vocab_size, CAPTION_MAX_LEN
from train4caption import eval_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model():
    m = tr.ResNetCaptionModel(vocab_size=get_vocab_size(), max_seq_len=CAPTION_MAX_LEN).to(device)
    return m


def fixed_gain(model, loader, max_batches=5, seed=0):
    """real/zero 各遍历一次, 每次遍历前 seed 全局 RNG -> 两遍命中完全相同的目标句."""
    def acc(blank):
        cur = tot = 0
        with torch.no_grad():
            random.seed(seed)                       # 固定每轮 caption 抽样
            for i, (img, cap) in enumerate(loader):
                if i >= max_batches:
                    break
                img, cap = img.to(device), cap.to(device)
                if blank:
                    img = torch.zeros_like(img)
                cap_in, cap_tar = cap[:, :-1], cap[:, 1:]
                out = model(img, cap_in)
                pred = out.argmax(-1).reshape(-1)
                tgt = cap_tar.reshape(-1)
                real = tgt != 0
                cur += ((pred == tgt) & real).sum()
                tot += real.sum()
        return 100 * cur / tot if tot else 0.0
    return acc(False), acc(True)


id2tok = None


def to_text(seq):
    w = []
    for i in seq[1:].tolist():
        t = id2tok.get(i)
        if t in ('<eos>', '<pad>'):
            break
        if t != '<unk>':
            w.append(t)
    return ' '.join(w)


for ep in [25, 50, 75, 100]:
    m = make_model()
    m.load_state_dict(torch.load(f"checkpoints/resnet50_model_epoch_{ep}.pth", map_location=device)["model_state_dict"])
    m.eval()
    _, test_loader = tr.make_loaders(batch_size=64)
    ds = test_loader.dataset
    b1, b2, b3, b4 = eval_bleu(m, ds, device, max_imgs=200, seed=0)
    ar, az = fixed_gain(m, test_loader)
    print(f"[epoch {ep}] BLEU {100*b1:.2f}/{100*b2:.2f}/{100*b3:.2f}/{100*b4:.2f}  "
          f"real={ar:.2f}% zero={az:.2f}% image_gain={ar-az:+.2f}pt")
    torch.cuda.empty_cache()

# 最终 epoch 的定性样例 (8 张真实 vs 全零)
m = make_model()
m.load_state_dict(torch.load("checkpoints/resnet50_model_epoch_100.pth", map_location=device)["model_state_dict"])
m.eval()
_, test_loader = tr.make_loaders(batch_size=64)
ds = test_loader.dataset
id2tok = {i: t for t, i in ds.vocab.items()}
imgs = torch.stack([ds[i][0] for i in range(8)]).to(device)
zeros = torch.zeros_like(imgs)
print("\n--- epoch100 greedy 样例 ---")
for b in range(8):
    r = to_text(tr._greedy_decode(m, imgs[b:b+1], device)[0])
    z = to_text(tr._greedy_decode(m, zeros[b:b+1], device)[0])
    print(f"[{b}] real: {r}")
    print(f"    zero: {z}")
