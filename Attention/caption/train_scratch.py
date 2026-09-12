# train4caption
import torch
import os
import sys
# 让 `import Attention` 生效：把 Attention 的父目录(30-)加入 sys.path（不依赖运行 cwd）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # Attention/caption
_ATTN_DIR = os.path.dirname(_THIS_DIR)                       # Attention
sys.path.insert(0, os.path.dirname(_ATTN_DIR))               # 30-
CKPT_DIR = os.path.join(_ATTN_DIR, 'checkpoints')            # 权重统一存这里
import torch.nn as nn
from Attention.caption.models.ImageCaptioningModel import ImageCaptioningModel
from Attention.caption.datasets.Flickr8k import get_data_loaders, get_vocab_size, CAPTION_MAX_LEN
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 特殊 token id 与 Flickr8k.py 的 _SPECIAL = ['<pad>','<sos>','<eos>','<unk>'] 对齐
SOS_ID, EOS_ID, PAD_ID = 1, 2, 0


def _greedy_decode(model, images, device, max_len=CAPTION_MAX_LEN):
    """自回归解码一批图（不用 teacher forcing）。返回 (B, T) 的 token id（含 <sos>），整行到 <eos> 就停。"""
    model.eval()
    batch = images.size(0)
    cur = torch.full((batch, 1), SOS_ID, dtype=torch.long, device=device)
    ended = torch.zeros(batch, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(images, cur)[:, -1]                 # (B, vocab)
            nxt = logits.argmax(-1)                            # (B,)
            nxt = torch.where(ended, torch.full_like(nxt, PAD_ID), nxt)
            cur = torch.cat([cur, nxt[:, None]], dim=1)
            ended = ended | (nxt == EOS_ID)
            if bool(ended.all()):
                break
    return cur


def eval_bleu(model, test_ds, device, max_imgs=200, seed=0):
    """取 test_ds 的固定子集做 greedy decode，报平均 sentence BLEU-1..4（每张图用其全部 5 句标注当参考）。"""
    rng = random.Random(seed)
    n = min(max_imgs, len(test_ds))
    idx = rng.sample(range(len(test_ds)), n)
    id2tok = {i: t for t, i in test_ds.vocab.items()}
    images = torch.stack([test_ds[i][0] for i in idx]).to(device)  # 只要图，忽略 __getitem__ 随机抽的那句目标
    seqs = _greedy_decode(model, images, device)
    smooth = SmoothingFunction().method1
    weights4 = [(1, 0, 0, 0), (.5, .5, 0, 0),
                (1 / 3, 1 / 3, 1 / 3, 0), (.25, .25, .25, .25)]
    bleus = [0.0] * 4
    for b, seq in enumerate(seqs):
        hyp = []
        for i in seq[1:].tolist():                              # 去掉开头的 <sos>
            t = id2tok.get(i)
            if t in ('<eos>', '<pad>'):
                break
            if t != '<unk>':
                hyp.append(t)
        if not hyp:
            continue                                            # 空句：BLEU 无定义，跳过
        refs = test_ds.caps[test_ds.ids[idx[b]]]                # 该图全部 5 句参考（已分词）
        for k, w in enumerate(weights4):
            bleus[k] += sentence_bleu(refs, hyp, weights=w, smoothing_function=smooth)
    return [s / n for s in bleus]


def train_models(model,epoch=100,device=None,data_loader = None,data_loader_test = None):
    model = model.to(device)
    PAD_ID = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(epoch):  # example number of epochs
        model.train()
        total_loss = 0
        cur = 0
        total = 0
        for img, cap in data_loader:
            img = img.to(device)
            cap = cap.to(device)
            optimizer.zero_grad()
            cap_in,cap_tar = cap[:, :-1], cap[:, 1:]
            outputs = model(img, cap_in)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), cap_tar.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = outputs.argmax(-1).reshape(-1)
            tgt = cap_tar.reshape(-1)
            real = tgt!=0
            cur += ((pred == tgt)&real).sum()
            total += real.sum()
        print(f"Epoch {epoch+1}, Train Acc: {100*cur/total if total > 0 else 0:.2f}%")
        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(data_loader):.4f}")
        model.eval()
        cur_test = 0
        total_test = 0
        with torch.no_grad():
            for img, cap in data_loader_test:
                img = img.to(device)
                cap = cap.to(device)
                cap_in,cap_tar = cap[:, :-1], cap[:, 1:]
                outputs = model(img, cap_in)
                _, predicted = torch.max(outputs.reshape(-1, outputs.shape[-1]), 1)
                cur_test += ((predicted == cap_tar.reshape(-1))&(cap_tar.reshape(-1)!=0)).sum()
                total_test += (cap_tar.reshape(-1)!=0).sum()

        print(f"Epoch {epoch+1}, Test Acc: {100*cur_test/total_test if total_test > 0 else 0:.2f}%")
        b1, b2, b3, b4 = eval_bleu(model, data_loader_test.dataset, device)
        print(f"Epoch {epoch+1}, BLEU-1/2/3/4: {100*b1:.2f} / {100*b2:.2f} / {100*b3:.2f} / {100*b4:.2f}")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, os.path.join(CKPT_DIR, f"imgcap_selftrainVit_model_epoch_{epoch+1}.pth"))
        zero_img_test(model,data_loader=data_loader_test,device=device,max_batches = 5)

def zero_img_test(model,data_loader= None,device = None,max_batches = 5):
    """零图测试。"""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if data_loader is None:
        raise ValueError("data_loader must be provided for zero image test.")

    def acc(blank):
        cur = total = 0
        with torch.no_grad():
            for i, (img, cap) in enumerate(data_loader):
                if i >= max_batches:
                    break
                img = img.to(device)
                cap = cap.to(device)
                cap_in, cap_tar = cap[:, :-1], cap[:, 1:]
                if blank:
                    img = torch.zeros_like(img)  # Replace images with zeros
                outputs = model(img, cap_in)
                pred = outputs.argmax(-1).reshape(-1)
                tgt = cap_tar.reshape(-1)
                real = tgt != 0
                cur += ((pred == tgt) & real).sum()
                total += real.sum()
        return 100 * cur / total if total > 0 else 0

    acc_real = acc(blank=False)
    acc_zero = acc(blank=True)
    print(f"Zero Image Test - Acc (real): {acc_real:.2f}%")
    print(f"Zero Image Test - Acc (blank): {acc_zero:.2f}%")
    print(f"image gain = {(acc_real - acc_zero)*100:+.2f} 个百分点")


if __name__ == "__main__":
        vocab_size = get_vocab_size()
        data_loader, data_loader_test = get_data_loaders(batch_size=64)
        model = ImageCaptioningModel(vocab_size=vocab_size, max_seq_len=CAPTION_MAX_LEN, patch_size=14,decoder_nums = 3)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("using device:", device)
        print("start training;exp0")
        train_models( model, epoch=100, device=device, data_loader=data_loader, data_loader_test=data_loader_test)

