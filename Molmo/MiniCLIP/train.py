# train.py
# MiniCLIP 训练：Flickr8k 图文对，双塔对比学习（对称 InfoNCE）。

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # Molmo/MiniCLIP
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_DIR)))   # 30-

import torch
import torch.nn.functional as F

from Attention.caption.datasets.Flickr8k import (
    get_data_loaders, get_vocab_size, CAPTION_MAX_LEN,
)
from Molmo.MiniCLIP.models.clip import CLIP
from logger import get_logger

CKPT_DIR = os.path.join(_THIS_DIR, 'checkpoints')

# 超参。max_seq_len 必须等于数据集的 CAPTION_MAX_LEN，
# 否则位置编码表长度和 caption 实际长度对不上。
D_MODEL = 192 
EMBED_DIM = 192     # 必须等于 VisionTransformer 的 d_model（CLS 特征维度），否则 linear_img 维度不匹配
SAME_DIM = 256      # 图文各自投影到同一空间后的维度
BATCH_SIZE = 64     # CLIP 的 batch 越大，负样本越多，损失越有意义
EPOCHS = 30
LR = 1e-4 


def clip_loss(similarity, logit_scale):
    """对称 InfoNCE。similarity 已是归一化余弦 (B, B)，对角线才是正样本。

    行表示"这张图去匹配哪句话"，列表示"这句话去匹配哪张图"，
    两个方向各算一次交叉熵再平均——这就是 CLIP 的 loss。
    """
    n = similarity.size(0)
    labels = torch.arange(n, device=similarity.device)      # 正样本在对角线
    logits = logit_scale.exp().clamp(max=100) * similarity  # 温度缩放，上限 100（CLIP 原文）
    loss_i = F.cross_entropy(logits, labels)                # 图 → 文
    loss_t = F.cross_entropy(logits.t(), labels)            # 文 → 图
    return (loss_i + loss_t) / 2


def retrieval_at_1(similarity):
    """batch 内 R@1：图 i 的相似度行取 argmax，看是否落在第 i 句文本上。"""
    n = similarity.size(0)
    labels = torch.arange(n, device=similarity.device)
    i2t = (similarity.argmax(dim=1) == labels).float().mean()   # image → text
    t2i = (similarity.argmax(dim=0) == labels).float().mean()   # text → image
    return i2t.item(), t2i.item()


def evaluate(model, data_loader, device):
    model.eval()
    i2t_sum = t2i_sum = 0.0
    total = 0
    with torch.no_grad():
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)
            similarity = model(images, captions)
            i2t, t2i = retrieval_at_1(similarity)
            n = similarity.size(0)
            i2t_sum += i2t * n
            t2i_sum += t2i * n
            total += n
    return i2t_sum / total, t2i_sum / total


def train_models(model, device, data_loader, data_loader_test,
                 epochs=EPOCHS, lr=LR, ckpt_dir=CKPT_DIR):
    logger = get_logger()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, captions in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            similarity = model(images, captions)                    # (B, B)
            loss = clip_loss(similarity, model.logit_scale)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        i2t, t2i = evaluate(model, data_loader_test, device)
        # logit_scale 会自己长大 → 温度变低 → 分布变尖锐，这是 CLIP 正常的学习现象
        scale = model.logit_scale.exp().item()
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | "
            f"test R@1 i2t {100 * i2t:.2f}% / t2i {100 * t2i:.2f}% | "
            f"logit_scale {scale:.1f}"
        )

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, os.path.join(ckpt_dir, f"miniclip_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("using device:", device)
    
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    vocab_size = get_vocab_size()
    print("vocab_size:", vocab_size)

    model = CLIP(vocab_size, d_model=D_MODEL, embed_dim=EMBED_DIM,
                 same_dim=SAME_DIM, max_seq_len=CAPTION_MAX_LEN)
    train_models(model, device, train_loader, test_loader)
