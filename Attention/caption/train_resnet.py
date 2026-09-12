# train4caption_resnet.py
# 实验臂 D/冻结版：ImageNet 预训练 ResNet50 冻结 当特征提取器 + 重训 3 层 cross-attn decoder。
# 与 train4caption.py(臂 A/从零 ViT) 输出格式对齐，便于直接对比 BLEU / image_gain。
#
# 关键设计(为什么)：
#   1) 冻结 ResNet50 + 只训 decoder/投影 —— 小数据(6k)下微调 86M 预训练网络易过拟合；先做"干净对照"。
#   2) 归一化必须换成 ImageNet mean/std(0.485/...)，预训练权重是按它训的，归一化错则特征废。
#   3) 冻结的 ResNet50 含 BatchNorm：BN 在 train 模式会用 batch 统计并更新 running stats，
#      因此 forward 里把 encoder 切成 eval() + no_grad()，只用训练好的 running stats、不吃梯度。
#   4) 维度桥接：ResNet layer4 输出 (B,2048,7,7) -> 49 个空间 token，Linear(2048->192)+LayerNorm 后接 decoder。

import os
import sys
import random
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

_THIS = os.path.dirname(os.path.abspath(__file__))          # Attention/caption
_ATTN_DIR = os.path.dirname(_THIS)                          # Attention
sys.path.insert(0, os.path.dirname(_ATTN_DIR))              # 30-，让 import Attention 生效
CKPT_DIR = os.path.join(_ATTN_DIR, 'checkpoints')           # 权重统一存这里

from Attention.caption.datasets.Flickr8k import Flickr8kCaptions, get_vocab_size, CAPTION_MAX_LEN
from Attention.common.text_embedding import TextEmbedding
from Attention.caption.models.Decoder_block import DecoderBlock
# 复用解码/BLEU 工具(它们只要求 model.forward(images, captions) -> logits)
from Attention.caption.train_scratch import _greedy_decode, eval_bleu, SOS_ID, EOS_ID, PAD_ID

FEAT_DIM = 2048            # ResNet50 layer4 通道数
D_MODEL = 192              # 与 ImageCaptioningModel 内部一致
DECODER_NUMS = 3           # 与臂 A 一致，控制变量
FREEZE_ENCODER = True      # 冻结预训练 ResNet50（本次实验主设定）
SAVE_PREFIX = "resnet50_model"


class ResNetCaptionModel(nn.Module):
    """冻结 ResNet50 特征 + 3 层 cross-attn decoder（接口与 ImageCaptioningModel 相同）。"""

    def __init__(self, vocab_size, max_seq_len, decoder_nums=DECODER_NUMS):
        super().__init__()
        self.max_seq_len = max_seq_len

        # --- 视觉：ImageNet 预训练 ResNet50，去掉 avgpool/fc，保留到 layer4 ---
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])   # (B,2048,7,7)
        if FREEZE_ENCODER:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # --- 桥接 + 文本 + decoder(全部可训) ---
        self.img_proj = nn.Sequential(nn.Linear(FEAT_DIM, D_MODEL), nn.LayerNorm(D_MODEL))
        self.text_emb = TextEmbedding(vocab_size, D_MODEL, max_seq_len)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(D_MODEL, D_MODEL, 3) for _ in range(decoder_nums)])
        self.pred_head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, images, captions):
        # 冻结路径：encoder 恒 eval + no_grad（BN 用 running stats、不吃梯度）
        if FREEZE_ENCODER:
            self.encoder.eval()
            with torch.no_grad():
                feat = self.encoder(images)                 # (B,2048,7,7)
        else:
            feat = self.encoder(images)
        feat = feat.flatten(2).transpose(1, 2)              # (B,49,2048)
        img_feat = self.img_proj(feat)                      # (B,49,192)

        text_feat = self.text_emb(captions)                 # (B,T,192)
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len,
                                     device=images.device))
        seq_len = text_feat.shape[1]
        out = text_feat
        for blk in self.decoder_blocks:
            out = blk(img_feat, out, mask[:seq_len, :seq_len])
        return self.pred_head(out)                          # (B,T,vocab)


# --- 数据集：ImageNet 归一化（预训练权重的先决条件） ---
_IMAGENET = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_loaders(batch_size=64, num_workers=0):
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), _IMAGENET])
    test_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), _IMAGENET])
    train_ds = Flickr8kCaptions('train', transform=train_tf)
    test_ds = Flickr8kCaptions('test', transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def zero_img_gain(model, loader, device, max_batches=5):
    """真实图 vs 全零图的 teacher-forcing token acc 差(image_gain)。返回 (acc_real, acc_zero)。"""

    def acc(blank):
        cur = total = 0
        with torch.no_grad():
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
                total += real.sum()
        return 100 * cur / total if total else 0.0

    return acc(False), acc(True)


def train(model, epoch=100, device=None, data_loader=None, data_loader_test=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 与臂 A 对齐
    for epoch_i in range(epoch):
        model.train()
        total_loss = cur = total = 0
        for img, cap in data_loader:
            img, cap = img.to(device), cap.to(device)
            optimizer.zero_grad()
            cap_in, cap_tar = cap[:, :-1], cap[:, 1:]
            out = model(img, cap_in)
            loss = criterion(out.reshape(-1, out.shape[-1]), cap_tar.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(-1).reshape(-1)
            tgt = cap_tar.reshape(-1)
            real = tgt != 0
            cur += ((pred == tgt) & real).sum()
            total += real.sum()
        print(f"Epoch {epoch_i+1}, Train Acc: {100*cur/total if total else 0:.2f}%")
        scheduler.step()
        print(f"Epoch {epoch_i+1}, Train Loss: {total_loss / max(len(data_loader),1):.4f}")

        # --- eval ---
        model.eval()
        cur_t = tot_t = 0
        with torch.no_grad():
            for img, cap in data_loader_test:
                img, cap = img.to(device), cap.to(device)
                cap_in, cap_tar = cap[:, :-1], cap[:, 1:]
                out = model(img, cap_in)
                pred = out.argmax(-1).reshape(-1)
                tgt = cap_tar.reshape(-1)
                real = tgt != 0
                cur_t += ((pred == tgt) & real).sum()
                tot_t += real.sum()
        print(f"Epoch {epoch_i+1}, Test Acc: {100*cur_t/tot_t if tot_t else 0:.2f}%")

        b1, b2, b3, b4 = eval_bleu(model, data_loader_test.dataset, device)
        print(f"Epoch {epoch_i+1}, BLEU-1/2/3/4: {100*b1:.2f} / {100*b2:.2f} / {100*b3:.2f} / {100*b4:.2f}")

        ar, az = zero_img_gain(model, data_loader_test, device)
        print(f"Epoch {epoch_i+1}, real={ar:.2f}% zero={az:.2f}% image_gain={ar-az:+.2f}pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch_i,
        }, os.path.join(CKPT_DIR, f"{SAVE_PREFIX}_epoch_{epoch_i+1}.pth"))


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # 支持 `python train4caption_resnet.py 3` 跑试点
    vocab_size = get_vocab_size()
    train_loader, test_loader = make_loaders(batch_size=64)
    model = ResNetCaptionModel(vocab_size=vocab_size, max_seq_len=CAPTION_MAX_LEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    print(f"start training; exp ResNet50(frozen)+decoder, epochs={epochs}")
    train(model, epoch=epochs, device=device, data_loader=train_loader, data_loader_test=test_loader)
    print("done.")
