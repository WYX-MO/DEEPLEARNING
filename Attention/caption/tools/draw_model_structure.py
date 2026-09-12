# draw_model_structure.py
# 绘制 ImageCaptioningModel（ViT 编码器 + Transformer 解码器）结构图，带张量形状标注
# 运行: python draw_model_structure.py  ->  生成 model_structure.png

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# macOS 中文 + 负号显示
matplotlib.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
    'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# B = batch, L = 句子长度, H = 头数, |V| = 词表大小
# demo 数值: B=2, L=10, H=3, d_model=192, patch=(32/4)^2=64


def box(ax, cx, cy, w, h, text, fc, ec, fs=9):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            zorder=3, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=20, linewidth=1.8,
                                 color='0.25', zorder=1))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.25, label, ha='center',
                va='center', fontsize=9, color='0.15', zorder=4,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.85, pad=0.4))


fig, ax = plt.subplots(figsize=(9, 13.5), dpi=160)
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.4)
ax.axis('off')

# ---- 输入层 ----
box(ax, 2.5, 14.3, 3.0, 1.1, '图像输入 images\n[B, 3, 32, 32]',
    '#e3f0fd', '#2f6fbb')
box(ax, 7.5, 14.3, 3.0, 1.1, '文本输入 captions（词索引）\n[B, L]',
    '#e3f0fd', '#2f6fbb')

# ---- 左：ViT 编码器 ----
enc_text = ('PatchEmbedding：切 8×8 patch\n'
            'Linear → [B, 64, 192]\n'
            '+ CLS token · 位置编码\n'
            '→ [B, 65, 192]\n'
            '12 × TransformerBlock\n'
            '(self-attn → 残差+LN → FFN → 残差+LN)\n'
            '→ [B, 65, 192]\n'
            '去掉 CLS token\n'
            'image_features  [B, 64, 192]')
box(ax, 2.5, 10.3, 3.6, 5.2, enc_text, '#fff3e0', '#d86a10', fs=8.5)

# ---- 右：文本嵌入 ----
box(ax, 7.5, 11.4, 3.4, 2.0,
    'TextEmbedding\n词嵌入 + 可学习位置嵌入\ntext_features  [B, L, 192]',
    '#fff3e0', '#d86a10', fs=8.5)

# ---- 解码器 ----
dec_text = ('① Self-Attention：Q = K = V = 文本\n'
            '因果 mask（下三角 [L,L]，挡未来）→ 分数 [B, H, L, L]\n'
            '残差 + LayerNorm → [B, L, 192]\n'
            '② Cross-Attention：Q = 文本，K = V = 图像\n'
            '图像 token [B, 64, 192] → 分数 [B, H, L, 64]\n'
            '残差 + LayerNorm → [B, L, 192]\n'
            '③ FeedForward → 残差 + LayerNorm → [B, L, 192]')
box(ax, 5.0, 5.1, 8.2, 4.0, dec_text, '#e6f6f6', '#1b7f7f', fs=8.5)
ax.text(5.0, 7.0, 'DecoderBlock（文本解码器）', ha='center', fontsize=10.5,
        fontweight='bold', color='#1b7f7f')

# ---- 输出层 ----
box(ax, 5.0, 2.0, 4.2, 1.2, 'pred_head：Linear\n192 → |V|',
    '#e8f7e6', '#2e8b34')
box(ax, 5.0, 0.7, 4.6, 1.0, 'logits（每位置 = 下一个词的分数）\n[B, L, |V|]',
    '#e8f7e6', '#2e8b34')

# ---- 连线 ----
arrow(ax, 2.5, 13.7, 2.5, 12.9)          # 图像 → ViT
arrow(ax, 7.5, 13.7, 7.5, 12.4)          # 文本 → Embedding
arrow(ax, 2.5, 7.7, 3.2, 7.15, 'K, V')   # 图像 token → Decoder
arrow(ax, 7.5, 10.4, 6.9, 7.2,  'Q')     # 词 token → Decoder
arrow(ax, 5.0, 3.1, 5.0, 2.6)            # Decoder → pred_head
arrow(ax, 5.0, 1.4, 5.0, 1.2)            # pred_head → logits

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'results', 'model_structure.png')
fig.savefig(_OUT, bbox_inches='tight')
print(f'saved -> {_OUT}')
