# Attention

> 从零实现 Attention 相关结构/模型的实验项目（参照 `../CNN` 的组织方式）。
> 代码按**实验**分子项目：`common/`（跨实验共享块）+ `caption/` / `gpt/` / `vit_cls/` 三个实验。

## 目录结构

```
Attention/
├── common/                  # 跨实验共享的可复用块
│   ├── attention.py         # SelfAttention / MultiHeadAttention
│   ├── feedforward.py       # FFN
│   ├── RotaryEmbedding.py   # RoPE
│   ├── Tokenizer.py         # 分词器（词级 / zh BPE 脚手架）
│   ├── text_embedding.py    # 词嵌入 + 可学习位置嵌入
│   └── vision/              # 视觉编码侧共享块（caption 与 vit_cls 共用）
│       ├── Vit.py
│       ├── PatchEmbeding.py
│       ├── CLSToken.py
│       ├── PositionEncoding.py
│       └── Transformer_block.py
├── caption/                 # 实验一：图像描述（Flickr8k）
│   ├── models/              # ImageCaptioningModel / CrossAttention / Decoder_block
│   ├── datasets/Flickr8k.py
│   ├── train_scratch.py     # 臂 A：从零 ViT + Transformer 解码器
│   ├── train_resnet.py      # 臂 D：冻结 ResNet50 + 解码器
│   ├── eval.py              # 独立评估：BLEU-1..4 / image_gain
│   └── tools/draw_model_structure.py
├── gpt/                     # 实验二：字符级 GPT（shakespeare）
│   ├── models/              # GPT / GPT_block
│   ├── datasets/shakespeare.py
│   └── train.py  test.py  generate.py
├── vit_cls/                 # 实验三：ViT 分类（CIFAR-10）
│   ├── datasets/cifar10.py
│   └── train.py
├── experiments/             # 受控实验记录：exp_main.md 索引 + 每次实验一个 md + experiments.csv
├── checkpoints/             # 训练权重（*.pth，gitignore）
├── results/                 # 图表 / 日志输出
├── data/                    # 原始数据集（gitignore，可随时重新下载）
├── AttentionExperiment.md   # 项目总览：目标 / 模型演进 / 结果表 / Key Learnings
├── 进组实验规划书_*.md       # 图像描述方向的总规划
└── requirement.txt          # 依赖（按 dl 环境实际版本固定）
```

## 运行方式

入口脚本都会把项目父目录（`30-`）注入 `sys.path`，所以可以直接按文件运行，不依赖当前目录：

```bash
/Users/liuzejiang/miniconda3/envs/dl/bin/python Attention/gpt/train.py
/Users/liuzejiang/miniconda3/envs/dl/bin/python Attention/caption/train_scratch.py
/Users/liuzejiang/miniconda3/envs/dl/bin/python Attention/vit_cls/train.py
```

也可以按模块运行（在 `30-` 目录下）：

```bash
python -m Attention.gpt.train
```

约定：

- 权重统一存 `Attention/checkpoints/`，原始数据统一放 `Attention/data/`（脚本内用绝对路径定位）。
- 每个模型文件底部保留 `if __name__ == "__main__":`，做一次前向形状自检，可单独 `python -m` 跑。
- `experiments/attention_demo.py` 是 SelfAttention 的最小可跑示例。

## 实验规范（沿用 CNN 的做法）

- 每次只改一个变量（模型 / 增广 / BN / …），其余训练配置固定，便于对照。
- 每次跑批在 `experiments/experiments.csv` 记一行（含 git commit 便于回溯）。
- 每个实验写独立 md，最终在 `experiments/exp_main.md` 建立索引汇总。
