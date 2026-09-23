# DEEPLEARNING — A From-Scratch Deep Learning Research Log

> **Every model in this repository is written by hand in PyTorch — no `transformers`, no model library.** Attention / ViT / GPT / ResNet / the multimodal stack are all built from primitives. Pretrained backbones appear only where they are the *object of study*, as controlled comparison arms against the from-scratch implementations. What is being studied here is not "how to call a model", but *what each architectural component actually buys you, and how much*.
>
> 本仓库所有模型均为手写 PyTorch 实现：Attention / ViT / GPT / ResNet / 多模态栈全部从原语搭起，不依赖 `transformers`、不调用现成模型库。预训练骨干只作为**对照实验臂**出现——它本身就是被研究的对象，用来回答"从零训练到底差多少"。研究对象不是"怎么调库"，而是**每一个结构改动到底换来了多少收益、代价是什么**。

---

## Abstract

This repository contains four self-contained research tracks, each organized as a **controlled experiment log**: one hypothesis, one variable changed per run, one row appended to an `experiments.csv` ledger with its git commit hash, and a written conclusion — including the ones that failed.

报告的核心主张有三条：

1. **小数据下从零训练的图像编码器学不到视觉表示。** 在 Flickr8k（6k 图）上端到端训练自写 ViT + cross-attention 解码器，逐图生成的 BLEU-1 为 **51.2**，而"在所有测试图上复读同一句万能模板"的 BLEU-1 是 **52.0**。模型跑输了复读机。
2. **在 CIFAR-10 上，结构改动的收益可以被逐项归因。** 从朴素 CNN 到 ResNet，测试准确率 73.98% → 88.27%，每一个中间步骤（增广 +4.19pt、BatchNorm +2.16pt、残差 +5.60pt、卷积核 5×5→3×3 +2.34pt）都由一次只改一个变量的对照实验给出。
3. **负结果和无效实验被完整保留。** 两个实验被明确标记为 `invalid`（增广代码没真正作用到图像上）；加深但没加残差的网络几乎拿不到任何收益（+0.34pt）；三国语料的模型被记录为"复读语料而非生成"。这些条目没有被删除，因为它们比成功案例更能说明实验纪律。

---

## Repository Map

| Directory | Track | Task | Status |
|---|---|---|---|
| `CNN/` | Supervised CV | CIFAR-10 classification, CNN → ResNet | 7 valid runs |
| `Attention/gpt/` | Sequence modeling | Small GPT from scratch (Shakespeare / 三国演义剧本) | 2 corpora trained |
| `Attention/caption/` | Vision-Language | Flickr8k image captioning, 4-arm controlled study | arm A done, arm D pilot |
| `Attention/vit_cls/` | Supervised CV | ViT on CIFAR-10 | 2 runs |
| `Attention/common/` | Infrastructure | Attention / FFN / RoPE / RMSNorm / SwiGLU / Tokenizer / ViT blocks | shared blocks |
| `Molmo/` | Multimodal | Molmo & PixMo replication (MiniCLIP + connector + VLM) | in progress |
| `cs321N/` | Coursework | Stanford CS231n assignments 1–3 + full lecture notes | complete |
| `papers/` | Reading | Paper PDFs & notes (Attention, ResNet, U-Net) | ongoing |
| `torch-api/` | Tooling | Personal PyTorch API quick-reference / knowledge base | complete |
| `AI/`, `logger.py`, `train.log` | Roadmap | Research roadmap, shared logging utility, run logs | — |

---

## Track 1 · CIFAR-10: attributing every architectural gain

**Hypothesis.** A plain CNN is limited by optimization, not capacity; adding the three canonical fixes (augmentation, normalization, residual connections) should each contribute a measurable, separable gain.

**Protocol.** Fixed seed, fixed hyperparameters (`lr=0.001`, `batch=64`, `10 epochs`), one variable changed per run, every run appended to `CNN/experiments/experiments.csv`.

Δ columns: *vs. control* compares against the run that isolates the same variable (e.g. Exp007.2 vs. Exp006 both use otherwise identical settings, differing only in kernel size); *vs. baseline* compares against the plain CNN (Exp001).

| # | Model | Change | Test Acc | Δ vs. control | Δ vs. baseline |
|---|---|---|---|---|---|
| Exp001 | CNN | — | 73.98% | — | — |
| Exp004 | CNN | + hflip / rotation / crop / jitter | 78.17% | **+4.19** | +4.19 |
| Exp005 | CNN | + BatchNorm | 80.33% | **+2.16** | +6.35 |
| Exp006 | ResNet-18 | + residual connections | 85.93% | **+5.60** | **+11.95** |
| Exp007 | deeper plain CNN | 5×5 conv, more layers | 74.32% | — | +0.34 |
| Exp007.1 | deeper plain CNN | 5×5 → 3×3 conv | 81.05% | **+6.73** | +7.07 |
| Exp007.2 | ResNet | 5×5 → 3×3 conv | **88.27%** | **+2.34** | **+14.29** |

**Findings.**

- **Depth alone buys nothing; depth plus residual connections buys +11.95pt.** The deeper plain CNN (74.32%) sits at +0.34pt over the shallow baseline (73.98%) — i.e. within run-to-run noise for a 10-epoch budget. Put residual connections at substantially the same depth and it jumps to 85.93%. The bottleneck is optimization, not representational capacity.
- **Kernel size is a free win at this scale**: 5×5 → 3×3 recovers +6.73pt on the plain CNN and +2.34pt on ResNet, at lower parameter count.
- **The cost of residual connections is real**: ~20× wall-clock per run for +5.60pt. Recorded as a caveat rather than glossed over.

**Invalid runs (kept deliberately).** `Exp002-invalid` and `Exp003-invalid` were discarded because the augmentation transform was written inside `train.py` and never actually applied to the tensors, and `lr=0.001` was too high for the augmented regime. They are retained in the ledger with the reason for invalidation. *An experiment that was never valid is not evidence — recording why it was invalid is.*

> See `CNN/CIFAR10Experiment.md` · `CNN/experiments/experiments.csv` · `CNN/models/`

---

## Track 2 · Attention from Scratch

All primitives hand-written in `Attention/common/`:

```
attention.py          SelfAttention / MultiHeadAttention / MHAttnWithCache (KV cache)
feedforward.py        FFN
SwiGLU.py  RMSNorm.py RotaryEmbedding.py     # modern LLM components
Tokenizer.py  text_embedding.py  word_vectors.py
vision/               Vit.py / PatchEmbeding.py / CLSToken.py / PositionEncoding.py / Transformer_block.py
```

### 2.1 Small GPT from Scratch — and a failure worth writing down

Two corpora, deliberately tokenized at different granularities to exercise both paths of `Attention/common/Tokenizer.py`: Shakespeare at **character level** (hand-built char vocabulary, `vocab=65`), and a 《三国演义》 screenplay corpus (dialogue + `剧情:` stage directions) at **subword level** via a sentencepiece model.

| Corpus | Tokenization | Epochs | Final train loss | Val loss | Perplexity |
|---|---|---|---|---|---|
| Shakespeare | char-level (vocab 65) | 30 | 0.5156 | 0.5080 | **1.6620** |
| 三国演义剧本 | sentencepiece subword | 100 | 0.0547 | 0.0850 | **1.0887** |

**Negative result #1 — the corpus was memorized, not modeled.** The first run on the 三国 corpus (`Exp007-Minigpt-sanGuo`) is recorded in the ledger as a failure: the model **背书 / verbatim-repeats** entire passages instead of generating, with the suspected cause being pathological token-frequency concentration (the corpus is small and dominated by a handful of character names). Perplexity 1.09 on a corpus this size is not a skill signal — it is a memorization signal. The run is recorded rather than quietly retrained.

**Negative result #2 — a tokenizer bug found by measuring, not by eyeballing.** The available `zh.model` sentencepiece model had been trained on a single chapter, and measured at **27% `<unk>`** on the full corpus. That number is documented in `Attention/gpt/datasets/sanguo.py` itself, because a tokenizer that maps a quarter of the text to `<unk>` will silently cap the achievable perplexity no matter how well the model trains. Both conclusions are visible in the repository instead of being engineering trivia lost to history.

> See `Attention/gpt/` · `somelogs.md`

### 2.2 Image Captioning on Flickr8k — *the main result of this repository*

**Setup.** 6,000 train / 1,000 dev / 1,000 test images, 5 captions each. Hand-written 12-layer ViT (`d_model=192`, `patch=14`, 224×224 → 256 tokens) → 3-layer cross-attention decoder. AdamW `lr=1e-4`, StepLR, 100 epochs, `batch=64`, `CrossEntropyLoss(ignore_index=0)` with `<pad>=0`. Teacher forcing at train, greedy decoding at inference.

**The finding.** A single diagnostic probe — *feed the model an all-zero image* — collapses the model: on 6 test images it emits the same sentence, `a man is standing on a beach .`, every time. Quantitatively:

| Probe | Value | Reading |
|---|---|---|
| Greedy BLEU-1/2/3/4 (200 test images) | 51.2 / 27.3 / 14.5 / 9.1 | headline numbers |
| **"Repeat one default sentence on every image" BLEU-1** | **52.0** | **beats the model** |
| Teacher-forcing token acc, real images | 39.2% | |
| Teacher-forcing token acc, **all-zero images** | 36.2% | |
| → `image_gain` (the difference) | **+3.0pt** | image contributes almost nothing |
| Pure language prior (always predict `a`) | 12.2% | a language-only lower bound |

**Three conclusions, all quantifiable:**

1. **The model is grammatically fluent and visually empty.** It reproduces templates correctly and fails on specific entities — a kissing couple becomes "standing on sidewalk", rock climbing becomes "swing". Visual gradient is swamped by function words.
2. **BLEU-1 is a severely inflated metric here.** A constant string scores 52.0, above the model's 51.2. Any captioning result that reports only BLEU-1 is uninterpretable. BLEU-4 (9.1) and rare-noun hit rate are the honest metrics.
3. **The diagnosis costs almost nothing.** `all-zero-image collapse + default-sentence baseline` is a two-run probe that localizes "the model is reciting a language prior" without any attention visualization or gradient analysis.

**Arm D pilot (frozen ImageNet ResNet-50 + retrained decoder), 3 epochs only:**

| Epoch | Test token acc | BLEU-1/2/3/4 | image_gain |
|---|---|---|---|
| 1 | 26.08% | 54.32 / 24.83 / 12.33 / 8.39 | +1.08pt |
| 2 | 29.04% | 48.59 / 24.85 / 11.95 / 7.51 | +0.25pt |
| 3 | 31.57% | 53.15 / 29.72 / 15.04 / **9.29** | +1.99pt |

**Reading, stated carefully.** After **3 epochs**, frozen ResNet-50 features already reach BLEU-4 **9.29** — comparable to what the from-scratch ViT needed **100 epochs** to reach (9.1). But `image_gain` has *not* improved (still ≤ +2.0pt), i.e. better BLEU without better image conditioning. Early-stopping on BLEU alone would have declared victory; the conditioning probe says the decoder has not yet learned to attend to the image. This is exactly the tension the evaluation protocol was designed to expose, and it is why token accuracy and BLEU are both reported.

**Designed but not yet run** — the 4-arm controlled study (one variable per arm: A from-scratch / B frozen pretrained ViT / C fully fine-tuned / D frozen ResNet-50), with the research question *"at 6k images, does fine-tuning a pretrained ViT beat freezing it?"* deliberately left unanswered, to be decided by experiment rather than assumed.

> See `Attention/in.md` (full experimental protocol) · `Attention/experiments/` (ledger) · `Attention/results/pilot_resnet3.log`

### 2.3 ViT on CIFAR-10

| Run | Config | Epochs | Train Acc | Test Acc | Note |
|---|---|---|---|---|---|
| Exp001 | hflip + random crop, lr=1e-4 | 10 | 94.59% | 76.74% | severe overfitting: train still climbing, test plateaued |
| Exp002 | + regularization | 50 | — | **83.10%** | +6.36pt |

Recorded alongside the CNN track for reference — noting that the two are **not** directly comparable as configured: the ViT run used 50 epochs against the CNN track's 10, and still landed below it (83.10% vs. 88.27%). The more informative number is the ViT's train/test gap at Exp001 (94.59% / 76.74%) — the overfitting signature that `Attention/in.md` attributes to ViT's lack of the CNN's built-in inductive bias, which small datasets cannot pay to learn.

---

## Track 3 · Molmo & PixMo Replication

Reimplementing the *Molmo* recipe — an **open-weights, open-data** VLM trained from scratch rather than distilled from a proprietary model — at a scale that fits a single 8 GB consumer GPU.

```
Molmo/
├── MiniCLIP/            two-tower symmetric InfoNCE contrastive pretraining on Flickr8k
│   ├── models/clip.py   d_model 192 → shared 256-d space, logit_scale clamped at 100 (CLIP recipe)
│   └── train.py
└── models/
    ├── connector.py         VisionProjector = Linear(vision_dim → llm_dim) + LayerNorm
    └── multimodal_model.py  ViT(cls=False) → VisionProjector → concat[visual, text] → ModernGPT
```

The multimodal path is the canonical late-fusion / token-concatenation recipe — image features projected into the LLM's embedding space and prepended to the text sequence — but built entirely on the hand-written stack: `VisionTransformer` from `Attention/common/vision/` and `ModernGPT` (pre-norm, RMSNorm, RoPE, SwiGLU, KV cache) from `Attention/gpt/models/`. No `transformers` dependency anywhere.

**Status:** MiniCLIP trained; connector and `MultimodalModel` forward path implemented; end-to-end VLM training in progress.

> See `Molmo/Molmo.md` · `Molmo/`

---

## Track 4 · CS231n

Assignments 1–3 plus a complete set of hand-written Chinese lecture notes (`cs321N/CS231N.md`) spanning: kNN / linear classifiers / softmax · regularization · SGD, Momentum, RMSProp, Adam, AdamW · backpropagation · convolution, receptive fields, pooling · BatchNorm, Dropout, ResNet, Kaiming initialization · RNN, LSTM · attention and the Transformer · ViT, DETR, segmentation, CAM, saliency · video (two-stream, I3D) · distributed training · self-supervised learning (rotation, jigsaw, MAE, SimCLR) · generative models (GAN, diffusion) · 3D representations · CLIP / CoCa / Flamingo / Molmo / SAM.

Tracked as the theoretical substrate the other tracks are tested against — every method implemented elsewhere in this repository has a corresponding derivation here.

---

## Methodology / 实验规范

The constraints below are enforced across all tracks, and are the reason the results above can be compared to each other.

1. **One variable per experiment.** Augmentation, BatchNorm, residual connections, and kernel size were each changed in isolation. A run that changes two things is marked invalid.
2. **Every run gets a ledger row.** `experiments/*.csv` records `id, git_commit, model, augmentation, lr, batch_size, epochs, train_acc, test_acc, note`. The commit hash is mandatory — a number that cannot be traced back to a code state is not a result.
3. **Invalid experiments are recorded as invalid, with the reason.** Not deleted, not silently rerun.
4. **Frozen evaluation protocol.** Fixed seed, fixed 200-image BLEU subset, model selection on dev, test reported once. Test is not used as a tuning signal.
5. **Negative results are reported as results.** "Extra depth without residual connections adds +0.34pt, i.e. nothing" and "perplexity 1.09 means memorization, not skill" are conclusions, not embarrassments.
6. **Metrics are chosen to be falsifiable.** `image_gain` (real-image vs. zero-image token accuracy) exists specifically because BLEU can be high while the model ignores the input entirely.
7. **Compute budget is an explicit design constraint.** Self-training an ImageNet-scale encoder on one 8 GB laptop GPU is ~3–4 weeks and would, per the ViT paper, not necessarily beat a same-budget ResNet — so the budget goes to *evaluation and attribution* instead of re-deriving pretraining.

---

## Environment

```
torch==2.5.1  torchvision==0.20.1  torchaudio==2.5.1
numpy==2.4.6  pillow==12.3.0  ipykernel==7.3.0
```

Entry scripts inject the repository root into `sys.path`, so any file can be run directly or as a module:

```bash
python Attention/gpt/train.py
python -m Attention.vit_cls.train
```

Conventions: checkpoints → `*/checkpoints/`, datasets → `*/data/` (both gitignored), every model file keeps a forward-shape self-test under `if __name__ == "__main__":`.

---

## Roadmap

- [ ] Complete the 4-arm captioning study (arms B/C/D) and report `mean ± std` over 3 seeds
- [ ] Use all 5 captions per image (≈5× data) — currently one random caption per epoch
- [ ] Scheduled sampling to attack exposure bias; label smoothing 0.1 to suppress template over-confidence
- [ ] Cross-attention visualization mapped back onto images
- [ ] Finish Molmo end-to-end VLM training; evaluate against the MiniCLIP zero-shot baseline
- [ ] Reproduce the CIFAR-10 crossover: at what dataset size does from-scratch training overtake frozen pretrained features?
- [ ] Generative track (GAN / VAE / diffusion) from scratch

---

## References

- Dosovitskiy et al., *An Image is Worth 16×16 Words* (ViT, 2021)
- Touvron et al., *Training data-efficient image transformers & distillation* (DeiT, 2021)
- Xu et al., *Show, Attend and Tell* (2015)
- Anderson et al., *Bottom-Up and Top-Down Attention* (2018)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021)
- He et al., *Deep Residual Learning for Image Recognition* (ResNet, 2016)
- He et al., *Rethinking ImageNet Pre-training* (2019)
- Vaswani et al., *Attention Is All You Need* (2017)
- Deitke et al., *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models* (2024)

---

<p align="center"><sub>Independent research log · all models implemented from scratch · negative results included</sub></p>
