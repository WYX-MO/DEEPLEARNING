# Experiment 2 — Augmentation & ResNet Baseline

> 在 Exp001 baseline 基础上做三组改进，汇总为一份对照报告。
> 结果取自 `experiments.csv`（Exp001 / Exp004 / Exp005 / Exp006）。

## 1. Experiment Goal

1. 数据增广（水平翻转 / 随机旋转 / 随机裁剪 / 颜色抖动）→ 提高泛化、缓解过拟合；
2. 在 CNN 中加入 BatchNorm 层 → 观察准确率变化；
3. 搭建 ResNet baseline → 对比与 CNN 的准确率差异。

各跑批只改「增广 / BatchNorm / 网络结构」，其余训练配置固定不变（见 §6）。
四组为**逐级叠加**：Exp004 在 Exp001 上加增广，Exp005 再加 BN，Exp006 换成 ResNet（增广仍开启），因此每步相对上一步才是干净的增量对比。

## 2. Environment

- Python: 3.12（dl conda 环境，osx-arm64）
- PyTorch: 2.5.1
- torchvision: 0.20.1
- Device: CPU（`train.py` 逻辑为 CUDA 不可用时走 CPU；本机 MPS 可用但未启用）
- GPU: 无（Apple Silicon，无 CUDA）

## 3. Dataset

- Dataset: CIFAR-10
- Train samples: 50,000
- Test samples: 10,000
- Image size: 32 × 32
- Channels: 3
- Classes: 10

## 4. Data Preprocessing

**Baseline（Exp001）**：无训练期增广，仅归一化。

**增广组（Exp004 / Exp005 / Exp006）**：

```python
transform_train = transforms.Compose([
    transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip(p=1)], p=0.5),
    transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=15)], p=0.5),
    transforms.RandomApply(transforms=[transforms.RandomCrop(32, padding=4)], p=0.5),
    transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# test（两种都无增广）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

## 5. Model

### (a) MyCNN（baseline，对应 Exp001）
Input: 3 × 32 × 32

- Conv2d 3 → 32, kernel=5, padding=2
- ReLU + MaxPool2d 2×2
- Conv2d 32 → 64, kernel=5, padding=2
- ReLU + MaxPool2d 2×2
- Conv2d 64 → 128, kernel=5, padding=2
- ReLU + MaxPool2d 2×2
- Flatten（128×4×4 = 2048）
- Linear 2048 → 512, ReLU
- Linear 512 → 10

Exp004 = 同一结构 + 增广；**无 BatchNorm**。

### (b) MyCNN + BatchNorm（对应 Exp005）
与 (a) 结构相同，每个 Conv2d 后加一层 `BatchNorm2d`（bn1/bn2/bn3，通道数 32/64/128）。

### (c) MyResNet baseline（对应 Exp006）
`My_resnet(MyResidualBlock, [2, 2, 2, 2])`，CIFAR-10 定制版：

- stem：Conv2d 3 → 64, kernel=3, pad=1 + BatchNorm + ReLU
- 4 个 stage：64 / 128 / 256 / 512 通道，stride 1 / 2 / 2 / 2，每层 `[2, 2, 2, 2]` 个残差块
- global average pool（avg_pool 4）+ Linear 512 → 10

`MyResidualBlock`：两个 Conv2d + BatchNorm + ReLU，shortcut 恒等映射，stride≠1 或通道变化时用 1×1 conv 下采样。注意：残差块内用的是 **5×5** conv（与标准 ResNet-18 的 3×3 不同），属自定义变体。

## 6. Training Configuration

所有跑批一致：
Batch size 64 / Epochs 10 / Optimizer Adam / Learning rate 0.001 / Loss CrossEntropyLoss / seed 42

## 7. Results

| 实验 | 模型 | 增广 | BatchNorm | Train Acc | Test Acc | vs Exp001 |
|---|---|---|---|---|---|---|
| Exp001 | MyCNN | ✗ | ✗ | ~95% | 73.98% | — |
| Exp004 | MyCNN | ✓（4 种） | ✗ | ~76.7% | 78.17% | +4.19 |
| Exp005 | MyCNN | ✓ | ✓ | ~78.5% | 80.33% | +6.35 |
| Exp006 | **MyResNet** | ✓ | ✓（残差块内） | 86.21% | 85.93% | +11.95 |

注：`experiments.csv` 中 Exp006 的 model 列误写为 `cnn`，实际为 MyResNet（提交 `486a38d` 已将 `train.py` 主流程切换为 `My_resnet`）。

## 8. Observation

- **增广（+4.19）**：Test 从 73.98 → 78.17。Exp001 无增广时 Train ~95% ≫ Test 74%，明显过拟合；加增广后训练期准确率反而降到 ~76.7%（训练时随机扰动更难拟合），但 Test 上升，说明增广有效抑制了过拟合。
- **BatchNorm（+2.16，对比 Exp004）**：Test 78.17 → 80.33，带来稳定小幅提升（收敛更快、数值更稳）。
- **ResNet（+11.95，对比 Exp001）**：提升最大，Test 85.93，且 Train 86.21 ≈ Test 85.93，几乎无过拟合——残差连接缓解了深层网络训练退化。
- **注意变量隔离**：Exp006 仍跑在增广预处理下，不是「纯 ResNet 无增广 baseline」，相对 Exp001 的 +11.95 是「ResNet 结构 + 增广」叠加结果。与同样开启增广+BN 的 Exp005 相比，**结构的净贡献 ≈ 85.93 − 80.33 = +5.60**。
- **代价**：ResNet 训练耗时约为 CNN 的 20 倍（csv 备注），后续需优化（数据加载 `num_workers`、设备改用 MPS、训练策略）。
- 本实现残差块用 5×5 conv，偏离标准 ResNet-18（3×3），对比他人结果时需说明。

## 9. Conclusion / Next Experiment

### ResNet解决的问题：
### 理论上深网络应该至少能复制浅网络的效果，但实际优化器很难让新增的非线性层学成 identity mapping

- 增广 + BN + ResNet 三级改进逐级有效，ResNet baseline 在 CIFAR-10 上可达 ~86% Test。
- 下一步：
  - 隔离变量：补跑「MyResNet 无增广」与「MyCNN(无 BN) + ResNet」对应跑批，量化结构与增广各自贡献；
  - 向标准 ResNet-18 对齐（3×3 conv、更长 epoch、学习率调度）；
  - 优化训练速度（`DataLoader(num_workers=...)`、`device="mps"`），缩短 ~20× 的跑批时间。
