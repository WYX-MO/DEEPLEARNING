# LayerNorm vs BatchNorm（nn 层）

两者都在做『标准化(减均值除方差)』，但归一化的『对象』不同：LayerNorm 对每个样本的最后一维归一化；BatchNorm 对每个通道跨 batch 归一化。Transformer 里序列长短不齐、每个样本独立 → 用 LayerNorm。

## 参数
| 参数 | 说明 |
|---|---|
| `LayerNorm(normalized_shape)` | 归一化的轴是最后一维 |
| `BatchNorm1d/2d(num_features)` | 归一化的轴是跨 batch 的每个通道 |

## 要点 / 坑
- BatchNorm 依赖 batch 统计，推理要用训练累计的 running 统计；小 batch 不稳。
- LayerNorm 与 batch 无关，逐样本可算 —— 这是 transformer 选它的核心原因。

## 示例（本机 torch 真实运行）
```python

import torch, torch.nn as nn
torch.manual_seed(0)
x = torch.randn(4, 3)            # 4个样本 × 3维特征
bn = nn.BatchNorm1d(3)           # 对每一列(特征)，跨4个样本归一化
ln = nn.LayerNorm(3)             # 对每一行(样本)，跨3维特征归一化
print("BN 的列(每特征跨batch) 均~0 方~1:")
print("  ", bn(x).mean(dim=0), bn(x).std(dim=0))
print("LN 的行(每样本跨特征) 均~0 方~1:")
print("  ", ln(x).mean(dim=1), ln(x).std(dim=1))

```

```text
BN 的列(每特征跨batch) 均~0 方~1:
   tensor([-5.9605e-08, -3.7253e-08,  0.0000e+00], grad_fn=<MeanBackward1>) tensor([1.1547, 1.1547, 1.1547], grad_fn=<StdBackward0>)
LN 的行(每样本跨特征) 均~0 方~1:
   tensor([ 3.9736e-08,  5.9605e-08, -3.9736e-08,  1.1921e-07],
       grad_fn=<MeanBackward1>) tensor([1.2247, 1.2247, 1.2247, 1.2247], grad_fn=<StdBackward0>)

```
