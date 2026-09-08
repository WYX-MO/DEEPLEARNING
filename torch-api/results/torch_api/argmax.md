# argmax / max（张量运算）

沿某一维找最大值。argmax 只返回『下标』；max 返回 (最大值, 下标)。不指定 dim 时会把张量摊平找全局最大。

## 参数
| 参数 | 说明 |
|---|---|
| `argmax(dim=None)` | dim 给定时沿该维；不给则全局(摊平后) |
| `max(input, dim)` | 返回 (values, indices) 两个张量 |
| `-1 = 最后一个维` | 对 (B,S,V) 用 argmax(-1) 就是在 V(词表)维挑最大 |

## 要点 / 坑
- 分类/生成里 argmax(-1) 是『取预测类别』的标准写法。
- argmax 不可导、无梯度，只能用于指标/推理，不能进 loss。

## 示例（本机 torch 真实运行）
```python

import torch
x = torch.tensor([[1., 5., 2.],
                  [9., 0., 3.]])
print("全局最大下标 argmax():", x.argmax().item())       # 3 (第4个元素=9)
print("每行最大下标 argmax(1):", x.argmax(dim=1).tolist())  # [1, 0]

vals, idx = torch.max(x, dim=1)
print("max(dim=1):", vals.tolist(), idx.tolist())

# 三维例子：outputs (B,S,V) -> 每个位置预测的词
out = torch.randn(2, 3, 5)
print("argmax(-1) 形状:", tuple(out.argmax(-1).shape))    # (2,3)

```

```text
全局最大下标 argmax(): 3
每行最大下标 argmax(1): [1, 0]
max(dim=1): [5.0, 9.0] [1, 0]
argmax(-1) 形状: (2, 3)

```
