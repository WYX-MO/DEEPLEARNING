# masked_fill + tril / triu（张量运算）

masked_fill(mask, value) 把 mask 为 True 的位置替换成 value。配 tril(下三角)能做出『因果 mask』：下三角保留、右上角填 -inf，softmax 后那些位置权重=0。

## 参数
| 参数 | 说明 |
|---|---|
| `masked_fill(mask, value)` | mask 是与原张量可广播的布尔张量 |
| `tril(m)` | m×m 下三角（含对角线）为 1，其余 0 |
| `为什么填 -inf` | softmax(exp(-inf)=0) → 那些位置注意力精确为 0 |

## 要点 / 坑
- mask 一定要能广播到 attention_scores：2D [L,L] 会被当 [1,1,L,L] 用。
- mask 填 -inf 必须在 softmax 之前才有效。

## 示例（本机 torch 真实运行）
```python

import torch
torch.manual_seed(0)
L = 4
mask = torch.tril(torch.ones(L, L))          # 下三角=1
scores = torch.rand(L, L)
masked = scores.masked_fill(mask == 0, float('-inf'))
print("tril mask:\n", mask)
print("masked 后的分数:\n", masked)
print("softmax 后（被挡位置=0）:\n", torch.softmax(masked, dim=-1))

```

```text
tril mask:
 tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
masked 后的分数:
 tensor([[0.4963,   -inf,   -inf,   -inf],
        [0.3074, 0.6341,   -inf,   -inf],
        [0.4556, 0.6323, 0.3489,   -inf],
        [0.0223, 0.1689, 0.2939, 0.5185]])
softmax 后（被挡位置=0）:
 tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4191, 0.5809, 0.0000, 0.0000],
        [0.3234, 0.3859, 0.2907, 0.0000],
        [0.1956, 0.2265, 0.2566, 0.3213]])

```
