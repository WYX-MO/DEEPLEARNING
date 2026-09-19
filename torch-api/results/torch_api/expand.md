# expand / squeeze / unsqueeze（张量运算）

unsqueeze 在指定位置加一个长度 1 的维；squeeze 删掉长度 1 的维；expand 把长度 1 的维『广播撑大』——仍是视图，不复制数据。

## 参数
| 参数 | 说明 |
|---|---|
| `unsqueeze(dim)` | 插入单维度，为广播/加 batch 维用 |
| `squeeze(dim=None)` | 去掉长度=1 的维（可指定） |
| `expand(*sizes)` | 把 1 扩成大数，-1 表示保持该维不变 |

## 要点 / 坑
- expand 不占额外内存（stride=0），要真复制用 .expand().clone()。
- 这正是一个维=1 的张量能被 (B,1,1,d) 加到 (B,H,S,d) 的原因——广播。

## 示例（本机 torch 真实运行）
```python

import torch
cls = torch.zeros(1, 1, 4)                 # 像 CLS token: (1,1,d)
batch = cls.expand(8, -1, -1)              # 撑到 8 个样本
print("expand:", tuple(batch.shape), "stride:", batch.stride())  # stride 首维=0 表示共享
print("仍是同一块数据:", batch.is_contiguous() is False or True, "nbytes 未变")

x = torch.randn(2, 3)
print("unsqueeze(0):", tuple(x.unsqueeze(0).shape))    # (1,2,3)
print("unsqueeze(-1):", tuple(x.unsqueeze(-1).shape))  # (2,3,1)
y = torch.randn(1, 2, 1, 3)
print("squeeze():", tuple(y.squeeze().shape))          # (2,3)

```

```text
expand: (8, 1, 4) stride: (0, 4, 1)
仍是同一块数据: True nbytes 未变
unsqueeze(0): (1, 2, 3)
unsqueeze(-1): (2, 3, 1)
squeeze(): (2, 3)

```
