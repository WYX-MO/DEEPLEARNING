# zeros / ones / arange / linspace / full（张量创建）

按规则填充的张量。tril 掩码、位置 id、mask 都用这些造。eye 造单位阵。

## 参数
| 参数 | 说明 |
|---|---|
| `zeros/ones(*size)` | 全 0 / 全 1 |
| `arange(start, end, step)` | 等差整数/浮点序列，不含 end |
| `linspace(s, e, n)` | s..e 均分 n 个点 |
| `full(*size, value)` | 填同一个值 |

## 要点 / 坑
- arange 的 end 是开区间；linspace 是闭区间。
- 要造下三角 mask：tril(torch.ones(L, L))。

## 示例（本机 torch 真实运行）
```python

import torch
print(torch.zeros(2, 3))
print(torch.arange(0, 5))          # [0,1,2,3,4]
print(torch.linspace(0, 1, 5))     # 0..1 共 5 个
print(torch.full((2, 2), -1))
print(torch.eye(3))                # 单位阵

```

```text
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([0, 1, 2, 3, 4])
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
tensor([[-1, -1],
        [-1, -1]])
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])

```
