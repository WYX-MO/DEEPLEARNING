# torch.matmul / @（张量运算）

矩阵乘法。2 维时等价 A @ B = (A的行, B的列)。更高维时把最后两维当矩阵、前面的维当 batch 做广播相乘。

## 参数
| 参数 | 说明 |
|---|---|
| `input, other` | 两个相乘张量 |
| `广播规则` | 若都 ≥3 维，要求除最后两维外的形状可广播 |

## 要点 / 坑
- @ 是 matmul 的简写；两者一样。
- 要区分 element-wise 的 *（逐元素乘）和 @（矩阵乘）。
- 两个 2D 满足内维相等 (m,k)@(k,n)；不满足会报错。

## 示例（本机 torch 真实运行）
```python

import torch
a = torch.randn(2, 3)          # (m,k)=(2,3)
b = torch.randn(3, 4)          # (k,n)=(3,4)
c = a @ b
print("2D 结果:", tuple(c.shape))            # (2,4)

# 高维 = 前两维当 batch 广播
A = torch.randn(5, 2, 3)
B = torch.randn(5, 3, 4)
print("batched:", tuple((A @ B).shape))       # (5,2,4)

# 手动逐元素验证
manual = (a @ b)
print("torch.matmul == @ :", torch.allclose(c, torch.matmul(a, b)))

```

```text
2D 结果: (2, 4)
batched: (5, 2, 4)
torch.matmul == @ : True

```
