# view / reshape / flatten（张量运算）

改变形状。view 是共享内存的视图（要求原张量连续）；reshape 在能 view 时用 view、否则自动复制；flatten(起始维) 把从该维开始的轴全部压成一维。

## 参数
| 参数 | 说明 |
|---|---|
| `view(*shape)` | 按新形状查看；-1 表示自动推断；不连续会报错 |
| `reshape(*shape)` | 功能近似 view，但不连续时自动 contiguous() 拷贝 |
| `flatten(start_dim=0)` | 把 start_dim 起的维压平 |

## 要点 / 坑
- view 只是换『解释方式』，改它的值会改到原张量。
- 元素总数必须一致：如 2*3*4=24，view(2,-1)->(2,12)。
- 经典报错：对 transpose 后的不连续张量直接 view() → 先 .contiguous()。

## 示例（本机 torch 真实运行）
```python

import torch
x = torch.randn(2, 3, 4)
print("x:", tuple(x.shape), "numel:", x.numel())

print("view(2,-1):", tuple(x.view(2, -1).shape))      # (2,12)
print("reshape(-1):", tuple(x.reshape(-1).shape))     # (24,)
print("flatten(1):", tuple(x.flatten(1).shape))       # (2,12)

y = x.transpose(0, 1)              # 不连续
print("y 连续?", y.is_contiguous())
try:
    y.view(-1, 4)
except RuntimeError as e:
    print("view 报错:", str(e)[:50], "...")
print("reshape 不报错:", tuple(y.reshape(-1, 4).shape))

```

```text
x: (2, 3, 4) numel: 24
view(2,-1): (2, 12)
reshape(-1): (24,)
flatten(1): (2, 12)
y 连续? False
view 报错: view size is not compatible with input tensor's si ...
reshape 不报错: (6, 4)

```
