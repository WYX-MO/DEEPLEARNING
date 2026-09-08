# ModuleList / Sequential（容器）

两个『装层的容器』。Sequential：给一组层，forward 自动依次执行；ModuleList：只负责登记一堆层（好让 .parameters() 收集到），forward 要自己写循环。decoder 堆叠多个 block 用 ModuleList + for 循环。

## 参数
| 参数 | 说明 |
|---|---|
| `nn.Sequential(*layers)` | 自动 chain，前一层输出喂下一层 |
| `nn.ModuleList([...])` | 只存层，必须自己 for 循环调用 |

## 要点 / 坑
- 把层装进 Python list 而不是 ModuleList → 参数不会被 optimizer 收集到！
- ModuleList 好处是能按层索引/动态取，参数照样注册。

## 示例（本机 torch 真实运行）
```python

import torch, torch.nn as nn
seq = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])  # 像 decoder 多层

x = torch.randn(5, 4)
y = x
for b in blocks:            # ModuleList 要自己循环
    y = b(y)
print("ModuleList 手动链:", tuple(y.shape))
print("Sequential 自动链:", tuple(seq(x).shape))

# 参数都能被收集到（重点：ModuleList 让 optimizer 能找到它们）
print("ModuleList 可训练参数数:", sum(p.numel() for p in blocks.parameters()))

```

```text
ModuleList 手动链: (5, 4)
Sequential 自动链: (5, 2)
ModuleList 可训练参数数: 60

```
