# nn.CrossEntropyLoss（损失）

分类交叉熵。输入是『未过 softmax 的 logits』(B, C) 和类别下标 (B,)；它内部自动做 softmax + log + NLL。ignore_index 让指定类别的样本不进 loss——图像描述里用来忽略 <pad>。

## 参数
| 参数 | 说明 |
|---|---|
| `input` | logits，形状 (B, C) 或 (B, L, C) 需自己 reshape |
| `target` | 类别下标 long，(B,) |
| `ignore_index` | 该下标不参与 loss，常用于 padding |

## 要点 / 坑
- 不要先手动 softmax 再喂它（会重复 softmax）。
- loss 平均的是『被计入的』样本/词，pad 位置不影响分母。

## 示例（本机 torch 真实运行）
```python

import torch, torch.nn as nn
logits = torch.randn(3, 5)          # 3 个样本，5 类
target = torch.tensor([0, 2, 1])
ce = nn.CrossEntropyLoss()
print("普通 CE:", ce(logits, target).item())

# 模拟变长句子 padding: 某些位置不参与
ce_ig = nn.CrossEntropyLoss(ignore_index=0)   # 0 是 <pad>
t = torch.tensor([0, 2, 0])                   # 位置0和2被忽略，只剩1个在算
print("ignore 后:", ce_ig(logits, t).item())

```

```text
普通 CE: 1.0807790756225586
ignore 后: 1.6483161449432373

```
