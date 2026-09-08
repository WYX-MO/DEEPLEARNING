# torchvision.models（resnet / vgg / vit，预训练）（torchvision·models）

直接拿到经典网络。weights=None 随机初始化；weights=xxx_Weights.DEFAULT 下载官方预训练权重。可改最后一层做迁移学习。

## 参数
| 参数 | 说明 |
|---|---|
| `resnet18/resnet50/..., weights=None` | 建网络；None=不加载预训练 |
| `weights=ResNet18_Weights.DEFAULT` | 加载官方预训练（首次会联网下载） |
| `改分类头` | net.fc = nn.Linear(512, 你的类数)（resnet） |

## 要点 / 坑
- 注意网络对输入的要求（多为 224x224），要用与预训练一致的 Normalize。
- 微调常把前面层冻结、只训新头，先 .eval()+只给头设 requires_grad。

## 示例（本机 torch 真实运行）
```python

import torch, torchvision, torch.nn as nn
m = torchvision.models.resnet18(weights=None)   # 不联网，随机初始化
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    print("resnet18 输出:", tuple(m(x).shape))     # (1,1000)
print("参数量: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
m.fc = nn.Linear(512, 10)                       # 换成你自己的类数
print("换分类头后输出:", tuple(m(x).shape))        # (1,10)

```

```text
resnet18 输出: (1, 1000)
参数量: 11.69 M
换分类头后输出: (1, 10)

```
