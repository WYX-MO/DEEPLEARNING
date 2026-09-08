# transforms.Compose（torchvision·transforms）

把一串图像变换串成流水线，按顺序依次作用于每张图。训练/测试各配一条。

## 参数
| 参数 | 说明 |
|---|---|
| `transforms.Compose([...])` | 传入按顺序执行的 transform 列表 |
| `用法` | composed(img) 返回处理后的图 |

## 要点 / 坑
- 顺序有讲究：先几何/尺寸(Resize/Crop)，最后 ToTensor、Normalize。
- 训练用『随机』变换，测试只做固定 Resize/CenterCrop。

## 示例（本机 torch 真实运行）
```python

import numpy as np
from PIL import Image
from torchvision import transforms as T
img = Image.fromarray(np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8))
p = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
out = p(img)
print("Compose 后:", tuple(out.shape), out.dtype)

```

```text
Compose 后: (3, 32, 32) torch.float32

```
