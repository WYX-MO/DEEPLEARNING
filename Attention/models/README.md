# models/

模型定义。建议参照 CNN 项目「块 + 网络」分层：

- 把可复用的小单元写成独立类/文件（如 CNN 的 `ResidualBlock.py`）；
- 整个网络用一个类组装（如 `ResNet.py` 的 `_make_layer` 排布），类名以 `My...` 开头；
- 文件底部用 `if __name__ == "__main__":` 测一次前向输出形状 `model(torch.randn(...)).shape`，方便快速验证维度。

一个模型尽量做成「和对照组只差一个变量」，方便做受控对比。
