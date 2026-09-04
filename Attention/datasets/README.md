# datasets/

数据管线模块（加载 + transform）。

参考 `../CNN/datasets/cifar10.py` 的写法：每个数据集一个文件，返回 `get_data_loaders(batch_size=...) -> (train_loader, test_loader)`。

```python
# 待填：例如 data_loader, data_loader_test = get_data_loaders(batch_size=64)
```

数据文件放在 `../data/`（已被 .gitignore 忽略）。
