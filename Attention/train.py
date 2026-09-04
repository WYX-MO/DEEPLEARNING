#train.py
# TODO: 参照 ../CNN/train.py 的结构自己写训练入口
# 固定 seed -> 加载数据 -> 建模型 -> 训练循环（train/valid）-> 记录结果到 experiments/experiments.csv
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else device
    print("device:", device)
    raise NotImplementedError("训练代码待填，参照 CNN/train.py")


if __name__ == "__main__":
    main()
