# Attention

> 科研项目框架（参照 `../CNN` 的结构搭建）。当前只是骨架，代码由自己逐步填。

## 目录结构说明

```
Attention/
├── AttentionExperiment.md   # 项目总览：目标 / 模型演进 / 结果表 / Key Learnings（写报告时更新）
├── README.md                # 个人过程记录（随手记，不追求格式）
├── requirement.txt          # 依赖（按 dl 环境实际版本固定）
├── train.py                 # 训练入口（TODO：参照 CNN/train.py 自己写）
├── datasets/                # 数据管线（TODO：如 datasets/*.py，参考 CNN/datasets/cifar10.py）
├── models/                  # 模型定义（TODO：建议拆成 block.py + 网络.py，参考 CNN/models/）
├── experiments/             # 受控实验记录：exp_main.md 索引 + 每次实验一个 md + experiments.csv
├── checkpoints/             # 训练权重（*.pth，gitignore）
├── results/                 # 图表 / 日志输出
└── data/                    # 原始数据集（gitignore，可随时重新下载）
```

## 实验规范（沿用 CNN 的做法）

- 每次只改一个变量（模型 / 增广 / BN / …），其余训练配置固定，便于对照。
- 每次跑批在 `experiments/experiments.csv` 记一行（含 git commit 便于回溯）。
- 每个实验写独立 md，最终在 `experiments/exp_main.md` 建立索引汇总。
