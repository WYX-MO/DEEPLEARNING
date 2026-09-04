# experiments/attention_demo.py
import os
import sys

# 把项目根目录(30-)加进搜索路径，才能 import Attention 这个包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
from Attention.models.attention import SelfAttention

torch.manual_seed(42)

x = torch.randn(2, 5, 8)

model = SelfAttention(
    d_model=8,
    d_k=4
)

output, attention = model(x)

print("x:", x.shape)
print("output:", output.shape)
print("attention:", attention.shape)

print(attention[0])
print(attention[0].sum(dim=-1))  # 每一行的和为 1

# import matplotlib.pyplot as plt

# plt.imshow(attention[0].detach().numpy())
# plt.xlabel("Key")
# plt.ylabel("Query")
# plt.title("Self-Attention")
# plt.colorbar()
# plt.show()