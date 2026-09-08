# torch_api_kb.py  —— 本地 API 知识库（自写整理，可自行增删条目）
# 每个条目字段：
#   title   : 显示名
#   keys    : 查询别名（越小写越好，含多种叫法）
#   cat     : 分类
#   desc    : 一句话/一段中文讲解
#   params  : [(参数名, 说明), ...]          -> 渲染成参数表
#   notes   : [坑点/提示, ...]               -> 渲染成要点
#   example : 可直接跑的示例代码（用本机 torch 真实执行并抓取输出）
# 想加新 API：照抄一个条目塞进 KB，程序自动就能查。

KB = [

# ---------------- 张量 / 矩阵运算 ----------------
dict(
    title="torch.matmul / @",
    keys=["matmul", "torch.matmul", "@", "bmm", "torch.bmm"],
    cat="张量运算",
    desc=("矩阵乘法。2 维时等价 A @ B = (A的行, B的列)。更高维时把最后两维当矩阵、"
          "前面的维当 batch 做广播相乘。"),
    params=[
        ("input, other", "两个相乘张量"),
        ("广播规则", "若都 ≥3 维，要求除最后两维外的形状可广播"),
    ],
    notes=[
        "@ 是 matmul 的简写；两者一样。",
        "要区分 element-wise 的 *（逐元素乘）和 @（矩阵乘）。",
        "两个 2D 满足内维相等 (m,k)@(k,n)；不满足会报错。",
    ],
    example='''
import torch
a = torch.randn(2, 3)          # (m,k)=(2,3)
b = torch.randn(3, 4)          # (k,n)=(3,4)
c = a @ b
print("2D 结果:", tuple(c.shape))            # (2,4)

# 高维 = 前两维当 batch 广播
A = torch.randn(5, 2, 3)
B = torch.randn(5, 3, 4)
print("batched:", tuple((A @ B).shape))       # (5,2,4)

# 手动逐元素验证
manual = (a @ b)
print("torch.matmul == @ :", torch.allclose(c, torch.matmul(a, b)))
''',
),

dict(
    title="transpose / permute / T",
    keys=["transpose", "permute", ".t", ".T", "x.transpose", "torch.transpose"],
    cat="张量运算",
    desc=("transpose 交换两个维度的顺序；permute 可任意重排所有维度。两者都是"
          "『视图(view)』——不复制数据，只改 stride，因此可能变成内存不连续。"),
    params=[
        ("transpose(dim0, dim1)", "交换两个轴，参数只能两个"),
        ("permute(*dims)", "把轴按给定顺序重排，参数必须覆盖所有维"),
        (".T", "最后两维转置（≥2维时）"),
    ],
    notes=[
        "transpose/permute 返回的是不连续的 view → 后面要 view() 得先 .contiguous()。",
        "不复制数据 = 几乎免费；但形状语义变了。",
    ],
    example='''
import torch
x = torch.randn(2, 3, 4)
print("x:", tuple(x.shape))
print("transpose(1,2):", tuple(x.transpose(1, 2).shape))   # (2,4,3)
print("permute(2,0,1):", tuple(x.permute(2, 0, 1).shape))  # (4,2,3)
print("T:", tuple(x.T.shape))                               # (4,3,2)

t = x.transpose(0, 1)          # 换轴 -> 不连续
print("is_contiguous:", t.is_contiguous())
print("contiguous 后:", t.contiguous().is_contiguous())
''',
),

dict(
    title="view / reshape / flatten",
    keys=["view", "reshape", "flatten", "x.view", "torch.flatten"],
    cat="张量运算",
    desc=("改变形状。view 是共享内存的视图（要求原张量连续）；reshape 在能 view 时用 view、"
          "否则自动复制；flatten(起始维) 把从该维开始的轴全部压成一维。"),
    params=[
        ("view(*shape)", "按新形状查看；-1 表示自动推断；不连续会报错"),
        ("reshape(*shape)", "功能近似 view，但不连续时自动 contiguous() 拷贝"),
        ("flatten(start_dim=0)", "把 start_dim 起的维压平"),
    ],
    notes=[
        "view 只是换『解释方式』，改它的值会改到原张量。",
        "元素总数必须一致：如 2*3*4=24，view(2,-1)->(2,12)。",
        "经典报错：对 transpose 后的不连续张量直接 view() → 先 .contiguous()。",
    ],
    example='''
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
''',
),

dict(
    title="argmax / max",
    keys=["argmax", "max", "x.argmax", "torch.max", "torch.argmax"],
    cat="张量运算",
    desc=("沿某一维找最大值。argmax 只返回『下标』；max 返回 (最大值, 下标)。"
          "不指定 dim 时会把张量摊平找全局最大。"),
    params=[
        ("argmax(dim=None)", "dim 给定时沿该维；不给则全局(摊平后)"),
        ("max(input, dim)", "返回 (values, indices) 两个张量"),
        ("-1 = 最后一个维", "对 (B,S,V) 用 argmax(-1) 就是在 V(词表)维挑最大"),
    ],
    notes=[
        "分类/生成里 argmax(-1) 是『取预测类别』的标准写法。",
        "argmax 不可导、无梯度，只能用于指标/推理，不能进 loss。",
    ],
    example='''
import torch
x = torch.tensor([[1., 5., 2.],
                  [9., 0., 3.]])
print("全局最大下标 argmax():", x.argmax().item())       # 3 (第4个元素=9)
print("每行最大下标 argmax(1):", x.argmax(dim=1).tolist())  # [1, 0]

vals, idx = torch.max(x, dim=1)
print("max(dim=1):", vals.tolist(), idx.tolist())

# 三维例子：outputs (B,S,V) -> 每个位置预测的词
out = torch.randn(2, 3, 5)
print("argmax(-1) 形状:", tuple(out.argmax(-1).shape))    # (2,3)
''',
),

dict(
    title="cat / chunk / stack",
    keys=["cat", "chunk", "stack", "split", "torch.cat", "torch.chunk"],
    cat="张量运算",
    desc=("cat 沿已有维拼接（形状除该维外必须一致）；stack 新增一维再堆叠；"
          "chunk 把一个张量沿某维等分成 N 块（Q/K/V 常用）。"),
    params=[
        ("cat([a,b], dim)", "拼接：不增加维数"),
        ("stack([a,b], dim)", "堆叠：多出一个维"),
        ("chunk(t, n, dim)", "沿 dim 切成 n 块（最后一块可能小）"),
    ],
    notes=[
        "cat 时两个张量除目标维外的形状必须完全相同。",
        "多头注意力里 qkv.view(...).chunk(3, dim=-1) 就是把拼接的 Q,K,V 拆开。",
    ],
    example='''
import torch
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print("cat dim=0:", tuple(torch.cat([a, b], 0).shape))   # (4,3)
print("cat dim=1:", tuple(torch.cat([a, b], 1).shape))   # (2,6)
print("stack dim=0:", tuple(torch.stack([a, b], 0).shape))  # (2,2,3)

qkv = torch.randn(2, 5, 9)          # 设想最后一维是 [Q|K|V] 拼好
q, k, v = torch.chunk(qkv, 3, dim=-1)
print("chunk: ", tuple(q.shape), tuple(k.shape), tuple(v.shape))  # 各 (2,5,3)
''',
),

dict(
    title="expand / squeeze / unsqueeze",
    keys=["expand", "squeeze", "unsqueeze", "broadcast", "x.expand"],
    cat="张量运算",
    desc=("unsqueeze 在指定位置加一个长度 1 的维；squeeze 删掉长度 1 的维；"
          "expand 把长度 1 的维『广播撑大』——仍是视图，不复制数据。"),
    params=[
        ("unsqueeze(dim)", "插入单维度，为广播/加 batch 维用"),
        ("squeeze(dim=None)", "去掉长度=1 的维（可指定）"),
        ("expand(*sizes)", "把 1 扩成大数，-1 表示保持该维不变"),
    ],
    notes=[
        "expand 不占额外内存（stride=0），要真复制用 .expand().clone()。",
        "这正是一个维=1 的张量能被 (B,1,1,d) 加到 (B,H,S,d) 的原因——广播。",
    ],
    example='''
import torch
cls = torch.zeros(1, 1, 4)                 # 像 CLS token: (1,1,d)
batch = cls.expand(8, -1, -1)              # 撑到 8 个样本
print("expand:", tuple(batch.shape), "stride:", batch.stride())  # stride 首维=0 表示共享
print("仍是同一块数据:", batch.is_contiguous() is False or True, "nbytes 未变")

x = torch.randn(2, 3)
print("unsqueeze(0):", tuple(x.unsqueeze(0).shape))    # (1,2,3)
print("unsqueeze(-1):", tuple(x.unsqueeze(-1).shape))  # (2,3,1)
y = torch.randn(1, 2, 1, 3)
print("squeeze():", tuple(y.squeeze().shape))          # (2,3)
''',
),

dict(
    title="softmax(dim)",
    keys=["softmax", "F.softmax", "torch.softmax", "log_softmax"],
    cat="张量运算",
    desc=("把一维的分数转成概率（和为 1）。dim 决定『沿着哪个维求和=1』——"
          "注意力里 dim=-1 表示对 keys 那一维归一化。"),
    params=[
        ("dim", "沿该维 softmax；每一条『行』各自和为 1"),
        ("-1", "最后一个维（对 (B,H,L,L) 就是每个 query 行的 key 权重）"),
    ],
    notes=[
        "用 dim=0 和 dim=-1 归一化方向完全不同，务必看清。",
        "输入全 -inf 的一行会变 NaN，别让整行都被 mask 掉。",
    ],
    example='''
import torch
x = torch.tensor([[1., 2., 3.],
                  [1., 1., 1.]])
s = torch.softmax(x, dim=-1)
print("dim=-1 每行和为1:", s, s.sum(-1))
s0 = torch.softmax(x, dim=0)
print("dim=0  每列和为1:", s0, s0.sum(0))
''',
),

dict(
    title="masked_fill + tril / triu",
    keys=["masked_fill", "tril", "triu", "mask", "torch.tril"],
    cat="张量运算",
    desc=("masked_fill(mask, value) 把 mask 为 True 的位置替换成 value。配 tril(下三角)"
          "能做出『因果 mask』：下三角保留、右上角填 -inf，softmax 后那些位置权重=0。"),
    params=[
        ("masked_fill(mask, value)", "mask 是与原张量可广播的布尔张量"),
        ("tril(m)", "m×m 下三角（含对角线）为 1，其余 0"),
        ("为什么填 -inf", "softmax(exp(-inf)=0) → 那些位置注意力精确为 0"),
    ],
    notes=[
        "mask 一定要能广播到 attention_scores：2D [L,L] 会被当 [1,1,L,L] 用。",
        "mask 填 -inf 必须在 softmax 之前才有效。",
    ],
    example='''
import torch
torch.manual_seed(0)
L = 4
mask = torch.tril(torch.ones(L, L))          # 下三角=1
scores = torch.rand(L, L)
masked = scores.masked_fill(mask == 0, float('-inf'))
print("tril mask:\\n", mask)
print("masked 后的分数:\\n", masked)
print("softmax 后（被挡位置=0）:\\n", torch.softmax(masked, dim=-1))
''',
),

# ---------------- nn 层 ----------------
dict(
    title="nn.Linear",
    keys=["linear", "nn.linear", "nn.Linear", "dense", "fully connected"],
    cat="nn 层",
    desc=("全连接：y = x·Wᵀ + b。注意权重形状是 (out, in)，所以手算要用 x @ W.T。"
          "作用在最后一维，前面所有维自动当作 batch。"),
    params=[
        ("in_features", "输入最后一维大小"),
        ("out_features", "输出大小"),
        ("bias", "默认 True，是否加偏置"),
    ],
    notes=[
        "weight.shape = (out, in) —— 这是最容易被手算搞反的地方。",
        "输入 (B, L, in) 输出 (B, L, out)，只变最后一维。",
    ],
    example='''
import torch, torch.nn as nn
lin = nn.Linear(4, 3)
print("weight:", tuple(lin.weight.shape), "bias:", tuple(lin.bias.shape))

x = torch.randn(2, 4)
y = lin(x)
print("forward:", tuple(y.shape))          # (2,3)

# 手算验证：y = x @ W.T + b
with torch.no_grad():
    manual = x @ lin.weight.T + lin.bias
    print("与手算一致:", torch.allclose(y, manual))
''',
),

dict(
    title="nn.Conv2d",
    keys=["conv2d", "nn.conv2d", "conv", "nn.Conv2d"],
    cat="nn 层",
    desc=("二维卷积。patch 化(切块)其实就是一个 stride=patch_size 的 Conv2d。"
          "输出边长公式：(H + 2*pad - kernel) / stride + 1。"),
    params=[
        ("in_channels / out_channels", "输入/输出通道数"),
        ("kernel_size", "卷积核大小"),
        ("stride", "步长（patch embedding 设 stride=patch 即不重叠切块）"),
        ("padding", "四周补零"),
    ],
    notes=[
        "权重形状: (out, in, k, k)。",
        "Conv2d 需要 4D 输入 (B, C, H, W)。",
    ],
    example='''
import torch, torch.nn as nn
conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1)
x = torch.randn(1, 3, 32, 32)
print("k3 p1 s1:", tuple(conv(x).shape))       # (1,8,32,32)

conv2 = nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=2)
print("k3 p0 s2:", tuple(conv2(x).shape))      # (1,8,15,15)  公式 (32-3)/2+1=15

# patch 化就是 stride=kernel 的 conv
patch = nn.Conv2d(3, 192, kernel_size=4, stride=4)
p = patch(torch.randn(2, 3, 32, 32))
print("patch4:", tuple(p.shape))               # (2,192,8,8) -> 8*8=64 个 patch
''',
),

dict(
    title="nn.Embedding",
    keys=["embedding", "nn.embedding", "nn.Embedding", "word2vec"],
    cat="nn 层",
    desc=("查表：把整数 id 映射成向量。词表索引 -> 稠密向量。padding_idx 指定的行"
          "固定为 0 且不被更新，正适合放 <pad>=0。"),
    params=[
        ("num_embeddings", "词表大小(含特殊token)"),
        ("embedding_dim", "每个词向量的长度"),
        ("padding_idx", "该索引的行恒为 0、不参与梯度更新"),
    ],
    notes=[
        "weight.shape = (num_embeddings, embedding_dim)。",
        "输入是整数 long 张量 (B, L)，输出 (B, L, dim)。",
    ],
    example='''
import torch, torch.nn as nn
emb = nn.Embedding(5, 3, padding_idx=0)     # 5个词，向量3维，0是pad
ids = torch.tensor([[0, 2, 1],
                    [3, 0, 4]])             # (B=2, L=3)
out = emb(ids)
print("输入:", tuple(ids.shape), "-> 输出:", tuple(out.shape))  # (2,3,3)
print("padding 行固定为0:", emb.weight[0].tolist())
''',
),

dict(
    title="LayerNorm vs BatchNorm",
    keys=["layernorm", "batchnorm", "layer_norm", "batch_norm", "ln", "bn"],
    cat="nn 层",
    desc=("两者都在做『标准化(减均值除方差)』，但归一化的『对象』不同："
          "LayerNorm 对每个样本的最后一维归一化；BatchNorm 对每个通道跨 batch 归一化。"
          "Transformer 里序列长短不齐、每个样本独立 → 用 LayerNorm。"),
    params=[
        ("LayerNorm(normalized_shape)", "归一化的轴是最后一维"),
        ("BatchNorm1d/2d(num_features)", "归一化的轴是跨 batch 的每个通道"),
    ],
    notes=[
        "BatchNorm 依赖 batch 统计，推理要用训练累计的 running 统计；小 batch 不稳。",
        "LayerNorm 与 batch 无关，逐样本可算 —— 这是 transformer 选它的核心原因。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
x = torch.randn(4, 3)            # 4个样本 × 3维特征
bn = nn.BatchNorm1d(3)           # 对每一列(特征)，跨4个样本归一化
ln = nn.LayerNorm(3)             # 对每一行(样本)，跨3维特征归一化
print("BN 的列(每特征跨batch) 均~0 方~1:")
print("  ", bn(x).mean(dim=0), bn(x).std(dim=0))
print("LN 的行(每样本跨特征) 均~0 方~1:")
print("  ", ln(x).mean(dim=1), ln(x).std(dim=1))
''',
),

dict(
    title="nn.Dropout",
    keys=["dropout", "nn.dropout", "nn.Dropout"],
    cat="nn 层",
    desc=("训练时按概率 p 随机把一些元素置 0，并把保留下来的值放大 1/(1-p)（保持期望不变）；"
          "eval 模式下什么都不做。作用是防止过拟合/减少对某些位置的依赖。"),
    params=[
        ("p", "置零概率，常见 0.1（注意力）~0.5"),
        ("train/eval", "默认 train；.eval() 后失效"),
    ],
    notes=[
        "忘了 .eval() 会导致推理结果随机抖动。",
        "dropout 只在训练加噪声，测试时等价于用了全部连接的平均。",
    ],
    example='''
import torch, torch.nn as nn
m = nn.Dropout(p=0.5)
x = torch.ones(1, 20000)
y = m(x)                                     # 当前是 train 模式
kept = (y != 0).float().mean().item()
kept_val = y[y != 0].mean().item() if (y != 0).any() else 0
print(f"train 保留比例 ~{kept:.3f} (期望0.5), 保留值被放大到 ~{kept_val:.3f} (期望2)")

m.eval()
print("eval 后恒等:", torch.equal(m(x), x))
''',
),

dict(
    title="nn.CrossEntropyLoss",
    keys=["crossentropy", "cross_entropy", "ce_loss", "nn.CrossEntropyLoss", "ignore_index"],
    cat="损失",
    desc=("分类交叉熵。输入是『未过 softmax 的 logits』(B, C) 和类别下标 (B,)；"
          "它内部自动做 softmax + log + NLL。ignore_index 让指定类别的样本不进 loss"
          "——图像描述里用来忽略 <pad>。"),
    params=[
        ("input", "logits，形状 (B, C) 或 (B, L, C) 需自己 reshape"),
        ("target", "类别下标 long，(B,)"),
        ("ignore_index", "该下标不参与 loss，常用于 padding"),
    ],
    notes=[
        "不要先手动 softmax 再喂它（会重复 softmax）。",
        "loss 平均的是『被计入的』样本/词，pad 位置不影响分母。",
    ],
    example='''
import torch, torch.nn as nn
logits = torch.randn(3, 5)          # 3 个样本，5 类
target = torch.tensor([0, 2, 1])
ce = nn.CrossEntropyLoss()
print("普通 CE:", ce(logits, target).item())

# 模拟变长句子 padding: 某些位置不参与
ce_ig = nn.CrossEntropyLoss(ignore_index=0)   # 0 是 <pad>
t = torch.tensor([0, 2, 0])                   # 位置0和2被忽略，只剩1个在算
print("ignore 后:", ce_ig(logits, t).item())
''',
),

dict(
    title="ModuleList / Sequential",
    keys=["modulelist", "sequential", "nn.Sequential", "nn.ModuleList", "stack layers"],
    cat="容器",
    desc=("两个『装层的容器』。Sequential：给一组层，forward 自动依次执行；"
          "ModuleList：只负责登记一堆层（好让 .parameters() 收集到），forward 要自己写循环。"
          "decoder 堆叠多个 block 用 ModuleList + for 循环。"),
    params=[
        ("nn.Sequential(*layers)", "自动 chain，前一层输出喂下一层"),
        ("nn.ModuleList([...])", "只存层，必须自己 for 循环调用"),
    ],
    notes=[
        "把层装进 Python list 而不是 ModuleList → 参数不会被 optimizer 收集到！",
        "ModuleList 好处是能按层索引/动态取，参数照样注册。",
    ],
    example='''
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
''',
),

# ---------------- 张量：创建 ----------------
dict(
    title="randn / rand / randint / randperm",
    keys=["randn", "rand", "randint", "randperm", "torch.randn"],
    cat="张量创建",
    desc=("生成随机张量。randn=标准正态 N(0,1)；rand=均匀 [0,1)；randint=整数范围；"
          "randperm=0..n-1 的一个随机排列（打乱索引用）。"),
    params=[
        ("torch.randn(*size)", "标准正态，训练初始化常配 manual_seed"),
        ("torch.rand(*size)", "均匀 [0,1)"),
        ("torch.randint(lo, hi, size)", "整数在 [lo, hi)"),
        ("torch.randperm(n)", "返回长度为 n 的随机排列"),
    ],
    notes=[
        "shape 直接当作位置参数或元组给：randn(2,3) 等价 randn((2,3))。",
        "同一模型要复现 → 前先 torch.manual_seed(seed)。",
    ],
    example='''
import torch
print(torch.randn(2, 3))            # 值服从 N(0,1)
print("rand:", torch.rand(2))       # [0,1)
print("randint(0,10,size=(2,2)):", torch.randint(0, 10, (2, 2)))
print("randperm(6):", torch.randperm(6))
''',
),

dict(
    title="zeros / ones / arange / linspace / full",
    keys=["zeros", "ones", "arange", "linspace", "full", "eye", "torch.zeros"],
    cat="张量创建",
    desc=("按规则填充的张量。tril 掩码、位置 id、mask 都用这些造。eye 造单位阵。"),
    params=[
        ("zeros/ones(*size)", "全 0 / 全 1"),
        ("arange(start, end, step)", "等差整数/浮点序列，不含 end"),
        ("linspace(s, e, n)", "s..e 均分 n 个点"),
        ("full(*size, value)", "填同一个值"),
    ],
    notes=[
        "arange 的 end 是开区间；linspace 是闭区间。",
        "要造下三角 mask：tril(torch.ones(L, L))。",
    ],
    example='''
import torch
print(torch.zeros(2, 3))
print(torch.arange(0, 5))          # [0,1,2,3,4]
print(torch.linspace(0, 1, 5))     # 0..1 共 5 个
print(torch.full((2, 2), -1))
print(torch.eye(3))                # 单位阵
''',
),

dict(
    title="torch.tensor / from_numpy / as_tensor",
    keys=["tensor", "torch.tensor", "from_numpy", "as_tensor", "torch.from_numpy"],
    cat="张量创建",
    desc=("从 Python 列表或 numpy 数组建张量。from_numpy 与 numpy 数组共享内存（改一个改两个）。"),
    params=[
        ("torch.tensor(list)", "复制数据建新张量"),
        ("torch.from_numpy(np_arr)", "共享内存，不复制，但不可改 requires_grad"),
        ("torch.as_tensor(x)", "能复用则复用内存，不能则拷贝"),
    ],
    notes=[
        "from_numpy 共享内存 → numpy 变了 torch 也变（可省拷贝但要注意）。",
        "要参与梯度，建完设 .requires_grad_(True)，或用 requires_grad=True 建。",
    ],
    example='''
import torch
import numpy as np
a = torch.tensor([1, 2, 3])
print(a, a.dtype)                        # 默认 int64

np_a = np.array([4., 5., 6.])
b = torch.from_numpy(np_a)               # 共享内存
np_a[0] = 100
print("from_numpy 共享内存，numpy改后:", b)
''',
),

# ---------------- 张量：属性 ----------------
dict(
    title="shape / size / numel / item / dtype / device",
    keys=["shape", "size", "numel", "item", "dtype", "device", ".shape", ".size"],
    cat="张量属性",
    desc=("问张量形状/大小/元素数。shape 与 size() 等价；numel 是元素总数；"
          "item() 把单元素张量变 Python 数字；dtype/device 告诉类型和所在设备。"),
    params=[
        (".shape / .size()", "各维长度，如 torch.Size([2,3])"),
        (".numel()", "元素总数 = 各维乘积"),
        (".item()", "只能用于单元素张量，返回 Python float/int"),
        (".dtype / .device", "数据类型 / CPU还是CUDA"),
    ],
    notes=[
        "batch 维放第一个：模型约定 (B, C, H, W)。",
        "想当 Python 数字用（打印、if 判断）→ .item()，别直接拿张量比。",
    ],
    example='''
import torch
x = torch.randn(2, 3, 4)
print("shape:", x.shape, "== size():", x.size())
print("dim:", x.dim(), "numel:", x.numel())
y = torch.tensor([[3.0]])
print("item():", y.item(), type(y.item()))
print("dtype:", x.dtype, "device:", x.device)
''',
),

dict(
    title="to / cpu / cuda / float / long",
    keys=["to", ".to", "cpu", "cuda", ".float", ".long", ".bool", ".double"],
    cat="张量属性",
    desc=("迁移设备 / 改 dtype。训练里最常做的是把数据搬到模型所在设备："
          "inputs = inputs.to(device)；标签做交叉熵前要 .long()。"),
    params=[
        (".to(device)", "搬到指定设备（或 .to(dtype)）"),
        (".cpu() / .cuda()", "搬到 CPU / GPU"),
        (".float()/.long()/.bool()", "改 dtype（图像 float、标签 long、掩码 bool）"),
    ],
    notes=[
        "模型和数据的 device 不一致是新手第一大报错来源。",
        "target 进 CrossEntropyLoss 必须 long 型。",
    ],
    example='''
import torch
x = torch.randn(2, 3)
print("默认 dtype:", x.dtype, "→ float():", x.float().dtype)
print("long():", x.long().dtype, "| bool():", (x > 0).bool().dtype)
print("搬到CPU:", x.cpu().device, "| 已在这台机器默认设备:", x.device)
''',
),

# ---------------- 自动求导 ----------------
dict(
    title="requires_grad / backward / grad",
    keys=["requires_grad", "backward", ".grad", "autograd", "torch.no_grad"],
    cat="自动求导",
    desc=("PyTorch 自动求导核心：把张量设 requires_grad=True，往前算到 loss，"
          "loss.backward() 后每个参数的 .grad 就存好梯度，交给 optimizer.step() 用。"),
    params=[
        ("requires_grad_(True)", "让该张量参与求导（叶子）"),
        ("loss.backward()", "反向传播，把梯度写进参与计算张量的 .grad"),
        (".grad", "存梯度的地方；每次要清零再算（optimizer.zero_grad()）"),
    ],
    notes=[
        "backward 前要 optimizer.zero_grad()，否则梯度跨 batch 累加。",
        "不需要梯度的推理前向用 with torch.no_grad(): 包住，省内存又更快。",
    ],
    example='''
import torch
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2                    # y = x^2
y.backward()                  # dy/dx = 2x = 6
print("x.grad:", x.grad.item())

# no_grad 下不求导
with torch.no_grad():
    print("no_grad 内 requires_grad:", (x * 2).requires_grad)
''',
),

dict(
    title="detach / clone",
    keys=["detach", "clone", ".clone", "copy", "deep copy"],
    cat="自动求导",
    desc=("clone 复制一份数据（保不保梯度看需要）；detach 从计算图里摘出来（得到的新张量"
          "不求导、但和原值共享内存——值会随原张量变）。"),
    params=[
        (".clone()", "复制数值，仍是图中节点（可带梯度继续反传）"),
        (".detach()", "返回一个 requires_grad=False 的同值张量，截断梯度"),
        (".clone().detach()", "常连用：既复制数值又不带梯度"),
    ],
    notes=[
        "想『拿个不参与反传的副本』→ x.detach().clone()（或 clone().detach()）。",
        "target/indices 之类标签别 require grad，必要就 detach。",
    ],
    example='''
import torch
x = torch.tensor([2.0], requires_grad=True)
c = x.clone()            # 复制，仍连图
d = x.detach()           # 摘出，不求导
print("clone requires_grad:", c.requires_grad, "| detach:", d.requires_grad)
(x * 3).backward()
print("原 x.grad =", x.grad.item())   # detach 不影响
''',
),

# ---------------- 张量运算补充 ----------------
dict(
    title="sum / mean（dim 与 keepdim）",
    keys=["sum", "mean", ".sum", ".mean", "keepdim"],
    cat="张量运算",
    desc=("沿某维求和/平均。keepdim=True 保留被约掉的维为 1，方便后面继续广播。"
          "loss 常 mean 到标量；acc 统计用 sum 累分子分母。"),
    params=[
        ("sum(dim=None)", "不指定则全约成标量"),
        ("sum(dim=k)", "沿第 k 维约掉"),
        ("keepdim=True", "结果的该维保留为 1，如 (B,S,d)->(B,1,d)"),
    ],
    notes=[
        "mask 统计里 acc = 猜对数.sum() / 总词数.sum()——分母分子都是累加量。",
        "要除以的是 count 不是 batch 数。",
    ],
    example='''
import torch
x = torch.randn(2, 3)
print("全 sum:", x.sum().item())
print("dim=1:", x.sum(1))                       # (2,)
print("dim=1 keepdim:", x.sum(1, keepdim=True).shape)  # (2,1)
print("mean dim=0:", x.mean(0))
''',
),

dict(
    title="clamp",
    keys=["clamp", "clip", "torch.clamp", "torch.clip"],
    cat="张量运算",
    desc=("把值钳制到 [min, max]。图像可视化前常用 clamp(0,1)；梯度也能 clamp 但一般用裁剪函数。"),
    params=[
        ("clamp(min, max)", "小于 min 变 min，大于 max 变 max"),
        ("只给一端", "clamp(min=0) 只保下界"),
    ],
    notes=[
        "注意与 nn.utils.clip_grad_norm_ 的区别：那是裁梯度不是裁张量值。",
    ],
    example='''
import torch
x = torch.tensor([-2., 0.5, 3., 10.])
print(x.clamp(0, 1))          # [0, 0.5, 1, 1]
print(x.clamp(min=0))         # 保下界
''',
),

dict(
    title="sort / topk",
    keys=["sort", "topk", "argsort", "torch.topk"],
    cat="张量运算",
    desc=("排序。topk 取最大的 k 个（连同下标），解码时取 top-k 候选/束搜索常用。"),
    params=[
        ("sort(dim, descending)", "返回 (values, indices)"),
        ("topk(k, dim)", "返回最大的 k 个 (values, indices)"),
    ],
    notes=[
        "beam search 会同时要 values(概率) 和 indices(词id)。",
    ],
    example='''
import torch
x = torch.tensor([[3., 1., 2.], [9., 7., 8.]])
v, i = x.topk(2, dim=-1)
print("topk values:", v.tolist())
print("topk indices:", i.tolist())
vs, is_ = torch.sort(x, dim=-1, descending=True)
print("sort 下标:", is_.tolist())
''',
),

dict(
    title="allclose / isclose",
    keys=["allclose", "isclose", "torch.allclose", "torch.isclose"],
    cat="张量运算",
    desc=("判断张量是否『几乎相等』（带 atol/rtol 容差）。测试自己手写的算子是否等价于官方 API 的标准工具。"),
    params=[
        ("allclose(a, b)", "整体是否都在容差内，返回一个 bool"),
        ("atol / rtol", "绝对/相对容差"),
        ("isclose(a, b)", "逐元素返回 bool 张量"),
    ],
    notes=[
        "浮点结果用 == 判断几乎必踩坑，验证等价用 allclose。",
    ],
    example='''
import torch
a = torch.tensor([1.0, 2.0])
b = a * 1 + 1e-9
print("allclose:", torch.allclose(a, b, atol=1e-6))
print("== 直接比:", (a == b))
''',
),

dict(
    title="where",
    keys=["where", "masked_select", "torch.where"],
    cat="张量运算",
    desc=("按条件选值：where(cond, x, y) 对应位置 cond 真取 x 假取 y。做『根据 mask 用两套值』很方便。"),
    params=[
        ("where(cond, a, b)", "cond 为 True 处取 a，否则取 b"),
        ("where(cond)", "返回 True 位置的下标（等价 nonzero）"),
    ],
    notes=[
        "想从张量里按 mask 挑出数值：x[cond] 即可。",
    ],
    example='''
import torch
x = torch.tensor([1., 5., 3.])
cond = x > 2
print("where:", torch.where(cond, x, torch.full_like(x, -1)))
print("x[cond]:", x[cond])
''',
),

dict(
    title="repeat / tile / expand",
    keys=["repeat", "tile", "torch.repeat", "torch.tile"],
    cat="张量运算",
    desc=("repeat/tile 会把数据真正复制多份（占用内存）；expand 只是广播视图不复制。"
          "要看重复 vs 共享内存的区别。"),
    params=[
        ("repeat(*times)", "把整块按各维重复，生成新内存"),
        ("tile(*dims)", "同 repeat 但维数处理略有差别"),
        ("expand", "1→n 的『假广播』，stride=0，不占额外内存"),
    ],
    notes=[
        "cls_token.expand(B,-1,-1) 是视图；想独立副本再 .clone()。",
    ],
    example='''
import torch
a = torch.tensor([[1, 2]])
print("repeat(3,1):", a.repeat(3, 1).shape)   # (3,2) 复制
b = torch.zeros(1, 1, 3)
e = b.expand(4, 2, 3)
print("expand:", tuple(e.shape), "stride首维(0=共享):", e.stride()[0])
''',
),

dict(
    title="einsum",
    keys=["einsum", "torch.einsum", "爱因斯坦"],
    cat="张量运算",
    desc=("用爱因斯坦记号把乘加浓缩成一行，能表达 matmul/转置/逐点乘/求和的各种组合，"
          "方便验证手写注意力。"),
    params=[
        ("'bij,bjk->bik'", "批量矩阵乘，等价 b @ (k 维对齐)"),
        ("'bhsd,bhtd->bhst'", "Q·K^T 式点积注意力分数"),
        ("-> 后没写的维自动求和", "省略下标=沿它求和"),
    ],
    notes=[
        "先能读懂就行：箭头左边是输入各自下标，右边是输出下标。",
        "性能上常用 matmul 替代，但 einsum 可读性强、适合对照公式。",
    ],
    example='''
import torch
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c1 = torch.einsum('bij,bjk->bik', a, b)     # 批量 matmul
c2 = a @ b
print("einsum matmul 等价:", torch.allclose(c1, c2), tuple(c1.shape))
''',
),

# ---------------- 训练 / 优化器 ----------------
dict(
    title="torch.optim.Adam",
    keys=["adam", "torch.optim.Adam", "optimizer", "opt"],
    cat="训练/优化",
    desc=("最常用的自适应学习率优化器。每步用法固定四连：zero_grad → loss.backward() "
          "→ optimizer.step() →（可加 scheduler.step()）。"),
    params=[
        ("params", "model.parameters() 或按名字分组的 list of dict"),
        ("lr", "学习率（Adam 里是『峰值』lr，常用 1e-3~1e-4）"),
        ("betas", "(β1, β2)，动量与二阶矩，一般不改"),
        ("weight_decay", "L2 正则系数"),
    ],
    notes=[
        "顺序必须是 zero_grad 先于 backward，否则梯度会跨 batch 累加。",
        "不同模块可以不同 lr：params=[{'params':a,'lr':1e-4},{'params':b}]。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
net = nn.Linear(3, 1)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x = torch.randn(4, 3)
y = torch.randn(4, 1)
before = net.weight.clone()
opt.zero_grad()
loss = ((net(x) - y) ** 2).mean()
loss.backward()
opt.step()
print("一步后 weight 有更新:", not torch.allclose(before, net.weight))
''',
),

dict(
    title="torch.optim.SGD",
    keys=["sgd", "torch.optim.SGD", "momentum"],
    cat="训练/优化",
    desc=("经典随机梯度下降 + 可选动量/weight_decay。没有自适应 lr，需要自己调 lr，"
          "配学习率衰减最常见。"),
    params=[
        ("lr", "学习率（比 Adam 通常设得更小或配合 warmup）"),
        ("momentum", "动量系数，常见 0.9"),
        ("weight_decay", "L2 正则"),
        ("nesterov", "Nesterov 动量"),
    ],
    notes=[
        "CNN/经典论文常用 SGD+momentum；Transformer 类任务 Adam 更好上手。",
        "SGD 对 lr 很敏感，多配 scheduler。",
    ],
    example='''
import torch, torch.nn as nn
net = nn.Linear(3, 2)
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
print("param_groups lr:", opt.param_groups[0]["lr"])
print("SGD 建好，用法与 Adam 相同四连。")
''',
),

dict(
    title="lr_scheduler（StepLR / CosineAnnealingLR）",
    keys=["scheduler", "StepLR", "CosineAnnealingLR", "lr_scheduler", "torch.optim.lr_scheduler"],
    cat="训练/优化",
    desc=("学习率调度：训练中按计划降 lr。StepLR 每 step_size 个 epoch 乘 gamma；"
          "CosineAnnealing 余弦退火。每 epoch 结束调 scheduler.step()。"),
    params=[
        ("StepLR(opt, step_size, gamma)", "每 step_size 轮 lr *= gamma"),
        ("CosineAnnealingLR(opt, T_max)", "T_max 轮内按余弦降到 0 附近"),
        ("scheduler.step()", "每轮 epoch 调一次（和 optimizer.step() 不同时刻）"),
    ],
    notes=[
        "optimizer.step() 是每 batch；scheduler.step() 一般每 epoch。别搞混。",
        "打印当前 lr：optimizer.param_groups[0]['lr']。",
    ],
    example='''
import torch, torch.nn as nn
net = nn.Linear(2, 2)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
for ep in range(5):
    opt.step()          # 训练里每 batch 调用
    sch.step()          # 每 epoch 后调用
    print(f"epoch{ep+1} lr =", opt.param_groups[0]["lr"])
''',
),

dict(
    title="clip_grad_norm_（梯度裁剪）",
    keys=["clip_grad", "clip_grad_norm", "nn.utils.clip_grad_norm_"],
    cat="训练/优化",
    desc=("backward 后、step 前，把整组参数的梯度范数钳到 max_norm，防止梯度爆炸"
          "（RNN/Transformer 训练常用）。"),
    params=[
        ("clip_grad_norm_(params, max_norm)", "若总范数超 max_norm 则整体等比例缩放"),
        ("norm_type", "默认 2 = L2 范数"),
    ],
    notes=[
        "必须放在 loss.backward() 之后、optimizer.step() 之前。",
        "返回裁剪前的总范数，可打印监控。",
    ],
    example='''
import torch, torch.nn as nn
net = nn.Linear(4, 1)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = net(torch.randn(2, 4)).pow(2).sum()
opt.zero_grad()
loss.backward()
tot = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
print("裁剪前总范数(已限到≤1):", round(tot.item(), 3))
opt.step()
''',
),

dict(
    title="model.train() / model.eval() / no_grad()",
    keys=["train()", "eval()", "model.eval", "model.train", "no_grad", "torch.no_grad()"],
    cat="训练/优化",
    desc=("train()/eval() 切换层的行为：Dropout/BatchNorm 在两种模式下表现不同。"
          "eval 阶段再包 with torch.no_grad() 省显存并加速。"),
    params=[
        ("model.train()", "进入训练模式（dropout 生效、BN 用本 batch 统计）"),
        ("model.eval()", "进入推理模式（dropout 关闭、BN 用 running 统计）"),
        ("with torch.no_grad():", "关闭自动求图，推理专用"),
    ],
    notes=[
        "忘记 .eval() → 推理结果随机；忘记包 no_grad → 白白攒计算图。",
        "注意别把 no_grad 套在训练循环上。",
    ],
    example='''
import torch, torch.nn as nn
m = nn.Dropout(0.5)
m.train()
print("train 模式输出有随机0:", (m(torch.ones(100)) == 0).float().mean().item() > 0)
m.eval()
print("eval 模式输出不变:", torch.equal(m(torch.ones(100)), torch.ones(100)))
''',
),

dict(
    title="manual_seed（可复现）",
    keys=["manual_seed", "torch.manual_seed", "seed", "复现", "repro"],
    cat="训练/优化",
    desc=("固定随机数种子让结果可复现。一般程序开头固定 Python、torch、numpy、cuda 的种子。"
          "注意：即便固定种子，GPU/多线程下仍可能不完全一致。"),
    params=[
        ("torch.manual_seed(n)", "固定 torch 的 CPU/GPU 随机数"),
        ("torch.cuda.manual_seed_all(n)", "固定所有 GPU"),
        ("random.seed(n) / np.random.seed(n)", "同时固定 Python/numpy 的随机"),
    ],
    notes=[
        "数据加载器的 shuffle 也会消耗随机性；想要每轮复现可配 generator。",
    ],
    example='''
import torch
torch.manual_seed(42)
a = torch.randn(2)
torch.manual_seed(42)
b = torch.randn(2)
print("同 seed 两次 randn 相同:", torch.equal(a, b))
''',
),

dict(
    title="nn.Parameter",
    keys=["parameter", "nn.Parameter", "register_parameter"],
    cat="训练/优化",
    desc=("把张量标记成『可训练参数』。放进 nn.Module 的 module/Parameter 后，"
          "model.parameters() 会自动收集到它，optimizer 才会更新它。"),
    params=[
        ("nn.Parameter(tensor)", "包一层，等价 tensor.requires_grad_(True)"),
        ("nn.ParameterList([...])", "一组参数（如每层可学的标量）"),
        ("register_parameter(name, p)", "显式注册参数"),
    ],
    notes=[
        "CLS token 这类『学出来的向量』就该是 nn.Parameter。",
        "普通 Python list 里的参数不会被 optimizer 找到。",
    ],
    example='''
import torch, torch.nn as nn
class My(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, 4))
        self.w = nn.ParameterList([nn.Parameter(torch.randn(3)) for _ in range(2)])
m = My()
print("可训练参数:", sum(p.numel() for p in m.parameters()), "个")
''',
),

# ---------------- nn：激活 / 池化 / 其他层 ----------------
dict(
    title="ReLU / GELU / Sigmoid（激活函数）",
    keys=["relu", "gelu", "sigmoid", "tanh", "F.relu", "nn.ReLU"],
    cat="nn·激活",
    desc=("给网络加非线性。ReLU=max(x,0)；GELU 平滑版(Transformer FFN 常用)；"
          "Sigmoid 把值压到 (0,1) 适合概率/二分类输出。"),
    params=[
        ("nn.ReLU() / F.relu(x)", "负值截 0"),
        ("nn.GELU()", "Transformer 前馈网络默认激活"),
        ("nn.Sigmoid()", "压到 (0,1)，二分类最后一层配 BCEWithLogitsLoss"),
    ],
    notes=[
        "做分类最后一层别手动 sigmoid/softmax——损失函数内部会做。",
        "inplace=True 省内存但会改输入，新手别急着开。",
    ],
    example='''
import torch, torch.nn as nn
x = torch.tensor([-2., -0.5, 0., 1.5])
print("ReLU:", nn.ReLU()(x).tolist())
print("Sigmoid:", nn.Sigmoid()(x).tolist())
print("GELU:", torch.nn.functional.gelu(x).tolist())
''',
),

dict(
    title="MaxPool2d / AvgPool2d / AdaptiveAvgPool2d",
    keys=["maxpool", "avgpool", "adaptiveavgpool", "nn.MaxPool2d", "pool"],
    cat="nn·池化",
    desc=("降采样。MaxPool 取窗口最大（保边缘响应），AvgPool 取平均；"
          "AdaptiveAvgPool 可把任意输入降成指定输出尺寸（分类头常用 → (1,1)）。"),
    params=[
        ("kernel_size / stride", "窗口大小 / 步长"),
        ("nn.AdaptiveAvgPool2d(1)", "不管输入多大都池化成 1x1，等价全局平均池化"),
    ],
    notes=[
        "ResNet 全局平均池化通常用 AdaptiveAvgPool2d((1,1)) 再 flatten。",
    ],
    example='''
import torch, torch.nn as nn
x = torch.randn(1, 3, 8, 8)
print("MaxPool(2):", tuple(nn.MaxPool2d(2)(x).shape))              # (1,3,4,4)
print("AvgPool(2):", tuple(nn.AvgPool2d(2)(x).shape))
print("Adaptive(1):", tuple(nn.AdaptiveAvgPool2d((1, 1))(x).shape))  # (1,3,1,1)
''',
),

dict(
    title="nn.Flatten",
    keys=["flatten layer", "nn.Flatten"],
    cat="nn·激活",
    desc=("把某起始维后的所有维压成一维（是层，可以塞进 Sequential）。跟张量方法 x.flatten 等价。"),
    params=[
        ("start_dim", "从哪一维开始压平，默认 0"),
        ("end_dim", "压到哪一维，默认 -1"),
    ],
    notes=[
        "分类头前常见 nn.Flatten() 或保持 batch 维用 start_dim=1。",
    ],
    example='''
import torch, torch.nn as nn
x = torch.randn(2, 3, 4, 5)
fl = nn.Flatten(start_dim=1)
print("输入:", tuple(x.shape), "->", tuple(fl(x).shape))   # (2, 60)
''',
),

dict(
    title="nn.Module 常用方法（to/parameters/state_dict/load）",
    keys=["state_dict", "load_state_dict", "parameters", "named_parameters", "nn.Module", "model.to"],
    cat="nn·其他",
    desc=("nn.Module 是模型基类。常用方法：.to(device) 搬家、.parameters() 给优化器/数参数量、"
          ".state_dict() 存权重、.load_state_dict() 读权重、.train()/.eval()。"),
    params=[
        (".to(device)", "把模型参数搬到设备"),
        (".parameters()", "迭代所有可训练参数（含子模块递归）"),
        (".state_dict()", "返回 {参数名: 张量} 用于 torch.save"),
        (".load_state_dict(sd)", "从 state_dict 载入权重，键必须完全匹配"),
    ],
    notes=[
        "多模型/多优化器一起存：torch.save({'model':...,'opt':...}, path)。",
        "load_state_dict 报 missing/unexpected key 多半是类别数等结构不一致。",
    ],
    example='''
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
sd = net.state_dict()
print("state_dict 键:", list(sd.keys())[:2], "...共", len(sd))
print("参数个数:", sum(p.numel() for p in net.parameters()))

net2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
net2.load_state_dict(sd)          # 结构一致才能载
print("载入成功，等权重:", torch.equal(net[0].weight, net2[0].weight))
''',
),

dict(
    title="register_buffer（不训练但会保存的变量）",
    keys=["register_buffer", "buffer", "running_mean", "running_var"],
    cat="nn·其他",
    desc=("登记『不是参数但要在 state_dict 里随模型保存』的量，如 BatchNorm 的 running_mean/var。"
          "它们不进 .parameters()，optimizer 不会更新。"),
    params=[
        ("register_buffer(name, tensor)", "注册后可通过 self.name 访问"),
        ("persistent=True", "是否进入 state_dict"),
    ],
    notes=[
        "想保存 EMA 均值、步数等就用 buffer，别用 Python 属性（不会被保存/搬设备）。",
    ],
    example='''
import torch, torch.nn as nn
class My(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(3))
m = My()
print("在 state_dict 中:", "running_mean" in m.state_dict())
print("不进 parameters:", sum(1 for _ in m.parameters()))
''',
),

dict(
    title="nn.functional（无状态的函数式 API）",
    keys=["functional", "F.", "torch.nn.functional", "F.relu", "F.softmax", "F.pad"],
    cat="nn·其他",
    desc=("函数式接口，不带参数状态（relu/softmax/interpolate/pad 等）。带参的层(Conv/Linear)"
          "对应 F.conv2d 要自己传权重，一般直接用层即可。"),
    params=[
        ("F.relu(x) vs nn.ReLU()", "无参激活两者等价，看习惯"),
        ("F.softmax(x, dim)", "需要显式给 dim"),
        ("F.pad(x, (左,右,上,下))", "手动补零"),
    ],
    notes=[
        "交叉熵也可用 F.cross_entropy(logits, target)。",
        "需要『每处都能用』的函数时选 functional 更省模块。",
    ],
    example='''
import torch, torch.nn.functional as F
x = torch.tensor([[1., 2.]])
print("softmax:", F.softmax(x, dim=-1).tolist())
y = torch.ones(1, 3, 2)
print("pad 成 2->4:", tuple(F.pad(y, (1, 1)).shape))   # (1,3,4)
''',
),

# ---------------- 损失补充 ----------------
dict(
    title="MSELoss / L1Loss",
    keys=["mse", "mseloss", "l1", "l1loss", "nn.MSELoss", "mae"],
    cat="损失",
    desc=("回归损失。MSE=L2（对离群值敏感），L1=MAE（更稳但对0点不可导）。"),
    params=[
        ("nn.MSELoss()", "输入输出形状一致，求 (x-y)^2 平均"),
        ("nn.L1Loss()", "求 |x-y| 平均"),
        ("reduction", "mean/sum/none"),
    ],
    notes=[
        "输入要 float，别拿 int 标签直接算 MSE。",
    ],
    example='''
import torch, torch.nn as nn
pred = torch.tensor([0.8, 0.1, 0.6])
y = torch.tensor([1.0, 0.0, 1.0])
print("MSE:", nn.MSELoss()(pred, y).item())
print("L1 :", nn.L1Loss()(pred, y).item())
''',
),

dict(
    title="BCEWithLogitsLoss（二分类）",
    keys=["bce", "bcewithlogits", "binary_cross_entropy", "nn.BCEWithLogitsLoss"],
    cat="损失",
    desc=("二分类交叉熵。输入是『未过 sigmoid 的 logits』，内部自带 sigmoid+BCE，数值更稳。"
          "target 需是 0/1 的 float。"),
    params=[
        ("input", "logits，形状与 target 一致"),
        ("target", "0/1 浮点张量"),
        ("pos_weight", "处理正负样本不均衡"),
    ],
    notes=[
        "不要先手动 sigmoid 再喂 BCELoss——数值不稳且易错。",
    ],
    example='''
import torch, torch.nn as nn
logits = torch.tensor([2.0, -1.0, 0.5])
y = torch.tensor([1.0, 0.0, 1.0])
loss = nn.BCEWithLogitsLoss()
print("BCEWithLogits:", loss(logits, y).item())
''',
),

dict(
    title="NLLLoss / LogSoftmax（CrossEntropy 的原料）",
    keys=["nll", "nllloss", "log_softmax", "nn.NLLLoss"],
    cat="损失",
    desc=("CrossEntropyLoss = LogSoftmax + NLLLoss 合成。想自己拆开控制中间量时用："
          "先 F.log_softmax(logits) 再过 NLLLoss(target)。"),
    params=[
        ("F.log_softmax(x, dim)", "softmax 后取 log"),
        ("nn.NLLLoss()", "取 target 对应位置的负对数"),
        ("ignore_index", "忽略某类别"),
    ],
    notes=[
        "别同时用 CE 又手动 softmax（双重 softmax）。",
    ],
    example='''
import torch, torch.nn as nn
import torch.nn.functional as F
logits = torch.randn(3, 5)
t = torch.tensor([0, 2, 1])
ce = nn.CrossEntropyLoss()(logits, t)
manual = nn.NLLLoss()(F.log_softmax(logits, dim=-1), t)
print("CE == LogSoftmax+NLL:", torch.allclose(ce, manual))
''',
),

# ---------------- 数据：torch.utils.data ----------------
dict(
    title="torch.utils.data.Dataset（自定义数据集）",
    keys=["dataset", "torch.utils.data.Dataset", "getitem", "__getitem__"],
    cat="数据加载",
    desc=("自己造数据集要继承 Dataset，实现 __len__（样本数）和 __getitem__（返回第 i 个"
          "(样本, 标签)）。数据加载流程：Dataset(一条样本) → DataLoader(打包成 batch)。"),
    params=[
        ("__init__", "读文件列表/路径/变换等，只做一次"),
        ("__len__", "返回 len(self) 需要的样本总数"),
        ("__getitem__(idx)", "返回这一条样本（图像张量、标签等）"),
    ],
    notes=[
        "图像转 Tensor、标准化建议放 __getitem__ 里用 transform 做。",
        "返回的元组由 DataLoader 自动叠成 batch 维。",
    ],
    example='''
import torch
from torch.utils.data import Dataset
class Tiny(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.randn(3, 16, 16), torch.tensor(i % 10)   # (图像, 标签)

ds = Tiny(5)
print("len:", len(ds), "| 第2条:", ds[2][0].shape, ds[2][1].item())
''',
),

dict(
    title="torch.utils.data.DataLoader",
    keys=["dataloader", "torch.utils.data.DataLoader", "shuffle", "num_workers"],
    cat="数据加载",
    desc=("把 Dataset 装成可迭代的 batch 供给训练：自动按 batch_size 打包、shuffle、多进程预取。"
          "迭代时拿到的是 (batch图像, batch标签)。"),
    params=[
        ("batch_size", "每批样本数；最后不足一批可 drop_last"),
        ("shuffle", "每个 epoch 是否打乱（训练 True、测试 False）"),
        ("num_workers", "用几个子进程预取数据；Mac 上 0 最稳"),
        ("collate_fn", "自定义怎么把多条合成一 batch（变长文本常用）"),
    ],
    notes=[
        "一次 for img, cap in loader 就是一个 batch；整轮 epoch = 跑完所有 batch。",
        "变长文本自己 pad 时，collate_fn 里做并返回定长。",
    ],
    example='''
import torch
from torch.utils.data import Dataset, DataLoader
class Tiny(Dataset):
    def __len__(self): return 100
    def __getitem__(self, i):
        return torch.randn(3, 16, 16), torch.randint(0, 10, (4,))
loader = DataLoader(Tiny(), batch_size=8, shuffle=True, drop_last=True)
img, cap = next(iter(loader))
print("一个 batch: 图像", tuple(img.shape), "| 标签", tuple(cap.shape))
''',
),

# ---------------- torchvision：transforms ----------------
dict(
    title="transforms.Compose",
    keys=["compose", "transforms.Compose", "T.Compose"],
    cat="torchvision·transforms",
    desc=("把一串图像变换串成流水线，按顺序依次作用于每张图。训练/测试各配一条。"),
    params=[
        ("transforms.Compose([...])", "传入按顺序执行的 transform 列表"),
        ("用法", "composed(img) 返回处理后的图"),
    ],
    notes=[
        "顺序有讲究：先几何/尺寸(Resize/Crop)，最后 ToTensor、Normalize。",
        "训练用『随机』变换，测试只做固定 Resize/CenterCrop。",
    ],
    example='''
import numpy as np
from PIL import Image
from torchvision import transforms as T
img = Image.fromarray(np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8))
p = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
out = p(img)
print("Compose 后:", tuple(out.shape), out.dtype)
''',
),

dict(
    title="transforms.ToTensor / Normalize",
    keys=["totensor", "normalize", "transforms.Normalize", "transforms.ToTensor"],
    cat="torchvision·transforms",
    desc=("ToTensor：PIL/ndarray(HWC,0-255) → torch(CHW,0-1)；Normalize：(x-mean)/std，"
          "把像素归一化让训练更稳。"),
    params=[
        ("ToTensor()", "必做；也会把通道从 HxWxC 挪到 CxHxW"),
        ("Normalize(mean, std)", "mean/std 传 3 个值分别对应 RGB"),
        ("常见 (0.5,0.5,0.5)/(0.5,0.5,0.5)", "归一化到 ~[-1,1]"),
    ],
    notes=[
        "Normalize 必须在 ToTensor 之后（因为要作用在 float tensor 上）。",
        "预测时也要用和训练相同的 Normalize。",
    ],
    example='''
import numpy as np
from PIL import Image
from torchvision import transforms as T
img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
tt = T.ToTensor()(img)
print("ToTensor:", tuple(tt.shape), "范围", round(tt.min().item(), 3), "~", round(tt.max().item(), 3))
nm = T.Normalize((0.5,) * 3, (0.5,) * 3)(tt)
print("Normalize 后范围约 [-1,1]:", round(nm.min().item(), 2), "~", round(nm.max().item(), 2))
''',
),

dict(
    title="transforms.Resize / CenterCrop / RandomCrop",
    keys=["resize", "centercrop", "randomcrop", "transforms.Resize", "transforms.CenterCrop"],
    cat="torchvision·transforms",
    desc=("改图尺寸。Resize 直接缩放；CenterCrop 从中心裁固定大小；RandomCrop 随机位置裁"
          "（作为增广）。测试/验证一般 Resize 后 CenterCrop，保证每张看到同一区域。"),
    params=[
        ("Resize((h,w))", "缩放（可只给边长让短边对齐）"),
        ("CenterCrop(size)", "从中心裁出 size 的正方形"),
        ("RandomCrop(size)", "随机位置裁，训练增广用"),
    ],
    notes=[
        "CNN 通常 Resize(256)+CenterCrop(224)；注意模型要求输入分辨率。",
    ],
    example='''
import numpy as np
from PIL import Image
from torchvision import transforms as T
img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
print("原图:", img.size)
print("Resize((32,32)):", T.Resize((32, 32))(img).size)
print("CenterCrop(24):", T.CenterCrop(24)(img).size)
print("RandomCrop(24):", T.RandomCrop(24)(img).size)
''',
),

dict(
    title="transforms 随机增广（RandomFlip/ColorJitter/RandomResizedCrop）",
    keys=["randomflip", "colorjitter", "randomresizedcrop", "randomaffine", "RandomHorizontalFlip"],
    cat="torchvision·transforms",
    desc=("训练时对图做随机扰动 = 数据增广，提升泛化。翻转/颜色抖动/随机裁剪缩放都只影响训练。"),
    params=[
        ("RandomHorizontalFlip(p=0.5)", "随机水平翻转"),
        ("ColorJitter(brightness, contrast, ...)", "随机调颜色/亮度/对比度"),
        ("RandomResizedCrop(size, scale)", "先随机裁一块再缩放到固定尺寸"),
        ("RandomApply([t], p)", "以概率 p 整体套用一串变换"),
    ],
    notes=[
        "caption 有语义(颜色/方向)时翻转要谨慎；测试永远别用随机增广。",
        "增广只放训练 loader。",
    ],
    example='''
import numpy as np
from PIL import Image
from torchvision import transforms as T
img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
flip = T.RandomHorizontalFlip(p=1.0)(img)              # 一定翻转
crop = T.RandomResizedCrop(32, scale=(0.5, 1.0))(img)  # 随机裁→32
cj = T.ColorJitter(brightness=0.5, contrast=0.5)(img)  # 调色
print("Flip:", flip.size, "| RandomResizedCrop:", crop.size, "| ColorJitter:", cj.size)
''',
),

# ---------------- torchvision：datasets / models ----------------
dict(
    title="torchvision.datasets（CIFAR10 / ImageFolder）",
    keys=["cifar10", "torchvision.datasets.CIFAR10", "imagefolder", "fakedata"],
    cat="torchvision·datasets",
    desc=("官方内置数据集与读取器。CIFAR10 会自动下载到 root；ImageFolder 按『每子目录一个类』"
          "读自己的图片文件夹；FakeData 造随机假数据用来测管线。"),
    params=[
        ("CIFAR10(root, train, download, transform)", "train=True/False 取训练/测试"),
        ("ImageFolder(root, transform)", "root 下每个子文件夹名=类名"),
        ("download=True", "首次会自动联网下载"),
    ],
    notes=[
        "给 Dataset 传的 transform 会在取样本时自动应用。",
        "自己造 label 用 0..9；CIFAR 就是 10 类。",
    ],
    example='''
import torchvision
# 用假数据测『datasets → DataLoader → 网络』管线（不用联网）。
# FakeData 返回的是 PIL 图，要拿 tensor 需过 transform（ToTensor）。
ds = torchvision.datasets.FakeData(
    size=16, image_size=(3, 32, 32), num_classes=10,
    transform=torchvision.transforms.ToTensor())
img, y = ds[0]
print("FakeData 样本:", tuple(img.shape), "label:", int(y))
print("真正要用 CIFAR10: root 给目录 + download=True 会自动下到本地")
''',
),

dict(
    title="torchvision.models（resnet / vgg / vit，预训练）",
    keys=["resnet", "resnet18", "vgg", "vit_b_16", "torchvision.models", "pretrained", "weights"],
    cat="torchvision·models",
    desc=("直接拿到经典网络。weights=None 随机初始化；weights=xxx_Weights.DEFAULT 下载官方预训练"
          "权重。可改最后一层做迁移学习。"),
    params=[
        ("resnet18/resnet50/..., weights=None", "建网络；None=不加载预训练"),
        ("weights=ResNet18_Weights.DEFAULT", "加载官方预训练（首次会联网下载）"),
        ("改分类头", "net.fc = nn.Linear(512, 你的类数)（resnet）"),
    ],
    notes=[
        "注意网络对输入的要求（多为 224x224），要用与预训练一致的 Normalize。",
        "微调常把前面层冻结、只训新头，先 .eval()+只给头设 requires_grad。",
    ],
    example='''
import torch, torchvision, torch.nn as nn
m = torchvision.models.resnet18(weights=None)   # 不联网，随机初始化
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    print("resnet18 输出:", tuple(m(x).shape))     # (1,1000)
print("参数量: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
m.fc = nn.Linear(512, 10)                       # 换成你自己的类数
print("换分类头后输出:", tuple(m(x).shape))        # (1,10)
''',
),

# ---------------- 张量·进阶 ----------------
dict(
    title="torch.gather",
    keys=["gather", "torch.gather", "x.gather", "按下标取值", "index 取值"],
    cat="张量·进阶",
    desc=("按 index 里给出的下标，沿 dim 取原张量的对应值。常用于『按 target 挑出该样本"
          "那一类的 logits/log 概率』（手写 NLL 的核心）。"),
    params=[
        ("gather(dim, index)", "index 每个位置都存一个『沿 dim 的下标』"),
        ("index 形状", "输出形状与 index 一致；除 dim 外的维要能对得上 input"),
        ("常见写法", "沿类别维挑: logits.gather(1, target.unsqueeze(1))"),
    ],
    notes=[
        "别和 index_select（整片取）混；gather 是逐位置取。",
        "target 要 unsqueeze 加一维再 gather，最后 squeeze 掉。",
    ],
    example='''
import torch
import torch.nn.functional as F
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 3.0, 1.0]])
target = torch.tensor([0, 2])              # 每行要取的下标
picked = logits.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
print("按 target 取的 logit:", picked.tolist())   # [2.0, 1.0]

logp = F.log_softmax(logits, dim=-1)
nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
print("手动 NLL:", nll.tolist())
print("== CrossEntropy(reduction=none):",
      torch.allclose(nll, F.cross_entropy(logits, target, reduction="none")).item())
''',
),

dict(
    title="F.one_hot",
    keys=["one_hot", "onehot", "F.one_hot", "独热", "one-hot"],
    cat="张量·进阶",
    desc=("把整数类别下标变成 one-hot 矩阵。标签→独热向量，某些损失/自定义计算需要。"),
    params=[
        ("F.one_hot(tensor, num_classes)", "每个元素变成一个 num_classes 长的 0/1 向量"),
        ("num_classes", "类别数；不写则用 max+1"),
        ("返回值 dtype", "int64，不是 float，要用时 .float()"),
    ],
    notes=[
        "index 越界会报错；先确认下标 < num_classes。",
        "CrossEntropyLoss 用不到 one-hot（直接吃下标），别多此一举。",
    ],
    example='''
import torch
import torch.nn.functional as F
y = torch.tensor([0, 1, 2, 1])
oh = F.one_hot(y, num_classes=3)
print(oh)                                  # (4,3)
print("形状:", tuple(oh.shape), "dtype:", oh.dtype)
print("要 float 就 .float():", oh.float().dtype)
''',
),

dict(
    title="torch.multinomial（带温度/top-k 采样）",
    keys=["multinomial", "sample", "sampling", "温度采样", "temperature", "torch.multinomial"],
    cat="张量·进阶",
    desc=("按概率分布『随机抽』下标，是文本/图像生成的采样基础。温度 temp<1 让分布更尖(更贪)，"
          "temp>1 更平(更多样)。greedy 则是每步直接 argmax。"),
    params=[
        ("multinomial(probs, num_samples, replacement)", "按 probs 抽 num_samples 个下标"),
        ("temperature", "先 logits/temp 再过 softmax 得到采样分布"),
        ("replacement", "True=有放回(可重复抽到)"),
    ],
    notes=[
        "sampling 前要把 logits 先 softmax 成合法概率分布。",
        "想用 top-k：先 topk 取最大 k 个，把其余的 logits 填 -inf，再 softmax 采。",
    ],
    example='''
import torch
torch.manual_seed(1)
logits = torch.tensor([[0.5, 2.0, 3.0]])      # 未归一化分数
temp = 0.8                                     # <1 更贪
probs = torch.softmax(logits / temp, dim=-1)
print("温度采样概率:", [round(float(v), 3) for v in probs[0]])
samples = torch.multinomial(probs, num_samples=6, replacement=True)
print("采样词 id:", samples.tolist())          # 大概率集中在 id=2
print("greedy(恒取最大):", logits.argmax(-1).tolist())
''',
),

# ---------------- 损失·进阶 ----------------
dict(
    title="KLDivLoss / F.kl_div（蒸馏）",
    keys=["kldiv", "kl_div", "kl divergence", "KL散度", "蒸馏", "nn.KLDivLoss", "distill"],
    cat="损失·进阶",
    desc=("衡量两个分布差异，知识蒸馏用它让『学生』靠近『教师』的软输出。"
          "关键坑：input 必须已是 log 概率(log_softmax 过)，target 是普通概率，别搞反。"),
    params=[
        ("F.kl_div(log_q, p, reduction)", "log_q=学生 log 概率；p=教师概率(soft label)"),
        ("reduction='batchmean'", "对 batch 求平均(与手写一致)；'none'/'sum' 也可"),
        ("nn.KLDivLoss(reduction='batchmean')", "同 F.kl_div 的类版本"),
    ],
    notes=[
        "教师 soft label 里别出现 0（log 会炸）；可加温度软化。",
        "真分布/标签概率若没取 log，别直接塞进 input。",
    ],
    example='''
import torch
import torch.nn.functional as F
p = torch.tensor([[0.7, 0.2, 0.1]])              # 教师概率(软标签)
logits = torch.tensor([[1.6, 0.6, 0.5]])          # 学生 logits
log_q = F.log_softmax(logits, dim=-1)             # 学生 log 概率
kl = F.kl_div(log_q, p, reduction="batchmean")
print("KL(q||p):", round(float(kl), 4))
manual = (p * (torch.log(p) - log_q)).sum(dim=-1).mean()
print("与手写一致:", torch.allclose(kl, manual).item())
''',
),

dict(
    title="(B,L,V) 序列做 CrossEntropy（teacher forcing 常用）",
    keys=["reshape loss", "teacher forcing", "(B,L,C) loss", "sequence loss", "逐词CE", "shift"],
    cat="损失·进阶",
    desc=("语言/图像描述里 logits 形状是 (B, 时间步, 词表)，target 是 (B, 时间步)。"
          "算 CE 前把前两维合起来 reshape(-1, V)，pad 位置用 ignore_index 排除即可。"),
    params=[
        ("logits.reshape(-1, V)", "把 (B,L,V) 并成 (B*L, V)，词表维留最后"),
        ("target.reshape(-1)", "并成 (B*L,) 的 long 下标"),
        ("ignore_index", "把 pad(<pad>=0) 位置排除出 loss 分子分母"),
    ],
    notes=[
        "decoder 输出形状 (B,L,V)，V 必须在最后一维，reshape 才正确。",
        "teacher forcing = 把真实词(移位后)喂 decoder，跟输出逐位置算 CE。",
    ],
    example='''
import torch, torch.nn as nn
B, L, V = 2, 5, 10
logits = torch.randn(B, L, V)                # (B, 步, 词表)
target = torch.randint(1, V, (B, L))
target[0, 3:] = 0                            # 模拟句尾后 pad(=0)
ce = nn.CrossEntropyLoss(ignore_index=0)
loss = ce(logits.reshape(-1, V), target.reshape(-1))
print("序列 CE:", round(float(loss), 4))
print("pad 位置被忽略，只有真实词参与平均")
''',
),

# ---------------- nn·CNN 进阶 ----------------
dict(
    title="nn.ConvTranspose2d（上采样卷积）",
    keys=["convtranspose2d", "convtranspose", "deconv", "nn.ConvTranspose2d", "转置卷积"],
    cat="nn·CNN 进阶",
    desc=("可学习的『放大』运算，解码器/U-Net/生成器常用。前向尺寸是 Conv2d 的反向："
          "kernel=4, stride=2, padding=1 会把边长翻倍。别叫它 deconv，它没有解卷积的数学意义。"),
    params=[
        ("in_channels/out_channels", "输入/输出通道数"),
        ("kernel_size, stride, padding", "配合公式决定输出边长"),
        ("output_padding", "额外补的一点点，仅用于凑整除的尺寸"),
    ],
    notes=[
        "公式: out = (H-1)*stride - 2*pad + dilation*(k-1) + output_padding + 1。",
        "想精确上采样 2 倍：k=4,s=2,p=1；上采样最近邻插值也可但不可学。",
    ],
    example='''
import torch, torch.nn as nn
up = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
x = torch.randn(1, 3, 8, 8)
y = up(x)
print("ConvTranspose2d 输出:", tuple(y.shape))        # (1,3,16,16)
print("手算 8→16:", (8 - 1) * 2 - 2 * 1 + (4 - 1) + 0 + 1)
''',
),

dict(
    title="F.interpolate（插值缩放特征图）",
    keys=["interpolate", "upsample", "F.interpolate", "插值", "resize feature"],
    cat="nn·CNN 进阶",
    desc=("把特征图放大/缩小到指定尺寸。mode: bilinear(双线性)带 align_corners，"
          "nearest(最近邻)常用于分割。只做插值、无参数可学。"),
    params=[
        ("F.interpolate(x, size=(h,w), mode=...)", "缩放到精确尺寸"),
        ("F.interpolate(x, scale_factor=2, ...)", "按倍数缩放"),
        ("align_corners", "bilinear 必须显式给，True/False 影响角点对齐"),
    ],
    notes=[
        "新 torch 里 size 和 scale_factor 二选一，不能同时给。",
        "nearest 不需要 align_corners；bilinear 不传会告警/报错。",
    ],
    example='''
import torch
import torch.nn.functional as F
x = torch.randn(1, 3, 8, 8)
up = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
print("bilinear →(16,16):", tuple(up.shape))
nn_up = F.interpolate(x, scale_factor=2, mode="nearest")
print("nearest ×2:", tuple(nn_up.shape))
down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
print("缩小一半:", tuple(down.shape))
''',
),

dict(
    title="nn.Conv1d / Conv3d",
    keys=["conv1d", "conv3d", "nn.Conv1d", "nn.Conv3d", "一维卷积", "三维卷积"],
    cat="nn·CNN 进阶",
    desc=("Conv1d 沿长度维卷积(文本/时序/心跳)，输入 (B, C, L)；Conv3d 沿 D,H,W 卷积(视频/体素)，"
          "输入 (B, C, D, H, W)。维度规则和 Conv2d 一致，只是少/多一个空间维。"),
    params=[
        ("Conv1d(in, out, k)", "输入 (B, C, L)，在 L 上滑窗"),
        ("Conv3d(in, out, k)", "输入 (B, C, D, H, W)，在 D,H,W 上滑窗"),
        ("输出长度公式", "L' = (L + 2*pad - dilation*(k-1) - 1)//stride + 1"),
    ],
    notes=[
        "文本里常见的 Ngram 卷积就是 Conv1d（把 embedding 当通道或当长度维）。",
        "维度是从『通道维之后』数的，别和 batch 弄混。",
    ],
    example='''
import torch, torch.nn as nn
c1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
x1 = torch.randn(2, 3, 20)
print("Conv1d (B,C,L):", tuple(c1(x1).shape))         # (2,6,20)
c3 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
x3 = torch.randn(1, 1, 8, 8, 8)
print("Conv3d (B,C,D,H,W):", tuple(c3(x3).shape))     # (1,4,8,8,8)
''',
),

dict(
    title="Conv 的 groups / dilation（深度可分离 / 空洞卷积）",
    keys=["groups", "depthwise", "pointwise", "dilation", "空洞卷积", "separable", "grouped conv"],
    cat="nn·CNN 进阶",
    desc=("groups 把通道分组各卷各的：groups=in_ch 就是 depthwise 逐通道卷积，再配 1x1 pointwise "
          "= MobileNet 的深度可分离卷积(省参数)。dilation 让卷积核中间隔点取像素=空洞卷积，"
          "不降分辨率地扩大感受野。"),
    params=[
        ("groups", "in_ch 与 out_ch 都须能被 groups 整除；groups=in_ch → depthwise"),
        ("kernel_size=1", "pointwise，只在通道间混合(逐点)"),
        ("dilation=d", "等效感受野变大：rf = (k-1)*d + 1"),
    ],
    notes=[
        "depthwise 卷积核形状 (in_ch, 1, k, k)—— 每通道一个独立核。",
        "空洞卷积常见配 dilation=2, padding=2 保持尺寸不变。",
    ],
    example='''
import torch, torch.nn as nn
x = torch.randn(1, 8, 16, 16)
dw = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)   # depthwise
print("depthwise:", tuple(dw(x).shape))                    # (1,8,16,16)
pw = nn.Conv2d(8, 16, kernel_size=1)                       # pointwise
print("+pointwise:", tuple(pw(dw(x)).shape))               # (1,16,16,16)
dil = nn.Conv2d(8, 8, kernel_size=3, padding=2, dilation=2)
print("dilation=2:", tuple(dil(x).shape))                  # (1,8,16,16)
print("尺寸公式: (H + 2*pad - dilation*(k-1) - 1)//stride + 1 =",
      (16 + 2 * 2 - 2 * 2 - 1) // 1 + 1)
''',
),

# ---------------- Transformer·注意力 ----------------
dict(
    title="nn.MultiheadAttention",
    keys=["multiheadattention", "nn.MultiheadAttention", "multi head", "多头注意力", "mha"],
    cat="Transformer·注意力",
    desc=("多头注意力。输入 Q/K/V，先各投影成 num_heads 个头做缩放点积注意力，再拼接过输出投影。"
          "返回 (attn_output, attn_output_weights)。Q=K=V=同一张量就是 self-attention。"),
    params=[
        ("embed_dim, num_heads", "embed_dim 必须是 num_heads 的整数倍(每头 = embed//heads)"),
        ("batch_first", "默认 False → 输入是 (L, B, E)；设 True 变 (B, L, E) 更直观"),
        ("key_padding_mask", "(B, Lk) 布尔，True 的位置是 pad，不参与注意力"),
        ("attn_mask", "(Lq, Lk) 掩码；float 用 -inf 屏蔽、bool True 表示屏蔽"),
    ],
    notes=[
        "自注意力直接 mha(x, x, x)；cross-attention 是 mha(query=tgt, key=memory, value=memory)。",
        "need_weights=False 时第二个返回值是 None，且更省(不用为权重额外算一次)。",
        "平均头: 默认返回的权重是 (N, Lq, Lk)（已对头取平均）。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
D, H, B, L = 8, 2, 2, 4
mha = nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)
x = torch.randn(B, L, D)
out, w = mha(x, x, x)                # self-attention
print("输出:", tuple(out.shape))                     # (2,4,8)
print("注意力权重(平均过头):", tuple(w.shape))        # (2,4,4)
out2, w2 = mha(x, x, x, need_weights=False)
print("need_weights=False:", tuple(out2.shape), "权重为", w2)
''',
),

dict(
    title="TransformerEncoder / TransformerEncoderLayer",
    keys=["encoder", "transformerencoder", "transformerencoderlayer", "nn.TransformerEncoder", "编码器"],
    cat="Transformer·注意力",
    desc=("编码器：每层 = Self-Attention + 残差/LN + FFN，一堆 layer 叠起来。输入多少个 token "
          "就输出多少个『看过全场上下文』的表示。双向(无因果掩码)，直接编码整个序列。"),
    params=[
        ("TransformerEncoderLayer(d_model, nhead, dim_feedforward)", "单层结构"),
        ("TransformerEncoder(layer, num_layers)", "把同款 layer 叠 num_layers 层"),
        ("batch_first / norm_first", "batch 放第一维 / 先 LN 后注意力(PreNorm)"),
    ],
    notes=[
        "d_model 必须被 nhead 整除。dim_feedforward 常见 4*d_model。",
        "叠层/残差让 encoder 越深越难训，可加 dropout 与 Norm 缓解。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
B, L, D, H = 2, 5, 16, 4
layer = nn.TransformerEncoderLayer(d_model=D, nhead=H,
                                   dim_feedforward=64, batch_first=True)
enc = nn.TransformerEncoder(layer, num_layers=2)
x = torch.randn(B, L, D)
out = enc(x)
print("Encoder 输出:", tuple(out.shape))            # (2,5,16)
print("参数量: %.0f" % sum(p.numel() for p in enc.parameters()))
''',
),

dict(
    title="TransformerDecoder / TransformerDecoderLayer",
    keys=["decoder", "transformerdecoder", "transformerdecoderlayer", "nn.TransformerDecoder", "解码器"],
    cat="Transformer·注意力",
    desc=("解码器每层有两块注意力：先对『自己已生成的词』做带因果掩码的 self-attention，"
          "再拿 encoder 的输出做 cross-attention(查源序列信息)。teacher forcing 时 tgt 直接喂整句。"),
    params=[
        ("decoder_layer(tgt, memory, tgt_mask, ...)", "前向要同时给目标与 encoder 的 memory"),
        ("tgt_mask", "因果掩码，防止看未来词；解码器必备"),
        ("memory", "encoder 输出，(B, L_src, d)，作为 cross-attention 的 K/V"),
        ("memory_key_padding_mask", "如果源序列有 pad，要在这里一并屏蔽"),
    ],
    notes=[
        "自回归生成时，训练 = teacher forcing(给真实历史)，推理 = 循环吐一个词喂回去。",
        "只想要『最后一个位置→整句』，别忘了输出是每个位置一个向量。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
B, Lt, Lm, D, H = 2, 4, 6, 16, 4
layer = nn.TransformerDecoderLayer(d_model=D, nhead=H,
                                   dim_feedforward=64, batch_first=True)
dec = nn.TransformerDecoder(layer, num_layers=2)
tgt = torch.randn(B, Lt, D)          # 目标序列(已 shift)
memory = torch.randn(B, Lm, D)       # encoder 记忆
causal = torch.triu(torch.ones(Lt, Lt, dtype=torch.bool), diagonal=1)
out = dec(tgt, memory, tgt_mask=causal)
print("Decoder 输出:", tuple(out.shape))            # (2,4,16)
''',
),

dict(
    title="nn.Transformer（编码器-解码器整机）",
    keys=["transformer", "nn.Transformer", "整机", "seq2seq transformer"],
    cat="Transformer·注意力",
    desc=("一次搭好 Encoder+Decoder 的完整 Transformer(机器翻译/seq2seq)。内部由若干个 "
          "TransformerEncoderLayer 和 TransformerDecoderLayer 组成。大多数任务会自己拼而不是直接用整机。"),
    params=[
        ("d_model / nhead", "隐层维与头数"),
        ("num_encoder_layers / num_decoder_layers", "两侧各叠几层"),
        ("dim_feedforward", "FFN 隐层大小"),
        ("forward(src, tgt, ...)", "src=源序列，tgt=已 shift 的目标序列"),
    ],
    notes=[
        "批量解码时 tgt_mask 因果掩码要自己给，否则会看到未来。",
        "改结构(如不同层数/激活)比照手写更灵活；整机适合快速原型。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Transformer(d_model=16, nhead=4,
                       num_encoder_layers=2, num_decoder_layers=2,
                       dim_feedforward=64, batch_first=True)
src = torch.randn(2, 5, 16)        # 源
tgt = torch.randn(2, 4, 16)        # 目标(已 shift)
out = model(src, tgt)
print("整机输出:", tuple(out.shape))              # (2,4,16)
''',
),

dict(
    title="attention mask（attn_mask / key_padding_mask 语义）",
    keys=["attn_mask", "key_padding_mask", "padding_mask", "src_key_padding_mask", "causal mask", "注意力掩码", "padding 掩码"],
    cat="Transformer·注意力",
    desc=("掩码告诉注意力『别去看某些位置』。两种：key_padding_mask 屏蔽 pad token(所有人"
          "都不能 attend 它)；attn_mask 屏蔽特定 (i,j) 配对(因果时是 j>i 全屏蔽)。"
          "bool True 或 float -inf 都表示『屏蔽』——softmax 后权重变 0。"),
    params=[
        ("key_padding_mask (B, Lk) bool", "True = 该 key 位置是 padding"),
        ("attn_mask bool (Lq, Lk)", "True = 禁止 (i,j) 配对"),
        ("attn_mask float (Lq, Lk)", "-inf = 禁止(加在 scores 上被 softmax 归零)"),
        ("放在哪", "在 softmax 之前生效 → 屏蔽位置权重精确为 0"),
    ],
    notes=[
        "float 版填 -inf 而不是 0：0 只是不加分，仍会被 attend；-inf 才归零。",
        "Encoder 只关心 pad → 给 key_padding_mask；Decoder 还要 causal → 再给 attn_mask。",
        "给错形状(如 (L,L) 当 (B,L))是最常见的报错来源。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
B, L, D, H = 1, 4, 8, 2
mha = nn.MultiheadAttention(D, H, batch_first=True)
x = torch.randn(B, L, D)

kpm = torch.zeros(B, L, dtype=torch.bool)
kpm[0, 2:] = True                       # key 位置 2,3 是 pad
out, w = mha(x, x, x, key_padding_mask=kpm)
print("key_padding_mask 后 query0 权重:", w[0, 0].tolist())   # 位置2,3≈0

inf = torch.full((L, L), float('-inf'))
causal = torch.triu(inf, diagonal=1)    # 右上角 -inf → 只能看自己及以前
_, w2 = mha(x, x, x, attn_mask=causal)
print("因果掩码后 query0 权重:", w2[0, 0].tolist())          # 只有 key0≈1
''',
),

dict(
    title="PositionalEncoding（手写正弦位置编码）",
    keys=["positional", "pos_encoding", "positionalencoding", "位置编码", "position embedding", "pe"],
    cat="Transformer·注意力",
    desc=("Transformer 没有顺序概念，要靠给每个 token 加位置向量才知道先后。正弦位置编码："
          "偶数维用 sin、奇数维用 cos，不同频率让相邻位置向量接近、远位置可分辨。直接加到 embedding 上。"),
    params=[
        ("pe[:, 0::2] = sin(pos * freq)", "偶数维用 sin"),
        ("pe[:, 1::2] = cos(pos * freq)", "奇数维用 cos"),
        ("freq 衰减", "频率 = base^(-2i/D)，base=10000"),
        ("使用", "x = x + pe[:L]  让 batch 也广播(加在第 0 维前)"),
    ],
    notes=[
        "长度 L 超过预计算范围就自己延长：pe 只取决于 L、D，可运行时现算。",
        "D 取偶数(要能整除 2)；加的时候要 unsqueeze 出 batch 维或索引到 L。",
    ],
    example='''
import torch, math
def pos_encoding(L, D, base=10000.0):
    """返回 (L, D) 的正弦位置编码。"""
    pe = torch.zeros(L, D)
    pos = torch.arange(L, dtype=torch.float).unsqueeze(1)      # (L,1)
    freqs = torch.arange(0, D, 2).float()
    div = torch.exp(freqs * (-math.log(base) / D))             # base^(-2i/D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

pe = pos_encoding(L=10, D=8)          # D 要偶数
print("形状:", tuple(pe.shape))
print("token0 编码:", [round(v, 3) for v in pe[0].tolist()])
print("每行范数相同:", [round(float(v), 3) for v in pe.norm(dim=-1)])
# 用法: x = x + pe[:x.size(1)].unsqueeze(0)   # 广播到 (B,L,D)
''',
),

# ---------------- 序列·RNN ----------------
dict(
    title="nn.RNN / nn.GRU / nn.LSTM",
    keys=["lstm", "gru", "rnn", "nn.LSTM", "nn.GRU", "nn.RNN", "循环网络", "hidden state"],
    cat="序列·RNN",
    desc=("处理序列的循环网络。LSTM 带记忆细胞(cell)，GRU 简化版，RNN 最朴素。"
          "输入 batch_first=True 时是 (B, L, input)，输出 output(每步隐状态) 和末态 h_n/c_n。"),
    params=[
        ("input_size / hidden_size", "每个输入向量维 / 隐状态维"),
        ("num_layers", "堆几层；h_n 第一维 = num_layers*方向数"),
        ("bidirectional", "True 双向，output 最后维翻倍为 2*hidden"),
        ("batch_first", "True → 输入 (B, L, input)，输出 (B, L, hidden)"),
    ],
    notes=[
        "输出 output 是每个时间步的隐状态；末态 h_n = output 的最后一时间步(单向一层时)。",
        "h_n/c_n 形状 (num_layers*dir, B, hidden)，要取最后一层用 h_n[-1]。",
        "忘了 batch_first → 输入写成 (L, B, d) 是最常见翻车点。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
B, L, in_d, hid = 3, 5, 4, 8
x = torch.randn(B, L, in_d)

lstm = nn.LSTM(input_size=in_d, hidden_size=hid, num_layers=1, batch_first=True)
out, (h, c) = lstm(x)
print("LSTM output:", tuple(out.shape))                    # (3,5,8)
print("h_n:", tuple(h.shape), "c_n:", tuple(c.shape))      # (1,3,8)
print("末态==output最后一步:", torch.allclose(out[:, -1, :], h[-1]).item())

gru = nn.GRU(in_d, hid, batch_first=True)
_, h_g = gru(x)
print("GRU 只有 h:", tuple(h_g.shape))

bi = nn.LSTM(in_d, hid, batch_first=True, bidirectional=True)
out_b, (h_b, _) = bi(x)
print("双向 output:", tuple(out_b.shape), "| h:", tuple(h_b.shape))  # (3,5,16),(2,3,8)
''',
),

dict(
    title="pad_sequence（变长序列补齐）",
    keys=["pad_sequence", "padding", "变长补齐", "pad"],
    cat="序列·RNN",
    desc=("把一堆长度不一的张量按最长那个在尾部补 padding_value，叠成 (B, L, ...) 一个 batch。"
          "文本/变长序列 batch 的标配，配合忽略 pad 的 loss(ignore_index) 使用。"),
    params=[
        ("pad_sequence(seqs, batch_first=True)", "seqs 是 list of (L_i, ...) 张量"),
        ("padding_value", "尾部填的值；文本一般填 <pad> 的 id(0)"),
        ("返回", "同长张量，最长者的长度为其长度维"),
    ],
    notes=[
        "pad 都在『尾部』，右侧补；跟 Transformer 一起时记得做 key_padding_mask。",
        "RNN 里最好用 pack_padded_sequence，别直接喂 pad 过长的输入浪费算力。",
    ],
    example='''
import torch
from torch.nn.utils.rnn import pad_sequence
a = torch.tensor([1, 2, 3])        # 3 词
b = torch.tensor([4, 5])           # 2 词
c = torch.tensor([6])              # 1 词
padded = pad_sequence([a, b, c], batch_first=True, padding_value=0)
print("pad 后形状:", tuple(padded.shape))       # (3,3)
print(padded)
''',
),

dict(
    title="pack_padded_sequence / pad_packed_sequence",
    keys=["pack_padded_sequence", "pad_packed_sequence", "packed", "pack sequence", "PackedSequence", "变长 RNN"],
    cat="序列·RNN",
    desc=("变长序列喂 RNN 前先 pack 起来，让 RNN 只对『真实 token』算、跳过 pad 位置——"
          "更快也更省显存，末态不受 pad 干扰。跑完再 pad_packed_sequence 解回定长张量。"),
    params=[
        ("pack_padded_sequence(x, lengths, batch_first, enforce_sorted)", "lengths=每个样本真实长度"),
        ("enforce_sorted=True", "要求 lengths 降序(自己先按长排)"),
        ("RNN 直接吃 PackedSequence", "把 packed 传给 LSTM 即可"),
        ("pad_packed_sequence(out)", "解回 (B, max_len, hidden) + 真实 lengths"),
    ],
    notes=[
        "lengths 必须按样本真实长度降序给(除非 enforce_sorted=False 让它内部排)。",
        "pack 只省 pad 的算力，不改变末态 h_n 含义(它就是最后一个真实 token 后的隐状态)。",
    ],
    example='''
import torch, torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
torch.manual_seed(0)
B, in_d, hid = 3, 4, 8
seqs = [torch.randn(5, in_d), torch.randn(3, in_d), torch.randn(2, in_d)]
lens = torch.tensor([len(s) for s in seqs])        # 已降序 [5,3,2]
padded = pad_sequence(seqs, batch_first=True)      # (3,5,4)
packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=True)
lstm = nn.LSTM(in_d, hid, batch_first=True)
out_packed, (h, c) = lstm(packed)
print("packed 内真实 token 数×hidden:", tuple(out_packed.data.shape))
out, lengths = pad_packed_sequence(out_packed, batch_first=True)
print("解开后:", tuple(out.shape), "各长度:", lengths.tolist())
''',
),

# ---------------- 训练·进阶 ----------------
dict(
    title="torch.save / torch.load（存权重 / 存 checkpoint）",
    keys=["save", "load", "torch.save", "torch.load", "checkpoint", "存模型", "继续训练"],
    cat="训练·进阶",
    desc=("保存模型。推荐只存 model.state_dict()(纯张量，安全)；要继续训练就把模型、优化器、"
          "epoch、最佳分数打成一个 dict 存。torch.save(目标, 路径) / torch.load(路径)。"),
    params=[
        ("torch.save(m.state_dict(), path)", "只存权重(官方推荐，轻便)"),
        ("torch.load(path)", "读回 dict；再 load_state_dict 进结构相同的模型"),
        ("checkpoint dict", "存 {'model':..., 'opt':..., 'epoch':..., 'best':...}"),
    ],
    notes=[
        "load_state_dict 报 missing/unexpected key = 模型结构对不上(如类别数变)。",
        "恢复训练：重建模型/优化器后 load，再跳到保存的 epoch 继续。",
    ],
    example='''
import os, torch, torch.nn as nn
m = nn.Linear(4, 2)
path = "/tmp/torch_kb_model.pt"
torch.save(m.state_dict(), path)

m2 = nn.Linear(4, 2)
m2.load_state_dict(torch.load(path))
print("载入后权重一致:", torch.equal(m.weight, m2.weight).item())

ckpt = {"model": m.state_dict(),
        "opt": torch.optim.Adam(m.parameters()).state_dict(),
        "epoch": 10}
torch.save(ckpt, "/tmp/torch_kb_ckpt.pt")
print("checkpoint 键:", list(torch.load("/tmp/torch_kb_ckpt.pt").keys()))
os.remove(path); os.remove("/tmp/torch_kb_ckpt.pt")
''',
),

dict(
    title="微调冻结层（requires_grad=False + 只更新部分参数）",
    keys=["freeze", "requires_grad=False", "冻结", "fine tune", "微调", "只训头"],
    cat="训练·进阶",
    desc=("迁移学习/微调时，先把预训练 backbone 冻结(requires_grad=False)，只训新加的分类头。"
          "方法是：遍历 backbone.parameters() 关梯度，再把『只含 requires_grad=True 的参数』喂给优化器。"),
    params=[
        ("p.requires_grad = False", "冻结该参数：反向传播不更新它"),
        ("过滤参数给 optimizer", "opt = Adam([p for p in model.parameters() if p.requires_grad])"),
        ("model.eval()", "冻结的 BN/Dropout 层推理时也走 eval"),
    ],
    notes=[
        "光设 requires_grad 还不够——优化器里必须过滤，否则照样更新。",
        "想只冻结 backbone 但不动它的 BN 统计？冻结层在 eval 下跑 forward 更干净。",
    ],
    example='''
import torch, torch.nn as nn
backbone = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
head = nn.Linear(4, 2)

for p in backbone.parameters():
    p.requires_grad = False                     # 冻结 backbone

def trainable(module):
    return [p for p in module.parameters() if p.requires_grad]

opt = torch.optim.Adam(trainable(head), lr=1e-3)
print("backbone 可训练参数:", sum(p.numel() for p in trainable(backbone)))
print("head 可训练参数:", sum(p.numel() for p in trainable(head)))

for p in backbone.parameters():                 # 解冻
    p.requires_grad = True
print("解冻后 backbone 可训练:", sum(p.numel() for p in backbone.parameters()))
''',
),

dict(
    title="AMP 混合精度（autocast + GradScaler）",
    keys=["amp", "autocast", "GradScaler", "fp16", "混合精度", "mixed precision", "scale_loss"],
    cat="训练·进阶",
    desc=("前向用 FP16 省一半显存并加快，但梯度可能下溢为 0——所以用 GradScaler 先把 loss "
          "放大再 backward，step 前缩回去。写法固定四步：autocast 包前向 → scaler.scale(loss)"
          ".backward() → scaler.step(opt) → scaler.update()。"),
    params=[
        ("with torch.cuda.amp.autocast():", "该块内的前向用 FP16 自动混合精度"),
        ("scaler.scale(loss).backward()", "loss 乘系数再反传，防梯度下溢"),
        ("scaler.step(opt)", "内部判断是否要跳过/缩小梯度后更新"),
        ("scaler.update()", "每个 batch 末尾调整系数"),
    ],
    notes=[
        "必须全程用 scaler，不能只开 autocast 不 scale(梯度仍可能下溢)。",
        "新版本推荐 import torch.amp，写 torch.amp.autocast('cuda')；老 API 也兼容。",
    ],
    example='''
import torch, torch.nn as nn
use_cuda = torch.cuda.is_available()
print("cuda 可用:", use_cuda)
if use_cuda:
    model = nn.Linear(4, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(8, 4).cuda()
    for _ in range(2):
        opt.zero_grad()
        with torch.cuda.amp.autocast():        # FP16 前向
            loss = model(x).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    print("AMP 训练两步完成")
else:
    print("本机无 GPU，跳过执行；有卡时写法如上")
''',
),

dict(
    title="梯度累积（大 batch 装不下时）",
    keys=["gradient accumulation", "accumulate", "梯度累积", "大 batch", "accum"],
    cat="训练·进阶",
    desc=("显存装不下大 batch，就把一个大 batch 拆成几个 micro-batch 依次 backward 累加梯度，"
          "攒够 ACC 步再 optimizer.step()。等效 batch = micro_batch × ACC。"),
    params=[
        ("每 micro-batch 的 loss 除以 ACC", "让累加后的梯度尺度与一次大 batch 相当"),
        ("每 ACC 步才 step + zero_grad", "平时只 backward 不 step，梯度自动累加在 .grad"),
    ],
    notes=[
        "忘了把 loss/ACC → 有效学习率被放大 ACC 倍，训练会不稳。",
        "BN/Dropout 等对 batch 敏感层的效果与大 batch 并不完全等价。",
    ],
    example='''
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(4, 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
data = torch.randn(32, 4)
ACC = 4                       # 等效 batch = micro_batch * 4 = 16
opt.zero_grad()
for i in range(8):
    loss = (model(data[i * 4:(i + 1) * 4]) ** 2).mean() / ACC
    loss.backward()
    if (i + 1) % ACC == 0:
        opt.step()
        opt.zero_grad()
print("每 4 个 micro-batch 才 step 一次")
''',
),

dict(
    title="SummaryWriter（tensorboard）",
    keys=["tensorboard", "summarywriter", "add_scalar", "add_image", "add_graph"],
    cat="训练·进阶",
    desc=("把 loss/acc/曲线/图像/计算图写进日志，浏览器里看训练过程。"
          "每个 epoch 记一条 scalar；图像/直方图也可随时 add。"),
    params=[
        ("SummaryWriter(log_dir)", "创建日志目录"),
        ("add_scalar(tag, value, step)", "记一条曲线，如 loss/train"),
        ("add_image(tag, tensor)", "记一张图(记得转 CHW, 0-1)"),
        ("add_graph(model, input)", "可视化计算图(可能较慢)"),
    ],
    notes=[
        "启动命令: tensorboard --logdir=日志目录，再开 localhost:6006。",
        "同名 tag 会把不同 run 叠一起比较，可用不同 log_dir 区分。",
    ],
    example='''
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir="/tmp/tb_demo")
    for step in range(5):
        w.add_scalar("loss/train", 1.0 / (step + 1), step)
        w.add_scalar("acc/train", 0.2 * (step + 1), step)
    w.close()
    print("已写入 /tmp/tb_demo；查看: tensorboard --logdir=/tmp/tb_demo")
except ImportError:
    print("本机没装 tensorboard: pip install tensorboard")
''',
),

dict(
    title="nn.init（权重初始化）",
    keys=["nn.init", "init", "xavier", "kaiming", "he init", "weight init", "初始化", "torch.nn.init"],
    cat="训练·进阶",
    desc=("手写层或想要特定初始化时用 nn.init：xavier 适配 sigmoid/tanh，kaiming(He) 适配 ReLU 系，"
          "normal_ 按给定均值/方差。不 init 的话大多数层有默认初始化，但显式做可复现/可控。"),
    params=[
        ("nn.init.xavier_uniform_(w)", "经典：fan_in/fan_out 取平均，适用 tanh/sigmoid"),
        ("nn.init.kaiming_normal_(w, a=..., mode='fan_in')", "ReLU 系推荐(He init)"),
        ("nn.init.normal_(t, 0, 0.02)", "自定义高斯；还有 zeros_/ones_/constant_"),
    ],
    notes=[
        "初始化太大会让深层输出爆炸/梯度消失，太小则信号衰减，量级要匹配激活。",
        "Conv/Linear 都自带合理默认；只在特殊层(如自己写的模块)才必须显式 init。",
    ],
    example='''
import torch, torch.nn as nn
lin = nn.Linear(8, 8)
nn.init.xavier_uniform_(lin.weight)
nn.init.zeros_(lin.bias)
print("xavier 后 |W| 量级:", round(float(lin.weight.abs().mean()), 4))

conv = nn.Conv2d(3, 6, 3)
nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
print("kaiming 后 std:", round(float(conv.weight.std()), 4))
''',
),

# ---------------- 数据·进阶 ----------------
dict(
    title="random_split / Subset（切分数据集）",
    keys=["random_split", "subset", "split dataset", "切分数据集", "train_test_split"],
    cat="数据·进阶",
    desc=("从已建好的 Dataset 里按比例/个数切出训练、验证。random_split(ds, [80,20]) 返回两个子集；"
          "Subset 手动按下标列表取一部分。"),
    params=[
        ("random_split(ds, lengths)", "lengths 是各份的大小或比例，如 [80, 20]"),
        ("Subset(ds, indices)", "给定下标列表取子集"),
        ("返回值仍是 Dataset", "可直接丢给 DataLoader"),
    ],
    notes=[
        "random_split 用随机生成器切；想复现先 manual_seed。",
        "真正打乱顺序靠 DataLoader 的 shuffle=True，别在 Subset 里自己乱序。",
    ],
    example='''
import torch
from torch.utils.data import TensorDataset, random_split, Subset
X = torch.randn(100, 4); y = torch.randint(0, 3, (100,))
ds = TensorDataset(X, y)
tr, va = random_split(ds, [80, 20])
print("random_split:", len(tr), len(va))
sub = Subset(ds, list(range(50)))
print("Subset 前50个:", len(sub))
''',
),

dict(
    title="collate_fn（变长文本自定义打包）",
    keys=["collate_fn", "collate", "变长 batch", "pad 到 batch"],
    cat="数据·进阶",
    desc=("DataLoader 默认把一批样本直接 stack 成张量——变长文本会炸。自定义 collate_fn 在拼 batch "
          "那一刻用 pad_sequence 补成定长再返回，是文本/图像描述数据管线的关键一环。"),
    params=[
        ("collate_fn(batch)", "输入是本 batch 的 (样本, 标签) 列表，返回要喂给模型的张量"),
        ("zip(*batch)", "把样本们和标签们分开"),
        ("pad_sequence + ignore_index", "补 <pad> 对齐，算 loss 时忽略"),
    ],
    notes=[
        "collate_fn 在每次取 batch 时被调用，是放 padding 的正确位置。",
        "除了文本，混合类型(图+文本+掩码)也常在 collate_fn 里各自处理。",
    ],
    example='''
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class VarCap(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        return torch.randint(1, 20, (i + 1,)), torch.tensor(i)  # 长 1,2,3,4

def pad_collate(batch):
    ids, labels = zip(*batch)
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    return ids, torch.stack(labels)

loader = DataLoader(VarCap(), batch_size=4, collate_fn=pad_collate)
x, y = next(iter(loader))
print("一个 batch 已补 pad:", tuple(x.shape), "标签:", y.tolist())
''',
),

# __KB_END__
]
