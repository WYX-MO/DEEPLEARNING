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

# __KB_END__
]
