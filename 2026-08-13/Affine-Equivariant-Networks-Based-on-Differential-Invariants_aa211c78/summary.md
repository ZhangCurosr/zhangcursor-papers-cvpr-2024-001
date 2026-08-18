---
title: "Affine-Equivariant-Networks-Based-on-Differential-Invariants"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Li_Affine_Equivariant_Networks_Based_on_Differential_Invariants_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:59:28"
field: "等变深度学习与几何深度学习"
keywords: ["equivariant network", "affine group", "differential invariant", "symmetric PDE", "deep learning", "computer vision", "group symmetry"]
innovations: ["首次在无群离散/采样前提下基于对称PDE与微分不变量构建仿射等变网络", "提出SupNorm归一化相对微分不变量（SNDI）避免除零并增强表达能力", "设计即插即用InvarLayer等变层并统一覆盖仿射群及其连续子群"]
benchmarks: ["Scale-MNIST", "Scale-Fashion", "RS-MNIST", "RS-Fashion", "affNIST"]
---

# 论文速读：Affine-Equivariant-Networks-Based-on-Differential-Invariants

## 一句话总结
本文提出了一种全新的仿射等变网络框架 InvarPDEs-Net 和可插拔层 InvarLayer，首次在不离散化/采样仿射群的前提下，基于对称 PDE 与微分不变量理论实现了仿射等变性，在 scale、rotation-scale 和 affine 三类非欧几里得群的分类任务上均取得最优或接近最优结果，out-of-distribution 下 affNIST 准确率较 affConv 提升 3.37%。

## 研究问题与动机
- **高维连续群的离散化困境**：现有等变网络多依赖群离散化或采样（如 group convolution），对高维仿射群而言不可行；affConv 虽通过在李代数上积分避免了直接采样，但仍需采样且显存随层数指数增长。
- **微分不变量理论的未充分开发**：微分不变量理论揭示了 PDE 群对称性与不变量的等价关系，但此前工作仅局限于 Euclidean 群及其子群，仿射群等高维非compact 群尚未被有效利用。
- **分数多项式不变量的除零问题**：仿射群的基本微分不变量以分数多项式形式出现（如 Hessian 行列式除以退化二次型），在图像均匀区域分母趋零，数值不稳定。
- **模型灵活性不足**：既有仿射等变方法难以适配现代 CNN 的变通道数和残差结构，无法直接替换标准卷积层。

## 核心贡献（创新点）
1. **首次无离散化/采样的仿射等变网络**：从对称 PDE 视角构建 InvarPDEs-Net，用微分不变量代替群积分，彻底规避 affConv 的层数限制与显存爆炸问题。
2. **SupNorm 归一化微分不变量（SNDI）**：提出将多项式相对微分不变量除以全局 sup 范数构造新不变量，避免除零并保持比经典基本不变量更强的表达能力（保留更多自由度）。
3. **InvarPDEs-Net 变通道架构**：通过 1×1 卷积线性组合跨 PDE 的通道，实现变通道深度的等变堆叠，天然含 skip connection。
4. **InvarLayer 即插即用等变层**：抽取单步迭代构造可自由指定输入/输出通道数的等变层，可直接替换各类 CNN 的卷积层。
5. **统一框架覆盖仿射群及其连续子群**：scale 群、rotation-scale 群、equi-affine 群等均可复用同一套设计与实现代码。

## 方法详解
- **基本设定**：将图像建模为 $\mathbb{R}^2$ 上的光滑向量函数 $\mathbf{u}: X \to \mathbb{R}^n$；群 $G$（仿射群元素 $g = (\mathbf{A}, \mathbf{b})$，$\mathbf{A} \in \mathbb{R}^{2\times 2}$ 可逆，$\mathbf{b}\in\mathbb{R}^2$）作用在函数集上为 $(g\cdot\mathbf{u})(\mathbf{x}) = \mathbf{u}(g^{-1}\cdot\mathbf{x})$。
- **对称 PDE 刻画等变性**：演化 PDE $\frac{\partial \tilde{\mathbf{u}}}{\partial t} = \mathbf{H}^{(t)} \circ \hat{\mathcal{T}}_{FDI}[\mathbf{u}^{(t)}]$，其中 $\mathcal{T}_{FDI}$ 为 $G$ 的基本微分不变量拼接；当且仅当右端由不变量构成时 PDE 为 $G$-对称。
- **时间离散化**：前向差分 $\mathbf{u}^{(t_{i+1})} = \mathbf{u}^{(t_i)} + \Delta t_i \cdot \mathbf{h}_{\theta_i} \circ \hat{\mathcal{T}}_{SNDI}[\mathbf{u}^{(t_i)}]$，每步迭代视为一个等变算子 $\Psi_i$，整体网络 $\Psi = \Psi_{N-1}\circ\cdots\circ\Psi_0$，天然含残差连接。
- **SNDI 构造**（定理 7）：对权重为 $w(g)$ 的多项式相对微分不变量集合 $\mathcal{I}_i$，定义 $\mathcal{T}_i(\mathbf{x},\mathbf{u}) = \frac{\mathcal{I}_i(\mathbf{x},\mathbf{u})}{\|\mathcal{I}(\cdot,\mathbf{u})\|_{\sup}}$，其中 $\|\cdot\|_{\sup} = \sup_{\mathbf{x}\in X}\|\cdot\|_\infty$；分子分母同时乘 $w(g)$ 抵消，保持 $G$-不变性。归一化后还具备亮度不变性 $\mathcal{T}(\mathbf{x},c\cdot\mathbf{u})=\mathcal{T}(\mathbf{x},\mathbf{u})$。
- **数值实现**：空间导数用 Gaussian 导数核卷积近似；MLP $\mathbf{h}_\theta$ 用带 ReLU 的 1×1 卷积实现（空间共享权重）；sup 范数用对应通道的全局 Max-Pooling 计算。
- **InvarLayer**：单步输出 $\mathbf{u}_{out} = \mathbf{h}_\theta \circ \hat{\mathcal{T}}_{SNDI}[\mathbf{u}_{in}]$，去掉 skip connection，允许任意 $C_{in}\to C_{out}$ 通道变换。
- **网络前后处理**：输入将单通道图像沿通道维复制至目标维度；输出经全局池化得不变特征后接两个全连接层分类。BatchNorm、点态非线性、Dropout 均保持等变性。

## 实验与结果
- **数据集**：Scale-MNIST / Scale-Fashion（MNIST/Fashion-MNIST 随机缩放 $[0.3,1]$ 后零填充回 $28\times 28$，训练 10k/测试 50k）；RS-MNIST / RS-Fashion（再加均匀旋转 $[0,2\pi]$，上采样至 $56\times 56$，训练 5k/测试 50k）；affNIST（在 50k 非变换 MNIST 上训练，在 320k 仿射扰动 MNIST 上测试，OOD 设置）。
- **基线**：scale 等变——SiCNN、SI-ConvNet、SEVF、DSS、SS-CNN、SESN、ScDCFNet、SE-CNN；rotation-scale——RST-CNN、SFCNN、RDCF；affine——affConv [37]、Capsule Network 系列（CapsNet、GE CapsNet、affine CapsNet、RU CapsNet）。所有模型参数量均控制在 $\approx 500\text{k}$ 以内。
- **Scale 结果**：InvarPDEs-Net 在 Scale-MNIST 达 **98.30%**（SOTA）、Scale-Fashion 达 **89.62%**（SOTA）；InvarLayer 在 Scale-MNIST 97.75%，Scale-Fashion 89.50%。
- **Rotation-Scale 结果**：InvarPDEs-Net 在 RS-MNIST 达 **95.80%**（SOTA）、RS-Fashion 达 **79.48%**（SOTA）；InvarLayer 在调参后 RS-MNIST 93.40%、RS-Fashion 76.08%。
- **Affine 结果**：InvarLayer 在 affNIST OOD 下达 **98.45%**，较 affConv（95.08%）提升 **+3.37%**，并同时超越 RU CapsNet（97.69%）；InvarPDEs-Net 达 95.72%。
- **结论**：InvarPDEs-Net 作为端到端架构在多数场景下最优；InvarLayer 作为即插即用模块在 affNIST 上刷新 SOTA，验证了框架的有效性与工程实用性。

## 相关工作脉络
- **Group Convolution（Cohen & Welling, 2016）**：将特征图视为群上函数做卷积；本文与其根本区别是不对群做离散/采样，直接用微分不变量构造等变映射，故可处理高维非紧群（仿射群）。
- **affConv（MacDonald et al., 2022）**：在李代数上积分实现仿射等变；本文避免了其仍依赖采样以及层数增加时显存指数增长的缺陷，理论上实现严格等变且模型规模不随群增大而膨胀。
- **Learnable PDE / neupDOs（Liu et al., 2010/2013; He et al., 2022）**：利用偏微分算子构造 Euclidean 等变系统；本文将其推广至仿射群及子群，核心差异是用微分不变量理论约束 PDE 右端而非直接约束 PDO 权重。
- **Steerable CNNs（Weiler & Cohen, 2018）**：以矢量场方式构造等变性；本文走微分不变量路线，不依赖旋量/特征表示的显式设计。
- **Capsule Networks（Sabour et al., 2017 及后续仿射改进版）**：通过动态路由体现几何先验，但缺乏严格等变理论保证；本文提供严密的李群理论支撑并在 affNIST 上超越最强 CapsNet（RU CapsNet）。
- **Scale-equivariant 网络（Sosnovik et al., 2019; Worrall & Welling, 2019; Naderi et al., 2020）**：本文框架同样适用于 scale 子群，并与 SESN、ScDCFNet、SE-CNN 等在同参数预算下直接对比。

## 局限性与未来方向
- **InvarLayer 表现波动**：作者在结论中承认 InvarLayer 在部分实验设置下性能存在一定波动，需要超参调整。
- **任务与数据范围有限**：仅在 2D 图像分类上验证，未见其他视觉任务（检测、分割）或更大规模数据集（CIFAR、ImageNet）的实证。
- **未探索更一般李群**：理论框架可推广至满足正则条件的 Lie 群，但文中仅覆盖仿射群及其连续子群，通用性有待验证。
- **流形扩展未涉及**：三维空间、球面等其他流形上的推广被明确列为未来工作。
- **计算开销**：SNDI 需全局 sup 范数（全局 Max-Pooling），对于高分辨率图像可能引入额外内存与通信开销，文中未详细讨论。

## 研究启发与可借鉴点
- **归一化替代除法**：SupNorm 归一化技巧可推广至其他会产生分数不变量的群（如投影群、共形群），为解决除零问题提供通用范式。
- **PDE 迭代 ↔ 网络深度的理论对应**：将对称 PDE 的前向差分迭代视作网络层，为等变网络设计提供可解释的架构归纳偏置，可启发新的结构化网络生成方法。
- **即插即用等变层的工程价值**：InvarLayer 与标准 CNN 组件（BatchNorm、ReLU、Dropout）完全兼容，证明"等变性 + 工程可用性"可兼得，便于在非等变基线模型上快速引入对称先验。
- **亮度不变性作为隐式正则**：SNDI 全局归一化自然带来亮度不变性，无需额外监督信号；类似思想可迁移至光照鲁棒表征学习。
- **统一多群框架**：同一套代码适配 scale、rotation-scale、affine 等多个群，提示可将多群等变性统一到"微分不变量库"的思路，减少重复实现。

## 关键术语表
- **Equivariance（等变性）**：算子 $\Psi$ 满足 $\Psi[g\cdot \mathbf{u}] = g\cdot \Psi[\mathbf{u}]$，输入经群作用后输出按相同方式变换。
- **Differential Invariant（微分不变量）**：由函数及其各阶导数组成的量，在群的延拓作用下保持不变。
- **Fundamental Differential Invariant（基本微分不变量）**：任意同阶微分不变量均可由其函数表达的一组独立不变量。
- **Polynomial Relative Differential Invariant（多项式相对微分不变量）**：满足 $\mathcal{I}(g\cdot\mathbf{x}, g\cdot\mathbf{u}) = w(g)\mathcal{I}(\mathbf{x},\mathbf{u})$ 的多项式微分表达式，$w(g)$ 为正的权重因子。
- **SupNorm Normalized Differential Invariant (SNDI)**：将相对微分不变量除以全局 sup 范数得到的新不变量，避免除零且信息容量更高。
- **Symmetric PDE（对称 PDE）**：右端仅由群 $G$ 的微分不变量构成的 PDE，其解流保持 $G$-等变性。
- **InvarPDEs-Net**：由多个不同维度对称 PDE 迭代堆叠而成的端到端等变网络。
- **InvarLayer**：从 PDE 迭代中提取的单步等变模块，支持任意输入/输出通道数，可作为卷积的即插即用替换。

## 可复现要素
- **数据集**：Scale-MNIST、Scale-Fashion、RS-MNIST、RS-Fashion 为作者自行生成（随机缩放/旋转+零填充），affNIST 为公开数据集（基于 MNIST）。
- **代码/权重**：论文未明确声明开源仓库（截至原文发表时），需关注作者机构页面或后续更新；权重文件未单独列出。
- **关键超参**：缩放因子范围 $[0.3, 1]$，旋转均匀分布于 $[0, 2\pi]$；训练样本数 Scale 类 10k/RS 类 5k；所有实验模型参数量控制在 $\approx 500\text{k}$；InvarLayer 在 ResNet-32 结构上测试；Gaussian 导数核的标准差 $\sigma$ 与网格点数 $N$ 论文正文未给出具体值（见 Supplementary Material）；前向差分步长 $\Delta t_i$ 未明确。
