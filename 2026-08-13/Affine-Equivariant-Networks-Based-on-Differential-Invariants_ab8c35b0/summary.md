---
title: "Affine-Equivariant-Networks-Based-on-Differential-Invariants"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Affine_Equivariant_Networks_Based_on_Differential_Invariants_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:59:27"
field: "等变深度学习"
keywords: ["等变网络", "仿射等变", "微分不变量", "对称PDE", "SupNorm归一化", "affNIST", "InvarLayer"]
innovations: ["首次不依赖群离散化/采样的仿射等变网络，从对称PDE视角构建等变层", "提出SupNorm归一化微分不变量(SNDI)，消除仿射群分数不变量的除零问题并增强表达力", "设计可插拔等变层InvarLayer，可直接替换CNN卷积层，参数更少且在affNIST OOD上提升3.37%"]
benchmarks: ["affNIST", "Scale-MNIST", "Scale-Fashion", "RS-MNIST", "RS-Fashion"]
---

# 论文速读：Affine-Equivariant-Networks-Based-on-Differential-Invariants

## 一句话总结
本文从对称偏微分方程（PDE）的视角出发，首次在不离散化/采样仿射群的前提下，利用微分不变量构建了仿射等变网络（InvarPDEs-Net / InvarLayer）。针对仿射群基本微分不变量的分数多项式形式导致的除零问题，提出一种基于SupNorm归一化多项式相对微分不变量的新技术（SNDI），在 scale、旋转-缩放及仿射变换的分类任务上均取得最优或接近最优结果。

## 研究问题与动机
- **仿射等变难以实现**：仿射群维度高、非紧，传统群卷积方法需对群离散化或采样，组越大模型参数量呈指数增长，无法直接用于深层网络。
- **现有仿射等变方法仍依赖采样**：affConv [37] 在李代数上积分但仍需采样；[39] 依赖 GL(n,R)-不变测度的采样，理论上等变性不严格。
- **微分不变量思路尚未推广至仿射群**：PDO-based 等变方法（如 PDE-net、neural ePDOs）集中于欧氏群子群（平移、旋转），仿射群上的潜力未被探索。
- **分数微分不变量存在数值除零风险**：仿射群基本微分不变量通常表现为分数多项式，在图像平坦区域易出现分母为零，影响训练稳定性。

## 核心贡献（创新点）
1. **首次不依赖群离散化/采样的仿射等变网络**：从对称 PDE 视角，将仿射等变性转化为微分不变量的组合问题，从根本上规避了 affConv 等方法因采样导致的内存指数膨胀与深度受限。
2. **提出 SupNorm 归一化微分不变量（SNDI）**：通过全局空间归一化多项式相对微分不变量替代经典基本微分不变量，既消除除零问题，又保留更多信息（表达力强于经典不变量）。
3. **设计可插拔等变层 InvarLayer**：作为对称 PDE 迭代过程的灵活提取，支持任意输入/输出通道数，可直接替换各种 CNN 架构中的卷积层。
4. **统一框架覆盖仿射群及其连续子群**：scale 群、旋转-缩放群、仿射群共用同一构造流程，仅更换对应的微分不变量即可适配不同群。
5. **out-of-distribution 上超越 affConv 3.37%**：在 affNIST 上，InvarLayer（配合 ResNet-32）达到 98.45%，较 affConv（95.08%）提升 3.37%，同时超过 RU CapsNet。

## 方法详解
### 理论框架：对称 PDE → 等变网络
- 将图像视为 $\mathbb{R}^2$ 上的光滑函数，特征提取过程建模为如下演化 PDE：
  $$\frac{\partial \tilde{\mathbf{u}}}{\partial t} = \mathbf{H}^{(t)} \circ \hat{\mathcal{T}}_{FDI}[\mathbf{u}^{(t)}]$$
  其中 $\mathcal{T}_{FDI}$ 为群 G 的基本微分不变量，$\mathbf{H}^{(t)}$ 为时变光滑函数（由神经网络近似）。
- **PDE 的 G-对称性**当且仅当其右端为微分不变量的函数（定理依据 [42]）。
- **时间离散化**采用前向差分：$\mathbf{u}^{(t_{i+1})} = \mathbf{u}^{(t_i)} + \Delta t_i \cdot \mathbf{h}_{\theta_i} \circ \hat{\mathcal{T}}_{FDI}[\mathbf{u}^{(t_i)}]$，每次迭代对应网络一层，天然包含残差连接。
- **空间离散化**用高斯导数近似各阶偏导数（$f_x \approx \sum_n \partial_x G(x_n;\sigma) f(x_n+x_0)$），实现为卷积核运算。

### SNDI（SupNorm 归一化微分不变量）
- 仿射群基本微分不变量为分数多项式（如 $\frac{u_{xx}u_{yy}-u_{xy}^2}{u_y^2 u_{xx}-2u_x u_y u_{xy}+u_x^2 u_{yy}}$），存在除零风险。
- 构造策略：选取一组同权重的多项式相对微分不变量 $\mathcal{I}_i$，定义 SupNorm：$\|\mathbf{v}\|_{\sup} = \sup_{\mathbf{x}} \|\mathbf{v}(\mathbf{x})\|_\infty$，构造：
  $$\mathcal{T}(\mathbf{x},\mathbf{u}) = \frac{1}{\|\mathcal{I}(\cdot,\mathbf{u})\|_{\sup}} \cdot \mathcal{I}(\mathbf{x},\mathbf{u})$$
  该构造满足等变性（Theorem 7），且**同时归一化多个同权同次的相对不变量可保留它们之间的相对关系**，信息损失更少（仅需牺牲最多 $k$ 个自由度，vs 除法损失至少 $M^2$ 自由度）。
- SNDI 还具备**光照不变性**：$\mathcal{T}(\mathbf{x}, c\cdot\mathbf{u}) = \mathcal{T}(\mathbf{x}, \mathbf{u}), \forall c > 0$。

### 网络结构
- **InvarPDEs-Net**：多个不同维度 PDE 迭代堆叠，相邻 PDE 间用 $1\times1$ 卷积线性融合通道以匹配维度（线性组合保持等变性）。
- **InvarLayer**：提取单步迭代 $\mathbf{u}_{out} = \mathbf{h}_\theta \circ \hat{\mathcal{T}}_{SNDI}[\mathbf{u}_{in}]$，无残差连接，输入/输出通道数可自由指定，等价于普通卷积层的可插拔替代品。
- **兼容标准组件**：BatchNorm、ReLU、Dropout、$1\times1$ Conv 均可无缝集成（Proposition 3 保证后Composition 不破坏等变性）；全局池化输出不变特征。

## 实验与结果
### Scale 等变（Scale-MNIST / Scale-Fashion）
- 数据集：MNIST/Fashion-MNIST 按 $[0.3, 1]$ 均匀随机缩放后补零恢复原尺寸；每集训练 10k、测试 50k。
- 参数约束：所有模型约 500K 参数。
- **Scale-MNIST**：InvarPDEs-Net 取得最高 **98.30%**（较次优 SESN 97.92% 提升 0.38%）。
- **Scale-Fashion**：InvarPDEs-Net **89.62%** / InvarLayer **89.50%**，显著超越所有基线（次优 SE-CNN 86.19%，提升 ~3.4%）。

### 旋转-缩放等变（RS-MNIST / RS-Fashion）
- 数据集：MNIST/Fashion-MNIST 施加 $[0,2\pi]$ 均匀旋转 + $[0.3,1]$ 缩放后上采样至 56×56；训练 5k、测试 50k。
- **RS-MNIST**：InvarPDEs-Net **95.80%**（次优 RST-CNN 93.19%，提升 2.61%）。
- **RS-Fashion**：InvarPDEs-Net **79.48%**（次优 RST-CNN 78.64%，提升 0.84%）。

### 仿射等变（affNIST OOD 设定）
- 训练：50k 原始 MNIST（40×40）；测试：320k 仿射扰动的 affNIST（40×40）。
- **InvarLayer + ResNet-32**：达到 **98.45%**（参数仅 365K，少于 affConv 的 373K），较 affConv（95.08%）提升 **3.37%**，超过 RU CapsNet（97.69%）0.76%。
- InvarPDEs-Net 亦优于 affConv（95.72% vs 95.08%）。

## 相关工作脉络
1. **Group Convolution 路线**：Cohen & Welling [3,4] 开创等变网络，后续扩展到 Euclid 群子群（旋转、SO(3) 等）。仿射群的高维非紧性使直接推广不可行，本文彻底绕开此路线。
2. **affConv [37]**：首个仿射等变网络，通过李代数积分实现，但仍依赖采样且深度受限；本文从理论上消除采样需求，并允许任意深度。
3. **Lie Group Decomposition [39]**：同样基于采样，且用 log-normal 替代 GL(n,R)-不变测度，等变性不严格；本文无需任何测度假设。
4. **Learnable PDE / PDO 路线**：Liu et al. [33,34]、neural ePDOs [25,27] 利用偏微分算子构造欧氏群等变网络；本文将其推广至仿射群，核心区别在于使用微分不变量而非固定 PDO 约束。
5. **Capsule Network 路线**：CapsNet / GE CapsNet / RU CapsNet 等通过动态路由获得仿射鲁棒性但缺乏严格等变性证明；本文提供严格的数学等变性保障。
6. **Scale equivariant CNNs**：SS-CNN [19]、SE-CNN [41]、SESN [53] 等针对尺度群；本文框架统一覆盖 scale、rotation-scale、affine 三类群。

## 局限性与未来方向
- **当前仅验证图像分类任务**：未探索目标检测、分割等更复杂视觉任务的适用性。
- **2D 平面限制**：方法尚未扩展至球面、3D 流形等其他几何空间。
- **InvarLayer 性能波动**：论文承认 InvarLayer 在某些设定下存在波动，层设计仍有优化空间。
- **一般 Lie 群的推广**：微分不变量对满足正则条件的 Lie 群存在，但如何系统化地将更多一般李群的不变量接入网络框架尚待研究。

## 研究启发与可借鉴点
1. **SNDI 归一化技巧可迁移**：SupNorm 归一化方法可推广至其他存在分数不变量的群（如投影群、共形群），避免除零的同时增强表达能力，值得在后续群等变网络设计中复用。
2. **对称 PDE 视角统一构造等变网络**：将"等变网络 = 对称 PDE 的时间离散化"作为设计原则，可作为未来其他群等变网络（如 SE(3)、similarity group）的通用设计范式。
3. **InvarLayer 的可插拔架构**：允许灵活指定输入/输出通道的等变层设计，可结合 ResNet / Transformer 等主流骨干网络，在医疗影像等仿射扰动显著的领域快速落地。
4. **微分不变量 + MLP 的组合方式**：用 $1\times1$ 卷积实现 MLP 空间共享权重是工程上高效且等变保真的实现技巧，适用于各类基于不变量的等变网络构建。
5. **与团队方向的结合机会**：若团队涉及非刚性/仿射变换下的特征学习、或三维点云/曲面上的等变表示，可将本框架的微分不变量构造思想迁移至相应流形，探索新的等变算子。

## 关键术语表
- **Equivariance（等变性）**：算子 $\Psi$ 满足 $\Psi[g\cdot\mathbf{u}] = g\cdot\Psi[\mathbf{u}]$，即输入经群作用后输出以相同方式变换。
- **Differential Invariant（微分不变量）**：由函数及其各阶偏导数组成的量，在群作用延长下保持不变。
- **Relative Differential Invariant（相对微分不变量）**：在群作用下按权函数 $w(g)$ 伸缩的量，满足 $\mathcal{I}(g\cdot\mathbf{x}, g\cdot\mathbf{u}) = w(g)\mathcal{I}(\mathbf{x},\mathbf{u})$。
- **SupNorm Normalized Differential Invariant (SNDI)**：用全局 SupNorm 归一化多项式相对微分不变量构造的新型不变量，避免除零且保留更多信息。
- **Symmetric PDE（对称偏微分方程）**：右端仅由群的微分不变量构成的 PDE，其解流保持群的对称性。
- **InvarPDEs-Net**：由多个可学习对称 PDE 迭代堆叠构成的完整等变网络。
- **InvarLayer**：提取自 PDE 迭代过程的等变层模块，支持任意输入/输出通道数，可插拔替换卷积层。
- **affNIST**：在 MNIST 上施加随机仿射变换生成的 out-of-distribution 分类基准数据集。

## 可复现要素
- **数据集**：Scale-MNIST、Scale-Fashion、RS-MNIST、RS-Fashion（论文中代码生成）、affNIST（公开）；均未提供独立下载链接，从 MNIST/Fashion-MNIST 合成。
- **代码/权重**：论文未提及开源仓库或预训练权重（截至发表时）。
- **关键超参**：高斯核标准差 $\sigma$、前向差分步长 $\Delta t_i$、MLP 层数与宽度、通道数配置；具体数值见 Supplementary Material（论文未提及）。
