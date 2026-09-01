---
title: "From-Activation-to-Initialization-Scaling-Insights-for-Optim"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Saratchandran_From_Activation_to_Initialization_Scaling_Insights_for_Optimizing_Neural_Fields_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:43:37"
---

# 论文速读：From-Activation-to-Initialization-Scaling-Insights-for-Optim

## 一句话总结
本论文建立了神经场（Neural Fields）优化过程中激活函数与初始化方案对网络规模（Overparameterization）需求的理论框架，并据此设计了更高效的初始化方案，使得正弦、高斯等频率型激活网络的参数规模需求从次线性降至线性（浅层）或从超二次降至二次（深层），显著优于 ReLU 基线。

## 研究问题与动机
- **理论框架缺失**：尽管神经场在计算机视觉等领域应用广泛，但目前缺乏对其架构设计、初始化与优化收敛性之间关系的系统性理论理解。
- **缩放机制不明**：在大模型时代，随着数据集规模扩大，现有方法往往通过启发式（ad hoc）方式调整网络大小，缺乏关于"多少参数才能保证梯度下降收敛到全局最优"的理论指导。
- **现有技术局限**：现有的缩放法则（如 Nguyen 等人的工作）主要针对 ReLU 激活，指出浅层需二次方、深层需三次方增长；而其他基于平滑激活的结果多依赖罕见初始化方案，与工程实践脱节。
- **工程效率诉求**：标准初始化方案（Xavier, Kaiming, LeCun）与频率激活组合下，参数缩放需求过高导致训练效率低，需探索更优初始化以匹配激活特性。

## 核心贡献（创新点）
1. **首次建立频率激活的缩放定律**：证明了使用正弦、sinc、高斯和小波激活的神经场，在标准初始化下，浅层需 $\Omega(N^{3/2})$、深层需 $\Omega(N^{5/2})$ 的参数过参数化水平，优于 ReLU 的二次/三次法则。
2. **揭示激活-初始化耦合机制**：从理论上阐明了隐藏层激活函数与最后一层初始化方差之间的内在联系——较小方差可降低所需的过参数化程度。
3. **提出新型初始化方案**：基于理论洞察，设计了两种新的初始化方案（Initialization 1: 高斯小方差；Initialization 2: 均匀分布小方差），将参数需求从次线性/超二次降至线性/二次。
4. **广泛实验验证**：在图像回归、超分辨率、占据场、NeRF 和 PINN 等多种神经场应用上验证了理论的预测能力和新初始化方案的优越性。

## 方法详解

### 4.1 理论基础

**网络模型**：考虑深度为 $L$ 的网络，输入维度 $n_0$，隐藏层宽度 $\{n_1, \ldots, n_L\}$，输出 $F_k$ 定义为：
$$F_k = \phi(F_{k-1}W_k + b_k), \quad k \in [L-1]; \quad F_L = F_{L-1}W_L + b_L$$

**过参数化定义**：若参数数量大于样本数 $N$，则称网络过参数化。

### 4.2 浅层网络缩放定律（定理 4.2）

对于深度为 2 的网络，激活函数为 $\sin(\omega x)$、$\text{sinc}(\omega x)$ 或 $e^{-x^2/2\omega^2}$，使用 LeCun 初始化：
$$(W_1^0)_{ij} \sim \mathcal{N}(0, 1/n_0), \quad (W_2^0)_{ij} \sim \mathcal{N}(0, 1/n_1)$$
当隐藏层宽度 $n_1 = \Omega(N^{3/2})$ 时，梯度下降以高概率收敛到全局最优。

**关键证明思路**：
- 建立不等式：$\sigma_0^2 \geq 16\sqrt{N}\sqrt{n_1}\sqrt{2\mathcal{L}(\theta_0)}\|W_2^0\|_2$
- 利用激活函数特性：$\sigma_0 \geq \Omega(\sqrt{n_1})$ 且 $\sqrt{2\mathcal{L}(\theta_0)} = \mathcal{O}(\sqrt{N})$
- 应用随机矩阵理论：$\|W_2^0\|_2 = \mathcal{O}(\sqrt{n_2}/\sqrt{n_1})$（LeCun 初始化）
- 最终导出 $n_1 \geq CNn_1^{3/4}$，即 $n_1 = \Omega(N^{3/2})$

### 4.3 深层网络缩放定律（定理 4.4）

对于深度 $L > 2$ 的网络，激活函数同上，使用 LeCun 初始化：
$$(W_l^0)_{ij} \sim \mathcal{N}(0, 1/n_{l-1}), \quad \forall l \in [L]$$
当倒数第二层宽度 $n_{L-1} = \Omega(N^{5/2})$ 时，梯度下降以高概率收敛。

### 4.4 新初始化方案设计

**理论洞察**：减小最后一层权重的方差可以降低所需的过参数化程度。设最后一层权重方差为 $\mathcal{N}(0, 1/n_{L-1}^p)$，则 $p$ 越大，参数需求越低。

**Initialization 1（高斯方案）**：
$$\begin{aligned} (W_l^0)_{ij} &\sim \mathcal{N}(0, 1/n_{l-1}), \quad l \in [L-1] \\ (W_L^0)_{ij} &\sim \mathcal{N}(0, 2/n_{L-1}^{3/2}) \end{aligned}$$
最后一层方差缩小 $1/\sqrt{n_{L-1}}$ 倍。

**定理 4.7（浅层）**：使用 Initialization 1 时，$n_1 = \Omega(N)$ 即可收敛（线性缩放）。

**定理 4.9（深层）**：使用 Initialization 1 时，$n_{L-1} = \Omega(N^2)$ 即可收敛（二次缩放）。

**Initialization 2（均匀分布方案）**：
$$\begin{aligned} (W_l^0)_{ij} &\sim \mathcal{U}(-1/\sqrt{n_{l-1}}, 1/\sqrt{n_{l-1}}), \quad l \neq L \\ (W_L^0)_{ij} &\sim \mathcal{U}(-1/n_{L-1}^{3/4}, 1/n_{L-1}^{3/4}) \end{aligned}$$
均匀分布范围同样在最后一层缩小 $1/\sqrt{n_{L-1}}$ 倍。

## 实验与结果

### 5.1 理论验证实验

**浅层实验**（1D 曲线拟合，$f(x) = \sin(2\pi x) + \sin(6\pi x) + \sin(10\pi x)$）：
- sinc 激活网络参数增长符合 $\mathcal{O}(N^{3/2})$
- ReLU-PE 网络参数增长符合 $\mathcal{O}(N^2)$
- 相同数据量下，sinc 网络所需参数量显著低于 ReLU-PE

**深层实验**（512×512 图像重建，目标 PSNR=25dB）：
- sinc 激活 + 标准初始化：符合超二次缩放
- sinc 激活 + Initialization 1：符合二次缩放
- 超过 30,000 样本后，sinc 网络以少于样本数的参数达到收敛

**均匀初始化对比**：
- Initialization 2 在所有数据集上均表现最优
- 相比标准均匀初始化，参数需求大幅降低

### 5.2 单图超分辨率（DIV2K 数据集，4× 放大）
- 数据集：DIV2K [1, 34]
- 评估指标：PSNR、SSIM [36]
- 网络：2 层隐藏层高斯激活（方差 $0.1^2$）
- **最佳结果**：
  - Initialization 1（高斯）：最高 train dB 和 SSIM
  - Initialization 2（均匀）：最高 train dB 和 SSIM
  - 相比 Kaiming/Xavier/LeCun 均有提升

### 5.3 占据场（Occupancy Fields）
- 对象：泰国雕像（XYZ RGB Inc.）
- 网络：Gabor 小波激活，3 层隐藏层，宽度 128
- 评估指标：PSNR、IOU [12, 35]
- **最佳结果**（Initialization 2）：
  | 初始化方案 | Train PSNR (dB) | IOU |
  |-----------|----------------|-----|
  | Initialization 1 | 22.7 | 0.89 |
  | LeCun Normal | 20.3 | 0.81 |
  | Xavier Normal | 21.2 | 0.84 |
  | Kaiming Normal | 19.9 | 0.80 |
  | **Initialization 2** | **24.5** | **0.91** |
  | LeCun Uniform | 21.2 | 0.86 |
  | Xavier Uniform | 21.6 | 0.87 |
  | Kaiming Uniform | 20.2 | 0.82 |

- **提升幅度**：相比次优初始化，PSNR 提升约 2-4 dB，IOU 提升约 0.04-0.11

### 5.4 NeRF 重建（Tanks & Temples 数据集 Lego 场景）
- 训练集：Kaiming uniform 初始化高斯激活 NeRF
- 测试集：24 个未见视角
- **最佳结果**：
  - Initialization 2 训练 PSNR 提升 1.1 dB
  - 测试 PSNR 提升 0.1-1.1 dB（跨不同场景）
- **结论**：在训练和测试集上均优于 Kaiming 初始化

### 5.5 PINN（Navier-Stokes 方程求解）
- 方程：2D 不可压缩 Navier-Stokes
- 网络：3 层隐藏层，宽度 128，高斯激活
- 评估：总损失 PSNR（MSE + PDE 约束）
- **最佳结果**：
  - Initialization 1 和 2 在总损失和 PDE 损失上均达到最高 dB
  - 相比标准初始化提升约 1-2 dB

### 最强结果总结
- **占据场 IOU**：0.91（Initialization 2），提升约 11% vs Kaiming
- **超分辨率 SSIM**：Initialization 1/2 显著超越基线
- **NeRF PSNR**：测试集提升 0.1-1.1 dB

## 相关工作脉络

1. **Du 等人 [10]**：使用 NTK 参数化证明平滑激活网络需 $\Omega(N^4)$ 过参数化，但使用不常见的标准正态初始化；本文使用更实用的 LeCun/Xavier/Kaiming 初始化，证明仅需 $\Omega(N^{3/2})$。

2. **Huang & Yau [14]**：使用 Neural Tangent Hierarchy 证明 $\Omega(N^3)$ 可保证收敛；本文针对频率激活得到更优的 $\Omega(N^{3/2})$。

3. **Nguyen [20]**：证明深度 ReLU 网络需 $\Omega(N^3)$；本文证明频率激活配合新初始化只需 $\Omega(N^2)$，显著优于 ReLU。

4. **Sitzmann 等人 [29]**：提出了正弦激活的均匀初始化；本文与其结合并证明更优，显示理论指导可改进实践经验。

5. **Arora 等人 [4]、Oymak & Soltanolkotabi [21]**：针对两层 ReLU 网络证明宽度需 $\Omega(N^4)$；本文证明对于频率激活的浅层网络，$\Omega(N^{3/2})$ 即可。

6. **Allen-Zhu 等人 [3]、Zou & Gu [37]**：证明固定输入输出层的 ReLU 网络需高阶多项式过参数化；本文针对神经场常用激活给出更低阶的缩放定律。

## 局限性与未来方向

1. **mini-batch 训练不适用**：理论结果基于 full-batch gradient descent，实际应用中多使用 mini-batch，当前理论无法直接推广到该场景（论文明确提及）。

2. **假设理想条件**：理论推导假设输入数据为 i.i.d 次高斯向量且归一化，实际数据可能违反这些假设。

3. **频率超参数选择**：激活函数的频率参数 $\omega$（或 $1/\omega^2$）需要手动调优，缺乏自动选择策略。

4. **仅针对特定激活**：理论主要适用于正弦、sinc、高斯和小波等频率激活，对 ReLU 等传统激活的缩放分析较间接。

5. **未来方向**：将理论推广到 mini-batch 训练、自动超参数选择、以及结合更多实用场景（如大规模 NeRF 训练）是未来的重要研究方向。

## 研究启发与可借鉴点

1. **初始化-激活联合设计范式**：本文展示了通过理论分析揭示激活与初始化之间的耦合关系，并据此设计更优初始化，这一方法论可迁移到其他网络设计问题。

2. **最后一层方差控制技巧**：减小输出层权重方差可降低过参数化需求，这一技巧可应用于其他坐标型网络或隐式表示学习任务。

3. **理论预测指导实验设计**：理论预测的 $\mathcal{O}(N^{3/2})$ vs $\mathcal{O}(N^2)$ 缩放关系在实验中得到精确验证，展示了理论-实践结合的有效性，可为后续研究树立范例。

4. **频率激活的普适优势**：对于神经场类应用，使用正弦/高斯/小波激活相比 ReLU+PE 可显著降低参数需求，这一结论可直接指导后续神经场架构设计。

5. **超参数敏感性分析**：高斯激活方差设为 $0.1^2$ 的经验最佳值，提示在实际应用中应对激活函数的频率/方差超参数进行系统敏感性分析。

## 关键术语表

**Neural Fields（神经场）**：使用坐标型神经网络参数化连续几何结构或视觉现象的方法，将空间坐标映射到信号值（如颜色、密度、符号距离）。

**Overparameterization（过参数化）**：神经网络参数数量超过训练样本数的现象，理论证明这是梯度下降收敛到全局最优的必要条件。

**Scaling Law（缩放定律）**：描述网络规模（宽度/深度）如何随数据集大小增长的数学关系，指导网络设计以适应不同数据规模。

**Random Fourier Features（RFF）**：一种位置编码技术，将输入坐标映射到高维空间以增强网络对高频信号的表达能力。

**Initialization 1/2**：本文提出的两种新初始化方案，通过在最后一层缩小方差（高斯或均匀分布），降低收敛所需的参数规模。

**PINN（Physics-Informed Neural Networks）**：将物理方程（如 PDE）作为约束嵌入损失函数的神经网络，用于求解正逆问题。

**Occupancy Field（占据场）**：用神经网络二值化表示 3D 形状隐式边界的方法，输出表示点是否在物体内部。

**NTK（Neural Tangent Kernel）**：刻画神经网络在无限宽度极限下训练动力学的核函数，用于理论分析梯度下降收敛性。

## 可复现要素

- **数据集**：
  - DIV2K（超分辨率）[1, 34]
  - Thai statue（占据场，来源 XYZ RGB Inc.）
  - NeRF real synthetic Lego（NeRF 重建）
  - Navier-Stokes 方程（PINN，解析解已知）
  - 公开数据集均已在相关工作中引用，具体代码未开源

- **代码/权重**：论文未提及开源代码或预训练权重

- **关键超参数**：
  - 高斯激活方差：$0.1^2$
  - 频率参数 $\omega$ 或 $1/\omega^2$：实验部分提及见补充材料 Sec. 2
  - 位置编码维度：浅层 8 维，深层 16 维
  - 训练目标 PSNR：浅层 35 dB，深层 25 dB
  - 优化器：Adam，full-batch gradient descent
  - 论文未明确提及 batch size（因使用 full-batch）
