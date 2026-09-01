---
title: "From-Activation-to-Initialization-Scaling-Insights-for-Optim"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Saratchandran_From_Activation_to_Initialization_Scaling_Insights_for_Optimizing_Neural_Fields_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:39:17"
---

# 论文速读：From-Activation-to-Initialization-Scaling-Insights-for-Optim

## 一句话总结
本文首次为采用正弦、sinc、高斯或小波激活的神经场（Neural Fields）建立了梯度下降收敛至全局最优的严格缩放定律，揭示了激活函数与初始化方案对参数量需求的决定性影响，并据此提出了一种末端层降方差初始化策略，将网络缩放复杂度从超线性/超二次显著降至线性/二次。

## 研究问题与动机
- 神经场在视觉与物理建模中广泛应用，但缺乏理解其随数据规模扩展如何缩放参数的理论框架，实践中多依赖经验性堆砌参数量。
- 现有理论结果（如针对ReLU网络的分析）所需参数缩放阶数较高（二次/三次），且部分基于实践中极少使用的$\mathcal{N}(0,1)$初始化，难以指导实际工程。
- 不清楚为何正弦/sinc/高斯/小波等激活在神经场中表现优异，亟需从优化理论层面解释激活选择与初始化策略的内在关联。
- 需要一套统一理论，明确给定数据集大小$N$时，网络宽度应满足何种渐近下界才能保证梯度下降有效收敛。

## 核心贡献（创新点）
- 首次为sine、sinc、Gaussian与wavelet激活的神经场给出了严格的收敛缩放定律证明，确立标准初始化下浅层需$\Omega(N^{3/2})$、深层需$\Omega(N^{5/2})$的参数需求。与Nguyen等针对ReLU网络的研究相比，本文覆盖更多实战常用激活并揭示激活类型对缩放阶数的本质影响。
- 提出Initialization 1（Normal）与Initialization 2（Uniform）两种新型初始化方案，将缩放复杂度降至浅层$\Omega(N)$（线性）与深层$\Omega(N^2)$（二次）。与以往依赖理想化方差或要求$\Omega(N^4)$过参数化的工作不同，本文方案直接对接LeCun/Xavier/Kaiming等实践常用初始化框架。
- 从理论上揭示“末尾层权重采样方差越小，所需过参数化程度越低”的内在机制，通过随机矩阵谱界与隐藏层最小奇异值下界建立完整证明链条。该机制填补了初始化设计与全局收敛理论之间的空白。
- 在图像回归、单图超分、占据场重建、NeRF与PINNs五大类神经场任务上系统验证了理论预测与初始化方案的有效性与优越性。与先前仅停留于理论推导的工作相比，本文提供了跨模态的完整实证支撑。

## 方法详解
- **问题设定与记号**：考虑深度$L$的前馈网络，输入$X\in\mathbb{R}^{N\times n_0}$（$N$个i.i.d.次高斯样本，归一化至单位范数），使用MSE损失$\mathcal{L}$。过参数化定义为网络参数量大于$N$。
- **标准初始化下的缩放定理**：
  - **浅层网络（Thm 4.2）**：当隐藏层宽度满足$n_1 = \Omega(N^{3/2})$、输出层宽度为常数时，采用LeCun/Xavier/Kaiming初始化与sine/sinc/Gaussian激活，小学习率下梯度下降w.h.p.收敛至全局最优。
  - **深层网络（Thm 4.4）**：除倒数第二层需$n_{L-1} = \Omega(N^{5/2})$外，其余层保持常数宽度，同样条件下可保证收敛。
- **证明核心技术路线**：核心在于建立隐藏层初始输出矩阵最小奇异值下界$\sigma_0^2 \geq 16\sqrt{N}\sqrt{n_1}\sqrt{2\mathcal{L}(\theta_0)}||W_2^0||_2$。该不等式依赖激活函数的谱性质（sinc/sine/Gaussian/wavelet均有界导数与良好频域衰减）。结合LeCun初始化下随机矩阵谱范数界$||W_2^0||_2 = \mathcal{O}(\sqrt{n_2/n_1})$，代入即得$n_1 = \Omega(N^{3/2})$。
- **新初始化设计（Initialization 1 & 2）**：
  - 洞察：若将最后一层权重方差由$1/n_{L-1}$缩小至$1/n_{L-1}^{3/2}$，则$||W_L^0||_2$量级下降，从而降低对$n_{L-1}$的依赖。
  - **Initialization 1（Normal）**：$(W_l^0)_{ij} \sim \mathcal{N}(0, 1/n_{l-1})$（$l < L$），$(W_L^0)_{ij} \sim \mathcal{N}(0, 2/n_{L-1}^{3/2})$。
  - **Initialization 2（Uniform）**：对应均匀分布版本，$(W_l^0)_{ij} \sim \mathcal{U}(-1/\sqrt{n_{l-1}}, 1/\sqrt{n_{l-1}})$，末层改为$\mathcal{U}(-1/n_{L-1}^{3/4}, 1/n_{L-1}^{3/4})$。
- **新缩放定理**：
  - **Thm 4.7（浅层）**：使用Initialization 1，仅需$n_1 = \Omega(N)$即可保证收敛。
  - **Thm 4.9（深层）**：使用Initialization 1，仅需$n_{L-1} = \Omega(N^2)$即可保证收敛。

## 实验与结果
- **理论验证实验**：
  - **浅层1D曲线拟合**：拟合$f(x)=\sin(2\pi
