---
title: "Denoising-Point-Clouds-in-Latent-Space-via-Graph-Convolution"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Mao_Denoising_Point_Clouds_in_Latent_Space_via_Graph_Convolution_and_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:49:36"
field: "3D点云处理"
keywords: ["point cloud denoising", "invertible neural network", "graph convolution", "latent space", "monotone operator"]
innovations: ["基于可逆单调算子的潜空间噪声显式解耦方法", "多级图卷积与可逆网络融合的结构设计", "EMD双射匹配替代最近邻均匀监督策略"]
benchmarks: ["PUNet", "Paris-rue-Madame"]
---

# 论文速读：Denoising-Point-Clouds-in-Latent-Space-via-Graph-Convolution-and-Invertible-Neural-Network

## 一句话总结
本文提出了一种基于可逆神经网络和图卷积的点云去噪方法，通过在潜空间中显式分离噪声与干净成分（将噪声映射到特定通道），实现间接去噪，在PUNet数据集上以CD和P2M指标取得了SOTA性能。

## 研究问题与动机
- 传统去噪方法（如双边滤波、MLS、LOP等）依赖法向估计或几何先验，在复杂真实场景中容易过度平滑、丢失细节。
- 现有深度学习位移估计方法（如Pointcleannet、Pointfilter）将高维几何特征回归为3D位移，易引发表面坍塌、面片收缩等问题，且监督时均匀性不足。
- 在特征空间去噪的方法（如PD-flow）受限于局部特征提取不足，且其Affine Coupling Layer需要维度划分，难以与图卷积特征有效结合。
- 论文动机：将去噪过程转移到潜空间，利用可逆神经网络的双射性质和图卷积的多级几何感知能力，显式分离噪声成分，从而避免直接回归位移的局限。

## 核心贡献（创新点）
1. 提出潜空间去噪范式：将噪声与干净点显式分离到不同通道（$\tilde{z}=[z_c,z_n]$），通过对噪声通道置零并逆变换恢复干净点云，区别于传统位移估计方法。
2. 引入可逆单调算子（Invertible Monotone Operator）构建轻量可逆神经网络：通过Lipschitz约束（$\|W\|_2<1$）实现无维度划分的自由架构，避免Affine Coupling的限制，便于与图卷积融合。
3. 设计多级图卷积模块（MLGC）：基于EdgeConv并引入Dense Connection，逐级累积局部到全局的几何上下文语义，显著增强形状结构表达。
4. 提出层级可逆编码框架：将MLGC融合的多级特征通过MLP维度适配后与可逆层输入相加，保持可逆性的同时增强特征表达能力。
5. 与已有工作的本质区别：不同于PD-flow直接使用Normalizing Flow（需维度划分）或位移估计方法直接预测偏移，本文通过隐式可逆变换在潜空间中实现噪声解耦，并利用图卷积强化局部几何感知。

## 方法详解
- **可逆单调算子建模**：基于理论（Ahn et al., 2022），定义变换$F(x)=\left(\frac{\text{Id}+G}{2}\right)^{-1}(x)-x$，其中$G$为1-Lipschitz函数。由Banach不动点定理，迭代$y=x-G(y)$可收敛至唯一解。通过谱正则化（$\|W\|_2<1$）保证激活函数层面满足Lipschitz条件。
- **多级图卷积（MLGC）**：基于EdgeConv构建k-NN图，每层提取边特征$\mathbf{f}_i^{(l)}=\sum_{(i,j)\in\mathcal{E}}\text{MLP}(\mathbf{h}_i^{(l-1)}\|\mathbf{h}_i^{(l-1)}-\mathbf{h}_j^{(l-1)})$；引入Dense Connection，使第$l$层输入汇聚所有前层输出：$\mathbf{h}_i^{(l)}=[\mathbf{f}_i^{(l)},\mathbf{f}_i^{(l-1)},...,\mathbf{f}_i^{(0)}]$。网络分两阶段构建层级MLGC。
- **可逆编码过程**：将MLGC提取的多级特征$H^{(l)}$经MLP维度适配为$C^{(l)}$，随后通过加法注入可逆层：$X_c^{(l)}=X^{(l)}+C^{(l)}$，再经$X^{(l+1)}=F_{\theta_l}(X_c^{(l)})$变换，逐级构建深层可逆编码。
- **维度增强（Dimension Augmentation）**：为保持高维特征空间，输入首层前拼接增强特征：$\mathbf{h}_i^a=f(x_i)+\sum_{x_j\in N(x_i)}g(x_j\|x_j-x_i)$，得到$x_i^{(0)}=[x_i,\mathbf{h}_i^a]\in\mathbb{R}^{3+D_a}$。
- **训练目标**：采用Earth Mover's Distance (EMD) 作为重建损失，强调双射匹配：$\text{EMD}(\hat{X},X)=\min_{\Phi:\hat{X}\to X}\sum_{\hat{x}}\|\hat{x}-\Phi(\hat{x})\|$；不使用基于高斯先验的分布学习损失，避免Jacobian计算。

## 实验与结果
- **数据集**：PUNet [50]（10K/50K点云，高斯噪声标准差0.05–0.2），测试噪声水平1%、2%、2.5%；另使用Paris-rue-Madame [44]进行真实场景泛化评估。
- **评估指标**：Chamfer Distance (CD)、Point-to-Mesh (P2M)、均匀性（Uniformity）。
- **主要结果（10K点云，单位$10^5$）**：
  - 2%噪声：Ours(heavy) CD=24.41（SOTA）、P2M=7.58（SOTA）；Ours(light) CD=25.67、P2M=8.24。
  - 对比IterativePFN [9]（最强基线之一）：2%噪声下CD提升约19.6%，P2M提升约10.3%。
  - 50K点云：Ours(heavy)在1%、2%、2.5%噪声下CD分别为4.70、6.46、8.63，均低于其他SOTA方法。
- **均匀性（Tab. 2）**：在所有噪声级别下，Ours的Uniformity值显著低于其他方法，证明点在曲面上分布更均匀。
- **消融实验（Tab. 3）**：移除MLGC导致性能大幅下降；减少可逆层数（1/6层）劣于完整模型；$D_a=8$过窄，$D_a=48$为最优折衷。
- **结论**：本文方法在不同噪声水平和点云密度下均取得最优CD和P2M指标，且在点分布均匀性上显著优于基线。

## 相关工作脉络
1. **传统去噪方法**（双滤波、MLS、LOP、谱方法）：依赖人工先验，难以适应复杂真实场景，易过度平滑。
2. **位移估计类方法**（Pointcleannet [43]、Pointfilter [52]）：回归3D位移，存在特征表达受限和表面坍塌风险，监督均匀性差。
3. **DMRDenoise [32]**：基于可微流形重建重采样，但重采样过程易丢失精细几何细节。
4. **PD-flow [35]**：利用Normalizing Flow建立双射映射，但Affine Coupling需维度划分，无法与图卷积特征深度融合。
5. **ScoreDenoise [33]**：基于Score Matching引导梯度上升去噪，对噪声尺度有偏倚。
6. **本文定位**：首次将可逆单调算子理论引入点云去噪，并在潜空间显式解耦噪声与干净成分，通过MLGC强化几何感知，填补了"可逆网络+图卷积+潜空间去噪"的空白。

## 局限性与未来方向
- **分辨率与噪声上限**：当前方法在极高噪声（>2.5%）或超大点云密度下表现可能受限，未进行极端条件测试。
- **参数规模**：Heavy版本1.4M参数相对于Light版增长明显，工业部署成本需进一步权衡。
- **训练策略**：依赖Patch Stitching重建全局点云，可能引入边界伪影。
- **未来方向**：探索更高维潜空间表示、自适应去噪强度、与下游任务（分割、配准）的联合训练、以及轻量化部署。

## 研究启发与可借鉴点
1. **潜空间显式解耦思路**：将噪声/干净成分分离到不同通道的策略可迁移至其他3D视觉任务（如补全、上采样、分割），实现"可解释的噪声建模"。
2. **可逆单调算子替代Affine Coupling**：基于Lipschitz约束的可逆网络无需维度划分，为与任意结构特征（如图卷积、Transformer）融合提供了通用框架。
3. **Dense EdgeConv层级构建**：逐级融合局部到全局几何特征的机制可推广至其他点云分析任务，增强多尺度表达。
4. **EMD损失替代最近邻匹配**：双射匹配的均匀性约束有效解决了位移方法中的监督偏差问题，值得在其他点云处理任务中复用。

## 关键术语表
- **可逆单调算子（Invertible Monotone Operator）**：基于Lipschitz约束算子，保证双射可逆性，用于建模无损变换。
- **MLGC（Multi-level Graph Convolution）**：基于EdgeConv并引入Dense Connection的多级图卷积模块，逐级累积局部到全局几何特征。
- **潜空间去噪（Latent Space Denoising）**：在隐式高维空间显式分离噪声与干净成分，而非直接在3D坐标空间位移点云。
- **Edith Mover's Distance (EMD)**：衡量两个点集间最小成本双射匹配的距离度量，用于确保监督均匀性。
- **Spectral Norm Regularization**：通过限制权重矩阵最大奇异值（$\|W\|_2<1$）保证Lipschitz连续性。
- **Patch Stitching**：将局部补丁去噪结果通过重叠区域拼接还原为全局点云的技术。
- **Affine Coupling Layer**：Normalizing Flow中需对维度进行划分的前馈块，限制了与任意特征结构的融合灵活性。

## 可复现要素
- **数据集**：PUNet [50]（公开）；Paris-rue-Madame [44]（公开）。
- **代码**：已开源，见 https://github.com/yanbiao1/PD-LTS。
- **实现细节**：PyTorch 1.11.0 + CUDA 11.3，RTX 3090 GPU。
- **Light版本**：单模块，12个可逆变换+EdgeConv层，$D_a=48$，679K参数。
- **Heavy版本**：3模块堆叠，每模块10个可逆变换+EdgeConv层，$D_a=32$，1.4M参数。
- **训练超参**：Adam优化器，学习率$2\times10^{-3}$，训练40 epoch。
