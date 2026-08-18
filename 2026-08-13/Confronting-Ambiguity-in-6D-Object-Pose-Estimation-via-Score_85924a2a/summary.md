---
title: "Confronting-Ambiguity-in-6D-Object-Pose-Estimation-via-Score"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Hsiao_Confronting_Ambiguity_in_6D_Object_Pose_Estimation_via_Score-Based_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:13:57"
field: "单目6D物体姿态估计"
keywords: ["6D姿态估计", "扩散模型", "SE(3)", "评分扩散", "姿态歧义", "李群", "对称性", "单目定位"]
innovations: ["首次将扩散模型应用于图像域中的SE(3)姿态估计，联合建模旋转-平移分布", "提出SE(3)上的替代Stein评分公式，解决非对称雅可比导致的收敛问题", "设计傅里叶条件机制增强MLP对SO(3)周期分布的建模能力"]
benchmarks: ["SYMSOL", "SYMSOL-T", "T-LESS"]
---

# 论文速读：Confronting-Ambiguity-in-6D-Object-Pose-Estimation-via-Score-Based-Diffusion-on-SE(3)

## 一句话总结
本文首次将评分扩散模型（Score-Based Diffusion Model）应用于SE(3)李群，用于解决单张RGB图像6D姿态估计中的姿态歧义问题（由物体对称性和遮挡引起）。通过联合建模旋转与平移的分布，并提出替代Stein评分公式提升收敛效率，该方法在SYMSOL、SYMSOL-T和T-LESS数据集上均取得最优或极具竞争力的结果。

## 研究问题与动机
1. **姿态歧义问题突出**：对称物体从多视角呈现相同视觉外观，遮挡会导致物体关键部分不可见，使得图像到姿态的映射从一对一变为多对一，传统回归方法性能显著下降。
2. **现有对称感知标注依赖性强**：Symmetry-aware loss等方法依赖精确的对称等价姿态标注，对纹理缺失（如杯子把手不可见）或复杂形状的物体标注成本极高，不具可扩展性。
3. **已有概率建模方法存在计算瓶颈**：Implicit-PDF、HyperPose-PDF等方法在SO(3)流形上进行非参数密度估计，训练需在全空间采样，推理依赖网格搜索分辨率，扩展到SE(3)时内存需求巨大。
4. **现有扩散方法局限于R³×SO(3)分离参数化**：已有工作（如Urain et al. [59]、Yim et al. [68]）使用R³SO(3)参数化，将旋转和平移视为独立实体进行扩散，忽略了由透视投影引起的旋转与平移分布之间的相关性。

## 核心贡献（创新点）
1. **首次将扩散模型应用于图像域中的SE(3)姿态估计**：此前扩散模型在SE(3)上的应用仅限于向量领域（如蛋白质生成），本文将其引入图像域的6D姿态估计任务。
2. **提出SE(3)上的评分扩散框架，联合建模旋转-平移分布**：不同于R³SO(3)的分离扩散，本文在SE(3)李群上实现旋转与平移的联合扩散，捕捉两者因透视效应产生的相关性。
3. **提出替代Stein评分（Surrogate Stein Score）公式**：SE(3)不满足左/右雅可比矩阵的对称关系，无法像SO(3)一样简化为闭合形式；本文通过将一步反向过程拆分为多个小步逼近，用替代评分 −z/σ² 提高收敛效率。
4. **设计傅里叶条件机制增强对周期分布的建模能力**：针对SO(3)姿态分布的周期性特征，提出基于Fourier的MLP条件机制，替代传统的scale-bias条件，显著提升旋转分布的学习效果。
5. **构建新数据集SYMSOL-T**：在原始SYMSOL数据集基础上添加随机平移，形成可用于评估SE(3)联合分布建模的合成数据集。

## 方法详解
### 整体框架
方法由条件生成部分和去噪部分组成。条件部分使用ResNet提取图像特征生成条件变量c，仅在推理开始时执行一次，避免每步重复特征提取；去噪部分由多个MLP模块构成，处理Lie代数空间中的噪声姿态。

### 关键设计
1. **SE(3)上的高斯扰动核**：定义于李群G上的扰动核为 $p_{\Sigma}(Y|X) = \mathcal{N}_{\mathcal{G}}(Y; X, \Sigma)$，利用Log/Exp映射将群元素转换到切空间进行高斯建模。

2. **评分函数与DSM目标**：网络 $s_{\theta}(\tilde{X}, \sigma)$ 通过Denoising Score Matching学习评分 $\nabla_{\tilde{X}} \log p_{\sigma}(\tilde{X}|X)$，目标函数为：
   $$\mathcal{L}(\theta;\sigma) = \frac{1}{2}\mathbb{E}_{p_{data}(X)}\mathbb{E}_{\tilde{X}\sim\mathcal{N}_{\mathcal{G}}(X,\Sigma)}[\|s_{\theta}(\tilde{X},\sigma) - \nabla_{\tilde{X}}\log p_{\sigma}(\tilde{X}|X)\|_2^2]$$

3. **替代Stein评分推导**：SO(3)和R³SO(3)满足 $\mathbf{J}_l(z) = \mathbf{J}_r^{\top}(z)$，评分可简化为 $-\frac{1}{\sigma^2}z$；但SE(3)不满足此性质，故将一步去噪拆分为多个子步，用 $\tilde{s}_X(\tilde{X},\sigma) = -\frac{1}{\sigma^2}z$ 作为替代评分。

4. **反向采样过程**：采用Lie群上的Geodesic Random Walk：$\tilde{X}_{i+1} = \tilde{X}_i \text{Exp}(\epsilon_i s_{\theta}(\tilde{X}_i, \sigma_i) + \sqrt{2\epsilon_i} z_i)$，其中 $z_i \sim \mathcal{N}(0, I)$。

5. **傅里叶条件机制**：针对SO(3)的周期性，提出条件模块 $f_i(x,c) = \sum_{j=0}^{d-1} \mathbf{W}_{ij}(\mathbf{A}_j(c)\cos(\pi x_j) + \mathbf{B}_j(c)\sin(\pi x_j))$，增强对周期特征的表达能力。

## 实验与结果
### 数据集与基线
- **SYMSOL**：250k张5种对称纹理缺失物体的图像，评估SO(3)上的旋转估计。基线：DBN、Implicit-PDF、HyperPosePDF、Normalizing Flows。
- **SYMSOL-T**：作者在SYMSOL基础上添加随机平移构建的新数据集，评估SE(3)上的联合旋转-平移估计。基线：直接回归、迭代回归（均使用对称感知损失）。
- **T-LESS**：30个纹理缺失工业物体的真实场景数据集，使用BOP挑战的三个标准指标（MSPD、MSSD、VSD）及对称感知指标R@k、T@k。基线：GDRNPP（2022 BOP挑战SOTA）。

### 主要结果
1. **SYMSOL（SO(3)）**：Ours (ResNet50) 平均角度误差 **0.37°**，超越Normalizing Flows的0.70°和HyperPosePDF的1.94°，所有形状类别均低于1°。
2. **SYMSOL-T（SE(3)）**：Ours (SE(3)) 旋转平均误差 **0.50°**、平移平均 **0.012m**，显著优于R³SO(3)变体（1.93°旋转）和回归基线（2.92°旋转），证明联合建模的优势。
3. **T-LESS**：Ours (SE(3)) 在MSPD上达到 **93.16%**，超过GDRNPP的90.17%；R@2达**47.21%**（GDRNPP为21.60%）；平移精度略低于GDRNPP（因GDRNPP使用3D几何引导深度估计）。
4. **推理速度**：SE(3)模型在5步去噪下可达**250 FPS**，具备实时应用潜力。

## 相关工作脉络
1. **Implicit-PDF [39] / HyperPosePDF [23]**：在SO(3)上学习非参数旋转密度，但需在全空间采样和网格搜索，计算成本高；本文扩展至SE(3)并避免显式密度估计。
2. **DBN [10] / 各种参数化分布方法**：使用Bingham/von-Mises分布建模旋转不确定性，参数化限制表达能力；本文采用非参数扩散模型灵活建模多峰分布。
3. **Urain et al. [59] (SE(3)-DiffusionFields)**：使用R³SO(3)参数化在切空间做扩散，旋转与平移独立建模；本文在完整SE(3)群上联合建模以捕捉透视相关性。
4. **Yim et al. [68]**：应用于蛋白质主链生成的SE(3)扩散模型，但不在图像域；本文首次将SE(3)扩散引入图像姿态估计。
5. **GDRNPP [62]**：2022 BOP挑战SOTA的回归方法，依赖3D模型几何引导；本文无需3D模型和对称标注，纯数据驱动处理歧义。
6. **Leach et al. [32] / Jagvaral et al. [28]**：在SO(3)上使用无闭合形式的IG_SO(3)分布和自动微分计算评分；本文在SE(3)上提出可高效计算的替代评分。

## 局限性与未来方向
1. **跨形状联合训练的相互干扰**：作者在SYMSOL实验中注意到，单一模型处理所有形状时，cone的ResNet50性能反而不如ResNet34，说明不同形状的分布学习存在冲突，未来需探索分形状建模或多任务学习策略。
2. **平移估计相对较弱**：在T-LESS上，SE(3)扩散模型的平移精度低于使用3D几何引导的GDRNPP，表明纯数据驱动的平移估计在真实场景中仍有提升空间。
3. **依赖已知边界框和分割掩码**：T-LESS实验中假设已提供可见部分的GT边界框和分割掩码，限制了端到端部署能力。
4. **替代评分的近似性质**：SE(3)上的替代Stein评分仅为真实评分的近似，可能在高噪声水平下引入偏差，未来可探索更高阶近似或自适应步长策略。
5. **未探索与其他条件生成模型的结合**：如与CLIP等视觉语言模型结合，或扩展至类别级（category-level）姿态估计。

## 研究启发与可借鉴点
1. **SE(3)扩散框架的可迁移性**：将扩散模型从欧氏空间扩展到李群SE(3)的完整方法论（包括扰动核定义、评分计算、反向采样）可直接迁移到其他机器人/SLAM中的位姿优化任务。
2. **替代评分设计的思路**：当流形上不满足雅可比对称性时，通过多步子过程逼近逆映射的思路，可推广到其他非欧空间（如Sim(3)、SE(2)）的扩散模型设计。
3. **傅里叶条件机制对周期分布的适用性**：针对SO(3)等流形上的周期性分布特征，引入Fourier编码的条件机制是一种有效且参数不增多的技巧，可借鉴到任何涉及角度/旋转预测的任务中。
4. **条件变量分离设计提升推理效率**：图像特征提取与去噪步骤分离的策略（仅提取一次条件变量）值得在其他迭代采样任务中复用。
5. **合成数据集构建范式**：SYMSOL-T的构建方式（在现有SO(3)数据集上叠加随机平移）为评估SE(3)方法提供了可复现的基准范式。

## 关键术语表
**SE(3)**：特殊欧式群，描述三维空间中刚体变换（旋转+平移）的李群，是6D姿态估计的自然数学框架。
**Score-Based Diffusion Model (SGM)**：基于评分的扩散模型，通过学习概率分布的梯度（评分）来指导从噪声到数据的去噪采样过程。
**Lie Algebra so(3)/se(3)**：李群SO(3)/SE(3)在单位元处的切空间，可表示为向量形式，便于神经网络处理和数值计算。
**Stein Score**：概率密度的对数梯度 $\nabla \log p(x)$，在扩散模型中用于指导反向去噪过程的方向。
**SYMSOL-T**：本文提出的新合成数据集，在原始SYMSOL（仅含旋转）基础上添加随机平移，用于评估SE(3)上的联合姿态估计。
**R³SO(3) vs SE(3)**：前者将旋转和平移视为独立空间分别扩散，后者在统一的李群结构上联合扩散，能捕捉两者间的透视相关性。
**BOP Challenge**：Benchmark for 6D Object Localization and Pose estimation，姿态估计领域的权威评测挑战。
**MSPD/MSSD/VSD**：BOP挑战的三种对称感知评估指标，分别衡量投影距离、表面距离和可见表面差异。

## 可复现要素
- **数据集**：SYMSOL（公开）、SYMSOL-T（作者声明构建，是否开源论文未明确说明）、T-LESS（公开，需申请BOP许可）
- **代码**：论文未明确声明代码开源
- **权重**：论文未提供预训练权重链接
- **关键超参**：输入尺寸224×224；骨干网络ResNet34/ResNet50；去噪步数5/10/50/100；使用JAX框架；训练使用Dynamic Zoom-In和hard augmentation（随机颜色、高斯模糊、噪声）
