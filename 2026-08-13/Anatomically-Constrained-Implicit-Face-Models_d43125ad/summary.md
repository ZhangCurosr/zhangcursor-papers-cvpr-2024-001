---
title: "Anatomically-Constrained-Implicit-Face-Models"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chandran_Anatomically_Constrained_Implicit_Face_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:01:35"
field: "隐式面部建模与动画"
keywords: ["implicit face model", "anatomical constraint", "coordinate neural network", "blendshape", "facial performance retargeting", "SIREN"]
innovations: ["首次将密集解剖学约束端到端集成到隐式面部blendshape模型中", "从稀疏骨骼监督下自动恢复全表面连续解剖子结构（骨骼点、厚度、法线）", "以显式几何生成替代约束优化，将每帧拟合时间从数分钟降至约1秒"]
benchmarks: ["ActorBlend (5 actors, 20 blendshapes)", "819-frame performance sequences for fitting error evaluation"]
---

# 论文速读：Anatomically-Constrained-Implicit-Face-Models

## 一句话总结
本文提出一种新型解剖学约束隐式面部模型（AIM），通过端到端学习的隐式神经网络联合建模面部解剖结构与皮肤表面，能够在少量演员数据下自动恢复密集解剖子结构并高精度重建面部表情，同时保持传统blendshape模型的可插拔兼容性与高效推理速度。

## 研究问题与动机
- **传统actor-specific blendshape模型需要大量数据与形状**：为准确建模复杂的面部变形，通常需要数百个3D扫描形状，数据采集成本高。
- **现有隐式面部模型缺乏解剖学约束**：当前基于坐标的隐式神经网络（如NPHM、IMAvatar等）能高效表征面部几何，但未显式建模底层解剖结构，限制了其在解剖合理的形状编辑与性能重定向中的应用。
- **解剖学局部模型（ALM）计算瓶颈严重**：Wu等人提出的ALM虽然引入了骨骼约束以提高捕捉精度，但其约束以稀疏正则项形式参与复杂优化，每帧需数分钟CPU计算，且约束仅作用于有骨骼覆盖的稀疏区域。
- **缺乏支持高效、密集解剖约束的连续可微表示**：现有物理模拟方法虽能密集约束皮肤-骨骼交互，但计算代价极高，难以应用于日常动画生产流程。

## 核心贡献（创新点）
- **首次将解剖学约束集成到隐式面部blendshape模型中**：提出AIM框架，在隐式表示的学习过程中直接耦合骨骼结构与皮肤表面，区别于以往隐式模型仅学习表面点的做法。
- **通过端到端学习为每个皮肤表面点建立密集解剖约束**：相比ALM仅在稀疏骨骼覆盖区施加约束，本文方法学习到贯穿整个面部表面的连续解剖子结构，实现每一点的解剖可追溯性。
- **从稀疏监督下自动恢复密集解剖属性**：仅使用中性面部配准的头骨/下颌骨估计作为稀疏监督信号，模型即可推断出全表面的软组织厚度、解剖法线与骨骼点位置。
- **显式从解剖结构推导皮肤表面，避免约束优化瓶颈**：通过公式$\mathbf{s_0} = \mathbf{b_0} + d_0 \mathbf{n_0}$结合LBS与修正位移直接生成皮肤点，替代ALM中需迭代求解的约束优化问题，将每帧拟合时间从数分钟降至约1秒。

## 方法详解
**模型架构**：采用5个独立的周期激活坐标MLP（SIREN），输入为模板空间中的点$\mathbf{c} \in \mathbb{R}^3$，分别预测：
- $\mathbf{B}(\mathbf{c}) \rightarrow \widetilde{\mathbf{b}}_0$：解剖骨骼点位置
- $\mathbf{D}(\mathbf{c}) \rightarrow \widetilde{d}_0$：软组织厚度
- $\mathbf{N}(\mathbf{c}) \rightarrow \widetilde{\mathbf{n}}_0$：解剖法线
- $\mathbf{K}(\mathbf{c}) \rightarrow \widetilde{k}$：蒙皮权重
- $\mathbf{E}(\mathbf{c}) \rightarrow B_{\mathbf{e}} \in \mathbb{R}^{(N-1)\times 3}$：表达式修正位移基（一次性预测所有$N-1$个blendshape的位移）

**皮肤表面重建**：中性表面由解剖属性重构$\widetilde{\mathbf{s}}_0 = \widetilde{\mathbf{b}}_0 + \widetilde{d}_0 \widetilde{\mathbf{n}}_0$；任意表情$i$下的表面点为：
$$\widetilde{\mathbf{s}}_i = \mathrm{LBS}(\widetilde{\mathbf{s}}_0, \widetilde{T}_b, \widetilde{k}) + \widetilde{\mathbf{e}}_i$$
其中$\widetilde{T}_b$为6-DOF下颌骨变换（采用连续6D旋转表示）。

**训练目标**：
- **皮肤位置损失**$\mathbf{L}_S$：监督预测表面点与ground truth的距离（$\lambda_S = 1.0$）。
- **解剖正则化**$\mathbf{L}_A$：在有骨骼覆盖的稀疏区域（约5–10%顶点）约束$\widetilde{\mathbf{b}}_0, \widetilde{d}_0, \widetilde{\mathbf{n}}_0$接近配准得到的头骨/下颌骨估计（权重均为1.0）。
- **厚度正则化**$\mathbf{L}_D$：在无约束区域鼓励软组织厚度趋近于零（$\lambda_D^{Reg} = 7.5\times10^{-4}$）。
- **对称正则化**$\mathbf{L}_{Sym}$：约束解剖点预测$\mathbf{B}$关于面部对称平面对称（$\lambda_{sym} = 1\times10^{-4}$）。
- **蒙皮权重正则化**$\mathbf{L}_K$：在前额等不受下颌影响区域鼓励蒙皮权重为零（$\lambda_K = 1\times10^2$）。

**模型拟合**：采用神经重参数化优化，学习一个4层MLP${\bf F}_T$预测每帧的头部与下颌变换$[\mathbf{T_g^j}, \mathbf{T_b^j}]$，另一个4层MLP${\bf F}_W$根据帧编码与模板查询点预测空间变化的系数$\mathbf{w^j}$。拟合损失包含3D/2D位置约束、系数L2正则（$\lambda_{Reg}^w = 0.75$）与时间平滑正则（$\lambda_{Reg}^t = 0.05$）。

## 实验与结果
- **数据集**：使用5名不同演员的3D扫描序列，每名演员提供20个注册好的表情blendshapes用于模型学习，另用多段表演序列（共819帧）进行测试。
- **评估基线**：全局blendshape模型（GBS）、patch blendshape模型（PBS）、解剖学局部模型（ALM）。
- **主要结果**（Table 1，平均拟合误差，单位mm）：

| 方法 | GBS | PBS | ALM | Ours (G) | **Ours** |
|------|-----|-----|-----|----------|----------|
| 误差 | 0.83 | 0.51 | 0.09 | 0.86 | **0.31** |

- **核心结论**：本文方法在平均误差上显著优于GBS与PBS，虽略高于ALM，但训练仅需~10分钟/演员，而ALM每帧拟合需数分钟CPU；在3D性能重定向任务中，本文方法每帧仅需2–3秒GPU计算，且能直接解耦刚性下颌运动与非刚性软组织变形，无需手动设计patch布局等超参数。
- **最强结果**：在表达式重建任务中，本文方法（Ours）获得0.31 mm的平均顶点误差，相比GBS（0.83 mm）与PBS（0.51 mm）分别提升62.7%与39.2%。

## 相关工作脉络
- **FLAME与隐式神经参数化头部模型（NPHM、IM-Face、IMAvatar等）**：此类方法使用坐标MLP学习面部几何与外观，但均以表面点或隐式场为直接表示，未显式建模底层骨骼-软组织约束关系；本文在此基础上引入解剖学隐式子结构，扩展了表达与编辑能力。
- **解剖学局部模型（ALM，Wu et al.）**：ALM通过稀疏解剖约束提升单目捕捉精度，但约束作为正则项参与优化导致计算昂贵；本文将其转化为端到端学习的连续解剖生成机制，实现实时拟合。
- **物理驱动面部模拟（如SoftDeCA、Physically-based models）**：物理方法可密集建模骨骼-皮肤交互但计算开销大；本文以数据驱动的隐式表示近似该约束，在保真度与效率间取得平衡。
- **语义深度面部模型（Semantic Deep Face Model）**：该方法通过多线性方式解耦身份与表情，但未考虑解剖学约束；本文可视为对其在解剖约束方向的扩展。
- **Sculptor与Animatomy**：这两类方法分别从CT数据或多边形基底学习解剖结构，依赖大量标定数据或手动设计；本文仅需稀疏头骨/下颌配准即可自动学习密集解剖，对数据需求更低。

## 局限性与未来方向
- **解剖表面在唇周等区域可能出现伪影**：由于解剖监督稀疏，无骨骼覆盖区域的厚度与法线预测可能不够稳定。
- **未对面部解剖结构本身进行蒙皮处理**：当前仅将解剖属性用于约束皮肤表面，未让解剖网格随表情刚性形变，限制了某些高级编辑应用。
- **拟合阶段在复杂表情序列中可能产生轻微时间抖动**：若优化提前终止，相邻帧编码$\mathbf{z_j}$的平滑性可能不足。
- **尚未扩展至外观（纹理/颜色）建模**：当前模型仅表征几何，未来可结合神经辐射场等技术实现全面部avatar。
- **未探索作为通用多身份morphable model的潜力**：当前为actor-specific，未来可扩展至支持多个身份的隐式解剖模型。

## 研究启发与可借鉴点
- **解剖约束的端到端可微集成**：将传统基于正则化的解剖约束转化为可由坐标MLP直接预测的连续属性（骨骼点、厚度、法线），并通过显式几何公式生成皮肤表面，避免了迭代优化的计算瓶颈，这一思路可迁移至身体、手部等其它解剖约束建模任务。
- **神经重参数化用于空间变化系数的预测**：使用轻量MLP预测每顶点表达式系数$\mathbf{w}$，而非直接优化高位参数，既控制了解变量数量又保证了拟合效率，该策略适用于任意高维隐式模型的快速适配。
- **对称性正则化与薄层厚度先验的联合使用**：通过对称正则鼓励解剖预测的左右一致性，配合厚度正则抑制无约束区域的厚度膨胀，可在稀疏监督下稳定学习解剖结构，适用于其它仅靠少量标记点监督的隐式几何学习场景。
- **可插拔的drop-in替换设计**：AIM输出可直接替代传统blendshape系数，便于接入现有动画管线；这种在保留传统接口前提下升级底层表示的工程思路，对工业界落地具有参考价值。
- **单步预测多blendshape修正位移**：将$(N-1)$个表达修正基一次性从单个MLP输出中索引，既节省参数量又保证一致性，可推广至任何需要多姿态/多表达式联合编码的隐式模型。

## 关键术语表
**AIM（Anatomical Implicit face Model）**：本文提出的解剖学约束隐式面部模型，使用坐标MLP联合学习皮下解剖结构与皮肤表面。
**LBS（Linear Blend Skinning）**：线性混合蒙皮，用于将骨骼刚性变换传递给表面点的标准蒙皮技术。
**神经重参数化优化（Neural reparameterized optimization）**：用小型MLP预测优化变量而非直接优化，以减少自由度并加速收敛。
**ALM（Anatomical Local Model）**：Wu等人提出的解剖学约束局部变形模型，以稀疏正则项形式约束皮肤-骨骼关系。
**SIREN（Sinusoidal Representation Networks）**：使用正弦激活函数的坐标MLP，擅长学习高频隐式几何表示。
**Expression blendshape**：描述特定表情下顶点相对于中性面位移的三维修正向量。
**6-DOF jaw transformation**：下颌骨的六自由度刚体变换，本文采用连续6D旋转表示以避免万向锁。

## 可复现要素
- **数据集**：ActorBlend（演员特定3D扫描blendshape集合），论文未提供公开下载链接；训练使用5名演员的配准网格与配准得到的头骨/下颌骨估计（Zoss et al. [56]方法）。
- **代码开源**：论文在GitHub提供开源代码（链接见正文），基于PyTorch实现。
- **关键超参**：学习阶段迭代$1\times10^4$次，学习率$2\times10^{-3}$；拟合阶段迭代$1\times10^4$次，学习率$1\times10^{-3}$；损失权重$\lambda_S=1.0,\lambda_b=\lambda_d=\lambda_n=1.0,\lambda_D^{Reg}=7.5\times10^{-4},\lambda_{sym}=1\times10^{-4},\lambda_K=1\times10^2,\lambda_{Reg}^w=0.75,\lambda_{Reg}^t=0.05$。
- **硬件与耗时**：训练约10分钟/演员（单卡Nvidia RTX 3090，40k顶点、20个blendshape）；每帧拟合约1秒/帧。
