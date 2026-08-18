---
title: "Capturing-Closely-Interacted-Two-Person-Motions-with-Reactio"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Capturing_Closely_Interacted_Two-Person_Motions_with_Reaction_Priors_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:54"
field: "单目3D人体姿态估计"
keywords: ["two-person motion capture", "reaction prior", "invertible neural network", "monocular 3D pose estimation", "human-human interaction", "dual human dataset", "occlusion handling"]
innovations: ["提出基于运动VAE+INN的双向反应先验，建模双人互动的姿态概率分布", "设计交互感知自注意力与查询式时序追踪的统一端到端双人MoCap框架", "构建包含2k序列、contact标注与文本描述的Dual-Human紧密互动数据集"]
benchmarks: ["Hi4D", "CHI3D", "Dual-Human"]
---

# 论文速读：Capturing-Closely-Interacted-Two-Person-Motions-with-Reactio

## 一句话总结
本文提出**反应先验（Reaction Priors）**，利用运动VAE与可逆神经网络（INN）双向建模两人互动的姿态概率分布，结合查询式Transformer估计器，从单目视频中有效捕捉紧密互动双人动作；同时构建了高质量大规模数据集 Dual-Human 以推动该领域研究。

## 研究问题与动机
- **紧密互动场景下严重遮挡问题**：双人对峙、拥抱、握手等互动中频繁出现肢体相互遮挡，导致传统方法无法准确恢复被遮挡关节的3D姿态。
- **单人不适用**：现有单人MoCap方法（如CLIFF、PARE）不利用人际信息，在重度遮挡下性能大幅下降。
- **多人在场方法局限**：现有双人/多人方法（如BEV、4D-Humans）更关注相对深度排序，而非精确的姿态互动语义，且端到端方法在密集遮挡下仍不稳定。
- **数据缺失**：现有双人交互数据集规模小、多样性不足、精度受限（如Hi4D仅100个序列，Inter-Human缺乏多视图图像且受多视角RGB遮挡影响），难以支撑模型训练与公平评测。

## 核心贡献（创新点）
1. **提出反应先验（Reaction Priors）**：利用运动VAE将动作编码为潜分布，再通过INN双向建模"给定对方动作推断自身反应"的概率分布，与BUDDI基于扩散模型的静态空间先验本质不同，前者捕捉时空动态语义。
2. **设计交互感知自注意力（Interaction-Aware Self-Attention）**：将self-attention分为 intra-human（骨架邻域）与 inter-human（两人相互）两级，避免全图注意力冗余，同时利用上一帧根位置作为Track queries实现无额外跟踪模块的时序一致性。
3. **构建Dual-Human大规模双人物交互数据集**：约2k序列、SMPL-X拟合、SDF碰撞优化、顶点级contact标注、合成多视图渲染与文本描述，填补了高质量紧密互动MoCap基准的空白。
4. **反应先验同时适用于训练正则化与测试时优化**：通过KL散度损失约束估计结果落在先验分布内，可在滑动窗口中以闭式形式优化潜变量，显著提升遮挡场景的鲁棒性。

## 方法详解
- **运动表征**：采用269维运动表示（角速度、线速度、根高度、局部关节位置/速度/旋转、脚部接触等），前263维与[17]一致，后6维用于恢复相机视角运动。
- **运动VAE（Motion VAE）**：Encoder为Transformer，将输入运动映射为潜分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma})$，采样后由Decoder（Transformer）重建运动；ELBO损失：$\mathcal{L}_{vae} = ||\hat{\boldsymbol{x}} - \boldsymbol{x}||_1 + \lambda_{dist} \cdot \mathcal{D}_{KL}[p(z|\boldsymbol{x}) || \mathcal{N}(\mathbf{0}, I)]$。
- **反应生成器（Reaction Generator / INN）**：冻结VAE编码器与解码器，INN将动作潜分布 $\mathcal{N}(\mu_a, \sigma_a)$ 映射为反应潜分布 $\mathcal{N}(\tilde{\mu}_r, \tilde{\sigma}_r)$；可逆性保证角色互换语义对称：$p(\tilde{z}_r|\boldsymbol{x}_a) = p(z_a|\boldsymbol{x}_a) \cdot \prod_k |\det(\partial f_k / \partial z_k)|^{-1}$。
- **先验损失（训练）**：$\mathcal{L}_{prior} = \mathcal{D}_{KL}[p(z_r|\hat{\boldsymbol{x}}_r) || p(\tilde{z}_r|\hat{\boldsymbol{x}}_a)]$，约束估计的反应潜分布与从动作推断的先验分布对齐。
- **测试时优化**：以潜变量 $(\mu_r^{opt}, \sigma_r^{opt})$ 为优化目标，$\mathcal{L} = ||\mathcal{D}(\mu_r^{opt}, \sigma_r^{opt}) - \hat{\boldsymbol{x}}_r||_2^2 + \lambda_{prior} \cdot \mathcal{D}_{KL}[\mathcal{N}(\mu_r^{opt}, \sigma_r^{opt}) || p(\tilde{z}_r|\hat{\boldsymbol{x}}_a)]$，$\lambda_{prior}=0.01$。
- **噪声数据增强**：模拟MoCap估计误差，在圆周上采样多个相机视角计算关节可见概率，按概率添加随机噪声训练INN，避免过拟合干净数据。
- **Pose Estimator**：HRNet-W48骨干 + 查询式Decoder-only Transformer，三类queries（Pose/Transl/Prob）各2个；交互感知自注意力分两层：intra-human（Pose query仅关注骨架祖先/子节点）与 inter-human（Pose+Transl交叉，Prob相互）；上一帧Transl作为Track query引导时序追踪。
- **时序平滑**：经SmoothNet对3D poses进行后处理，输出269维运动表示。
- **滑动窗口优化**：窗口大小 $L=64$，重叠 $L/2$，窗口内用反应先验优化估计结果。

## 实验与结果
- **数据集**：Hi4D（室内高精度）、CHI3D（室内8类互动）、Dual-Human（室内外，2019序列，训练/测试比3:1）。
- **指标**：MPJPE（mm）、PA-MPJPE（mm）、Translation Error（mm）。
- **基线对比**（Table 2）：与CLIFF（Top-Down）、4D-Humans（Transformer）、BEV（Bottom-Up）、TRACE（Temporal）比较。本文方法在三个数据集上均取得最佳或接近最佳结果。
  - **Dual-Human最优结果**：MPJPE=63.4，PA=51.2，Transl=112.1（较第二名的TRACE分别提升5.4/2.7/8.8）。
  - **Hi4D最优结果**：MPJPE=75.0（较TRACE的83.8提升8.8）。
  - **CHI3D最优结果**：MPJPE=71.9（较CLIFF的76.4提升4.5）。
- **先验对比**（Table 3）：相同Pose Estimator下，BUDDI先验MPJPE=67.3，本文反应先验MPJPE=64.1，提升3.2。
- **消融实验**（Table 4）：
  - 移除运动VAE → MPJPE 64.4（+1.0）；INN替换为MLP → 66.2（+2.8）；无噪声增强 → 77.4（+14.0，泛化性大幅下降）。
  - 移除交互注意力 → 70.3（+6.9）；移除反应先验 → 67.9（+4.5）；移除时序网络 → 63.6（Transl上升16.2）。
- **定性结果**（Fig.7）：基线方法在严重遮挡下常出现姿态不合理、人员缺失或位置偏移，本文方法能更好恢复。

## 相关工作脉络
- **BUDDI [40]**：基于扩散模型学习静态双人空间先验（proxemics），仅描述静态空间关系；本文反应先验基于运动VAE+INN，捕捉时空动态语义，且可双向可逆。
- **ProHMR [29] / HuMoR [46] / VPoser [43]**：单人运动先验，分别基于归一化流、条件VAE、标准VAE；本文首次将双人互动先验引入端到端MoCap框架。
- **InterPrior [73]**：仅考虑手部互动先验；本文覆盖全身269维运动表征，适用范围更广。
- **单帧多人方法**（CLIFF [32]、BEV [54]、4D-Humans [16]）：缺乏显式双人互动语义建模，面对重度遮挡退化明显。
- **时序多人方法**（TRACE [55]）：关注全局轨迹，对局部互动姿态的精细化不足。
- **互动动作生成**（InterHuman [33]、Intergen等）：侧重文本到动作生成，精度受多视角RGB遮挡影响，不适合直接用于MoCap任务。

## 局限性与未来方向
- 反应先验对数据集中**未出现的互动类型泛化有限**，若输入完全不互动，过大 $\lambda_{prior}$ 会产生虚假互动动作。
- 框架假设场景中**恰好两人**，超过两人的场景需额外扩展模块。
- 未来计划扩展Dual-Human规模与多样性，并利用文本描述探索多模态（文本+图像）互动动作生成任务。

## 研究启发与可借鉴点
1. **运动VAE + INN构建双向概率先验**的思路可迁移至任意"对称互动关系"建模（如手-物交互、群体动作），为遮挡推理提供结构化概率约束。
2. **训练时模拟估计误差的数据增强策略**（按关节可见概率添加噪声）对任何依赖"干净监督"的端到端模型均有参考价值，能有效缓解train-test gap。
3. **查询式时序追踪（Track queries）**免去了额外跟踪模块，为视频级多人姿态估计提供简洁的统一框架设计参考。
4. **Dual-Human数据集构建流程**（IMU+VIVE采集 → SMPL-X拟合 → SDF碰撞优化 → 顶点级contact标注 → 多视图渲染）可作为高质量双人交互数据采集的标准范式。
5. **$\lambda_{prior}$在训练与测试时灵活调节**的设计模式：训练时以固定KL散度正则化，测试时可将先验作为优化项，兼顾训练稳定与推理灵活性。

## 关键术语表
- **Reaction Priors（反应先验）**：基于运动VAE与INN构建的概率模型，用于从一方的动作潜分布双向推断另一方的反应潜分布，缓解遮挡歧义。
- **Motion VAE**：将运动序列编码为均值-方差潜分布并通过Transformer解码重构的变分自编码器，用于学习紧凑的运动语义表示。
- **INN（Invertible Neural Network，可逆神经网络）**：保雅可比行列式可计算的双向映射网络，用于建模双人互动中角色互换的对称概率分布。
- **MPJPE**：Mean Per Joint Position Error，预测关节与真实关节的欧氏距离均值（单位mm），衡量3D姿态绝对精度。
- **PA-MPJPE**：Procrustes-aligned MPJPE，在计算MPJPE前进行刚性对齐，消除全局尺度与旋转偏差后的姿态误差。
- **Translation Error**：根关节（hip）预测位置与真实位置的欧氏距离，衡量全局定位精度。
- **Contact Annotation**：基于SDF阈值自动提取的顶点级身体接触标注，用于刻画双人互动的物理接触关系。
- **Dual-Human**：本文构建的双人紧密互动动作数据集，含约2k序列、合成多视图渲染、contact标注与文本描述，已公开。

## 可复现要素
- **数据集Dual-Human**：已公开，项目页面 https://neteasegameai.github.io/Dual-Human/；Hi4D、CHI3D为公开基准数据集。
- **代码/权重**：论文未明确声明代码开源状态（仅提供项目页面链接）。
- **关键超参**：输入分辨率512×512；HRNet-W48骨干；EMA=AdamW，lr=1×10⁻⁴（VAE/INN/SmoothNet）或1×10⁻⁵（Pose Estimator，衰减因子5）；λ_pose=10，λ_transl=1，λ_prob=1，λ_dist=0.0001，λ_prior=0.01；窗口大小L=64，重叠L/2；训练轮数：VAE 4000，INN 500，SmoothNet 2000。
