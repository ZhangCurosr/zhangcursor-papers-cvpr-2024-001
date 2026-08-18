---
title: "Capturing-Closely-Interacted-Two-Person-Motions-with-Reactio"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Fang_Capturing_Closely_Interacted_Two-Person_Motions_with_Reaction_Priors_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:34"
field: "单目三维人体动作捕捉与人际交互理解"
keywords: ["two-person motion capture", "reaction prior", "invertible neural network", "human-human interaction", "monocular MoCap", "Dual-Human dataset"]
innovations: ["提出基于运动VAE+INN的双向reaction prior，显式建模双人姿态条件概率分布", "设计interaction-aware自注意力与query-based tracking相结合的端到端双人MoCap框架", "构建Dual-Human大规模高质量双人交互数据集（2k序列+多视图+contact+text标注）"]
benchmarks: ["Hi4D", "CHI3D", "Dual-Human"]
---

# 论文速读：Capturing-Closely-Interacted-Two-Person-Motions-with-Reaction-Priors

## 一句话总结
本文提出利用 **reaction priors（反应先验）** 从单目视频中捕捉紧密互动的两人动作，通过运动 VAE + 可逆神经网络（INN）双向建模两人姿态的条件概率分布，并结合双人对齐的 query-based Transformer 估计器，在多个基准上超越现有方法；同时构建了约 2k 序列的高质量交互动作数据集 **Dual-Human**。

## 研究问题与动机
- **核心问题**：从单目视频捕捉紧密互动的两人动作（closely interacted two-person MoCap），该任务因频繁的人体自遮挡而被严重低估。
- **现有单人生成方法不足**：如 CLIFF、PARE 等，不考虑人际信息，面对重度遮挡时易失效。
- **现有多人方法不足**：如 BEV、TRACE 等，更关注人物间的相对深度顺序，而非精确的互动姿态语义，且 crop 操作会丢失空间上下文。
- **动机**：密切身体接触往往发生在特定情境（握手、拥抱等），此类情境蕴含强语义先验，可帮助推断被遮挡关节的姿态；已有研究表明人-物交互先验能有效辅助遮挡恢复，类似逻辑可扩展至人-人交互。

## 核心贡献（创新点）
- **Reaction Priors 的提出**：用运动 VAE + INN 构建可逆的双向反应生成器，显式建模"一人姿态条件化下另一人的姿态概率分布"，区别于 BUDDI 的静态扩散近景先验，本文捕捉的是**时序动态语义**。
- **端到端双人 MoCap 框架**：基于 query-based decoder-only Transformer，结合 interaction-aware self-attention 与 query-based tracking，避免 top-down/bottom-up 的多阶段信息损失。
- **Dual-Human 数据集**：约 2019 段高质量两人互动动作，含合成多视图渲染、contact 标注与文本描述，弥补了既有数据集在规模、精度与标注完整性上的不足。
- **先验的灵活注入机制**：reaction prior 既可作为训练正则化项（KL 散度 prior loss），也可在测试时用于优化估计结果，提升对严重遮挡的鲁棒性。

## 方法详解
### 1. Reaction Priors 构建
- **Latent Motion Representation**：采用 269 维运动表示（角速度、线速度、根节点高度、局部关节位姿、关节速度、旋转、foot contact 等），通过 Transformer 编码为 VAE 潜分布 $\mathcal{N}(\mu, \log\sigma^2)$，解码器重建运动，训练目标为 ELBO：
$$\log p(x) \geq \mathbb{E}_{z\sim q}[\log p(x|z)] - \mathcal{D}_{KL}[q(z|x) || \mathcal{N}(0, I)]$$
- **Reaction Generator (INN)**：在冻结的 VAE encoder/decoder 基础上，用可逆神经网络将 action 潜分布 $\mathcal{N}(\mu_a, \sigma_a)$ 映射为 reaction 潜分布 $\mathcal{N}(\tilde{\mu}_r, \tilde{\sigma}_r)$，利用 INN 的可逆性保证角色互换对称性：
$$p(\tilde{z}_r | x_a) = p(z_a | x_a) \cdot \prod_k |\det(\frac{\partial f_k}{\partial z_k})|^{-1}$$
- **训练噪声增强**：为模拟 MoCap 估计器的噪声，按关节可见概率添加随机噪声进行数据增强，提升泛化性。

### 2. Prior 注入方式
- **训练时正则化**：prior loss 约束估计动作的 latent 分布与 reaction prior 推断分布之间的 KL 散度：
$$\mathcal{L}_{prior} = \mathcal{D}_{KL}[p(z_r|\hat{x}_r) || p(\tilde{z}_r|\hat{x}_a)]$$
- **测试时优化**：以优化变量 $(\mu_r^{opt}, \sigma_r^{opt})$ 为目标，联合数据项与 prior 项：
$$\mathcal{L} = ||\mathcal{D}(\mu_r^{opt}, \sigma_r^{opt}) - \hat{x}_r||_2^2 + \lambda_{prior} \cdot \mathcal{D}_{KL}[\mathcal{N}(\mu_r^{opt}, \sigma_r^{opt}) || p(\tilde{z}_r|\hat{x}_a)]$$

### 3. Pose Estimator
- **架构**：HRNet-W48 骨干 + decoder-only Transformer，查询类型包括 Pose、Transl、Prob（共各 2 个，对应两人）。
- **Interaction-Aware Self-Attention**：分离 intra-human 自注意力（仅关注骨骼祖先/后代关节）与 inter-human 自注意力（Pose/Transl 查询相互关注），提升效率。
- **Query-Based Tracking**：上一帧的根节点平移作为 Track 查询输入当前帧，实现无需额外跟踪模块的跨帧一致性。
- **Temporal Smoothing**：通过 SmoothNet 施加时序平滑；滑动窗口（长度 L=64，重叠 L/2）内进行 prior 优化。

### 4. Dual-Human 构建流程
- 使用 Xsens 惯性动捕 + HTC Vive 定位采集约 2k 段双人互动动作。
- 拟合 SMPL-X 参数，加入基于 SDF 的碰撞损失消除模型穿透。
- 按 BEDLAM 流程渲染合成多视图图像；自动提取顶点级 contact 标注；人工标注文本描述。

## 实验与结果
- **基准数据集**：Hi4D（100 段高精度）、CHI3D（626 段）、Dual-Human（2019 段）。
- **评估指标**：MPJPE↓、PA-MPJPE↓、Translation Error↓。
- **主要结果（Dual-Human 测试集）**：
  | 方法 | MPJPE | PA | Transl |
  |---|---|---|---|
  | CLIFF (top-down) | 73.3 | 53.1 | 275.2 |
  | 4D-Humans | 66.9 | 51.3 | — |
  | BEV (bottom-up) | 83.1 | 56.8 | 263.2 |
  | TRACE (temporal) | 67.8 | 53.9 | 120.9 |
  | **Ours** | **63.4** | **51.2** | **112.1** |
- **最显著突破**：Translation Error 从 TRACE 的 120.9 降至 112.1；在 Hi4D 上 MPJPE 75.0、Transl 106.7，全面领先。
- **Prior 对比（Table 3）**：reaction prior（MPJPE 64.1）优于 BUDDI 的 proxemics prior（67.3）。
- **消融关键结论**：移除 motion VAE、用 MLP 替代 INN、去除噪声增强均导致显著性能下降；interaction-aware attention 加速收敛并提升精度；prior 对严重遮挡动作改善尤大。

## 相关工作脉络
- **BUDDI [40]**：利用扩散模型学习静态图像级的三人社交 proxemics 先验，仅刻画静态空间关系；本文 reaction prior 基于时序运动 VAE+INN，捕捉动态节奏且支持测试时优化，定位从**静态场景先验**扩展到**动态交互先验**。
- **ProHMR [29] / HuMoR [46] / VPoser [43]**：单人体姿态先验（normalizing flow、cVAE、标准 VAE）；本文将其推广至**双人对称可逆**的先验建模，利用 INN 的自然互换性。
- **InterPrior [73]**：仅建模手部交互先验；本文覆盖全身双侧关节的紧密互动建模，适用范围更广。
- **Hi4D [65] / CHI3D [12] / ExPI [18]**：既有交互数据集存在规模小、精度受限或缺少 annotation 等问题；本文 Dual-Human 在规模（2k）、精度（IMU+VIVE）、标注丰富度（多视图+contact+text）上综合超越。
- **Inter-Human [33]**：大规模文本-动作数据集但含穿透问题且精度受限于多视角 RGB；本文聚焦**高保真 MoCap 基准**，强调几何正确性。
- **RNN/Transformer-based reaction generation [30, 7]**：依赖精确动作条件，难以直接适配 noisy MoCap 估计；本文通过 VAE 潜空间 + INN 的分布对齐方式规避该问题。

## 局限性与未来方向
- **分布外交互泛化有限**：reaction prior 无法处理训练集中未覆盖的交互类型。
- **无交互场景下的伪交互风险**：对无交互输入，过大 $\lambda_{prior}$ 可能生成虚假互动姿态。
- **仅限双人假设**：扩展至三人及以上需额外设计。
- **未来方向**：扩充 Dual-Human 的交互多样性；探索文本等多模态条件驱动的双人互动生成。

## 研究启发与可借鉴点
- **INN 双向先验设计值得迁移**：可复用到其他需要对称关系的任务（如人-物交互、手-物 grasp 建模），利用可逆性自然消除角色不对称性。
- **VAE 潜空间 + 噪声增强的先验训练策略**：模拟下游估计器噪声进行 prior 训练，是缓解 distribution shift 的有效范式，可推广至单人生成 prior 的 robust 训练。
- **Interaction-aware 分层自注意力**：intra/inter 分离设计兼具效率与表达力，可在多目标跟踪、多 Agent 协作等场景中复用。
- **Dual-Human 多模态标注（多视图+contact+text）** 为后续 text-to-interaction-motion、contact-aware generation 等方向提供了高质量锚点，可直接作为下游任务预训练数据。
- **测试时 prior 优化作为 plug-and-play 后处理**：可与任意 MoCap 方法结合，无需重新训练即获得性能提升，实用价值高。

## 关键术语表
- **Reaction Priors（反应先验）**：以一人姿态为条件、预测另一人姿态概率分布的时序动态先验，由运动 VAE + INN 共同学习。
- **Invertible Neural Network (INN)**：可逆神经网络，通过归一化流实现双向映射，天然支持双人角色互换对称性。
- **Motion VAE**：将时序动作编码为分布参数（均值/方差），再解码重建，提供紧凑且符合人体运动约束的潜表示。
- **Interaction-Aware Self-Attention**：将 self-attention 分解为 intra-human（骨骼层级）和 inter-human（跨人关节）两部分以提升注意力效率。
- **Dual-Human**：本文构建的双人紧密互动数据集，含 ~2019 段动作、合成多视图图像、contact 标注与文本描述。
- **PA-MPJPE**：Procrustes-aligned MPJPE，先做刚性对齐后再计算平均关节误差，消除全局尺度/旋转差异。
- **ELBO（Evidence Lower Bound）**：变分推断中用于训练 VAE 的目标函数下界，包含重建项与 KL 散度正则项。
- **SDF-based Collision Loss**：基于 Signed Distance Field 的碰撞惩罚，用于消除 SMPL-X 拟合中的身体穿透问题。

## 可复现要素
- **数据集**：Dual-Human 已公开，项目页面：https://neteasegameai.github.io/Dual-Human/；Hi4D、CHI3D 为公开基准。
- **代码/权重**：论文未明确声明开源仓库链接（仅在项目页说明数据集公开），代码与预训练权重状态**论文未提及**。
- **关键超参**：HRNet-W48 backbone；学习率 $1\times10^{-4}$（AdamW，weight decay $5\times10^{-4}$）；$\lambda_{prior}=0.01$；窗口长度 $L=64$，重叠 $L/2$；输入分辨率 $512\times512$；$\lambda_{dist}=0.0001$。
