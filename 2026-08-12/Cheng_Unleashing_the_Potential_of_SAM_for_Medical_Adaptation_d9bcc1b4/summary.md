---
title: "Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cheng_Unleashing_the_Potential_of_SAM_for_Medical_Adaptation_via_Hierarchical_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:17:03"
field: "医学图像分割与视觉基础模型适配"
keywords: ["Segment Anything Model", "医学图像分割", "少样本适应", "分层解码", "LoRA", "无prompt适配", "可学习掩码注意力"]
innovations: ["两阶段分层解码框架：以Stage-1概率先验掩码引导Stage-2精细解码", "可学习掩码交叉注意力：用连续概率图替代二元硬掩码实现可微空间软调制", "类平衡掩码引导自注意力（CMAttn）：通过逆方差高斯噪声扰动缓解医学长尾标签分布问题"]
benchmarks: ["Synapse Multi-Organ CT", "Left Atrial (LA) MRI", "PROMISE12 Prostate MRI"]
---

# 论文速读：Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding

## 一句话总结
本文提出 H-SAM，一种面向医学图像分割的**无 prompt** SAM 自适应方法，通过两阶段分层解码流程，以第一阶段生成的概率先验掩码引导第二阶段精细解码；在仅使用 Synapse 数据集 **10% 切片**（约1个体素）的情况下，平均 Dice 达 **80.35%**，超越现有所有 prompt-free SAM 变体约 **5%**，且在无任何未标注数据时仍击败多个依赖大量未标注数据的半监督 SOTA 方法。

## 研究问题与动机
1. **SAM 零样本医学性能退化**：SAM 虽在自然图像上表现优异，但在未见医学图像上准确率与鲁棒性显著下降（因其训练仅含自然图像，未见病理/解剖特征）。
2. **全量微调成本过高**：对 SAM 全部参数微调需巨额计算资源，且极易在单一医学数据集上过拟合。
3. **Prompt 依赖难以落地**：现有有 prompt 的 SAM 适配方法依赖 ground truth 提取的点或框提示，需医学专家标注，耗时且噪声大，临床可用性低。
4. **无 prompt 方法性能有限**：已有 prompt-free 适配方法（如 AutoSAM、SAMed）因缺乏医学先验知识引导，分割精度明显落后于有 prompt 方法。

## 核心贡献（创新点）
1. **两阶段分层解码框架**：冻结 SAM 图像编码器，第一阶段用原始轻量 mask decoder 生成概率先验掩码，第二阶段在此基础上进行更精细的解码，本质区别在于将"单次解码"扩展为"先验引导的级联解码"。
2. **类平衡掩码引导自注意力（CMAttn）**：对第一阶段掩码特征施加与类别频率成反比的方差高斯噪声扰动，缓解医学图像中标签长尾分布问题，使图像嵌入对尾部类别更鲁棒，区别于传统 logit adjustment 仅作用于分类头。
3. **可学习掩码交叉注意力（Learnable Mask Cross-Attention）**：用未经变换的概率掩码 M 直接与注意力图做逐元素乘法，解决了原始 mask attention 中梯度消失和二元掩码信息丢失的问题，使得空间调制可微且能区分不同前景区域的贡献。
4. **分层像素解码器（Hierarchical Pixel Decoder）**：在第二阶段引入 U 型像素解码器并通过跳跃连接融合第一阶段的局部特征，弥补原始 SAM 像素解码器仅输出 H/4×W/4 低分辨率、难以捕捉细小医学结构的不足。
5. **仅微调 LoRA 适配器 + 默认 embedding**：图像编码器全冻结，仅加 rank=4 的 LoRA bypass，prompt encoder 不引入真实 prompt 而只训练一个默认 embedding，在保持预训练知识的同时实现高效少样本适应。

## 方法详解
**整体架构**：冻结 SAM ViT 图像编码器 → 添加 LoRA 适配器 → 默认 prompt embedding 输入原始轻量 mask decoder（Stage-1）→ 生成概率先验掩码 P → Stage-2 以增强后的图像嵌入和 P 为条件进行精细解码 → 两级输出概率平均融合。

**Stage-1（初始解码）**：使用 SAM 原始 mask decoder（2 层 Transformer decoder + 轻量 pixel decoder），输入为 LoRA 适配后的图像嵌入和默认 prompt embedding，输出低分辨率（H/4×W/4）概率掩码 P ∈ R^{N×C×H×W}。

**CMAttn（Class-Balanced Mask-Guided Self-Attention）**：
- 对 P 施加高斯噪声扰动：$P(gt=i) += \mathcal{N}(0, var(i))$，其中 var(i) 与类别 i 的样本频率成反比（离线统计）。
- 将扰动后 P 通过 self-attention 重校准图像嵌入，再用线性层压缩通道后与原图像嵌入做 Hadamard 积 ⊙，保留残差路径。

**Learnable Mask Cross-Attention（Stage-2 Transformer decoder 内）**：
- 原始 mask attention：$X = softmax(t(M) + KQ^T)V + X$，其中 t(M) 将二值掩码映射为 {-∞, 0}，梯度无法回传至 M。
- 本文可学习版本：$X = M \odot softmax(KQ^T)V + X$，使用连续概率图 M（与 saliency map 同分辨率），逐元素乘法实现空间软调制，梯度可畅通回传，对不同前景区域赋予差异化权重。

**Hierarchical Pixel Decoder（Stage-2）**：
- 采用 U 型结构，从 Stage-1 pixel decoder 输出通过跳跃连接引入局部特征，逐级上采样至全分辨率 H×W，有效捕获多尺度医学目标。

**训练损失**：
- $\mathcal{L} = \lambda_{ce}\mathcal{L}_{ce} + \lambda_{dice}\mathcal{L}_{dice}$，两阶段分别施加深度监督（Stage-1 监督 1/4 分辨率 gt，Stage-2 监督全分辨率 gt）。
- 总损失：$\mathcal{L}_{total} = \lambda_w \mathcal{L}_{stage1} + (1-\lambda_w)\mathcal{L}_{stage2}$，$\lambda_w$ 从 0.8 开始以衰减系数 0.005 指数递减，鼓励模型逐步聚焦精细输出。
- 最终预测为两阶段概率输出取平均。

## 实验与结果
**数据集**：
- **Synapse 多器官 CT**：3779 轴位切片，18 训练病例 / 12 测试病例，评估 8 个腹部器官（aorta, gallbladder, spleen, left/right kidney, liver, pancreas, stomach）。
- **LA（Left Atrial）MRI**：80 训练扫描 / 20 测试扫描，few-shot 设置取 4 个标注扫描（5%），其余 76 个作为未标注数据（H-SAM 不使用）。
- **PROMISE12（前列腺 MRI）**：50 例，40 训练 / 10 测试，few-shot 设置取 3 个标注病例（7.5%），其余 37 个未使用。

**评估指标**：Dice coefficient、平均 Hausdorff Distance（HD）。

**主要结果**：
- **Synapse 10% 切片（few-shot）**：H-SAM Mean Dice **80.35%**，较 SAMed（75.57%）提升 **4.78%**，较 AutoSAM（55.69%）提升巨大；HD 仅 15.54，显著优于基线。
- **Synapse 全监督**：H-SAM Mean Dice **86.49%**，超越 DAE-Former（82.43%）和 MERIT（84.90%），HD 仅 **8.18**。
- **LA 4 扫描（5% labeled，0 unlabeled）**：H-SAM Mean Dice **89.22%**，超过半监督方法 BCP（88.02%）和 nnU-Net（64.02%）。
- **PROMISE12 3 病例（7.5% labeled，0 unlabeled）**：H-SAM Mean Dice **87.27%**，超过半监督 MLB-Seg（78.27%）**10.94%**，超过 SAMed（86.00%）1.27%。

**消融实验**：三模块贡献——Learnable Mask-Attention +2.1%，CMAttn +1.2%，Hierarchical Pixel Decoder +2.2%，三者叠加总提升 **4.78%**（75.57 → 80.35）。原始 mask-attention（不可微）仅提升 0.04%，验证可学习设计的必要性。

**效率分析**：H-SAM 总参数 112.3M，仅比 SAMed（108.8M，2 层 Transformer）多 3.5M，但以相同参数量远超更深 Transformer（SAMed-4layer: 76.80%；SAMed-6layer: 78.05%）；相比 SAM Adapter（131.5M）减少 20M 参数且 Dice 高 7.5%。

## 相关工作脉络
1. **MedSAM [49]**：构建大规模医学图像数据集适配 box-prompt SAM，属于**有 prompt**适配路线，依赖 ground truth 提取边界框；本文定位无 prompt 场景，无需任何人工提示。
2. **Medical SAM Adapter [64]**：将 adapter 注入 SAM 图像编码器 + point-prompt 适配，同样依赖点提示；本文在不使用任何提示的情况下实现更好性能。
3. **AutoSAM [30]**：冻结编码器、添加预测头实现无 prompt 分割，但解码结构简单，缺乏层级先验引导；H-SAM 以分层解码弥补其信息损失。
4. **SAMed [70]**：在编码器中插入 LoRA 适配器，保留原始轻量 mask decoder，是无 prompt 适配的强基线；H-SAM 在其基础上创新性地增加了第二解码阶段和三个核心模块。
5. **Mask2Former [11]**：引入 mask attention 机制，本文取其思想但解决其梯度消失和二元掩码信息不足的问题，提出可学习连续掩码版本。
6. **CMAttn 与长尾处理 [41]**：借鉴 logit adjustment 思路，但将类平衡扰动直接作用于 mask feature 而非分类 logits，并嵌入 self-attention 中实现更精细的图像嵌入重校准。

## 局限性与未来方向
1. **3D 自适应未探索**：当前方法仅在 2D 切片上验证，SAM 的 3D 医学体积适配（如肺结节、肝脏分割的体素级任务）尚待研究。
2. **ViT-L 在极少数据下的过拟合风险**：全监督设置使用 ViT-L 虽有效，但在极端少样本（如 1~2 例）场景下泛化能力未充分验证。
3. **类别数固定假设**：CMAttn 依赖离线统计类别频率，在多中心、类别差异大的异构医学数据上需动态调整策略。
4. **未来方向**：扩展至 3D 体积分割；结合提示工程实现 prompt/hierarchical 混合模式；探索与其他医学基础模型（如 Med-PaLM、CLIP 医学版）的协同适配。

## 研究启发与可借鉴点
1. **分层解码范式**：将"粗→精"两阶段决策迁移到其他视觉基础模型的领域适配中（如自然图像 SAM 的细粒度 variant、其他 foundation model 的少样本 adaptation），具有通用方法论价值。
2. **可学习掩码注意力**：将 mask attention 的二元硬选择改为连续概率软调制，解决了梯度流问题，此设计可复用于任何需空间软调制的 transformer decoder（如检测、实例分割任务）。
3. **类平衡噪声增广**：将 logit adjustment 思想从分类头迁移到特征表示层，以高斯扰动增强尾部类别的表征鲁棒性，适用于各类医学长尾分割数据集。
4. **U 型像素解码器嫁接**：在冻结的大模型编码器基础上，仅在解码端引入 U 型跳跃连接结构，以极小参数代价显著提升细节恢复能力，可作为大模型适配的通用模块模板。
5. **深度监督 + 渐进式权重衰减**：$\lambda_w$ 从 0.8 指数衰减至末端的训练策略，有效平衡了粗掩码先验学习 vs 精细解码优化的时序关系，值得在级联网络训练中推广。

## 关键术语表
**SAM（Segment Anything Model）**：Meta 于 2023 年发布的大规模图像分割基础模型，在超 10 亿 mask 上预训练，支持 zero-shot 分割和 prompt 驱动交互。
**LoRA（Low-Rank Adaptation）**：通过在冻结 Transformer 层的注意力模块中插入低秩分解矩阵（如 rank=4）来实现参数高效微调，仅更新少量参数即可适配新域。
**CMAttn（Class-Balanced Mask-Guided Self-Attention）**：引入类频率逆方差高斯噪声扰动掩码特征后通过 self-attention 重校准图像嵌入，缓解医学图像中长尾类别表征不足问题。
**Learnable Mask Cross-Attention**：用连续概率图替代二元掩码与交叉注意力的逐元素乘法，使空间软调制可微且能区分前景区域重要性。
**Hierarchical Pixel Decoder**：在 Stage-2 引入 U 型像素解码器，通过跳跃连接融合 Stage-1 局部特征并逐级上采样至全分辨率，增强小目标和细节捕捉能力。
**Deep Supervision**：对网络各中间层输出同时施加监督损失（本文 Stage-1 监督 1/4 分辨率 gt，Stage-2 监督全分辨率 gt），加速训练并缓解梯度消失。
**Dice Coefficient**：衡量预测掩码与真实掩码重叠度的指标，取值 [0,1]，1 为完美重合，医学图像分割中最常用的 Dice 均值（Mean Dice）为标准评测指标。
**Few-shot Segmentation**：仅使用极少量标注样本（如 3~4 个病例）进行模型训练，评估在数据稀缺场景下的分割泛化能力。

## 可复现要素
- **数据集**：Synapse（公开）、LA（公开，Atrial Segmentation Challenge）、PROMISE12（公开），数据预处理和 split 方式遵循各自基准论文。
- **代码**：已开源，GitHub 地址：https://github.com/Cccccczh404/H-SAM
- **预训练权重**：原始 SAM（ViT-B / ViT-L）公开可用；LoRA 适配器为训练学习。
- **关键超参**：LoRA rank = 4；最大训练轮数 = 300；优化器 AdamW（β1=0.9, β2=0.999, weight decay=0.1）；λ_w 初始 0.8，指数衰减系数 0.005；全监督训练分辨率 224×224，few-shot 训练分辨率 512×512。
- **硬件**：4 × NVIDIA RTX A5000 GPU。
- **数据增强**：弹性形变（elastic deformation）、旋转（rotation）、缩放（scaling）。
