---
title: "Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:59:42"
field: "图像恢复"
keywords: ["Adaptive Sparse Transformer", "Image Restoration", "Self-Attention", "Feature Redundancy", "Rain Removal", "Dehazing"]
innovations: ["通过稀疏与密集两分支自适应融合抑制无关区域噪声并保留关键信息", "设计增强‑缓解前馈网络在通道维度消除冗余"]
benchmarks: ["SPAD", "AGAN-Data", "Dense-Haze"]
---

# 论文速读：Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F

## 一句话总结
本文提出自适应稀疏 Transformer（AST），通过**自适应稀疏自注意力（ASSA）**与**特征精炼前馈网络（FRFN）**协同消除空间与通道维度的冗余和噪声交互，在雨条纹去除、真实雾气去除、雨滴去除等图像恢复任务上取得领先性能。

## 研究问题与动机
1. 现有 Transformer-based 图像恢复方法虽能建模长程依赖，但计算复杂度高，且密集注意力会引入无关区域的噪声交互。
2. 密集聚合的特征图中存在大量冗余信息，阻碍模型聚焦于 informative features。
3. 已有稀疏注意力方法（如 Top-K 选择、超像素空间投影）对特定参数敏感，且在超像素空间内仍对所有 token 计算注意力，无法彻底消除冗余。

## 核心贡献（创新点）
1. **提出自适应稀疏自注意力（ASSA）模块**：通过稀疏分支（SSA）与密集分支（DSA）自适应加权融合，在过滤噪声的同时保留必要信息，避免依赖固定 K 值或复杂空间投影。
2. **设计特征精炼前馈网络（FRFN）**：采用“增强‑缓解”方案，利用部分卷积（PConv）和门控机制在通道维度消除冗余，互补 ASSA 的空间冗余抑制。
3. **构建统一高效的 AST 框架**：在雨条纹、雾气、雨滴三种退化任务上实现卓越性能，代码与预训练权重均已开源。

## 方法详解
- **ASSA（自适应稀疏自注意力）**：将特征图划分为 M×M 非重叠窗口，生成 Q、K、V 矩阵。分别计算稀疏自注意力 `SSA = ReLU²(QKᵀ/√d + B)` 与密集自注意力 `DSA = Softmax(QKᵀ/√d + B)`，两者经可学习权重自适应加权融合：`A = (w₁·SSA + w₂·DSA)V`，权重通过 Softmax 归一化。
- **FRFN（特征精炼前馈网络）**：输入特征先经线性投影与 PConv 增强有用通道，再经 Reshape/Flatten 引入局部性，随后通过 DWConv 与门控机制抑制冗余，最终经 GELU 激活与线性投影输出。
- **整体架构**：对称编码器‑解码器结构，编码器阶段使用 FRFN，解码器阶段使用 ASSA+FRFN 基本块，瓶颈阶段复用解码器块以捕获更长程依赖。损失函数为 Charbonnier 损失（ε=10⁻³）。

## 实验与结果
- **数据集**：SPAD（雨条纹去除）、AGAN‑Data（雨滴去除）、Dense‑Haze（真实雾气去除）。
- **基线**：对比 DRSformer、Uformer、Restormer、IDT 等多种 CNN/Transformer 方法。
- **主要结果**：
  - SPAD：AST‑B 达 **PSNR 49.51 dB**，较次优 Transformer 方法 DRSformer 提升 **0.98 dB**。
  - AGAN‑Data：AST‑B 达 **PSNR 32.32 dB**，较最优方法 AWRCP 提升 **0.39 dB**。
  - Dense‑Haze：AST‑B 达 **PSNR 17.12 dB**，较次优 Transformer 方法 MB‑TaylorFormer‑B 提升 **0.46 dB**。
  - 感知质量（NIQE）：AST‑B 在真实雨景数据上达到最低值 **5.493**。
- **消融实验**：ASSA 较 Swin SA、Top‑k SA、Condensed SA 分别提升 0.96、0.76、0.49 dB；FRFN 较 GDFN 提升 0.77 dB。

## 相关工作脉络
1. **SwinIR / Uformer**：基于窗口注意力的高效 Transformer，但仍是密集计算；AST 进一步通过自适应稀疏机制过滤无关 token。
2. **DRSformer**：采用 Top‑K 选择最有价值 token，但 K 值敏感且未处理通道冗余；AST 以两分支自适应融合替代固定选择。
3. **CODE / Condensed SA**：将特征投影至超像素空间再计算注意力，仍会在超像素内引入不必要交互；AST 直接在窗口内实现稀疏‑密集融合。
4. **Restormer / LeFF**：引入通道注意力与高效 FFN，但 FFN 在通道维度仍可能冗余；AST 的 FRFN 专门针对通道冗余进行增强‑缓解变换。
5. **StarReLU / ACON**：自适应激活函数用于修改注意力分数本身；AST 在两分支层面进行宏观信息流调控，适应性更强。

## 局限性与未来方向
1. 当前模型针对特定退化类型训练，泛化至多种退化场景的能力有限，未来可探索统一模型。
2. 在重度退化场景（如图 6 所示）下性能下降，需增强鲁棒性。
3. 可结合领域先验（如暗通道先验用于去雾、Retinex 先验用于低光照）进一步提升专用任务表现。

## 研究启发与可借鉴点
1. **两分支自适应融合思路**：可迁移至视频去噪、医学图像恢复等需平衡稀疏性与信息完整性的任务，通过可学习权重动态调整不同分支贡献。
2. **“增强‑缓解”特征变换范式**：FRFN 的设计模式（部分卷积增强 + 门控抑制冗余）可替代其他视觉 Transformer 中的标准 FFN，降低通道冗余。
3. **注意力熵分析作为诊断工具**：论文利用注意力熵量化稀疏/密集注意力的信息分布差异，该方法可用于评估不同注意力设计的合理性。
4. **模块化设计易于扩展**：ASSA 与 FRFN 作为独立组件，可与其他高效注意力机制（如线性注意力）结合，构建更轻量的恢复模型。

## 关键术语表
- **Adaptive Sparse Self‑Attention (ASSA)**：通过稀疏与密集两分支自适应加权融合，平衡噪声过滤与信息保留的自注意力机制。
- **Feature Refinement Feed‑forward Network (FRFN)**：采用增强‑缓解方案在通道维度消除冗余的前馈网络，包含部分卷积与门控机制。
- **Sparse Self‑Attention (SSA)**：使用平方 ReLU 激活实现稀疏性的自注意力，过滤低匹配分数的 token 交互。
- **Dense Self‑Attention (DSA)**：标准 softmax 自注意力，考虑所有 token 对，可能引入无关区域噪声。
- **Partial Convolution (PConv)**：仅对部分通道进行卷积，视为稀疏选择过程以聚焦有用特征。
- **Attention Entropy**：衡量注意力分数分布集中程度的指标，低熵表示高度集中，高熵表示均匀分布。

## 可复现要素
- **数据集**：SPAD、AGAN‑Data、Dense‑Haze（均为公开基准）。
- **代码/权重**：论文声明代码与预训练模型已在 GitHub 开源（https://github.com/joshyZhou/AST）。
- **关键超参**：窗口大小 8×8；嵌入维度 C=32（AST‑B）；AdamW 优化器，初始学习率 0.0002，余弦衰减至 0.000001；训练使用 128×128 图像块；Charbonnier 损失（ε=10⁻³）。
