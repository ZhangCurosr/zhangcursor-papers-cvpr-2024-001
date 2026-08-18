---
title: "Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:59:16"
field: "低级别视觉-图像恢复"
keywords: ["图像恢复", "Transformer", "稀疏注意力", "特征冗余", "雨条纹去除", "雾霾去除"]
innovations: ["提出双分支自适应稀疏自注意力（ASSA），通过可学习权重融合稀疏与密集分支，避免过稀疏导致的信息损失", "设计增强-简化策略特征细化前馈网络（FRFN），利用 PConv 和门控机制沿通道维度筛选关键特征", "在雨条纹、雨滴、雾霾三项恢复任务上均取得 SOTA，SPAD 达 49.51 dB，超越 DRSformer 0.98 dB"]
benchmarks: ["SPAD", "AGAN-Data", "Dense-Haze", "Internet-Data"]
---

# 论文速读：Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F

## 一句话总结
论文提出了自适应稀疏 Transformer（AST），通过双分支自适应稀疏自注意力（ASSA）和特征细化前馈网络（FRFN）协同过滤空间与通道维度的冗余信息，在雨条纹去除、真实雾霾去除和雨滴去除任务上均取得 SOTA 性能。

## 研究问题与动机
- **计算复杂度与冗余并存**：标准 Transformer 的全局自注意力计算成本高，且会引入无关区域的噪声交互；现有高效注意力设计（如窗口注意力、通道注意力）虽降低了计算负担，但未解决冗余特征干扰问题。
- **稀疏注意力存在信息丢失**：直接采用 ReLU-based 稀疏自注意力可过滤低匹配度 token，但过于稀疏会导致信息损失，损害恢复质量；当前工作要么依赖 Top-K 选择（参数 K 对任务敏感），要么投影到超像素空间（仍考虑所有 token）。
- **通道维度冗余未被充分处理**：特征图在深层通常具有高通道维度，大量通道包含冗余信息，标准 FFN 对每个像素位置独立处理，无法有效筛选通道级关键信息。

## 核心贡献（创新点）
- **提出 AST 统一框架**：通过 ASSA 和 FRFN 协同消除空间域与通道域的冗余信息，无需复杂参数调节即可自适应平衡稀疏性与信息完整性。
- **设计双分支自适应自注意力（ASSA）**：与已有 Top-K 硬选择或超像素投影方法本质不同，ASSA 通过可学习权重自适应融合稀疏分支（SSA，基于 Squared ReLU 过滤负相关项）和密集分支（DSA，保留关键信息），避免过稀疏导致的信息损失。
- **开发增强-简化策略特征细化前馈网络（FRFN）**：与常规 FFN 或其他轻量 FFN（如 GDFN、LeFF）不同，FRFN 采用部分卷积（PConv）增强关键通道，并通过门控机制沿通道维度简化冗余特征，实现精细化的特征选择。

## 方法详解
- **整体架构**：AST 采用编码器-解码器结构，输入图像经卷积层生成低层特征 $F_0$，通过 $N_1$ 个对称编解码阶段（每个阶段含 $N_2$ 个基本块），最终经卷积层生成残差图并与输入叠加得到恢复图像。训练损失为 Charbonnier loss：$\ell(I',\hat{I}) = \sqrt{\|I'-\hat{I}\|^2 + \epsilon^2}$，其中 $\epsilon=10^{-3}$。
- **自适应稀疏自注意力（ASSA）**：将特征图划分为不重叠窗口（默认大小 $M \times M = 8 \times 8$），生成 $Q, K, V$。稀疏分支 SSA 使用 Squared ReLU 激活：$SSA = ReLU^2(QK^T/\sqrt{d} + B)$，过滤负相关相似度；密集分支 DSA 使用标准 Softmax：$DSA = Softmax(QK^T/\sqrt{d} + B)$。最终通过可学习参数 $\{a_1, a_2\}$ 加权融合：$w_n = \frac{e^{a_n}}{\sum_{i=1}^{N} e^{a_i}}$，输出 $A = (w_1 \cdot SSA + w_2 \cdot DSA)V$。
- **特征细化前馈网络（FRFN）**：采用增强-简化（enhance-and-ease）策略：$\hat{X'} = GELU(W_1 \cdot PConv(\hat{X}))$，将 $\hat{X'}$ 按通道分割为 $\hat{X'_1}, \hat{X'_2}$，经重排和展平操作后，利用门控机制 $\hat{X'_r} = \hat{X'_1} \otimes F(DWConv(R(\hat{X'_2})))$，最终输出 $\hat{X'_{out}} = GELU(W_2 \cdot \hat{X'_r})$。PConv 仅对部分通道进行卷积，起到稀疏选择作用；DWConv 配合门控机制减轻冗余通道处理负担。

## 实验与结果
- **数据集与基线**：
  - 雨条纹去除：SPAD 数据集，对比包括 DDN、ResCAN、PReNet、RCDNet、SPDNet、SPAIR、DualGCN、SEIDNet、MPRNet、Fu et al.、Restormer、SCD-Former、IDT、Uformer、DRSformer 等 15 个方法。
  - 雨滴去除：AGAN-Data 数据集，对比 Eigen、Pix2pix、Uformer、WeatherDiff、TransWeather 等 15 个方法。
  - 真实雾霾去除：Dense-Haze 数据集，对比 RIDCP、DCP、SGID、AOD-Net、GridDehazeNet、FFA-Net、Uformer、Restormer、Fourmer 等 15 个方法。
- **主要结果**：
  - **SPAD 雨条纹去除**：AST-B 达到 **PSNR 49.51 dB / SSIM 0.9942**，较前 best Transformer 方法 DRSformer（48.53 dB）提升 **0.98 dB**，较前 best CNN 方法 Fu et al.（45.00 dB）提升 **4.51 dB**；AST-B+（含几何自集成）达 **49.72 dB / 0.9944**。
  - **AGAN-Data 雨滴去除**：AST-B 达 **PSNR 32.32 dB / SSIM 0.935**，较前 best 方法 AWRCP（31.93 dB）提升 **0.39 dB**，较扩散模型 WeatherDiff₁₂₈（30.71 dB）提升 **2.66 dB**。
  - **Dense-Haze 雾霾去除**：AST-B 达 **PSNR 17.12 dB / SSIM 0.55**，较前 best CNN 方法 AECR-Net（15.80 dB）提升 **1.37 dB**，较近期 Transformer 方法（MB-TaylorFormer-B 16.66 dB）提升 **0.46 dB**。
  - **无参考指标 NIQE（真实雨天）**：AST-B 达 **5.493**，显著优于 Uformer（5.749）、Restormer（6.162）等方法，表明感知质量更优。
- **效率对比**：在相同参数量下（AST-B: 6.65M params, 13.35G FLOPs），AST 在三项任务上均超越计算量相近或更高的方法。

## 相关工作脉络
- **DRSformer [8]**：采用 Top-K 通道选择算子筛选最有信息量的 token，但 K 值对特定任务敏感；本文 AST 无需手动调节 K，通过双分支自适应机制自动平衡稀疏度与信息保留。
- **CODE [114]**：将特征投影到超像素空间执行注意力，虽减少空间冗余但仍考虑超像素内所有 token；AST 通过 SSA 直接过滤负相关交互，同时 DSA 补充必要信息流，不依赖超像素分割。
- **Restormer [100] / Uformer [82]**：分别引入通道注意力与窗口注意力降低计算复杂度，但未解决特征冗余问题；AST 在保留这些高效注意力设计思想的基础上，通过 ASSA 和 FRFN 双重冗余抑制进一步提升性能。
- **SwinIR [37]**：采用移位窗口注意力捕捉跨窗口交互；AST 的 ASSA 不依赖固定窗口策略，而是通过自适应权重在窗口内动态平衡稀疏/密集交互。
- **GDFN [100] / LeFF [82]**：轻量级前馈网络变体，通过深度卷积或门控机制简化计算；本文 FRFN 在此基础上增加 PConv 增强机制，更精细地沿通道维度筛选关键特征。

## 局限性与未来方向
- **重退化场景表现有限**：论文 Fig. 6 展示案例表明，AST 在处理重度退化场景时仍存在恢复困难，说明当前设计对极端退化模式的鲁棒性有待提升。
- **任务特定模型**：当前 AST 针对单一退化类型（雨条纹/雾霾/雨滴）设计，未实现统一的多退化处理框架；未来需探索统一模型以应对多种低质量退化。
- **未探索先验注入**：论文建议可注入领域先验（如去雾霾暗通道先验、低光照 Retinex 先验）以扩展适用范围，当前工作未涉及。
- **窗口大小固定**：默认窗口大小设为 8，可能无法自适应不同尺度的退化模式，未来可研究动态窗口或自适应感受野机制。

## 研究启发与可借鉴点
- **双分支自适应融合策略**：将稀疏与密集机制结合并通过可学习权重自适应调节，避免了单一稀疏化带来的信息损失，该思想可迁移至其他视觉任务（如超分辨率、去噪）的特征聚合设计中。
- **增强-简化（enhance-and-ease）特征变换范式**：FRFN 通过 PConv 筛选关键通道、门控机制抑制冗余通道的设计，为 FFN 改进提供了新方向，可在 ViT、Swin Transformer 等架构的前馈部分复用。
- **稀疏注意力熵分析验证**：论文通过注意力熵（entropy）定量分析 SSA 过稀疏问题（Tab. 6），该方法可作为评估稀疏注意力机制有效性的通用诊断工具。
- **几何自集成策略**：AST-B+ 通过几何自集成进一步提升性能（SPAD 达 49.72 dB），该策略可与其他 Transformer 恢复模型结合，作为提升精度的后处理技巧。

## 关键术语表
- **AST（Adaptive Sparse Transformer）**：本文提出的自适应稀疏 Transformer 模型，通过 ASSA 和 FRFN 协同去除空间与通道冗余特征。
- **ASSA（Adaptive Sparse Self-Attention）**：自适应稀疏自注意力模块，采用双分支（稀疏分支 SSA + 密集分支 DSA）并通过可学习权重自适应融合，兼顾稀疏过滤与信息保留。
- **FRFN（Feature Refinement Feed-forward Network）**：特征细化前馈网络，采用增强-简化策略，通过 PConv 增强关键通道信息、门控机制简化冗余通道。
- **SSA（Sparse Self-Attention）**：稀疏自注意力分支，基于 Squared ReLU 激活，过滤负相关 query-key 相似度，抑制无关区域噪声交互。
- **DSA（Dense Self-Attention）**：密集自注意力分支，采用标准 Softmax，确保关键信息的充分传播，避免 SSA 过稀疏导致的信息丢失。
- **PConv（Partial Convolution）**：部分卷积操作，仅对部分输入通道执行卷积，作为稀疏选择机制强化关键特征。
- **Sparsity vs. Information Loss Trade-off**：稀疏注意力设计中的核心矛盾——过度稀疏会丢失必要信息，本文通过双分支自适应机制平衡这一权衡。
- **Entropy Analysis of Attention**：注意力熵分析，用于量化不同自注意力机制的信息分布集中度，本文借此验证 ASSA 的平衡效果。

## 可复现要素
- **数据集**：SPAD [78]、AGAN-Data [58]、Dense-Haze [1]、Internet-Data [78]；其中 SPAD、AGAN-Data、Dense-Haze 为公开 benchmark。
- **代码与权重**：代码已开源，链接为 https://github.com/joshyZhou/AST；预训练模型已提供。
- **关键超参**：编码器/解码器阶段数 $N_1=4$，窗口大小 $8 \times 8$，每头维度 $d$ 与 Restormer/Uformer 保持一致；AST-T 嵌入维度 $C=16$，AST-B $C=32$；学习率初始 0.0002，余弦衰减至 0.000001；训练使用 128×128 图像块，旋转与翻转数据增强；Charbonnier loss 中 $\epsilon=10^{-3}$。
- **训练策略**：采用渐进式学习策略（progressive learning），类似 Restormer/Uformer。
