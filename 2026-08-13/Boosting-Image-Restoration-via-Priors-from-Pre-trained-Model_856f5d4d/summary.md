---
title: "Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Restoration_via_Priors_from_Pre-trained_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:13"
field: "低层视觉-图像恢复"
keywords: ["图像恢复", "预训练模型", "轻量模块", "CLIP", "空间变化增强", "通道-空间注意力", "泛化"]
innovations: ["提出PTG-RM轻量插件，利用预训练多模态模型特征（OSF）作为通用恢复先验", "PTG-SVE用可学习空间映射替代固定SNR参考，自适应融合短程/长程操作", "PTG-CSA利用先验生成空间变化卷积核定制通道-空间注意力"]
benchmarks: ["LOL-real", "SID", "Rain100H", "Rain100L", "Test100", "GoPro", "HIDE", "RealBlur-R", "DPDD", "Set12", "BSD68", "Urban100", "SIDD"]
---

# 论文速读：Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model

## 一句话总结
本文提出一个轻量级插件模块 **PTG-RM**（Pre-Train-Guided Refinement Module，<1M 参数），通过从 CLIP、Stable Diffusion、BLIP2 等预训练多模态模型中提取的现成特征（OSF）作为先验，自适应指导空间变化增强与通道-空间注意力，从而在不增加推理开销的前提下显著提升低光增强、去雨、去模糊、去噪等多种图像恢复任务的性能。

---

## 研究问题与动机
- **图像恢复本质上是病态问题**，仅靠堆叠网络结构或增加参数量容易过拟合，难以持续突破性能上限。
- **现有先验方法依赖物理变量**（如深度、语义图），但这些密集预测网络跨场景泛化能力不足，且需要复杂定制机制，普适性受限。
- **预训练多模态模型（CLIP 等）已被发现包含退化相关特征信息**（如 CLIP-IQA），但在图像恢复方向的应用尚未探索。
- **异构特征对齐困难**：预训练模型的 OSF 与恢复网络的特征在表示空间和形状（1D vs. 2D）上存在差异，难以直接使用。

---

## 核心贡献（创新点）
1. **提出首个通用框架利用预训练多模态模型的特征提升各类图像恢复任务**，将 CLIP/SD/BLIP2 等模型的现成特征转化为恢复先验，推理阶段无需预训练模型参与。
2. **设计 PTG-SVE（空间变化增强）**，用可学习的空间映射替代传统的固定 SNR 参考，自适应融合短程与长程操作结果，实现按区域最优处理。
3. **设计 PTG-CSA（通道-空间注意力）**，利用预训练先验生成空间变化的卷积核来合成空间权重，使注意力机制能针对不同区域定制，而非均匀处理。
4. **以端到端方式联合训练目标恢复网络与轻量 Refinement 模块**，仅在训练阶段引入预训练模型（<1M 额外参数），Loss 与目标网络保持一致，并可兼容无监督训练范式。

---

## 方法详解

**整体流程**：给定退化图像 $I_d$，目标恢复网络 $\mathcal{F}$ 产出初始结果 $\hat{I}_c = \mathcal{F}(I_d)$；同时预训练模型 $\mathcal{G}$ 提取 OSF $g = \mathcal{G}(I_d)$。PTG-RM $\mathcal{R}$ 以 $\hat{I}_c$、$I_d$、$g$ 为输入，输出修正结果 $\bar{I}_c$。

**编码器-解码器结构（潜空间蒸馏）**：
- 编码器：$f = \mathcal{R}_e(\hat{I}_c \oplus I_d)$，通过对比初始恢复结果与输入图像，提取 latent feature $f \in \mathbb{R}^{h \times w \times c}$。
- 解码器输出校正掩码 $I_m$ 和残差细化 $I_r$。
- 最终结果：$\bar{I}_c = I_d + (\hat{I}_c - I_d) \times I_m + I_r$（公式 1）。

**PTG-SVE（空间变化增强）**：
- 引入可学习位置嵌入 $S_m = \mathcal{P}(S)$（$S$ 为 $h \times w \times 2$ 的坐标图），将先验 $g$ 映射到更适合的空间 $\mathcal{T}_m(g)$。
- 范围分数图：$M = \mathcal{R}_m(f \oplus \mathcal{T}_m(g) \oplus S_m)$（公式 2），决定每个位置的短程/长程操作占比。
- 短程操作 $\mathcal{R}_s$（ResNet）、长程操作 $\mathcal{R}_l$（Restormer backbone），融合：$\hat{f} = M \times f_s + (1-M) \times f_l$（公式 3）。

**PTG-CSA（通道-空间注意力）**：
- 通道注意力：$M_c = \mathcal{R}_c(\mathcal{O}(\hat{f}) \oplus \mathcal{T}_c(g))$，$\hat{f}_c = \hat{f} \times M_c$（公式 4）。
- 空间注意力：先生成位置感知卷积核 $\mathcal{C}_p = \mathcal{R}_p(\hat{f}, \mathcal{T}_c(g), S_c)$（公式 5），再通过逐位置卷积得到 $\hat{M}_s = \hat{f} * \mathcal{C}_p$，最后 $M_s = \mathcal{R}_o(\hat{M}_s)$，$\hat{f}_s = \hat{f} \times M_s$（公式 6）。
- 融合：$\bar{f} = \mathcal{R}_f(\hat{f}_c \oplus \hat{f}_s)$（公式 7）。

**损失函数**：$\mathcal{L} = \mathcal{L}_g(\hat{I}_c, I_c) + \lambda_1 \mathcal{L}_g(\bar{I}_c, I_c)$，其中 $\lambda_1 = 1$，$\mathcal{L}_g$ 可与目标网络保持一致（像素级重建损失或感知损失）（公式 8）。

---

## 实验与结果

**数据集**：
- 低光增强：LOL-real、SID
- 去雨：Rain13K（训练）/ Rain100H、Rain100L、Test100、Test1200、Test2800
- 去模糊：GoPro（训练）/ HIDE、RealBlur-R、RealBlur-J（运动模糊）；DPDD（训练）/ EDB、JNB（离焦模糊）
- 去噪：Set12、BSD68、CBSD68、Kodak、McMaster、Urban100（高斯）；SIDD（真实去噪）

**基线模型**：UHD、URetinex、SNR（低光）；SPAIR、Restormer（去雨）；MPRNet、Restormer（去模糊）；DRUNet、GRL（去噪）。

**主要结果（保留关键数值）**：
- **LOL-real + SNR 基线**：PSNR 从 21.48 → 25.50（+4.02），SSIM 从 0.849 → 0.892（+43/100），使用 CLIP 先验（Ours-c）效果最佳。
- **SID + SNR 基线**：PSNR 从 22.87 → 23.34（+0.47），SSIM 从 0.625 → 0.630（+5/100）。
- **URetinex on LOL-real**：PSNR 21.16 → 24.70（+3.54）。
- **去雨（Restormer 基线）**：Test100 PSNR 32.00→32.30，Rain100H 31.46→31.77，Rain100L 38.99→39.27。
- **运动去模糊（Restormer 基线）**：GoPro 32.92→33.18，HIDE 31.22→31.51，RealBlur-R 36.19→36.47，RealBlur-J 28.96→29.21。
- **高斯去噪（σ=25, Restormer 单模型）**：BSD68 29.51→29.78，Urban100 31.39→31.67。
- **真实去噪 SIDD**：Restormer+Ours-c 达到 PSNR 40.22 dB，SSIM 0.965。
- **无监督增强**：ZeroDCE、RUAS、SCI 在 LOL-real/SID 上均有显著提升（如 ZeroDCE PSNR 18.06→18.79 on LOL-real）。
- **参数增量**：PTG-RM 仅 <1M 参数，远低于 SKF（2.15M）和 SMG（16.76M），但提升更优。

**最强结果**：LOL-real 上 SNR+Ours-c 达到 25.50 PSNR / 0.892 SSIM，相比基线提升 4.02 dB / 43/100 SSIM。

---

## 相关工作脉络
1. **SKF（CVPR 2023）**：利用语义图作为先验优化低光增强特征空间——需配对的多模态标注，而本文无需任何额外标注。
2. **SMG（CVPR 2023）**：融合边缘、深度、语义信息增强低光图像——需大量物理变量监督，且参数量大（16.76M）；本文仅用 <1M 参数即超越。
3. **SNR（CVPR 2022）**：基于预计算的 SNR 值作为短/长程操作的融合掩码——固定参考缺乏自适应能力；本文用预训练先验在线学习空间变化映射。
4. **CLIP-IQA（AAAI 2023）**：证明 CLIP 特征含退化相关信息可用于图像质量评估——首次探索用于恢复任务的辅助特征；本文是首个将其用于图像恢复的方法。
5. **传统图像先验方法**（噪声水平估计、模糊核估计）——需显式估计，现实数据中难以获取；本文直接从预训练模型隐式提取恢复相关先验。
6. **DisCo（AAAI 2022）**：利用近红外信息进行低光成像——需特定传感器硬件；本文方法完全数据驱动，无需额外硬件。

---

## 局限性与未来方向
- **提升幅度因任务和模型而异**：部分场景增强显著，部分不明显，与目标网络容量及任务复杂度相关。
- **推理阶段仍依赖原始恢复网络**：预训练模型仅在训练时参与，未探索推理时联合推理的潜力。
- 未来方向：（1）设计针对性更强的蒸馏框架以提取更精细的恢复特征先验；（2）突破现有性能上限；（3）开发面向实际场景的技术产品。

---

## 研究启发与可借鉴点
1. **"预训练模型特征即先验"范式可迁移**：CLIP/Stable Diffusion 等多模态预训练模型蕴含丰富的退化信息，可成为通用图像恢复任务的"免费先验源"，值得在其他低层视觉任务（超分辨率、去雾）中探索。
2. **空间变化融合策略（替代固定参考）**：用可学习空间映射替代 SNR 等固定先验，实现自适应的短/长程操作融合，这一思路可扩展到其他需要动态选择操作范围的场景。
3. **潜空间蒸馏+残差细化机制**：通过在 latent space 进行蒸馏（而非直接对齐特征），规避了异构特征的形状差异；输出校正掩码 $I_m$ 与残差 $I_r$ 的分解设计精巧，可复用于其他 refinement 框架。
4. **无监督兼容性**：PTG-RM 可与无监督损失函数无缝结合（如 ZeroDCE、En-GAN），为无监督/自监督恢复任务提供新的增强路径。
5. **轻量插件设计哲学**：<1M 参数的通用 refinement 模块可"即插即用"于多种 SOTA 恢复网络，为工程落地提供了高性价比的方案。

---

## 关键术语表
**PTG-RM**：Pre-Train-Guided Refinement Module，本文提出的轻量级（<1M 参数）精炼模块，利用预训练模型特征增强恢复结果。
**OSF（Off-the-shelf Features）**：现成特征，指直接从预训练模型（如 CLIP、Stable Diffusion）中提取的特征，无需微调即可用于下游任务。
**PTG-SVE**：Pre-Train-Guided Spatial-Varying Enhancement，利用预训练先验生成空间变化的融合权重，自适应选择短程（CNN）与长程（Transformer）操作结果。
**PTG-CSA**：Pre-Train-Guided Channel-Spatial Attention，利用预训练先验生成通道注意力和空间变化的注意力图，增强恢复相关特征。
**SNR-aware（SNR）**：Xu 等人（CVPR 2022）提出的低光增强方法，基于预计算的信噪比值进行空间变化操作范围选择，是本文的主要对比基线之一。
**Restormer**：Zamir 等人（CVPR 2022）提出的高效 Transformer 架构，在多项恢复任务上达到 SOTA，本文多次将其作为基础网络验证 PTG-RM 的通用性。

---

## 可复现要素
- **数据集**：LOL-real、SID、Rain100H/L、Test100/1200/2800、GoPro、HIDE、RealBlur-R/J、DPDD、Set12、BSD68、CBSD68、Kodak、McMaster、Urban100、SIDD（均为公开数据集）
- **代码**：论文未明确声明 GitHub 链接，但作者在 Introduction 中提到代码将在后续发布（"More results... can be seen in experiments"）
- **预训练模型**：CLIP（openai/ViT-L/14）、Stable Diffusion、BLIP2——均为开源模型
- **关键超参**：$\lambda_1 = 1$（loss weight，跨任务和网络保持一致）
- **参数量**：PTG-RM < 1M 额外参数
- **训练策略**：与目标恢复网络 $\mathcal{F}$ 联合训练，使用相同 loss

---
