---
title: "Diffusion-Models-Without-Attention"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Diffusion_Models_Without_Attention_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:23:32"
field: "生成模型架构"
keywords: ["扩散模型", "状态空间模型", "注意力替代", "高效生成", "ImageNet", "DiT"]
innovations: ["提出完全去除注意力的扩散架构 DIFFUSSM，用门控双向 SSM 替代自注意力，避免 patchification 信息压缩", "设计 SSM 全长度全局建模 + 沙漏式 MLP 降采样的高效块结构，显著降低 FLOPs", "在 ImageNet 256x256 以 20% 更少 Gflops 超越 DiT-XL/2 FID，512x512 以 25% 更少 Gflops 达竞争性 sFID"]
benchmarks: ["ImageNet 256x256", "ImageNet 512x512", "LSUN Church", "LSUN Bedroom"]
---

# 论文速读：Diffusion-Models-Without-Attention

## 一句话总结
本文提出 DiffuSSM（Diffusion State Space Model），一种完全去除注意力机制的扩散模型架构，用门控双向状态空间模型（SSM）替代自注意力，在无 patchification 压缩的前提下实现了与 DiT 相当甚至更优的图像生成质量，同时显著降低训练 FLOPs。

## 研究问题与动机
- **高保真图像生成的计算瓶颈**：现有扩散模型（DDPMs）在高生成分辨率时严重依赖自注意力或 Transformer 架构，而自注意力的复杂度随序列长度呈二次增长，导致计算开销难以承受。
- **Patchification/压缩引入的信息损失**：DiT 等方案通过将图像划分为 patch 来降低序列长度，但这会造成高频空间信息和结构完整性的退化；多尺度方法则在上下采样过程中引入伪影。
- **无补丁处理下长序列建模的效率缺口**：如何在保持细粒度图像表示（不压缩序列）的同时实现高效的长程依赖建模，是当前未充分解决的问题。
- **注意力替代方案的扩散应用验证不足**：SSM 在语言建模等领域已展示高效长序列建模能力，但在扩散图像生成中的潜力尚未被系统探索。

## 核心贡献（创新点）
- **提出首个完全去注意力的扩散架构 DIFFUSSM**：以门控双向 SSM 替换注意力核心，在隐空间直接处理完整序列，避免 patchification 带来的信息压缩。
- **设计"双端 SSM + 沙漏式 MLP"的块结构**：SSM 层在全序列长度上运行以捕捉全局上下文，而 MLP 部分通过下采样/上采样（hourglass 结构）将序列长度缩减至 L/M，在计算效率与表征能力之间取得平衡。
- **系统性证明 FLOP 效率优势**：在 ImageNet 256×256 上，DIFFUSSM-XL 以约 20% 更少的训练总 Gflops 超过 DiT-XL/2 的 FID（9.07 vs 9.62，无 CFG），在 512×512 上以少 40% 的训练图像和 25% 的 Gflops 达到竞争性 sFID 分数（5.84 vs 5.02）。
- **消融与定性分析验证设计选择**：证明了双向 SSM 显著优于单向变体；消融了 patch 大小 P 和缩放比例 M，发现去除 patching 在同等训练阶段能提升空间重构鲁棒性。

## 方法详解
- **整体流程**：将输入的噪声图像（或 VAE 编码后的隐变量）展平为序列 $\mathbf{I} \in \mathbb{R}^{L \times D}$，依次通过多层相同的 DIFFUSSM 块，最后在输出端使用线性解码器恢复空间维度以预测噪声 $\varepsilon_\theta$ 和对角协方差 $\Sigma_\theta$。
- **SSM 核心**：采用 S4D（对角化状态空间模型），通过 FFT 实现长卷积，将复杂度从 $O(L^2)$ 降至 $O(L \log L)$。SSM 参数由连续时间状态空间离散化得到（$\bar{A}, \bar{B}, \bar{C}$ 由 $\Delta$ 学习）。
- **门控双向 SSM 块**：两个 SSM 层分别从前向后和从后向前处理序列，拼接后经过 MLP，并引入乘法门控（gated）以提升表达能力。
- **沙漏式前馈网络（Hourglass MLP）**：设缩放因子 $M = L/J$，MLP 内部先将序列从长度 $L$ 压缩至 $J$（降采样），在短序列上做非线性变换后再上采样回长度 $L$，从而大幅减少 MLP 的计算量。单块总 FLOPs 约为 $7.5 L D^2$（$M=2$ 时），远低于全长度自注意力的 $2DL^2$。
- **参数规模**：块中线性变换参数量约为 $13D^2$（$M=2$），与 DiT 核心的 $12D^2$ 相当；作者使用额外一层来匹配参数量。
- **条件注入**：每个位置融合类别标签 $\mathbf{y}$ 和时间步 $\mathbf{t}$ 的嵌入，遵循 DiT/DiffEq 的标准做法；无条件生成时移除类别嵌入。
- **训练策略**：噪声预测头采用简化的 MSE 目标，协方差预测头采用完整目标；使用 EMA 权重平均、混合精度训练；VAE 编码器参数冻结。

## 实验与结果
- **数据集与设置**：ImageNet-1k（256×256 和 512×512，经 VAE 编码后有效尺寸分别为 32×32 和 64×64，对应序列长度 $L=1024$ 和 $L=4096$）；LSUN Church（126k 图）和 Bedroom（3M 图），训练无条件模型；8 张 NVIDIA A100 80GB，全局 batch size 256。
- **ImageNet 256×256 条件生成（主表）**：
  - **无 CFG**：DIFFUSSM-XL（660M 图像，$1.85 \times 10^{11}$ Gflops）FID=**9.07**、sFID=**5.52**，超越 DiT-XL/2（FID=9.62，$2.13 \times 10^{11}$ Gflops），Gflops 减少约 20%；训练步数约为非 CFG 最佳 Latent Diffusion 的 1/3。
  - **有 CFG（-G）**：DIFFUSSM-XL-G（$1.85 \times 10^{11}$ Gflops）FID=**2.28**、sFID=**4.49**，IS=259.13，sFID 超越所有 DDPM 基线（DiT-XL/2-G 未报告 sFID）。
- **ImageNet 512×512 条件生成**：DIFFUSSM-XL-G 使用 302M 图像、$3.22 \times 10^{11}$ Gflops，sFID=**5.84**，显著优于 DiT-XL/2-G 的 $4.03 \times 10^{11}$ Gflops（sFID=5.02），展示了以更少资源获得竞争力的能力。
- **LSUN 无条件生成（256×256）**：在 Church 类别上与 LDM 差距仅 -0.08 FID，在 Bedroom 上差距 +0.07 FID，但训练预算仅为 ADM 的 25%；GAN 类方法仍领先。
- **定性分析**：去除 patching（P=1）在相同训练步数下显著改善空间结构重建质量，且与高 M 值组合可获得更低计算量的近似效果。

## 相关工作脉络
- **DiT（Peebles & Xie, 2022）**：将 Transformer 用于扩散模型的标志性工作，依赖 patchification 处理高分辨率；本文在相同"去 patch 处理长序列"目标下，用 SSM 替代注意力，避免了信息压缩。
- **U-ViT（Bao et al., 2023）**：结合 UNet 与 ViT 骨干，使用长跳跃连接；其 256×256 表现较好但在 512×512 不及本方法，本文未采用长跳跃以保持公平对比。
- **LDM（Rombach et al., 2022）**：在潜空间训练扩散模型，依赖 U-Net+Attention；本文在潜空间同样工作但完全去除 attention，且不使用 patchification。
- **S4/S4D（Gu et al., 2021/2022）**：状态空间模型的骨干，在 LRA 和音频基准上表现优异；本文将其引入扩散图像生成领域，填补了 SSM 在高分辨率图像扩散中的空白。
- **Pretraining Without Attention（Wang et al., 2022）**：早期探索非注意力架构的 NLP 工作，为本文选择 SSM 替代 attention 提供了预训练层面的合理性支撑。
- **MaskGIT / Masked Diffusion Transformers（Chang et al., 2022; Gao et al., 2023）**：结合掩码 token 预测增强扩散训练，作者指出该方向与本工作正交，未来可结合。

## 局限性与未来方向
- **仅针对条件/无条件图像生成**，未扩展至文本到图像（text-to-image）等更具挑战性的多模态任务。
- **未结合掩码预训练（masked training）等增强技术**，作者承认这些方法可能进一步提升性能。
- **S4D 的离散化近似可能引入精度损失**，结构化 SSM（如 HiPPO）的理论保证更强但计算代价更高，未在文中探索。
- **运行时优化空间较大**：FlashAttention 和 FlashConv 等硬件感知优化尚未集成，实际训练速度可能还有提升潜力。
- **LSUN 无条件生成上 GAN 仍占优**，说明在极低 FLOP 预算下 GAN 范式在部分任务上仍有优势，扩散模型需进一步追赶。

## 研究启发与可借鉴点
- **SSM 作为 attention 替代品的验证路径**：本文证明了在扩散模型的潜空间序列建模中，SSM 能在不压缩分辨率的情况下达到 attention 级别的性能，为"attention-free 生成模型"提供了坚实的实证基础，值得在其他生成任务（视频、音频、3D）中复现验证。
- **沙漏式 MLP 的尺度交替设计**：在 SSM 层保持全序列长度、仅在 MLP 层做下采样/上采样的策略，是兼顾全局感知与局部计算效率的有效设计模式，可迁移至其他需要处理长序列的生成模型。
- **去 patchification 的质量-效率权衡分析**：本文的消融实验（不同 P 和 M 组合）清晰地展示了压缩与质量的 trade-off 曲线，这种分析框架可用于指导未来架构设计决策。
- **与 U-ViT 长跳跃连接的潜在结合**：作者明确提到 adapt U-ViT 的 long-skip connection 可能同时惠及 DiT 和 DIFFUSSM，这是一个低成本的后续改进方向。
- **与 masked diffusion 的正交可组合性**：MaskGIT 式掩码训练与 DIFFUSSM 的 attention-free 架构天然正交，结合两者可能产生更强的合成效果。

## 关键术语表
- **DIFFUSSM（Diffusion State Space Model）**：本文提出的完全去除注意力的扩散模型架构，以门控双向 SSM 为核心骨干处理图像序列。
- **State Space Model（SSM）**：一类基于线性时不变系统的序列建模架构，可通过 FFT 加速为长卷积，复杂度为 $O(L \log L)$，代表实现包括 S4/S4D。
- **S4D（Diagonal SSM）**：SSM 的对角化简化版本，将状态矩阵约束为对角形式，保持与结构化 SSM 相当的精度但大幅降低计算开销。
- **Classifier-free Guidance（CFG）**：一种无需分类器的扩散模型条件引导技术，通过训练中随机丢弃条件信息，在推理时放大条件信号以提升生成质量。
- **Hourglass Architecture（沙漏架构）**：在模块内部先对序列做下采样再上采样恢复的设计模式，用于在局部计算（MLP）中降低序列长度以节省 FLOPs。
- **FID / sFID**：FID（Fréchet Inception Distance）衡量生成图像与真实图像的分布距离；sFID 是其空间畸变鲁棒版本，对空间错位不敏感。
- **Patchification**：将高分辨率图像划分为固定大小的 patch 以降低序列长度和注意力计算量的常用策略，本文的核心对比对象。
- **Gated Bidirectional SSM**：两个方向（前向+后向）的 SSM 输出拼接后经门控机制融合，是 DIFFUSSM 的核心注意力替代模块。

## 可复现要素
- **数据集**：ImageNet-1k（公开）、LSUN Church 和 Bedroom（公开）；论文已声明使用标准 VAE 编码器（从 LDM 获取，参数冻结）。
- **代码开源**：论文声明"All models from this work will be available open-source (Apache 2.0 license) upon release"，截至论文发表时承诺开源，具体仓库需查阅发布状态。
- **关键超参**：模型尺寸 $D=1152$，层数 29，总参数量约 673M；缩放因子 $M=2$；global batch size=256；8×NVIDIA A100 80GB；混合精度训练；EMA 权重平均；扩散步数与方差调度遵循 ADM 设置。
- **训练步数**：ImageNet 256×256 条件下约 660M 图像等效训练量（步骤×batch），512×512 约 302M。
- **评估协议**：FID-50K（250 步 DDPM 采样），sFID，IS，Precision/Recall；classifier-free guidance 默认未启用除非标注 "-G"。
