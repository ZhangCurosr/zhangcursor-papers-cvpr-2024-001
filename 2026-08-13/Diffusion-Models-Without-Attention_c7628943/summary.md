---
title: "Diffusion-Models-Without-Attention"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Yan_Diffusion_Models_Without_Attention_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:50:27"
field: "生成式扩散模型架构"
keywords: ["diffusion models", "state space models", "attention-free architectures", "image generation", "efficient generative modeling"]
innovations: ["提出完全去除注意力的DIFFUSSM架构，用门控双向SSM替代自注意力", "设计沙漏式前馈网络在SSM旁压缩序列以降低FLOPs而不损失全局上下文", "在ImageNet 256x256上以20%更低训练FLOPs取得优于DiT的FID和sFID"]
benchmarks: ["ImageNet 256x256", "ImageNet 512x512", "LSUN-Church 256x256", "LSUN-Bedroom 256x256"]
---

# 论文速读：Diffusion-Models-Without-Attention

## 一句话总结
本文提出了 Diffusion State Space Model (DIFFUSSM)，一种完全去除自注意力机制的扩散模型架构，采用门控双向状态空间模型（SSM）骨干网络配合沙漏式前馈网络，在不进行图像 patchifying 或多尺度压缩的前提下，实现了与 DiT 相当或更优的高分辨率图像生成质量，同时显著降低了训练 FLOPs。

## 研究问题与动机
- 高分辨率图像生成中，自注意力机制的计算复杂度随序列长度呈二次方增长，成为核心瓶颈。
- 现有解决方案（如 U-Net 中的 patchifying 或 DiT 中的 patch embedding）通过降低表征粒度来减少计算量，但以牺牲高频空间细节和结构完整性为代价。
- 多尺度分辨率方案虽能缓解注意力层的计算压力，但下采样会损失空间细节，上采样可能引入伪影。
- 状态空间模型（SSM）已被证明是高效、可扩展的长序列建模替代方案，但尚未在扩散模型架构中得到充分探索。

## 核心贡献（创新点）
1. **提出首个完全无注意力的扩散架构 DIFFUSSM**：用门控双向 SSM 骨干网络替换 U-Net/Transformer 中的自注意力层，避免了对全局压缩（patchifying）的依赖。
2. **沙漏式前馈网络设计**：在 SSM 核心周围引入 hourglass 结构的 MLP 层，在保留全局上下文感知的同时降低序列维度以减少计算量，该设计对 SSM 有效但对注意力模型效果不佳。
3. **显著降低训练计算成本**：在 ImageNet 256×256 类别条件生成任务上，DIFFUSSM-XL 的训练总 FLOPs 比 DiT-XL/2 减少约 20%，且无需分类器引导即可取得更优 FID（9.07 vs 9.62）和 sFID（5.52 vs 6.85）。
4. **在更高分辨率下保持竞争力**：在 512×512 分辨率上，DIFFUSSM-XL-G 仅使用 302M 训练图像（DiT 的 40%）和 25% 的 FLOPs，sFID 达到 5.84，接近 DiT-XL/2-G 的 5.02。
5. **系统性的消融与定性分析**：验证了双向 SSM、无 patchifying 设计以及沙漏架构的有效性，并展示了消除压缩带来的空间重建鲁棒性提升。

## 方法详解
- **整体架构**：DIFFUSSM 接收经过 VAE 编码的噪声潜变量，将其展平为序列后输入重复堆叠的 Gated SSM 块，每个块包含双向 SSM 核心和沙漏式 MLP，最后通过线性解码器恢复空间维度以输出噪声预测和协方差预测。
- **状态空间模型核心**：采用 S4D（对角化 SSM）作为基础，学习连续时间参数 $\overline{A}, \overline{B}, \overline{C}$ 和离散化率 $\Delta$，通过 FFT 实现 $O(L \log L)$ 复杂度的长卷积计算；采用双向拼接加门控机制增强建模能力。
- **沙漏式前馈网络**：设下采样比率 $M$，输入序列长度 $L$ 被压缩至 $J = L/M$ 后进入 MLP 上分支；SSM 在原始长度 $L$ 上计算以保留全局上下文；输出经下采样 MLP 压缩后，与原始输入通过残差连接和逐元素乘法融合。
- **关键公式**：每层计算包含四个步骤：上采样分支 $\mathbf{U}_l = \sigma(\mathbf{W}_k^\uparrow \sigma(\mathbf{W}^0 \mathbf{I}_j))$，双向 SSM $\mathbf{Y} = \text{Bidirectional-SSM}(\mathbf{U})$，下采样分支 $\mathbf{I}'_{j,...} = \sigma(\mathbf{W}_k^\downarrow \mathbf{Y}_l)$，以及门控融合 $\mathbf{O}_j = \mathbf{W}^3(\sigma(\mathbf{W}^2 \mathbf{I}'_j) \odot \sigma(\mathbf{W}^1 \mathbf{I}_j))$。
- **FLOPs 分析**：单层总 FLOPs 约为 $13\frac{L}{M}D^2 + LD^2 + \alpha \cdot 2L\log L \cdot D$，当 $M=2$ 时约 $7.5LD^2$；相比同等参数量下使用自注意力（$2DL^2$），在长序列场景下优势显著。
- **条件注入**：在每个位置融合类别标签 $\mathbf{y}$ 和时间步 $\mathbf{t}$ 的嵌入，遵循 DiT 的设计范式。

## 实验与结果
- **数据集**：ImageNet-1k（256×256 和 512×512 分辨率，使用 VAE 潜空间 $32\times32$ 和 $64\times64$），LSUN Church（126k 图像）和 Bed（3M 图像，256×256 分辨率）。
- **评估基线**：BigGAN-deep, MaskGIT, StyleGAN-XL, ADM, LDM, DiT-XL/2, U-ViT-H/2 等类别条件生成模型；ImageBART, PGGAN, StyleGAN 系列, DDPM 等无条件生成模型。
- **主要结果（ImageNet 256×256 类别条件）**：
  - 无分类器引导：DIFFUSSM-XL 取得 FID=9.07（优于 DiT-XL/2 的 9.62）、sFID=5.52（优于 DiT 的 6.85）、IS=118.32，训练图像 660M（DiT 为 1792M），总 FLOPs 减少约 20%。
  - 带分类器引导（-G）：DIFFUSSM-XL-G 取得 FID=2.28、sFID=4.49、IS=259.13，优于 DiT-XL/2-G 的 FID=2.29 和 sFID=4.60。
- **主要结果（ImageNet 512×512）**：DIFFUSSM-XL-G 使用 302M 图像和 $3.22\times10^{11}$ Gflops，取得 FID=3.41、sFID=5.84，优于 DiT-XL/2-G 的 FID=3.04 和 sFID=5.02，但训练量仅为 DiT 的 40% 图像和 25% FLOPs。
- **LSUN 无条件生成**：在 Church 数据集上 FID=4.02（接近 LDM），在 Bed 数据集上 FID=1.90（略低于 ADM 的 0.64，因训练预算仅为 25%）。
- **最强结果**：带分类器引导的 DIFFUSSM-XL 在 ImageNet 256×256 上取得 sFID=4.49，为所有 DDPM 基线中最佳；FID=2.28 略优于 DiT-XL/2-G 的 2.29。

## 相关工作脉络
- **Diffusion Models（DDPM 系列）**：本文建立在 DDPM 框架之上，采用 classifier-free guidance 范式，但彻底替换了 U-Net 和 Transformer 中的注意力组件。
- **U-Net with Self-attention**：传统高分辨率扩散模型依赖多尺度 U-Net 并在低分辨率特征图上叠加注意力层，本文方法避免了这种分辨率降级策略。
- **Diffusion Transformers（DiT）**：DiT 将图像 patchify 后输入纯 Transformer，本文指出 patchifying 会损失空间细节，DIFFUSSM 直接在完整序列长度上操作。
- **State Space Models（S4/S4D）**：S4 及其简化版本 S4D 已被证明在语言建模和音频生成上有效，本文首次将其系统性地应用于图像扩散生成。
- **Efficient Long-Range Architectures**：包括 Linear Attention、Mega、Linformer 等近似注意力方法，本文选择完全不同的 SSM 路线，避免任何 $O(L^2)$ 计算。
- **U-ViT**：结合 U-Net 和 ViT 的 hybrid 架构，使用 long-skip connections，本文未采用该设计以保持与 DiT 的公平对比。

## 局限性与未来方向
- 当前工作仅聚焦于（条件/无条件）图像生成，尚未扩展到完整的 text-to-image 生成任务。
- 未结合 masked token reconstruction 等训练增强技术（如 MaskGIT 或 Masked Diffusion Transformer），作者认为这些方法可与本文正交结合。
- 对于高分辨率生成，SSM 的 FFT 实现虽已高效，但相比 FlashAttention 等专门优化，仍有进一步加速空间（作者提及 FlashConv 可能带来改进）。
- 未探索 U-ViT 式的 long-skip connections 与 DIFFUSSM 的结合，作者认为这可能是未来的改进方向。
- 在 LSUN-Bedroom 任务上，由于训练预算限制（仅使用 25% 的 ADM 预算），未能超越 GAN 类方法的最佳结果。

## 研究启发与可借鉴点
- **去注意力化的架构探索**：证明在扩散模型中，SSM 可以完全替代自注意力并达到 comparable 性能，为后续研究提供了"attention-free"范式的有力实证。
- **沙漏式维度调度设计**：在 SSM 核心保持全序列长度以保留全局上下文，仅在 MLP 部分压缩序列以降低计算量的思路，可迁移至其他序列建模任务。
- **FLOPs 作为核心比较指标**：将训练计算成本（总 Gflops）与生成质量并置评估，为高效率生成模型的研究树立了更全面的基准比较标准。
- **无压缩的高分辨率处理**：避免 patchifying 和多尺度下采样，直接在全分辨率潜空间上建模，为追求高频细节保留的生成任务提供了新思路。
- **与 masked training 的正交性**：本文方法可与 MaskGIT、Masked Diffusion Transformer 等技术结合，提示了架构改进与训练策略改进可协同增益。

## 关键术语表
- **Diffusion State Space Model (DIFFUSSM)**：本文提出的完全去除自注意力的扩散模型架构，以门控双向 SSM 为核心骨干。
- **State Space Model (SSM)**：一类基于连续时间状态空间离散化的序列建模架构，可通过长卷积高效实现 $O(L\log L)$ 复杂度。
- **S4D**：S4 的对角化简化版本，通过近似连续时间参数化实现稳定且高效的序列建模。
- **Classifier-free Guidance**：扩散模型中通过无条件预测联合有条件预测来提升生成质量的引导技术，无需额外分类器。
- **Patchification**：将图像分割为固定大小 patch 并展平为序列的处理方式，可降低 Transformer 的计算复杂度但损失空间细节。
- **Hourglass Architecture**：先压缩后扩展序列维度的网络结构，在本文用于在 SSM 旁降低 MLP 的计算负担。
- **FID / sFID**：Fréchet Inception Distance 衡量生成图像与真实图像的分布距离；sFID 是其改进版，对空间失真更鲁棒。
- **Total Gflops**：训练过程中的总浮点运算量，本文用作衡量模型计算效率的核心指标。

## 可复现要素
- **数据集**：ImageNet-1k（公开），LSUN Church/Bed（公开）；VAE 编码器使用开源预训练权重（论文引用 [4]）。
- **代码与权重**：论文声明所有模型将在发布时以 Apache 2.0 许可证开源。
- **训练配置**：8× NVIDIA A100 80GB GPU，global batch size=256，混合精度训练，EMA 权重衰减，遵循 DiT 和 ADM 的训练配方。
- **模型规模**：DIFFUSSM-XL 约 673M 参数，29 层 Bidirectional Gated SSM 块，模型维度 $D=1152$，下采样比率 $M=2$。
- **超参数**：扩散步数、方差调度、时间/类别嵌入方式遵循 ADM；线性解码器+重排输出遵循 DiT；ViT 风格层初始化。
- **评估设置**：FID-50K（250 步 DDPM 采样），sFID，IS，Precision/Recall；明确标注是否使用 classifier-free guidance（-G 后缀）。
