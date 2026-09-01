---
title: "Diffusion-Models-Without-Attention"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Diffusion_Models_Without_Attention_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:51:36"
field: "生成模型架构设计"
keywords: ["Diffusion Model", "State Space Model", "Attention-free", "Image Generation", "FLOP-efficient", "S4D"]
innovations: ["首次提出完全无注意力的扩散模型架构 DIFFUSSM，用门控双向SSM替代Self-Attention", "设计Hourglass SSM块，在SSM层保持全长捕获全局上下文、在MLP层降采样降低计算量", "在ImageNet 256x256上以20%更少Gflops超越DiT-XL/2（FID 9.07 vs 9.62）"]
benchmarks: ["ImageNet 256x256", "ImageNet 512x512", "LSUN-Church 256x256", "LSUN-Bedroom 256x256"]
---

# 论文速读：Diffusion Models Without Attention

## 一句话总结
本文提出 DIFFUSSM（Diffusion State Space Model），一种完全去除注意力机制的扩散模型架构，用门控状态空间模型（Gated SSM）骨干网络替代 Self-Attention，在保持高分辨率图像细节的前提下实现更低的 FLOPs 计算开销，在 ImageNet 256×256 和 512×512 上均达到与 DiT 相当或更优的效果。

## 研究问题与动机
- **高分辨率扩散模型的计算瓶颈**：现有 DDPM 在高分辨率图像生成上严重依赖 Self-Attention，其二次复杂度（O(L²)）使训练成本随分辨率急剧上升。
- **现有压缩方案损害表示能力**：主流做法（如 patchifying、多尺度分辨率）通过降低序列长度来缓解计算压力，但会损失高频空间信息和结构完整性。
- **缺乏真正无注意力的扩散架构**：U-Net 和 Transformer 类扩散模型均以注意力为核心组件；目前尚无完整替代方案能达到同等质量。
- **线性时间复杂度的可行路径**：SSM 基于 FFT 实现仅需 O(L log L) 复杂度，为长序列建模提供了高效且可扩展的选择。

## 核心贡献（创新点）
1. **首个无注意力扩散架构**：首次将完整的门控双向 SSM 骨干直接应用于扩散去噪网络，完全移除 Self-Attention。与 DiT/U-ViT 等注意力架构的本质区别在于不依赖任何序列压缩手段即可处理高分辨率输入。
2. **沙漏式（Hourglass）SSM 块设计**：在 MLP 部分通过降采样/上采样缩短序列长度以降低计算量，而在 SSM 核心层保持原始全长，兼顾全局上下文建模与局部密集计算效率。与 Attention + Hourglass 的本质区别在于：注意力在全长下即使使用沙漏仍会产生 O(L²) 额外开销，而 SSM 的 FFT 开销仅 O(L log L)。
3. **无需 Patchification 即优于或匹敌有 Attention 基线**：在 ImageNet 256×256 上以约 1/3 的训练步数超过 DiT-XL/2（FID 9.07 vs 9.62）；在 512×512 上以 40% 更少的训练图像和 25% 更少的 Gflops 取得竞争性结果。
4. **揭示 SSM 在扩散生成任务中的潜力**：证明了注意力并非高分辨率图像生成的必要条件，为低资源/高效率扩散模型开辟了新方向。

## 方法详解
- **骨干网络：S4D（Diagonal State Space Model）**：采用 Gu 等人提出的简化对角化 SSM，将连续时间 SSM 参数（A、B、C、Δ）离散化为线性 RNN，并通过 FFT 实现 O(L log L) 的长卷积运算。
- **Gated Bidirectional SSM Block**：双向 SSM 输出与门控机制结合（借鉴 Pretraining Without Attention 的工作），以乘法门控增强序列建模能力。
- **Hourglass 设计**：每层接收降采样后的输入序列（降采样比 M=2），SSM 层在全长 L 上计算以捕获全局信息，MLP 部分在降采样后 J=L/M 的长度上操作，最后通过上采样恢复到原长度，公式如下：
  - 上采样分支：$\mathbf{U}_l = \sigma(\mathbf{W}_k^{\uparrow} \sigma(\mathbf{W}^0 \mathbf{I}_j))$
  - SSM 核心：$\mathbf{Y} = \text{Bidirectional-SSM}(\mathbf{U})$，在原始全长 L 上计算
  - 下采样分支：$\mathbf{I}'_{j, \cdot} = \sigma(\mathbf{W}_k^{\downarrow} \mathbf{Y}_l)$
  - 门控输出：$\mathbf{O}_j = \mathbf{W}^3(\sigma(\mathbf{W}^2 \mathbf{I}'_j) \odot \sigma(\mathbf{W}^1 \mathbf{I}_j))$
- **条件注入**：每个位置同时融合类别标签 $\mathbf{y}$ 和时间步 $\mathbf{t}$ 的嵌入（沿用 DiT 范式）。
- **总参数量**：DIFFUSSM-XL 约 673M 参数，29 层 Bidirectional Gated SSM，D=1152，与 DiT-XL 参数量相当。
- **损失函数**：噪声预测采用简化 MSE $\min_\theta \|\varepsilon_\theta(x_t) - \varepsilon_t\|_2^2$；协方差预测采用完整变分下界目标（遵循 DiT 设定）。
- **FLOPs 分析**：单层 FLOPs 约 $7.5 \cdot L \cdot D^2$（M=2 时），远低于同等长度下 Self-Attention 的 $2D \cdot L^2$。

## 实验与结果
- **数据集**：ImageNet-1k（256×256 和 512×512）、LSUN-Church（126k 图）和 LSUN-Bedroom（3M 图），均使用 VAE 编码后的潜空间（32×32 和 64×64）。
- **评估指标**：FID-50K（250 步 DDPM 采样）、sFID、Inception Score、Precision/Recall。
- **ImageNet 256×256（无 Classifier-Free Guidance）**：
  - DIFFUSSM-XL：FID=9.07，sFID=5.52，IS=118.32，训练图像 660M，Total Gflops=$1.85\times10^{11}$
  - 对比 DiT-XL/2：FID=9.62，sFID=6.85，训练图像 1792M，Total Gflops=$2.13\times10^{11}$
  - **提升**：FID 降低 5.5%（9.62→9.07），Total Gflops 减少约 20%，训练图像减少约 63%
- **ImageNet 256×256（含 CFG）**：
  - DIFFUSSM-XL-G：FID=2.28，sFID=4.49，IS=259.13
  - DiT-XL/2-G：FID=2.29，sFID=4.60
  - **最强结果**：sFID 达到所有 DDPM 方法最优（4.49 vs 4.60），FID 以极小差距（0.01）匹敌 DiT
- **ImageNet 512×512（含 CFG）**：
  - DIFFUSSM-XL-G：FID=3.41，sFID=5.84，训练图像 302M，Total Gflops=$3.22\times10^{11}$
  - DiT-XL/2-G：FID=3.04，sFID=5.02，训练图像 768M，Total Gflops=$4.03\times10^{11}$
  - **提升幅度**：以 40% 更少图像和 25% 更少 Gflops 获得可比较结果
- **LSUN 无条件生成（256×256）**：
  - LSUN-Church：FID=4.02，与 LDM 相当（gap=0.08）
  - LSUN-Bedroom：FID=1.90，与 LDM 相当（gap=0.07）

## 相关工作脉络
1. **DiT (Peebles & Xie, 2022)**：将 Vision Transformer 引入扩散模型，依赖 patchification + 全注意力。DIFFUSSM 的核心差异：完全移除注意力，无需 patchification 即可高效处理全长序列。
2. **U-ViT (Bao et al., 2023)**：U-Net + ViT 混合架构，使用 long-skip 连接。DIFFUSSM 与之对比在于不使用长跳跃连接，且架构完全不同（SSM vs Transformer）。
3. **LDM (Rombach et al., 2022)**：潜空间扩散模型，使用 U-Net + 注意力。DIFFUSSM 在同一潜空间框架下替换了 U-Net 的注意力模块，证明注意力非必需。
4. **S4 / S4D (Gu et al., 2021/2022)**：状态空间模型的序列建模基座。本文将其首次应用于扩散生成任务，拓展了 SSM 的应用领域。
5. **Pretraining Without Attention (Wang et al., 2022)**：门控双向 SSM 的设计灵感来源，DIFFUSSM 沿用了 Gated Bidirectional SSM 层并针对扩散任务做了适配。
6. **ADM (Dhariwal & Nichol, 2021)**：早期高分辨率 U-Net 扩散基线。DIFFUSSM 在相同或更低计算预算下超越 ADM 及其变体。

## 局限性与未来方向
- **仅聚焦图像生成**：目前仅限于（无）条件图像生成，尚未扩展到文本到图像（text-to-image）等更复杂任务。
- **未结合 Masked Token Prediction**：论文承认 MaskGIT、MDiT 等结合掩码预训练的方法可能进一步提升性能，但认为与本文主要对比正交。
- **无条件生成在 LSUN-Bedroom 上不及 GAN**：作者指出当前仅使用 25% 的训练预算与 ADM 对比，可能存在不公平。
- **未探索 FlashConv 等加速技术**：提及未来可结合 FlashAttention 类技术（如 FlashConv）进一步降低运行时开销。
- **架构扩展性有待验证**：在更长序列（如更高频视频/3D 数据）上的表现尚需进一步研究。

## 研究启发与可借鉴点
1. **Hourglass + SSM 的组合设计可迁移**：该设计在保持全局感受野的同时降低局部 MLP 的计算量，可借鉴到音频生成、视频生成、3D 生成等需要处理超长序列的扩散任务中。
2. **完全移除注意力是可行方向**：本文打破了"高分辨率生成必须依赖注意力"的范式认知，为后续探索线性复杂度生成模型提供了实证支持。
3. **消融实验设计值得学习**：作者对比了单向 vs 双向 SSM、UViSSM（U-Net 风格+SSM）与完整 DIFFUSSM 的 loss 曲线，清晰展示了各组件的有效性。
4. **与本团队方向的结合机会**：可将 SSM 骨干与掩码重建目标（Masked Diffusion）结合，探索更高效率的高分辨率生成；也可将 hourglass SSM 块迁移至视频/音频扩散模型中。
5. **训练效率分析方法**：作者以 Total Gflops 而非单纯参数量或训练步数作为公平比较标准，这一评估视角值得在后续工作中沿用。

## 关键术语表
- **DIFFUSSM（Diffusion State Space Model）**：本文提出的无注意力扩散模型架构，以门控双向 SSM 替代 Self-Attention 作为核心骨干。
- **SSM（State Space Model，状态空间模型）**：一类将输入序列映射到输出的线性动力系统架构，可通过 FFT 实现 O(L log L) 复杂度，替代 Transformer 中的注意力机制。
- **S4D（Diagonal SSM）**：Gu 等人提出的简化对角化 SSM 变体，在保持 S4 性能的同时大幅降低计算和存储开销，本文选用其作为骨干。
- **Gated Bidirectional SSM**：双向 SSM 输出与门控机制相乘的复合层，兼具全局上下文建模和局部特征选择能力，是 DIFFUSSM 的核心组件。
- **Hourglass Architecture（沙漏架构）**：在中间层（SSM）保持全序列长度、在两侧（MLP）降采样/上采样的结构，兼顾全局感受野与计算效率。
- **Patchification（分块化）**：将高分辨率图像划分为 P×P 的 patch 以降低序列长度的技术，是 DiT 等 Transformer 扩散模型的标准做法。
- **Classifier-Free Guidance（无分类器引导）**：通过同时训练有条件与无条件扩散模型，在推理时利用两者差异放大条件信号的技术。
- **FID / sFID**：Frechet Inception Distance 衡量生成图像与真实图像的分布距离；sFID 是其空间失真鲁棒版本。

## 可复现要素
- **数据集**：ImageNet-1k（公开）、LSUN-Church 和 LSUN-Bedroom（公开），均使用标准 VAE 潜空间编码
- **代码开源**：论文声明所有模型将在发布时以 Apache 2.0 许可证开源
- **关键超参**：
  - 模型大小 D=1152，层数 29，总参数约 673M
  - 降采样比 M=2（hourglass）
  - 训练设备：8× NVIDIA A100 80GB
  - Global batch size=256
  - 混合精度训练
  - VAE 编码器参数冻结
  - 使用 EMA（指数移动平均）权重
- **训练设置**：遵循 DiT 和 ADM 的相同训练配方（线性方差调度、时间/类别嵌入、协方差参数化）
- **评估**：FID-50K（250 步 DDPM 采样）、sFID、IS、Precision/Recall
