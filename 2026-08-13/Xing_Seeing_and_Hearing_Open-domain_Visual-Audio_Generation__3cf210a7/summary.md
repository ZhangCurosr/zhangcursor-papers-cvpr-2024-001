---
title: "Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:50:21"
field: "多模态生成"
keywords: ["视听联合生成", "扩散模型", "多模态对齐", "ImageBind", "训练免费生成", "跨模态引导"]
innovations: ["提出基于ImageBind的扩散潜空间对齐器，无需训练即可桥接单模态扩散模型实现跨模态生成", "设计双损失/三角损失函数结合引导提示词调优，解决音频语义稀疏和时序一致性问题"]
benchmarks: ["VGGSound", "Landscape", "MKL", "FAD", "FVD", "KVD", "AV-align"]
---

# 论文速读：Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners

## 一句话总结
本文提出了一种**基于优化的训练免费范式**，利用预训练的多模态模型 ImageBind 作为潜空间对齐器（latent aligner），桥接已有的单模态扩散生成模型，实现了开放域的视频-音频联合生成、视频→音频、音频→视频、图像→音频四种任务，无需从头训练大模型或大规模成对数据集。

## 研究问题与动机
- **现有方法割裂生成**：当前扩散模型仅在单一模态（视频或音频）上独立生成，无法实现同步的视听内容创作，限制了影视制作等实际应用。
- **两阶段串行方案缺陷**：先用 T2V 生成视频再用 V2A 生成音频的方式，因 V2A/A2V 现有方法能力有限或泛化性差，导致最终视听不同步。
- **Joint-VA 任务被忽视**：联合音视频生成工作极少，现有 MM-Diffusion 等方法只能在小域上做无条件生成，缺乏语义可控性。
- **资源约束**：从头训练多模态生成模型成本高，需要一种轻量级、可复用的跨模态桥接方案。

## 核心贡献（创新点）
1. **首次提出开放域文本引导的联合视频-音频生成范式**，将预训练单模态扩散模型桥接为有机系统；与 MM-Diffusion 等基线的本质区别在于支持开放域文本条件控制，而非局限于训练域。
2. **引入扩散潜空间对齐器（Diffusion Latent Aligner）**，在 ImageBind 共享语义空间中通过梯度回传逐步引导噪声潜变量向目标条件靠近；与 classifier guidance 的本质区别在于无需训练分类器，利用预训练多模态嵌入空间直接提供生成指导。
3. **设计双重/三角损失函数与引导提示词调优（Guided Prompt Tuning）**，解决音频语义信息稀疏导致的对齐失效问题；与简单多模态对齐的本质区别在于引入文本作为第三种监督信号构建三角损失，并优化提示词嵌入以提升 A2V 生成效果。

## 方法详解
- **核心框架**：基于潜扩散模型（LDM），对每个去噪步 t，由预训练扩散模型预测干净潜变量 $\tilde{\mathbf{z}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\mathbf{z}_t - \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\hat{\epsilon}$，再将其与条件输入 ImageBind 编码器得到嵌入向量，计算多模态距离作为指导信号。
- **多模态指导机制**：类比 classifier guidance，但不需额外训练分类器。利用 $\mathcal{L} = 1 - \mathcal{F}(\mathbf{E}^{M_1}(\tilde{\mathbf{z}}_0), \mathbf{E}^{M_2}(\mathbf{x}^{M_2}))$ 作为惩罚，回传梯度更新 $\mathbf{z}_t$：$\hat{\mathbf{z}}_t = \mathbf{z}_t - \lambda_1 \nabla_{\mathbf{z}_t}\mathcal{L}$。
- **双损失（Dual Loss）**：针对音频语义信息不足的问题，引入文本条件作为补充监督。A2V：$\mathcal{L}_{a2v} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p)$；V2A：$\mathcal{L}_{v2a} = \mathcal{F}(\mathbf{e}_a, \mathbf{e}_v) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$，其中 $\mathcal{F}$ 为 1 减余弦相似度。
- **三角损失（Triangle Loss，联合生成）**：$\mathcal{L}_{\text{joint-va}} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$，同时优化视频-文本、视频-音频、音频-文本三元对齐。
- **引导提示词调优（Guided Prompt Tuning）**：针对 A2V 任务中每帧梯度难以保证时序一致性的问题，通过反向传播优化输入文本嵌入向量：$\hat{\mathbf{y}} = \mathbf{y} - \lambda_2 \nabla_{\mathbf{y}}\mathcal{L}$，更新后的嵌入在所有去噪步共享，提供一致的语义指导。
- **实现细节**：使用预训练 AudioLDM（音频）和 AnimateDiff（视频），去噪步数分别为 30/25/25，学习率 λ₁=0.1（AudioLDM）和 0.01（AnimateDiff）。

## 实验与结果
- **数据集**：VGGSound（各任务随机采样 3k 对），Landscape（200 对用于联合生成）。
- **评估指标**：V2A/I2A 使用 MKL(KL↓)、ISc↑、FD↓、FAD↓；A2V 使用 FVD↓、KVD↓、AV-align↑；联合生成使用 FVD↓、FAD↓、AV-align↑、TA-align↑、TV-align↑。
- **V2A 最强结果**：Ours 相比 SpecVQGAN，KL 从 3.290 降至 2.619（↓20.5%），FAD 从 7.736 降至 7.316（↓5.4%）。
- **I2A 最强结果**：Ours 相比 Im2Wav，KL 从 2.612 降至 2.691（接近），FAD 从 7.576 降至 6.869（↓9.3%）。
- **A2V 最强结果**：Ours 相比 TempoToken，FVD 从 1866.285 降至 402.385（↓78.4%），KVD 从 389.096 降至 34.764（↓91.1%），AV-align 从 0.423 提升至 0.522（↑23.4%）。
- **联合生成（开放域）**：Ours 相比 Ours-vanilla，AV-align 从 0.226 提升至 0.283（↑25.2%），TV-align 从 0.322 提升至 0.324。
- **联合生成（Landscape 域 + MM-Diffusion）**：FAD 从 7.752 降至 6.463（↓16.7%），视频质量基本保持。
- **核心结论**：训练免费的方案在各项指标上全面超越需要大规模配对数据训练的基线方法。

## 相关工作脉络
- **MM-Diffusion [36]**：首个无条件联合音视频生成框架，但仅适用于训练域且缺乏语义可控性；本文扩展至开放域并支持文本条件引导。
- **SpecVQGAN [26]**：视频→音频生成基线，依赖 ResNet-50 特征和 VQGAN codebook；本文方法不依赖特定下游领域，泛化性更强。
- **Im2Wav [37]**：图像→音频基线，利用 CLIP+ 语言模型 pipeline；本文直接利用 ImageBind 嵌入空间进行对齐，架构更简洁。
- **TempoTokens [48]**：音频→视频生成基线，存在视频质量和时序一致性差的问题；本文通过多模态对齐和提示词调优显著改善该问题。
- **AudioLDM [29] / AnimateDiff [18]**：各自独立的文本→音频和文本→视频生成模型；本文通过 latent aligner 将其桥接，实现跨模态条件生成。
- **Classifier Guidance [10]**：经典的条件生成机制，需训练时间感知的分类器；本文对齐器无需额外训练，直接复用预训练多模态模型。

## 局限性与未来方向
- **优化开销**：虽然无需训练，但每个去噪步的多模态对齐计算增加了推理时间（论文承认会增加少量采样时间）。
- **音频语义稀疏性依赖辅助**：当音频条件本身语义信息极弱时（如纯背景音乐），即使引入文本补充仍可能存在对齐不精确的问题。
- **时序一致性**：A2V 任务中帧间梯度缺乏显式时序约束，虽通过 prompt tuning 部分缓解，但未从根本上建模时序一致性。
- **模型依赖**：方法性能受限于所选基础生成模型（AudioLDM、AnimateDiff）和 ImageBind 的对齐能力，若基础模型质量不佳则上限受限。
- **未来方向**：可扩展到其他模态组合（如视频-文本-音频三模态联合生成）、引入时序一致性约束、探索更高效的对齐优化策略。

## 研究启发与可借鉴点
- **训练免费的跨模态桥接范式**：利用预训练多模态嵌入空间（如 ImageBind）替代 classifier guidance 中的训练分类器，为任何两个扩散模型提供即插即用的跨模态条件生成方案，可迁移至其他多模态组合（如 video-text、image-text）。
- **三角损失设计思路**：当直接的双模态对齐信号不足时，引入第三种模态（文本）构建三角关系进行联合优化，可有效缓解信息稀疏问题；该思路可用于其他信息不对等的跨模态生成任务。
- **潜空间梯度指导的采样轨迹修改**：在去噪过程的每个 t 步，将多模态距离损失的梯度回传到噪声潜变量 $\mathbf{z}_t$ 以修正生成路径，这一 technique 可推广到其他需要跨模态条件引导的扩散生成场景。
- **与团队方向结合机会**：本方法的 latent aligner 设计可与团队现有的单模态生成模型结合，快速实现跨模态生成能力；三角损失和 prompt tuning 策略可应用于其他多模态联合生成任务。

## 关键术语表
- **Diffusion Latent Aligner**：在扩散模型潜空间中对齐不同模态表示的核心组件，利用预训练多模态模型的嵌入空间计算距离并回传梯度。
- **ImageBind**：Meta 提出的多模态嵌入模型，将图像、文本、音频、视频、深度和热成像等多种模态映射到统一语义空间。
- **Classifier Guidance**：通过训练的分类器梯度引导无条件扩散模型生成特定类别样本的经典技术。
- **Latent Diffusion Model (LDM)**：在压缩的潜空间而非像素空间进行扩散去噪的生成模型，代表作为 Stable Diffusion。
- **MKL (Maximum Kernel Likelihood)**：衡量生成音频与参考音频之间多模态分布匹配度的评价指标，越低越好。
- **FAD (Frechet Audio Distance)**：衡量生成音频与真实音频分布之间距离的指标，越低代表音频质量越高。
- **AV-align**：衡量生成音视频之间音频-视觉对齐程度的指标，越高越好。

## 可复现要素
- **数据集**：VGGSound（公开）、Landscape（公开）；论文使用了随机采样的子集（3k/200 对），但未公开具体采样种子以外的数据划分。
- **代码**：项目网站 https://yzxing87.github.io/Seeing-and-Hearing/，论文未明确声明代码开源状态（截至阅读时间需进一步确认 GitHub）。
- **权重**：使用预训练 AudioLDM 和 AnimateDiff，权重均可从原项目获取；ImageBind 权重可从 Meta 官方获取。
- **关键超参**：去噪步数（V2A=30，A2V/Joint=25）、学习率 λ₁=0.1（AudioLDM）、λ₁=0.01（AnimateDiff）、λ₂ 用于 prompt tuning（论文未明确给出具体值）。
