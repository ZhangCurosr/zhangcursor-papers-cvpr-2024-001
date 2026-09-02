---
title: "Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:15:44"
field: "多模态生成"
keywords: ["跨模态生成", "视觉-音频生成", "扩散模型", "ImageBind", "无训练对齐", "联合生成"]
innovations: ["提出Diffusion Latent Aligner，利用ImageBind在潜空间引导跨模态去噪", "首次实现文本引导的开放域Joint-VA生成", "引入Dual/Triangle Loss与Guided Prompt Tuning解决音频语义稀薄问题"]
benchmarks: ["VGGSound", "Landscape"]
---

# 论文速读：Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with-Diffusion-Latent-Aligners

## 一句话总结
本文提出一种基于优化的跨模态对齐方法，利用预训练 ImageBind 模型作为多模态嵌入空间的"桥梁"，在不重新训练单模态扩散模型的前提下，桥接预训练的视觉（AnimateDiff）和音频（AudioLDM）生成模型，实现开放域的联合视频-音频生成、视频→音频、音频→视频、图像→音频四种任务。

## 研究问题与动机
1. **现有扩散模型仅关注单模态生成**：真实世界内容是视听多模态的，但当前 T2V/T2A 模型各自独立生成，导致生成的视频无伴随音频、音频无同步视觉内容。
2. **两阶段拼接方案存在缺陷**：先用 T2V 生成视频再用 V2A 生成音频的方案，受限于 V2A/A2V 模型能力弱、领域泛化差的问题，且缺乏联合优化。
3. **Joint-VA 生成研究匮乏**：MM-Diffusion 等已有工作局限于特定领域（如 Landscape）且无法进行语义可控生成。
4. **缺乏资源友好的通用方案**：从头训练大规模多模态联合生成模型成本高昂，需要一种能复用现有强单模态模型的低成本方案。

## 核心贡献（创新点）
1. **提出跨模态扩散潜空间对齐新范式**：利用 ImageBind 的多模态嵌入空间作为"对齐器"，将独立的单模态扩散模型桥接为有机系统，无需重新训练基座模型。
2. **设计 Diffusion Latent Aligner**：借鉴 classifier guidance 思想，在去噪过程中将噪声潜变量映射回干净潜变量后，通过 ImageBind 计算与条件模态的距离并反向传播梯度，实现实时引导。
3. **引入 Dual/Triangle Loss 与 Guided Prompt Tuning**：针对音频语义信息不完整的问题，引入文本 embedding 作为第三锚点构建三角形损失；同时通过优化 prompt embedding 解决 A2V 任务中引导信号过弱及帧间时序不一致问题。
4. **首次实现文本引导的开放域 Joint-VA 生成**：以 AudioLDM + AnimateDiff 为基座，配合 Latent Aligner，完成首个支持文本条件、兼顾音视频对齐与语义一致性的联合生成工作。

## 方法详解
1. **基础架构**：采用 Latent Diffusion Model（LDM）框架，视频生成基座为 AnimateDiff，音频生成基座为 AudioLDM，对齐器使用预训练 ImageBind（无需微调）。
2. **Classifier Guidance 类比的 Latent 引导**：对每个去噪步 t，由预测噪声得到干净潜变量 $\tilde{\mathbf{z}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\mathbf{z}_t - \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\hat{\epsilon}$，输入 ImageBind 分别提取各模态 embedding，计算 cosine 距离作为对齐损失，再通过梯度 $\nabla_{\mathbf{z}_t}\mathcal{L}$ 更新潜变量：$\hat{\mathbf{z}}_t = \mathbf{z}_t - \lambda_1\nabla_{\mathbf{z}_t}\mathcal{L}$。
3. **Dual/Triangle Loss**：考虑到音频语义信息稀薄，引入文本 prompt embedding $\mathbf{e}_p$ 作为辅助锚点。V2A 损失：$\mathcal{L}_{v2a} = \mathcal{F}(\mathbf{e}_a, \mathbf{e}_v) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$；A2V 损失：$\mathcal{L}_{a2v} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p)$；Joint-VA 损失：$\mathcal{L}_{joint-va} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$，其中 $\mathcal{F}=1-\text{cosine\_sim}$。
4. **Guided Prompt Tuning**：对文本 prompt embedding $\mathbf{y}$ 也施加梯度更新：$\hat{\mathbf{y}}=\mathbf{y}-\lambda_2\nabla_{\mathbf{y}}\mathcal{L}$，使文本语义能够根据视觉内容自适应调整，解决 A2V 任务中引导信号微弱及帧间时序不一致的问题。
5. **训练策略**：全程 training-free，仅在推理时进行 N 步优化（warmup 前 K 步启用），学习率 $\lambda_1=0.1$（AudioLDM）、$\lambda_2=0.01$（AnimateDiff）。

## 实验与结果
- **数据集**：VGGSound（V2A/I2A/A2V 各采 3k 样本）、Landscape（Joint-VA 采 200 对）。
- **评估指标**：V2A 用 KL、ISc、FD、FAD；I2A 同；A2V 用 FVD、KVD、AV-align；Joint-VA 用 FVD、FAD、AV-align、TA-align、TV-align。
- **V2A**：Ours 的 KL=2.619（最优）、FAD=7.316，优于 SpecVQGAN（KL=3.290, FAD=7.736）。
- **I2A**：Ours 的 KL=2.691、FAD=6.869，显著优于 Im2Wav（KL=2.612, FAD=7.576）及 Ours-vanilla（KL=3.115, FAD=7.364）。
- **A2V**：Ours 的 FVD=402.385、AV-align=0.522，明显优于 TempoToken（FVD=1866.285, AV-align=0.423）。
- **Joint-VA（Open-domain）**：Ours 的 AV-align=0.283、TA-align=0.324、TV-align=0.138，较 Ours-vanilla（0.226/0.322/0.074）提升显著；可插件式增强 MM-Diffusion 的 FAD（从 7.752 降至 6.463）。
- **核心结论**：training-free 方法在音视频质量与对齐度上均超越需要大规模训练的对标方法，且在 Joint-VA 场景中首次实现语义可控的开放域生成。

## 相关工作脉络
1. **MM-Diffusion [36]**：首个联合音频视频生成框架，但无文本条件、局限于训练域；本文将其扩展至开放域并加入语义引导。
2. **SpecVQGAN [26]**：基于视频的音频生成 baseline，依赖 ResNet-50 特征+codebook重建；本文无需预训练音频-视频配对数据，泛化性更强。
3. **Im2Wav [37]**：基于 CLIP 表征+语言模型的图像→音频方法；本文利用 ImageBind 统一多模态空间，对齐更精准。
4. **TempoTokens [48]**：音频驱动视频生成，但视觉质量差、时序对齐不足；本文通过 Latent Aligner 在生成过程中实时纠正对齐偏差。
5. **Sound2sight [5] / Wav2CLIP [45]**：早期 A2V 工作，分别基于 GAN 和对比学习；本文首次在 latent diffusion 框架下实现训练友好的跨模态引导。
6. **Make-An-Audio / AudioLDM 系列**：单模态 T2A 模型；本文不改变其权重，仅在其去噪过程外挂载对齐器，实现跨模态复用。

## 局限性与未来方向
1. **音频语义信息有限**：纯背景音乐等音频缺乏丰富语义，难以提供有效引导（需依赖 text 作为补充锚点）。
2. **推理速度较慢**：每步需额外计算 ImageBind embedding 和梯度反传，增加采样时间。
3. **时序一致性依赖 prompt tuning**：A2V 任务中帧间一致性需通过优化 prompt embedding 间接保障，尚未显式建模时序约束。
4. **ImageBind 的模态覆盖依赖**：对齐能力受限于 ImageBind 已学习的跨模态绑定质量，对未见模态组合泛化能力未知。

## 研究启发与可借鉴点
1. **Classifier Guidance 思想的跨模态迁移**：将 classifier guidance 推广至无 classifier 场景，用预训练多模态 encoder 的距离函数替代分类器梯度，是一种通用的"对齐即引导"范式，可迁移到其他跨模态生成任务（如 Video-Text、Audio-Text）。
2. **Training-free 的 Latent 空间优化策略**：不修改预训练模型权重，仅在去噪轨迹上进行梯度修正，兼顾复用性与灵活性，适合研究资源有限的团队快速验证新思路。
3. **Triangle Loss 设计**：利用文本作为语义锚点弥补音频信息密度不足的缺陷，这一设计可用于任何"语义稀薄条件 → 语义密集生成"的跨模态任务。
4. **Prompt Embedding Tuning 机制**：将条件输入本身也作为可优化变量，使其在生成过程中自适应调整，提升了引导信号的鲁棒性，可推广至其他条件生成场景。

## 关键术语表
- **Latent Diffusion Model（LDM）**：在压缩潜空间而非像素空间执行扩散去噪的生成模型，代表如 Stable Diffusion、AudioLDM。
- **Classifier Guidance**：通过无条件扩散模型与训练好的分类器结合，以梯度形式引导生成过程满足特定条件的经典方法。
- **ImageBind**：Meta 提出的多模态嵌入模型，将图像、文本、音频、视频等投影到统一语义空间，支持跨模态检索与对齐。
- **Diffusion Latent Aligner**：本文提出的核心模块，利用 ImageBind 在多模态嵌入空间中计算生成潜变量与条件之间的差异，并以梯度方式指导去噪轨迹。
- **V2A / A2V / Joint-VA / I2A**：Video-to-Audio（视频→音频）、Audio-to-Video（音频→视频）、Joint Visual-Audio（联合音视频）、Image-to-Audio（图像→音频）四种生成任务的缩写。
- **FAD（Frechet Audio Distance）**：衡量生成音频与真实音频分布差异的常用指标，值越低越好。
- **FVD（Frechet Video Distance）**：衡量生成视频与真实视频分布差异的常用指标，值越低越好。
- **AV-align**：音视频对齐度量指标，值越高表示生成结果与条件模态的同步性越好。

## 可复现要素
- **数据集**：VGGSound、Landscape（均公开，可下载）
- **代码/权重**：论文未声明开源；项目主页 https://yzxing87.github.io/Seeing-and-Hearing/（截至知识截止日未明确开源链接）
- **基座模型**：AudioLDM [29]、AnimateDiff [18]（均开源）、ImageBind [17]（开源）
- **关键超参**：去噪步数 V2A=30、A2V/Joint-VA=25；学习率 λ₁=0.1、λ₂=0.01；优化步数 N 及 warmup 步数 K 论文未详述具体值；随机种子固定。
