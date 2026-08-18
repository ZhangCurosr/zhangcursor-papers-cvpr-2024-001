---
title: "One-Shot Structure-Aware Stylized Image Synthesis"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cho_One-Shot_Structure-Aware_Stylized_Image_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:44"
field: "图像风格化与生成"
keywords: ["one-shot stylization", "diffusion model", "structure preservation", "CLIP directional loss", "DiffAE", "image editing"]
innovations: ["通过DiffAE结构-语义潜码解耦实现可调节的结构保留风格化", "提出轻量SPN模块补偿扩散编码高频细节损失", "无数据CLIP方向微调范式用于域间风格迁移"]
benchmarks: ["FFHQ", "AAHQ", "MetFaces", "AFHQ-dog", "LSUN-church", "DeepFashion"]
---

# 论文速读：One-Shot Structure-Aware Stylized Image Synthesis

## 一句话总结
本文提出OSASIS，一种基于扩散模型的单次图像风格化方法，通过解耦结构潜码与语义潜码实现结构保持的风格迁移，在低密度（罕见结构）图像上显著优于现有GAN和扩散模型基线。

## 研究问题与动机
- **GAN风格化的结构保持瓶颈**：现有GAN-based方法（如MTG、JoJoGAN）在处理训练集中罕见结构元素（如手、麦克风等）时容易产生结构失真或伪影。
- **域间差距与语义混淆**：GAN方法的风格编码往往将结构与语义纠缠在一起，导致参考图像的结构伪影渗入生成结果（OOD场景）。
- **扩散模型风格化的质量衰减**：DiffuseIT、InST等扩散模型方法虽能避免严重伪影，但在弥合输入与风格图像的域间差距、保持色彩忠实度方面存在不足。
- **缺乏可控的内容-风格混合机制**：如何在推理阶段精确控制注入的语义内容与风格强度，仍是开放问题。

## 核心贡献（创新点）
- **结构-语义解耦框架**：通过DiffAE语义编码器提取语义潜码$\mathbf{z}_{\mathrm{sem}}$，结合DDPM前向过程编码至特定时间步$\mathbf{t}_0$获得结构潜码$\mathbf{x}_{\mathbf{t}_0}$，实现对结构保留强度的可调节控制。
- **结构保留网络(SPN)**：引入轻量1×1卷积网络，将空间结构信息以$\lambda_{SPN}$系数加权后叠加至扩散解码过程，补偿前向编码引入的噪声导致的高频细节丢失。
- **无数据训练的CLIP方向微调**：借鉴MTG策略，利用预训练DDPM从零生成一对语义对齐的域内/域外风格图像，仅通过CLIP方向损失微调目标域扩散模型，无需真实风格数据集。
- **内容-风格混合推理策略**：在UNet不同层次特征图上分别条件化语义潜码——低层使用风格语义$\mathbf{z}_{\mathrm{sem}}^{\mathrm{style}}$、高层使用输入语义$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$，实现解耦的内容风格混合。
- **文本驱动操作扩展**：直接优化输入图像的语义潜码$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$并配合CLIP方向损失，支持在风格化同时修改属性（如年龄、表情），且不破坏结构与风格。

## 方法详解
**训练阶段**：
- 给定风格图像$I_B^{\mathrm{style}}$（目标域B），通过前向DDPM编码至 timestep $\mathbf{t}_0$：$\mathbf{x}_{\mathbf{t}_0} = \sqrt{\alpha_{\mathbf{t}_0}}\mathbf{x}_0 + \sqrt{1-\alpha_{\mathbf{t}_0}}\mathbf{z}$。
- 利用预训练DDPM反向生成域A中语义对齐的风格图像$I_A^{\mathrm{style}}$。
- 冻结预训练DDIM $\epsilon_\theta^A$与DiffAE语义编码器$\mathrm{Enc}_\phi$，复制得到$\epsilon_\theta^B$并进行微调。
- 输入图像$I_A^{\mathrm{in}}$经Forward DiffAE编码为结构潜码$\mathbf{x}_{\mathbf{t}_0}^{\mathrm{in}}$（可调节$\mathbf{t}_0$控制结构保留强度）。

**结构保留网络(SPN)**：
- $\mathbf{x}_t^{SPN} = \mathrm{SPN}(I_A^{\mathrm{in}})$ 为1×1卷积输出。
- 修正潜码：$\mathbf{x}_t' = \mathbf{x}_t + \lambda_{SPN} \cdot \mathbf{x}_t^{SPN}$。
- 反向DiffAE解码：$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}f_\theta(\mathbf{x}_t', t, \mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}) + \sqrt{1-\alpha_{t-1}}\epsilon_\theta^B(\mathbf{x}_t', t, \mathbf{z}_{\mathrm{sem}}^{\mathrm{in}})$。

**损失函数**（共三项）：
1. **跨域损失(Cross-domain Loss)**：约束$I_A^{\mathrm{in}} \to I_B^{\mathrm{in}}$的CLIP变化方向与$I_A^{\mathrm{style}} \to I_B^{\mathrm{style}}$一致。
2. **域内损失(In-domain Loss)**：测量域A和域B内部变化的相似性，防止过度风格化。
3. **重建损失(Reconstruction Loss)**：$L_1$损失 + Perceptual Loss + $L_1$ CLIP Embedding Loss，监督风格图像的重建 fidelity。

**推理采样**：
- 结构潜码$\mathbf{x}_{\mathbf{t}_0}^{\mathrm{in}}$控制结构完整性。
- 风格语义$\mathbf{z}_{\mathrm{sem}}^{\mathrm{style}}$注入UNet低层特征图，输入语义$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$注入高层特征图，切换点为$f_{ch}$。

**文本驱动操作**：
- 对$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$施加CLIP方向损失进行直接优化，再经上述混合流程生成风格化结果。

## 实验与结果
**数据集**：
- 主要评估：FFHQ（训练域），AAHQ、MetFaces、Prev（风格参考来源）。
- 泛化测试：AFHQ-dog、LSUN-church、DeepFashion。
- 低密度图像筛选：从FFHQ随机选20,000张，基于Stochastic Reconstruction的LPIPS得分排序，取Top 100高LPIPS组作为"低密度"测试集。

**基线方法**：MTG+HFGI、JoJoGAN+HFGI、DiffuseIT、InST（均使用公开预训练模型）。

**评估指标**：ArtFID↓（风格相关性）、ID Similarity↑（内容保留）、Structure Distance↓（结构保留）。

**主要结果（低密度图像）**：

| 方法 | ArtFID (AAHQ/MetFaces/Prev) | ID Sim (AAHQ/MetFaces/Prev) | SD (AAHQ/MetFaces/Prev) |
|------|----------------------------|----------------------------|-------------------------|
| MTG+HFGI | 36.39 / 38.02 / 37.27 | 0.3730 / 0.4656 / 0.4063 | 0.0386 / 0.0350 / 0.0360 |
| JoJoGAN+HFGI | 40.41 / 44.74 / 41.09 | 0.5145 / 0.5207 / 0.4743 | 0.0411 / 0.0454 / 0.0430 |
| DiffuseIT | 44.93 / 53.35 / 48.18 | 0.6992 / 0.7158 / 0.6994 | 0.0309 / 0.0300 / 0.0310 |
| InST | 38.16 / 50.33 / 35.86 | 0.2253 / 0.2188 / 0.2238 | 0.0492 / 0.0443 / 0.0488 |
| **OSASIS** | **34.89 / 43.20 / 33.20** | **0.6825 / 0.7323 / 0.7029** | **0.0361 / 0.0295 / 0.0391** |

- OSASIS在ID Similarity和Structure Distance上显著优于所有基线，ArtFID同样最优或接近最优。
- DiffuseIT ID/结构保持好但ArtFID劣（域间差距大）；InST ArtFID不错但ID保持极差。
- **最强提升**：OSASIS相对于JoJoGAN+HFGI，ID Similarity从~0.51提升至~0.68（+33%），结构距离从0.041降至0.036。

**消融实验**：
- SPN的$\lambda_{SPN}=0.1$为最优平衡点：ArtFID=34.89，ID Sim=0.6825，SD=0.0361；过大（$\geq 0.5$）会牺牲风格质量。

## 相关工作脉络
- **MTG [39]**：CLIP方向损失微调GAN生成器的开创性工作；本文借鉴其loss设计，但迁移至扩散模型框架并强化结构保持。
- **JoJoGAN [2]**：首次成功的GAN one-shot风格迁移；本文强调GAN反演方法对稀有结构保持不足，需扩散模型更优。
- **DiffAE [23]**：提供语义编码器$\mathrm{Enc}_\phi$与结构化潜码分解机制，是本文结构-语义解耦的基石。
- **DiffuseIT [15] / InST [37]**：扩散模型风格化代表；前者缺乏训练适应性（域间差距），后者依赖文本反演导致色彩/身份失真。
- **DiffusionCLIP [13]**：CLIP方向损失用于文本驱动编辑；本文将其扩展至图像引导风格化与内容-风格混合。
- **HFGI [31]**：高保真GAN反演方法，用作MTG/JoJoGAN公平对比的基础设施。

## 局限性与未来方向
- **训练时间长**：相比GAN基线，扩散模型微调耗时显著，限制实际部署效率。
- **每风格需单独训练**：当前方法需为每个新风格图像微调$\epsilon_\theta^B$，缺乏零样本或多风格泛化能力。
- **未讨论高分辨率扩展**：实验集中于FFHQ人脸域，未验证对高分辨率或复杂场景（如全场景图像）的有效性。
- **未来方向**：优化训练效率（如少步微调、冻结策略）、探索单模型多风格适应能力、扩展至非人脸域的大规模验证。

## 研究启发与可借鉴点
- **结构-语义潜码解耦设计**：利用DiffAE框架分离$\mathbf{z}_{\mathrm{sem}}$与$\mathbf{x}_{\mathbf{t}_0}$的思路可迁移至其他生成任务（如editing、super-resolution），实现细粒度的结构可控性。
- **轻量SPN模块**：1×1卷积的结构保留机制计算开销极小，可作为即插即用组件嵌入任意扩散解码流程，适用于需要保结构的下游应用。
- **无数据CLIP方向微调范式**：通过DDPM生成伪风格对弥合域间差距，避免了收集大规模风格数据集的成本，可推广至其他域适应任务。
- **UNet层次化语义条件注入**：低层注入风格、高层注入内容的设计提供了一种通用的跨域混合推理策略，可启发跨-domain translation或conditioned generation的架构设计。
- **低密度图像筛选评估协议**：基于Stochastic Reconstruction的LPIPS排序挑选"困难样本"作为评测基准，为结构保持能力评估提供了可复用的protocol。

## 关键术语表
- **OSASIS**：One-Shot Structure-Aware Stylized Image Synthesis，本文提出的单次结构感知风格化扩散模型框架。
- **DiffAE (Diffusion Autoencoder)**：将图像编码为语义潜码与结构潜码的扩散自编码器，支持语义有意义的潜空间操纵。
- **结构潜码 $\mathbf{x}_{\mathbf{t}_0}$**：图像经前向DDPM编码至特定时间步$\mathbf{t}_0$得到的潜表示，控制结构保留强度。
- **语义潜码 $\mathbf{z}_{\mathrm{sem}}$**：由DiffAE语义编码器提取的潜码，携带高层语义信息（内容/风格）。
- **CLIP方向损失**：约束生成图像与目标图像在CLIP embedding空间中的变化方向一致，用于域间迁移与属性编辑。
- **结构保留网络 (SPN)**：轻量1×1卷积模块，将输入图像的空间信息加权叠加至扩散解码过程以保结构。
- **ArtFID**：评估风格化图像与参考风格分布之间FID的指标，衡量风格保真度。
- **低密度图像**：在预训练分布中罕见出现的图像（如含稀有结构元素），用于评估模型的OOV鲁棒性。

## 可复现要素
- **数据集**：FFHQ（公开）、AAHQ、MetFaces（公开）；其他风格来源提及未明确公开性。
- **代码开源**：是，GitHub: https://github.com/hansam95/OSASIS
- **预训练权重**：公开可用的预训练DDIM（FFHQ）及baseline模型权重
- **关键超参**：$\lambda_{SPN}=0.1$（最优）、$\mathbf{t}_0$（结构编码时间步，可调）、$f_{ch}$（内容/风格特征图切换点）；训练细节见supplementary。
