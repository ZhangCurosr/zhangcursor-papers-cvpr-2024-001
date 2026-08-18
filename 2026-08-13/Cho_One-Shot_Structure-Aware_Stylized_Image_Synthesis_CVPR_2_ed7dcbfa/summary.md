---
title: "One-Shot Structure-Aware Stylized Image Synthesis"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cho_One-Shot_Structure-Aware_Stylized_Image_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:13:17"
---

# 论文速读：One-Shot Structure-Aware Stylized Image Synthesis

## 一句话总结
提出OSASIS，一种基于扩散模型的单参考图结构感知风格化方法，通过显式解耦图像的结构与语义潜码并引入结构保持网络（SPN），在保留输入图像原始结构（尤其是训练分布外的稀有元素）的同时实现高质量风格迁移，并可无缝衔接文本驱动的属性操控。

## 研究问题与动机
- GAN-based单图风格化方法在推理时难以将参考图的结构与风格完全分离，常导致参考图的结构性伪影污染生成结果，且对罕见结构（如手、麦克风）的保留能力显著下降。
- 现有扩散模型风格化方法（如DiffuseIT、InST）虽能生成高保真图像，但缺乏对输入图像结构与内容质量的强约束，容易产生身份/表情畸变或色彩/风格迁移不彻底的问题。
- 现有方法多依赖大量配对数据或针对特定域微调，缺乏仅需单张参考图即可桥接任意风格域、且能精细控制“结构保留强度”与“风格迁移强度”的通用框架。

## 核心贡献（创新点）
1. 提出基于扩散模型的结构-语义显式解耦框架，利用DiffAE语义潜码与特定时间步结构潜码分离内容与风格信息；与以往将结构/风格纠缠于单一倒推潜码的方法本质不同，本方法支持可控的分层注入与强度调节。
2. 设计结构保持网络（SPN，轻量1x1卷积），在反向去噪每一步叠加输入图像的空间特征以补偿前向DiffAE编码固有的噪声损耗；区别于仅靠提升$t_0$硬保结构的思路，SPN提供了可微的空间细节修复通道。
3. 提出无需配对数据集的单图风格域微调范式：利用预训练DDPM生成语义对齐的伪参考图，结合跨域/同域CLIP方向损失与重建损失联合微调目标DDIM；相比MTG等方法依赖大规模生成数据，本方法数据效率更高且天然支持OOD风格图。
4. 实现语义潜码的直接优化以支持文本驱动的风格化属性编辑，无需重新训练生成网络；与DiffusionCLIP等纯文本编辑方法相比，本工作将文本编辑与图像风格化在统一潜空间内无缝衔接。

## 方法详解
- **双潜码解耦与编码**：采用DiffAE的语义编码器$\mathrm{Enc}_\phi$提取$\mathbf{z}_{\mathrm{sem}}$；输入图像经冻结的预训练DDIM $\epsilon_\theta^A$执行前向DiffAE，编码至特定时间步$t_0$得到结构潜码$\mathbf{x}_{\mathbf{t}_0}^{\mathrm{in}}$，$t_0$越大底层结构保留越强。
- **风格域微调（无数据依赖）**：从真实风格域B取单张$I_B^{\mathrm{style}}$，利用预训练DDPM生成域A中语义对齐的$I_A^{\mathrm{style}}$。冻结$\epsilon_\theta^A$与$\mathrm{Enc}_\phi$，复制得到$\epsilon_\theta^B$并进行微调。
- **损失函数**：总损失由三部分构成。① 跨域损失（Cross-domain loss）：保证$I_A^{\mathrm{in}} \to I_B^{\mathrm{in}}$的CLIP方向与$I_A^{\mathrm{style}} \to I_B^{\mathrm{style}}$一致；② 同域损失（In-domain loss）：约束域内方向相似度，防止单一跨域损失引起的意外属性偏移；③ 重建损失（Reconstruction loss）：对$I_A^{\mathrm{style}}$编码后经SPN反向生成$\hat{I}_B^{\mathrm{style}}$，计算$L_1$ + Perceptual + $L_1$ CLIP Embedding损失。
- **结构保持网络（SPN）**：定义$\mathbf{x}_t^{SPN} = SPN(I_A^{\mathrm{in}})$（1x1卷积），在每一步反向DiffAE更新时加入：$\mathbf{x}_t' = \mathbf{x}_t + \lambda_{SPN} * \mathbf{x}_t^{SPN}$，再通过微调后的$\epsilon_\theta^B$结合$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$递推生成下一时刻隐状态，最终输出$I_B^{\mathrm{in}}$。
- **推理阶段内容/风格混合**：将$\mathbf{z}_{\mathrm{sem}}^{\mathrm{style}}$注入UNet低层特征图（传递颜色/纹理等风格），将$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$注入高层特征图（传递物体/身份等语义），切换点设为$f_{ch}$，配合结构潜码完成采样。
- **文本驱动操控**：直接对$\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$施加CLIP方向损失进行梯度优化，再将优化后潜码代入上述混合流程，实现带风格迁移的文本属性编辑。

## 实验与结果
- **数据集**：FFHQ（作为$\epsilon_\theta^A$训练底库与输入源）；风格参考取自AAHQ、MetFaces、Prev（既往研究风格图）；泛化测试覆盖AFHQ-dog、LSUN-church、DeepFashion。
- **评估指标**：ArtFID（风格化质量）、ID Similarity（ArcFace，身份/内容保持）、Structure Distance（结构保持）。
- **主要定量结果（低密度/OOD稀有结构图像，Table 1）**：OSASIS全面领先。ArtFID为34.89（AAHQ）/43.20（MetFaces）/33.20（Prev）；ID Similarity为0.6825/0.7323/0.7029；Structure Distance为0.0361/0.0295/0.0391。显著优于MTG+HFGI、JoJoGAN+HFGI、DiffuseIT、InST。
- **最强结果与提升**：在MetFaces风格上，OSASIS的ID Similarity达0.7323（较次优InST的0.7029提升约4.3%），Structure Distance达0.0295（优于DiffuseIT的0.0300，且大幅领先其他基线），证明其在稀有结构保留上的鲁棒性。
- **消融实验（Table 2）**：引入SPN（$\lambda_{SPN}=0.1$）使ID Similarity从0.6595提升至0.6825，Structure Distance从0.0371降至0.0361；$\lambda_{SPN} \ge 0.5$会过度强调结构而损害ArtFID与风格化自然度。

## 相关工作脉络
- **MTG**：基于GAN的CLIP方向微调单图风格化。本文定位差异在于：MTG依赖GAN inversion且结构保真受限于GAN先验，OSASIS基于扩散模型并显式解耦结构/语义潜码，在OOD稀有结构上表现更稳定。
- **JoJoGAN**：通过随机风格混合生成训练集再微调GAN映射器。本文相比无需生成大规模配对数据，仅凭单张参考图即可完成domain gap桥接。
- **DiffuseIT / In
