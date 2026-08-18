---
title: "CAT-DM-Controllable-Accelerated-Virtual-Try-on-with-Diffusio"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zeng_CAT-DM_Controllable_Accelerated_Virtual_Try-on_with_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:22:19"
field: "图像生成与虚拟试穿"
keywords: ["virtual try-on", "diffusion model", "ControlNet", "accelerated sampling", "garment transfer"]
innovations: ["提出GC-DM通过ControlNet+DINO-V2增强扩散模型可控性", "截断加速策略利用预训练GAN生成隐式分布起点实现25倍加速", "Poisson融合消除生成边界痕迹保留非服装区域真实性"]
benchmarks: ["DressCode", "VITON-HD"]
---

# 论文速读：CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model

## 一句话总结
CAT-DM提出了一种可控加速的虚拟试穿框架，通过融合GC-DM（衣物条件扩散模型）与基于预训练GAN的截断加速策略，在保留复杂衣物图案纹理的同时，将扩散模型采样步数从默认50步降至2步，实现25倍推理加速，并在DressCode和VITON-HD基准上均取得SOTA性能。

## 研究问题与动机
1. **GAN方法可控性不足**：现有基于GAN的虚拟试穿方法（如TPS、STN、FlowNet）在应对复杂姿态时衣物变形不自然，且生成结果缺乏真实感和精细细节。
2. **扩散模型可控性挑战**：扩散模型虽生成质量优异，但在虚拟试穿任务中难以精确保留衣物的复杂图案与纹理细节，可控性仍是瓶颈。
3. **推理速度慢**：扩散模型的高保真生成需大量采样步（通常1000步），限制其实时应用场景。
4. **现有加速方法有缺陷**：TDPM等方法需学习隐式分布，难以准确捕捉截断步的损坏数据分布，且在复杂图案生成上仍面临可控性问题。

## 核心贡献（创新点）
1. **提出GC-DM（Garment-Conditioned Diffusion Model）**：通过ControlNet引入额外控制条件并增强衣物特征提取，解决扩散模型在虚拟试穿中的可控性问题，与已有扩散试穿方法的本质区别在于采用DINO-V2替代CLIP进行像素级特征提取。
2. **引入截断加速策略**：利用预训练GAN生成初始试穿图像并加噪获取隐式分布，作为反向去噪起点，相比TDPM无需学习隐式分布，而是直接利用GAN先验，显著提升推理速度。
3. **设计Poisson融合模块**：将原始人物图像与生成结果无缝融合，消除拼接痕迹，解决LDM重建中人脸区域的像素精度损失问题。
4. **CAT-DM整体框架**：结合GC-DM与预训练GAN，在保证生成质量的同时将采样步降至2步，实现25倍加速，在DressCode和VITON-HD上均超越现有方法。

## 方法详解
**GC-DM（衣物条件扩散模型）**：
- **ControlNet架构**：冻结预训练PBE（Paint by Example）的所有参数，复制其SD Encoder Blocks和SD Middle Block到ControlNet，仅训练ControlNet参数，大幅降低显存需求与训练成本。
- **多条件输入**：ControlNet接收噪声图像$\mathbf{x}_t$、时间步$t$、掩码$m$、蒙皮图像$\mathbf{x}'_0$、衣物图像$g$及densepose $p$，生成控制向量$c_t$，通过跳连和中间块融入PBE的U-Net。
- **损失函数**：
$$\mathbb{E}_{\mathbf{x}_0, \mathbf{x}'_0, m, g, p, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta (\mathbf{x}_t, \mathbf{x}'_0, m, g, p, t) \|_2^2 \right]$$
- **衣物特征提取**：使用DINO-V2替代CLIP作为特征提取器，DINO-V2生成全局token和patch token，保留更多像素级细节；经全连接层映射后通过交叉注意力融入U-Net。
- **Poisson融合**：在掩码区域$\Omega$外保持原图不变，求解：
$$\begin{cases}|N_p|f_p - \sum_{q \in N_p} f_q = \sum_{q \in N_p} v_{pq}, & p \in \Omega \\ f_p = f^*_p, & p \in \partial\Omega\end{cases}$$
其中$v_{pq} = h_p - h_q$，确保非服装区域（如人脸）无失真且无缝拼接。

**截断加速策略**：
- 使用预训练GAN模型（如GP-VTON）生成初始试穿图像$\bar{\mathbf{x}}$。
- 通过正向扩散过程加噪至$T_{\text{trunc}}$步：
$$\mathbf{x}_{T_{\text{trunc}}} = \sqrt{\bar{\alpha}_{T_{\text{trunc}}}}\bar{\mathbf{x}} + \sqrt{1 - \bar{\alpha}_{T_{\text{trunc}}}}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
- 以$\mathbf{x}_{T_{\text{trunc}}}$为起点，使用DDIM采样器进行反向去噪，训练时$t$仅在$\{1, 2, ..., T_{\text{trunc}}\}$均匀采样。
- 通过调整$T_{\text{trunc}}$控制GAN与GC-DM的贡献比例：$T_{\text{trunc}}$越小越依赖GAN，越大越依赖扩散模型。

## 实验与结果
**数据集**：DressCode（upper/lower/dresses三类别）、VITON-HD。

**评估指标**：FID、KID（真实性）；SSIM、LPIPS（配对设置下的保真度）。

**主要结果**：
- **DressCode数据集**（GC-DM，DDIM 16步）：
  - DressCode-Upper：FID_u=12.62，KID_u=1.89，FID_p=9.85，SSIM_p=0.927
  - DressCode-Lower：FID_u=14.83，KID_u=2.82，FID_p=10.25，SSIM_p=0.902
  - DressCode-Dresses：FID_u=14.30，KID_u=3.36，FID_p=10.71，SSIM_p=0.863
  - 在FID和KID上全面超越MGD、LaDI-VTON等扩散方法。

- **VITON-HD数据集**（CAT-DM，DDIM 2步）：
  - FID_u=8.93，FID_p=5.60，SSIM_p=0.877，LPIPS_p=0.0803
  - 相比DCI-VTON（FID_p=8.19），FID_p提升约31.6%；相比GC-DM（FID_p=7.11），在2步采样下仍保持竞争力。
  - 实现25倍加速（vs DCI-VTON默认50步）。

**消融实验**：
- DINO-V2相比CLIP、IP-Adapter、SeeCoder在所有指标上最优。
- Poisson融合优于直接生成和图像拼接。
- 不同截断步$T_{\text{trunc}}$实验表明$T_{\text{trunc}}=100$时2步采样效果最佳。

## 相关工作脉络
1. **GAN-based方法（VITON-HD、HR-VITON、GP-VTON）**：以衣物形变+生成器合成为主，CAT-DM通过扩散模型解决其变形不自然、细节模糊问题，同时利用GAN作为初始化加速。
2. **扩散试穿方法（MGD、LaDI-VTON、DCI-VTON）**：MGD使用多模态数据引导，LaDI-VTON用文本反演映射衣物特征，DCI-VTON将形变衣物贴入扩散输入作为局部条件；CAT-DM通过ControlNet+DINO-V2提供更细粒度的像素级控制，并解决可控性与加速的兼顾问题。
3. **加速采样方法（DDIM、TDPM、Progressive Distillation）**：DDIM通过非马尔可夫过程加速，TDPM学习隐式分布截断扩散轨迹；CAT-DM利用预训练GAN生成隐式分布起点，避免TDPM的学习困难，同时保持生成质量。
4. **ControlNet（Zhang et al.）**：为扩散模型添加条件控制，CAT-DM将其引入虚拟试穿，并通过多源条件（densepose、衣物图像等）增强可控性。
5. **DINO-V2 vs CLIP**：CLIP仅提取全局语义特征，DINO-V2保留patch级像素信息；CAT-DM选择DINO-V2作为衣物特征提取器，实现更精确的图案生成。

## 局限性与未来方向
1. **依赖预训练GAN性能**：CAT-DM的性能在一定程度上受限于所用GAN模型的质量，若GAN表现不佳，加速效果会打折扣。
2. **复杂姿态泛化能力待验证**：论文主要验证了标准测试集，对于极端姿态、遮挡等复杂场景的泛化能力尚未充分探讨。
3. **高分辨率训练成本**：虽冻结了PBE参数，但ControlNet训练仍需较大显存，且DINO-V2的特征提取增加了计算开销。
4. **未来方向**：可扩展至更多衣物类别和分辨率；探索更轻量的加速采样器；研究免GAN的隐式分布学习策略以降低对预训练模型的依赖。

## 研究启发与可借鉴点
1. **预训练模型作为隐式分布初始化**：将GAN生成结果加噪作为扩散反向过程的起点，是一种有效的加速策略，可迁移至其他图像生成任务（如超分、修复）以减少采样步数。
2. **DINO-V2用于像素级特征控制**：在需要保留细节的任务中，用DINO-V2替代CLIP进行特征提取并通过交叉注意力注入U-Net，可显著提升生成可控性。
3. **ControlNet多条件融合设计**：结合mask、densepose、参考图像等多种条件，通过ControlNet注入扩散模型，可在不改变主干网络的情况下增强条件控制能力。
4. **Poisson融合处理生成边界**：在局部生成任务中，使用Poisson融合替代简单拼接，可有效消除边界 artifacts，保持非生成区域的真实性。
5. **冻结主干+训练适配器**：冻结大规模预训练模型（如PBE）的参数，仅训练轻量ControlNet，是节省算力、快速适配新任务的可行范式。

## 关键术语表
**GC-DM（Garment-Conditioned Diffusion Model）**：衣物条件扩散模型，CAT-DM的核心组件，通过ControlNet和多源条件增强扩散模型在虚拟试穿中的可控性。

**ControlNet**：为预训练扩散模型添加可训练条件控制网络的架构，通过复制主干编码器并注入控制向量实现条件引导。

**DINO-V2**：自监督视觉特征提取器，生成全局token和patch token，保留更多像素级细节，优于CLIP的全局语义表征。

**Poisson融合**：基于偏微分方程的图像融合技术，通过在目标区域求解泊松方程实现无缝拼接，消除边界痕迹。

**截断加速策略**：利用预训练GAN生成初始图像并加噪至中间扩散步，作为反向去噪起点，显著缩短采样轨迹。

**TDPM（Truncated Diffusion Probabilistic Model）**：通过复用扩散模型参数生成隐式分布作为反向扩散起点的加速方法，但难以学习准确的截断分布。

**DDIM（Denoising Diffusion Implicit Models）**：非马尔可夫扩散过程，可在更少采样步数下生成高质量图像，无需额外训练。

**PBE（Paint by Example）**：基于示例的图像编辑扩散模型，能够根据参考图像语义修改目标图像内容。

## 可复现要素
- **数据集**：DressCode（公开）、VITON-HD（公开）
- **代码/权重**：论文未明确提及代码开源状态
- **关键超参**：
  - 图像分辨率：$512 \times 384$
  - 优化器：AdamW
  - 学习率：$2 \times 10^{-5}$
  - GPU：2× NVIDIA GeForce RTX 4090
  - GC-DM采样步数：16（DDIM）
  - CAT-DM截断步$T_{\text{trunc}}$：100，采样步数：2（DDIM）
  - 预训练GAN：GP-VTON（用于CAT-DM）
