---
title: "CAT-DM-Controllable-Accelerated-Virtual-Try-on-with-Diffusio"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zeng_CAT-DM_Controllable_Accelerated_Virtual_Try-on_with_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:22:39"
field: "虚拟试穿与图像生成"
keywords: ["Virtual Try-on", "Diffusion Model", "ControlNet", "Image Generation", "Model Acceleration"]
innovations: ["提出结合ControlNet和DINO-V2特征提取的GC-DM，增强扩散模型试穿可控性", "设计基于预训练GAN初始化的截断加速策略，将采样步数降至2步", "将泊松融合引入潜在扩散模型后处理，解决边界不连续和面部失真问题"]
benchmarks: ["DressCode", "VITON-HD"]
---

# 论文速读：CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model

## 一句话总结
本文提出CAT-DM（可控加速虚拟试穿扩散模型），通过结合预训练GAN的初始化与改进的扩散模型，在提升衣物图案与纹理可控生成质量的同时，将采样步骤大幅缩减至2步，实现了生成质量与推理速度的双重突破。

## 研究问题与动机
1.  **控制性问题**：现有扩散模型（如DCI-VTON、LaDI-VTON）应用于虚拟试穿时，难以精准控制和复现复杂衣物的图案与纹理细节（如图1所示）。
2.  **生成质量问题**：基于GAN的方法（如VITON-HD）在处理复杂姿态时容易产生不自然的衣物变形，且生成结果往往缺乏真实感，细节模糊。
3.  **推理速度瓶颈**：扩散模型生成高质量图像通常需要数十甚至上百次去噪迭代，限制了其在实时虚拟试穿等场景的应用。

## 核心贡献（创新点）
1.  **提出GC-DM（服饰条件扩散模型）**：设计了基于ControlNet的虚拟试穿网络，引入了额外的条件控制并增强了服饰图像的特征提取能力，从根本上提升了扩散模型在试穿任务中的可控性。**本质区别**：不同于DCI-VTON仅粘贴裁剪后衣物的局部条件，GC-DM通过ControlNet学习更丰富的服饰无关人物表示，并使用DINO-V2替代CLIP进行细粒度特征提取。
2.  **提出基于截断的加速策略**：利用预训练的GAN模型生成初始试穿图像，并以此作为隐式分布起点来启动反向去噪过程。**本质区别**：不同于TDPM直接学习隐式分布，本方法直接复用成熟的预训练GAN生成初始化结果，并结合DDIM采样器加速。
3.  **引入泊松融合技术**：采用Poisson blending算法将生成区域与原图无缝拼接，有效消除了直接拼接产生的明显接缝，并避免了LDM潜在空间重建导致的面部失真。**本质区别**：直接解决了Latent Diffusion Model（LDM）因潜在编码-解码过程带来的像素精度损失和拼接不连续问题。
4.  **实现显著的速度与质量提升**：在VITON-HD数据集上，CAT-DM仅需2步采样即可达到优于所有基线（包括扩散模型和GAN模型）的生成质量，相比DCI-VTON的50步实现了25倍加速。

## 方法详解
1.  **GC-DM架构**：
    *   **基础**：以冻结参数的Paint-by-Example (PBE) 扩散模型为核心生成器。
    *   **控制网络**：构建可训练的ControlNet，其结构复制自PBE的SD Encoder Blocks和Middle Block。通过ControlNet融合多类条件（噪声图像`x_t`、时间步`t`、遮罩`m`、遮罩图像`x'_0`、服饰图像`g`、densepose `p`）生成控制向量`c_t`，注入PBE的跳跃连接和中层块。
    *   **服饰特征提取**：摒弃原PBE使用的CLIP编码器，改用DINO-V2作为服饰图像`g`的特征提取器`ψ`。DINO-V2能提供包含patch token的细粒度特征，再通过全连接层映射到U-Net空间，并利用交叉注意力机制融入生成过程。
    *   **泊松融合**：对于非服饰区域`Ω`，通过求解泊松方程，使生成结果在该区域的梯度场与原图`h`保持一致，从而实现与生成区域`f*`的无缝融合。

2.  **截断加速策略**：
    *   使用预训练的GAN模型（如GP-VTON）生成初始试穿图像`x̄`。
    *   对该图像添加指定时间步`T_trunc`的噪声，得到隐式分布的起始点`x_{T_trunc}`：`x_{T_trunc} = sqrt(ᾱ_{T_trunc}) * x̄ + sqrt(1 - ᾱ_{T_trunc}) * ε`。
    *   以`x_{T_trunc}`作为起点，使用DDIM采样器仅进行`N`步（文中设为2步）反向去噪生成最终结果。训练时，噪声时间步`t`从均匀分布`{1,...,T}`改为`{1,...,T_trunc}`。

## 实验与结果
*   **数据集**：DressCode [23]（Upper, Lower, Dresses）和VITON-HD [4]。
*   **评估指标**：FID（↓）、KID（↓）、SSIM（↑）、LPIPS（↓），分paired（p）和unpaired（u）设置。
*   **主要结果**：
    *   **VITON-HD（Table 2）**：CAT-DM在所有指标上取得SOTA。`FID_u = 8.93`, `FID_p = 5.60`, `SSIM_p = 0.877`, `LPIPS_p = 0.0803`。相比DCI-VTON（`FID_p = 8.19`），CAT-DM的`FID_p`更低（5.60），且只需2步采样。
    *   **DressCode（Table 1）**：GC-DM（未使用GAN加速）表现优异，`FID_u = 9.67`, `FID_p = 7.11`。
*   **消融实验（Table 3, Fig. 7-10）**：
    *   **特征提取器**：DINO-V2配合泊松融合效果最佳（`FID_p = 7.11`），显著优于CLIP和IP-Adapter。
    *   **融合方式**：泊松融合（Poisson Blending）优于直接生成和简单拼接（Concatenation）。
    *   **截断步数`T_trunc`**：实验表明`T_trunc`设为100时，配合2步采样达到最优性能（Fig. 10）。

## 相关工作脉络
1.  **VITON-HD / HR-VITON / GP-VTON**：GAN-based方法，通过扭曲和合成管线生成试穿图。CAT-DM的定位在于克服其在复杂姿态下的变形问题和生成细节的模糊性。
2.  **DCI-VTON**：扩散模型基线，通过将warped衣物作为局部条件粘贴到输入中来提升可控性。CAT-DM与其区别在于引入了更强大的ControlNet控制框架和更细粒度的DINO-V2特征，且大幅减少了采样步数。
3.  **LaDI-VTON**：利用文本反转将服饰视觉特征映射到CLIP空间进行引导。CAT-DM直接使用像素级的DINO-V2特征，避免了语义空间的映射损失，在图案保留上更具优势。
4.  **TDPM (Truncated Diffusion Probabilistic Models)**：学习隐式分布以加速采样。CAT-DM的加速策略借鉴了其思想，但核心区别在于使用预训练GAN生成初始点而非让扩散模型自身学习隐式分布，且结合了更强的条件控制。
5.  **PBE (Paint by Example)**：作为可编辑的扩散模型基础。CAT-DM冻结其参数，仅训练附加的ControlNet，既保留了其强大的生成能力，又降低了训练成本。

## 局限性与未来方向
1.  **对预训练GAN的依赖**：加速策略的性能在一定程度上受限于所使用的预训练GAN模型（如GP-VTON）的质量。如果基础GAN效果差，会限制加速后的上限。
2.  **截断参数`T_trunc`的选择**：需要一个经验性的折中，`T_trunc`过大则加速效果有限，过小则可能过度依赖GAN的初始分布，损失扩散模型的 refinement 能力。未来可探索自适应的`T_trunc`选择策略。
3.  **极端复杂场景的泛化**：论文未详细讨论在极度遮挡、罕见姿态或极端风格图案下的性能表现，这通常是虚拟试穿任务的难点。
4.  **未来方向**：可探索将控制条件（如densepose）替换或补充为更轻量的条件；研究无需预训练GAN的自蒸馏加速方案；以及将加速框架推广至3D化身生成或视频试穿等领域。

## 研究启发与可借鉴点
1.  **混合架构范式**："预训练专用模型初始化 + 扩散模型精修"的架构非常有效。GAN负责快速生成合理布局和大结构，扩散模型负责细节修正和质量提升。这一范式可迁移至其他需要兼顾结构正确性和细节真实性的生成任务。
2.  **控制条件的精细化设计**：利用ControlNet整合多种条件（姿态、遮罩、参考图）并通过特征提取器的升级（CLIP→DINO-V2）实现从语义级到像素级的控制增强，这一思路对需要强条件控制的图像生成任务具有借鉴意义。
3.  **泊松融合的后处理技巧**：针对潜在扩散模型输出与原图融合时的边界 artifacts 问题，泊松融合是一个有效且通用的解决方案，尤其适用于 inpainting 和编辑任务。
4.  **实验设计的完整性**：论文不仅报告了最终指标，还通过系列消融实验（特征提取器、融合方式、截断步数、不同初始化GAN）充分验证了每个组件的贡献，这种详尽的归因分析值得学习。

## 关键术语表
*   **Virtual Try-on (虚拟试穿)**：计算机视觉任务，旨在将目标衣物图像无缝、真实地合成人物身上，同时保持人物姿态、身份和衣物细节。
*   **GC-DM (Garment-Conditioned Diffusion Model)**：本文提出的核心扩散模型架构，通过ControlNet和增强特征提取来提升试穿过程的可控性。
*   **ControlNet**：一种给预训练扩散模型添加可训练控制分支的网络架构，通过冻结主模型参数、只训练控制分支来高效注入条件信号。
*   **DINO-V2**：一种自监督学习获得的视觉特征提取器，能输出全局token和细粒度patch token，相比CLIP更适合需要像素级对齐的任务。
*   **Truncation-Based Acceleration (截断加速策略)**：利用预训练模型生成初始样本，并从中添加噪声开始较短的反向去噪链，以减少总采样步数的方法。
*   **Poisson Blending (泊松融合)**：一种图像编辑技术，通过在目标区域求解泊松方程，使融合区域的梯度与原图保持一致，实现无缝拼接。

## 可复现要素
*   **数据集**：DressCode [23] 和 VITON-HD [4] 均为公开数据集。
*   **代码与权重**：论文未明确提及代码和预训练权重的开源情况。
*   **关键超参**：图像分辨率 `512 × 384`；优化器 AdamW，学习率 `2e-5`；截断步数 `T_trunc` 设为 `100`；最终采样步数设为 `2`；使用DDIM采样器。
