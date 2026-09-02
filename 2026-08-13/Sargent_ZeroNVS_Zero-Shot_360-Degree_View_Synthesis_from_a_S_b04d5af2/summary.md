---
title: "ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sargent_ZeroNVS_Zero-Shot_360-Degree_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:14:47"
field: "3D视觉生成"
keywords: ["novel view synthesis", "zero-shot learning", "diffusion model", "3D generation", "score distillation sampling", "NeRF"]
innovations: ["6DoF+1相机条件参数化与视角中心归一化方案", "SDS锚定机制提升背景多样性", "首个野外场景单图像零样本360度视图合成系统"]
benchmarks: ["DTU", "Mip-NeRF 360", "CO3D", "RealEstate10K", "ACID"]
---

# 论文速读：ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image

## 一句话总结
ZeroNVS 提出了一种3D感知扩散模型，通过新的相机条件参数化/归一化方案和SDS锚定机制，实现野外场景的单图像零样本360度视图合成，在 DTU 和 Mip-NeRF 360 基准上均取得最强 LPIPS 结果。

## 研究问题与动机
- **单对象方法的局限**：现有单图像新视图合成方法（如 Zero-1-to-3、RealFusion）主要针对带掩码背景的单个物体设计，无法直接处理多物体、复杂背景的野外场景。
- **缺乏大规模场景训练数据**：与物体中心数据集（Objaverse-XL）不同，目前缺乏具有完整几何、纹理和相机参数的全局场景数据集，导致训练困难。
- **SDS 蒸馏多样性不足**：Score Distillation Sampling (SDS) 在360度场景蒸馏时倾向于预测单调背景（灰色/平滑），缺乏真实感和多样性。
- **相机表示不适配**：3DoF 相机参数化（俯仰角、方位角、半径）无法表达任意姿态和滚转角，且深度尺度歧义问题在真实场景中更为严重。

## 核心贡献（创新点）
- **提出 ZeroNVS 框架**：首个在野外场景上实现单图像360度视图合成的3D感知扩散模型，突破了以往仅适用于单对象的假设。
- **新型相机条件参数化与归一化方案**：将3DoF扩展为6DoF+1（相对位姿+视场角），并设计视角中心归一化（viewer-centric normalization）解决多源数据的深度尺度歧义。
- **SDS 锚定机制（SDS Anchoring）**：先用 DDIM 采样多个锚定视图，再在 SDS 优化时用最近邻视图作为条件，显著改善背景多样性。
- **新基准与 SOTA 结果**：将 Mip-NeRF 360 适配为单图像 NVS 基准，并在 DTU 零样本设置下超越专门微调的方法（LPIPS 0.380 vs. 0.421）。

## 方法详解
- **整体框架**：先训练2D条件扩散模型 p_θ 进行新视图合成，再通过3D SDS 蒸馏将结果提升为 NeRF。
- **条件表示函数**：M(D, f, E, i, j) 将深度图 D、视场角 f、外参 E 和输入/目标视图索引 i, j 映射为条件嵌入，与输入图像 X_i 一起输入扩散模型。
- **相机参数化演进**：
  - M_Zero-1-to-3 = P(E_i) - P(E_j)，仅用3DoF（仰角、方位角、半径），不适用于任意姿态。
  - M_6DoF+1 = [E_i^(-1)E_j, f]，使用相对位姿矩阵+视场角，对刚性变换不变，但不处理尺度歧义。
  - M_6DoF+1,norm = 用相机位置平均距离 s 对平移分量进行归一化。
  - **M_6DoF+1,viewer（最终方案）**：q_i = Q_20(D̄_i)（输入视图深度的第20百分位数），然后对 E_i^(-1)E_j 的平移分量除以 q_i，完全依赖输入视图深度实现视角中心归一化。
- **深度预处理**：使用 off-the-shelf 深度估计器填充 ORB-SLAM 稀疏深度图的空洞，确保不同数据集的尺度定义一致（训练时使用，推理时不使用）。
- **SDS 锚定**：
  1. 用 DDIM 从均匀分布在方位角上的 k 个视点采样锚定视图 X̂_k。
  2. 在 SDS 优化时，对每个目标视图，选择最近的锚定视图（或输入视图）作为条件，而非始终使用输入视图。
  3. DDIM 采样不陷入 SDS 的模式坍塌，提供多样性锚点。

## 实验与结果
- **训练数据**：CO3D + RealEstate10K + ACID 的混合数据集，256×256 分辨率，各数据集均匀采样。
- **评估基准**：
  - **DTU**（零样本）：LPIPS = 0.380（SOTA，超越 DS-NeRF 0.649、PixelNeRF 0.535、SinNeRF 0.525 等专门在 DTU 微调的方法）。
  - **Mip-NeRF 360**（新基准，零样本）：LPIPS = 0.625，优于 Zero-1-to-3（0.667）和 PixelNeRF（0.718）。
  - **2D NVS 评测**：在 CO3D、RealEstate10K、ACID 的 held-out 子集上评估。
- **消融实验**：
  - 每种训练数据均可带来性能提升（Table 3）。
  - 相机条件表示从 M_Zero-1-to-3 逐步改进到 M_6DoF+1,viewer，在多个数据集上持续提升（Table 4）。
- **用户研究**：SDS 锚定在 Mip-NeRF 360 上获得更高 Realism（78%）、Creativity（82%）、Overall Preference（80%）。
- **可视化分析**：图9显示标准 SDS 产生单调背景，SDS 锚定产生更多样化的背景内容。

## 相关工作脉络
- **DreamFusion / SDS**：利用2D扩散模型进行3D蒸馏的开创性工作，本文在其基础上解决场景级 NVS 的多样性和尺度问题。
- **Zero-1-to-3**：单对象3D感知扩散模型，使用3DoF相机表示，本文将其扩展为6DoF+1并适配场景。
- **RealFusion**：单图像360度重建方法，主要针对单个对象，本文处理更复杂的野外场景。
- **PixelNeRF / SinNeRF / DietNeRF / NeRDi**：基于单/少视图的 NeRF 重建方法，本文在零样本设置下超越这些专门训练的方法。
- **DS-NeRF**：深度监督的 NeRF，本文对比其 DTU 结果。
- **GeNVS**：支持360度相机运动的扩散方法，但仅针对特定类别（如火栓），本文支持通用场景。

## 局限性与未来方向
- **PSNR/SSIM 与语义质量不相关**：如图7所示，对齐误差可能导致语义合理的预测得分更低，说明现有指标不能完全反映生成质量。
- **SDS 蒸馏计算成本高**：长运行时间使得 FID 等生成指标难以计算，限制了多样性评估。
- **深度估计仅在训练中使用**：推理阶段无法获得密集深度，可能影响开场景的泛化。
- **未探索文本/语义条件**：当前模型仅以输入图像为条件，未来可结合文本提示实现语义控制。

## 研究启发与可借鉴点
- **相机条件的归一化设计**：视角中心归一化（用输入视图深度决定全局尺度）是处理多源异构数据的有效思路，可迁移到其他多模态/多域融合任务。
- **SDS 锚定的多样性改进**：DDIM 预采样 + 最近邻条件切换的策略，为扩散模型3D蒸馏的多样性问题提供了简洁有效的解决方案。
- **基准迁移策略**：将 Mip-NeRF 360（原本用于多视图重建）改编为单图像 NVS 基准，展示了数据集重用的灵活思路。
- **消融的系统性设计**：从3DoF → 6DoF+1 → 相机归一化 → 视角中心归一化的渐进式消融，清晰展示了各模块的贡献，值得借鉴。

## 关键术语表
- **Score Distillation Sampling (SDS)**：利用预训练2D扩散模型的分数信息优化3D表示（如NeRF）的技术，由 DreamFusion 提出。
- **Zero-shot NVS**：在未见过的场景上直接进行新视图合成，无需目标域微调。
- **NeRF（Neural Radiance Fields）**：用连续神经网络表示3D场景辐射场的技术，支持任意视角渲染。
- **DDIM（Denoising Diffusion Implicit Models）**：加速扩散模型采样的确定性采样方法，可避免 SDS 的模式坍塌。
- **6DoF+1 相机表示**：使用相对位姿矩阵（6自由度）和视场角（1自由度）描述相机之间的关系。
- **SDS Anchoring**：先用 DDIM 采样锚定视图，再在 SDS 优化时用最近邻视图作为条件的技巧。
- **Viewer-centric Normalization**：用输入视图的深度统计量对相机平移进行归一化，解决尺度歧义。
- **Mip-NeRF 360**：包含360度环绕场景的基准数据集，本文首次将其用于单图像 NVS 评测。

## 可复现要素
- **数据集**：CO3D、RealEstate10K、ACID、DTU、Mip-NeRF 360（公开可用）。
- **代码/权重**：论文声明 Code and models are available at the project URL（链接在论文中，但 Markdown 中被截断为 "this url"）；基于 Zero-1-to-3 和 Threestudio 开源代码。
- **关键超参**：训练分辨率 256×256，DTU 评估分辨率 400×300；使用 ViTPred 深度估计器填充空洞；噪声调度按 Wang et al. [38] 退火。
- **基线对比**：Zero-1-to-3、PixelNeRF、DS-NeRF、SinNeRF、DietNeRF、NeRDi 均在同等混合数据集上重新训练对比。
