---
title: "ZeroNVS-Zero-Shot-360-Degree-View-Synthesis-from-a-Single-Im"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sargent_ZeroNVS_Zero-Shot_360-Degree_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:52:59"
field: "单图新视角合成与3D生成"
keywords: ["novel view synthesis", "360-degree view synthesis", "score distillation sampling", "diffusion model", "zero-shot 3D", "scene reconstruction", "camera parameterization"]
innovations: ["提出6DoF+1相机条件参数化与viewer-centric深度规范化，消除混合数据源的尺度歧义", "设计SDS anchoring机制，通过DDIM锚点视图轮换条件提升背景多样性", "在零样本设置下于DTU和Mip-NeRF 360基准上取得SOTA，并提出后者作为新评测基准"]
benchmarks: ["DTU", "Mip-NeRF 360", "CO3D", "RealEstate10K", "ACID"]
---

# 论文速读：ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image

## 一句话总结
本文提出 ZeroNVS，一种面向野外复杂场景的零样本 360 度新视角合成方法，通过改进的相机条件参数化与深度规范化方案训练扩散模型，并引入 SDS anchoring 机制缓解 Score Distillation Sampling 在背景生成中的模式坍塌问题，在 DTU 和 Mip-NeRF 360 数据集上均取得零样本设置下的 SOTA 结果。

## 研究问题与动机
- **场景级 NVS 的数据与泛化挑战**：现有单图新视角合成方法主要针对 Masked 背景的单物体场景设计（如 Objaverse-XL），而真实场景缺乏大规模带几何、纹理和相机参数的标注数据，难以直接复用物体-centric 方法。
- **相机表示在场景任务中的表达力不足**：Zero-1-to-3 等物体方法使用 3DoF（俯仰、方位、半径）相机参数，无法描述任意位姿的野外相机，也无法处理复杂背景与多尺度场景。
- **SDS 蒸馏的场景多样性瓶颈**：SDS 在长距离视角变化（如 180°）时倾向于生成单调/灰度背景，导致生成结果的多样性和真实感下降。
- **混合数据源中的尺度歧义**：不同数据源（CO3D、RealEstate10K、ACID）使用不同采集设置和 SFM/SLAM 重建方法，导致尺度不统一，需要设计适配的规范化策略。

## 核心贡献（创新点）
1. **提出 6DoF+1 相机条件参数化与多层级规范化方案**：在相对位姿基础上引入视场角（FoV），并通过深度图分位数与单视图尺度估计消除不同数据源间的尺度歧义。
2. **设计 SDS anchoring 机制以提升背景多样性**：先通过 DDIM 采样多个伪真值视角作为锚点，再用最近视角（输入或采样视角）替代原始输入作为 SDS 条件，缓解模式坍塌。
3. **在混合多场景数据（CO3D + RealEstate10K + ACID）上训练单一扩散模型**：实现零样本 360 度场景新视角合成，无需目标数据集微调。
4. **建立 Mip-NeRF 360 为单图 NVS 新基准并公布最强零样本结果**：DTU 零样本 LPIPS 0.380 超越所有微调基线；Mip-NeRF 360 零样本 LPIPS 0.625 为当前最佳。
5. **系统性消融验证相机表示与数据混合的有效性**：逐层对比 3DoF、6DoF+1、camera-normalized、depth-aggregated、viewer-centric 五类表示，证明渐进改进的价值。

## 方法详解
- **条件嵌入函数 $\mathbf{M}(D,f,E,i,j)$**：将深度图 $D$、视场角 $f$、外参 $E$ 与输入/目标视角索引 $(i,j)$ 映射为条件向量，输入扩散模型 $p_\theta(X_j | X_i, \mathbf{M})$。
- **相机表示演进**：
  - $\mathbf{M}_{\text{Zero-1-to-3}} = \mathbf{P}(E_i) - \mathbf{P}(E_j)$：3DoF 投影差，适用于中心对齐单物体。
  - $\mathbf{M}_{\text{6DoF+1}} = [E_i^{-1} E_j, f]$：相对 SE(3) 位姿 + FoV，对刚体变换不变但对尺度敏感。
  - $\mathbf{M}_{\text{6DoF+1, norm}}$：以相机位置均值为中心、平均范数 $s$ 为尺度进行归一化。
  - $\mathbf{M}_{\text{6DoF+1, agg}}$：借鉴 Stereo Magnification，取每张深度图 5-th 分位数后聚合取 10-th 分位数作为场景尺度 $q$，缩放平移分量。
  - $\mathbf{M}_{\text{6DoF+1, viewer}}$：使用单输入视图的深度图 $\bar{D}_i$（经 ViT-Depth 补孔）的第 20-th 分位数 $q_i$ 作为尺度，避免跨视角聚合引入的歧义。
- **SDS anchoring**：
  1. 用 DDIM 在均匀分布的方位角上采样 $k$ 个新视角 $\hat{X}_j$。
  2. 对每个优化步，选取与目标视角空间最近的锚点图像（输入或 DDIM 采样结果）作为扩散模型的条件。
  3. 标准 SDS 始终以输入图为条件，导致所有视角趋向同一分布；SDS anchoring 使不同视角得到多样化引导。
- **训练数据**：CO3D（object-centric）、RealEstate10K（indoor/outdoor）、ACID（natural scenes），三源均匀采样混合训练，分辨率 256×256。
- **蒸馏流程**：基于 Threestudio，使用 Mip-NeRF 360 与 Instant-NGP 结合的 NeRF 表示，噪声调度按 ProlificDreamer 退火。

## 实验与结果
- **数据集与基线**：DTU（25 场景）、Mip-NeRF 360（8 场景）、CO3D/RealEstate10K/ACID 各自 held-out 子集用于 2D 评估；基线包括 DS-NeRF、PixelNeRF、SinNeRF、DietNeRF、NeRDi、Zero-1-to-3。
- **DTU 零样本结果**（Table 1）：ZeroNVS 获得 LPIPS = 0.380，显著优于所有在 DTU 上微调的方法（次优 NeRDi 为 0.421）；PSNR = 13.55，SSIM = 0.469。
- **Mip-NeRF 360 零样本结果**（Table 2）：LPIPS = 0.625，超越 Zero-1-to-3（0.667）与像素级重训练的 PixelNeRF（0.718）。
- **数据混合消融**（Table 3）：移除任一数据源均导致 DTU LPIPS 上升（性能下降），全量最优。
- **相机表示消融**（Table 4）：从 3DoF 到 viewer-centric 逐层提升，6DoF+1, viewer 在 CO3D/RealEstate/ACID/DTU 上均达到或接近最佳。
- **用户研究**（Mip-NeRF 360，21 人）：SDS anchoring 在 Realism（78%）、Creativity（82%）、Overall Preference（80%）三项均显著优于标准 SDS。
- **可视化分析**（Figure 5）：viewer 表示降低尺度歧义带来的方差；Figure 9 展示标准 SDS 产生单调背景而 SDS anchoring 生成多样化场景。

## 相关工作脉络
1. **Zero-1-to-3 (CVPR 2023)**：3DoF 相机表示的单物体 NVS 方法，是本文最直接的前序工作；本文将其扩展至任意位姿场景并解决尺度歧义。
2. **DreamFusion (ICLR 2023)**：提出 SDS 用于 text-to-3D；本文沿用 SDS 蒸馏框架但针对场景背景多样性缺陷提出 SDS anchoring 改进。
3. **PixelNeRF / DietNeRF / NeRDi**：单图/少图 NeRF 重建方法；本文的 ZeroNVS 在零样本设置下超越这些在目标数据集上微调的方法。
4. **RealFusion (CVPR 2023)**：单图 360° 场景重建；本文指出其缺乏对复杂背景的多样性建模，且训练数据规模有限。
5. **GeNVS (ICCV 2023)**：支持 360° 运动的扩散 NVS，但仅限于消防栓等特定类别；本文支持通用场景类别。
6. **3DGP / VQ3D / IVID**：ImageNet 上的 3D 生成方法，相机运动范围有限；本文在混合真实场景数据上训练，支持完整 360° 视角变化。

## 局限性与未来方向
- **计算成本较高**：SDS 蒸馏本身耗时较长，SDS anchoring 需额外 DDIM 采样步骤，生成流程仍未达到实时。
- **依赖预训练深度估计器**：viewer-centric 规范化需在训练时利用 ViT-Depth 补孔，推理时虽不使用但间接影响训练数据构建。
- **DTU 场景的语义复杂性有限**：DTU 多为室内静态场景，Mip-NeRF 360 更具挑战性但仍不足以覆盖动态/非刚性场景。
- **缺乏显式 3D 几何一致性度量**：主要依赖 LPIPS/PSNR/SSIM，未评估几何误差（如 Chamfer Distance）或大角度 view consistency。
- **潜在方向**：扩展至动态场景、结合显式几何先验（如平面/骨架）、探索更高效的多视角蒸馏策略。

## 研究启发与可借鉴点
1. **多数据源混合训练 + 规范化对齐**：将 CO3D、RealEstate、ACID 等不同采集协议的数据混合时，通过 viewer-centric 深度分位数统一尺度，为多源 3D 生成提供了可复用的数据对齐范式。
2. **SDS anchoring 的思路可迁移**：对于任何基于 SDS 的场景级生成任务，当背景多样性成为瓶颈时，DDIM 采样 + 最近邻条件替换是一种低成本的改进策略。
3. **相机表示设计的渐进式消融方法论**：从 3DoF → 6DoF+1 → camera-normalized → depth-aggregated → viewer-centric 的五步递进，展示了条件表示设计的系统性验证路径。
4. **Mip-NeRF 360 作为 NVS 新基准**：本文为单图 NVS 引入了更全面的 360° 评估基准，后续工作可直接对标此 benchmark。
5. **对 PSNR/SSIM 的批判性使用**：论文明确指出 PSNR/SSIM 在 NVS 中与非对齐敏感性问题，建议以 LPIPS 为主指标，这一观点值得推广到其它生成式 3D 评估中。

## 关键术语表
**Score Distillation Sampling (SDS)**：利用预训练 2D 扩散模型对 NeRF 渲染图像的梯度指导 3D 几何与外观优化的蒸馏技术。
**6DoF+1 相机表示**：包含 6 自由度相对位姿（旋转+平移）加视场角（FoV）的相机条件编码，适用于任意位姿的场景。
**SDS Anchoring**：先用 DDIM 采样多个视角作为锚点，再用最近锚点替代输入作为 SDS 条件，以提升背景多样性。
**Viewer-Centric Normalization**：以单输入视图深度图的第 20-th 分位数作为全局尺度，避免跨视图聚合带来的歧义。
**Mip-NeRF 360**：包含 8 个无界场景的基准数据集，支持 360° 视角变化，被本文引入为单图 NVS 评估基准。
**In-the-wild Scenes**：指具有复杂背景、非中心对齐、尺度不定的真实世界场景，区别于 Masked 背景的物体-centric 场景。
**ViT-Depth**：基于 Vision Transformer 的单目深度估计器，用于在训练阶段填充深度图空洞。
**NeRF Distillation**：将 2D 扩散模型的生成先验通过梯度蒸馏到 3D NeRF 表示中的过程。

## 可复现要素
- **数据集**：CO3D、RealEstate10K、ACID、DTU、Mip-NeRF 360（均公开可用）
- **代码/权重**：论文声明 Code and models are available at the project URL（链接见论文首页）
- **训练分辨率**：256×256（DTU 评估用 400×300）
- **初始化**：Zero-1-to-3-XL pretrained weights，替换 conditioning module
- **蒸馏框架**：Threestudio + 自定义 Mip-NeRF 360 / Instant-NGP 混合 NeRF
- **深度估计器**：ViT-Depth（仅训练阶段补孔用，推理不使用）
- **噪声调度**：annealed schedule（参考 ProlificDreamer）
- **超参细节**：论文注明见 supplementary material
