---
title: "HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:46:08"
field: "文本驱动3D生成"
keywords: ["text-to-3D", "human generation", "Gaussian Splatting", "score distillation sampling", "3D human avatar"]
innovations: ["Structure-Aware SDS: 基于SMPL-X初始化与双分支RGB-深度扩散模型联合优化外观与几何", "Annealed Negative Prompt Guidance: 分解SDS并利用退火负提示在低CFG尺度下避免过饱和伪影", "Size-Conditioned Gaussian Pruning: 基于高斯缩放因子剪除浮动伪影以提升几何平滑性"]
benchmarks: ["User Study (MOS on Texture, Geometry, Text Alignment)", "CLIP Score", "Aesthetic Score", "HPSv2"]
---

# 论文速读：HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

## 一句话总结
本文提出 HumanGaussian，一个高效且有效的文本驱动 3D 人类生成框架，将 3D Gaussian Splatting (3DGS) 适配到文本生成领域，通过结构感知的 Score Distillation Sampling (SDS) 和退火负提示引导，在保持真实外观的同时生成细粒度几何结构的 3D 人体。

## 研究问题与动机
- **核心问题**：如何从文本提示高效生成具有细粒度几何细节和真实外观的高质量 3D 人体？
- **现有方法不足**：
  1. 传统 mesh/NeRF 方法在细节建模（如配饰、皱纹）与效率之间难以兼顾：mesh 方法难以刻画复杂拓扑，NeRF 渲染高分辨率结果耗时耗显存。
  2. 基于 SDS 的文本到 3D 方法（如 DreamFusion）依赖 3DGS 时面临挑战：① 3DGS 初始化缺乏人体结构先验，导致稀疏梯度阻碍几何与外观优化；② 原始 SDS 需极大 CFG 尺度（如 100）以对齐文本，但会导致过饱和伪影；③ SDS 损失的高方差使基于梯度的密度控制不稳定，产生浮动伪影。
  3. 已有 3D 人体生成工作（如 DreamHuman、TADA）多采用两阶段管道或隐式表示，难以高效捕捉细粒度人体细节。

## 核心贡献（创新点）
1. **提出 Structure-Aware SDS**：通过 SMPL-X 网格初始化高斯点并扩展 Stable Diffusion 同时去噪 RGB 和深度，实现外观与几何的联合优化。与仅使用 RGB 或深度单一模态扩散模型的工作（如 GaussianDreamer、HumanNorm）本质不同。
2. **设计 Annealed Negative Prompt Guidance**：将 SDS 分解为生成分数与分类器分数，利用退火负提示替代空条件，在较低 CFG 尺度（7.5）下避免过饱和伪影。与 DreamFusion 依赖大 CFG 尺度（τ=100）的 naive 方案不同。
3. **引入 Size-Conditioned Gaussian Pruning**：在退火负提示基础上，于剪枝阶段基于高斯缩放因子消除低透明度浮动伪影。与直接使用梯度进行密度控制的方法不同，该策略专门针对 SDS 高方差导致的平滑伪影问题。

## 方法详解
- **3D Gaussian Splatting 表示**：场景由一组各向同性高斯函数描述，参数包括中心位置 $\mu$、协方差 $\Sigma$（可分解为缩放 $s$ 和旋转 $q$）、颜色 $c$（球谐系数）、不透明度 $\alpha$。渲染通过投影到相机平面、分块排序与 $\alpha$-混合实现。
- **SMPL-X 先验初始化**：从 SMPL-X 网格表面均匀采样 100k 个点作为初始高斯中心，设置单位缩放、默认颜色和零旋转，确保初始分布贴合人体表面结构。
- **Texture-Structure Joint Model**：基于 SD 2.0 微调，通过复制 UNet 的 conv in、首 DownBlock、末 UpBlock 和 conv out 层，同时学习 RGB 图像与深度图的条件分布（输入含姿态骨架 $p$ 与文本 $y$），使用 v-prediction 训练。
- **双分支 SDS 优化**：
  - 深度渲染：$\mathbf{d}(p) = \sum_{i \in \mathcal{N}} d_i \sigma_i \prod_{j=1}^{i-1}(1-\sigma_j)$，其中 $d_i$ 为第 $i$ 个高斯中心的投影深度。
  - 联合损失：$\nabla_\theta \mathcal{L}_{\text{SDS}} = \lambda_1 \mathbb{E}_{\epsilon_x,t}[w_t(\epsilon_\phi(\mathbf{x}_t;\mathbf{p},y)-\epsilon_x)\frac{\partial\mathbf{x}}{\partial\theta}] + \lambda_2 \mathbb{E}_{\epsilon_d,t}[w_t(\epsilon_\phi(\mathbf{d}_t;\mathbf{p},y)-\epsilon_d)\frac{\partial\mathbf{d}}{\partial\theta}]$，其中 $\lambda_1=\lambda_2=0.5$。
- **退火负提示引导**：将 SDS 分解为 $\delta_g$（生成分数）与 $\delta_c$（分类器分数），使用负提示 $y_{\text{neg}}$ 构造 $\delta_{nc} = \delta_c - \delta_n$，损失为 $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{\epsilon,t}[w_t(\tau_1\delta_c - \tau_2\delta_n)\frac{\partial\mathbf{x}}{\partial\theta}]$，其中 $\tau_1=7.5$，$\tau_2$ 从 1.0 线性衰减至 0（timestep 200 后）。
- **Size-Conditioned Pruning**：在自适应密度控制（第 300–2100 步，每 300 步一次）后，从第 2400 步开始执行仅剪枝阶段（每 300 步一次），移除缩放因子超过阈值（0.008）的高斯实例。

## 实验与结果
- **数据集与评估**：使用自定义提示进行定性/定量评估，定量指标包括 CLIP Score、Aesthetic Score、HPSv2；用户研究邀请 17 名参与者从纹理质量、几何质量、文本对齐三个维度评分（1–5 分）。
- **基线方法**：通用文本到 3D 方法（DreamGaussian、GaussianDreamer）、专用 3D 人体生成方法（TADA、DreamHuman）。
- **主要结果**：
  - 用户研究（表 1）：HumanGaussian 在纹理质量（4.24）、几何质量（3.88）、文本对齐（4.71）均最高，优于 TADA（3.76/3.53/4.35）和 DreamHuman（3.41/3.65/4.24）。
  - 定量指标（表 2）：HumanGaussian 的 CLIP（30.82）、Aes.（6.436）、HPSv2（0.262）均优于所有基线。
  - 训练效率：单张 NVIDIA A100（40GB）约 1 小时完成优化（3600 步迭代）。
- **消融实验**：验证了 SMPL-X 初始化、退火负提示、双分支 SDS、尺寸条件剪枝各环节的有效性（图 4）。

## 相关工作脉络
1. **DreamFusion (Poole et al., 2022)**：开创 SDS 用于文本到 3D，但依赖 NeRF，训练慢且细节不足。本文将其适配到 3DGS 以提升效率。
2. **GaussianDreamer (Yi et al., 2023)、DreamGaussian (Tang et al., 2023)**：并行工作探索 3DGS 用于文本到 3D，但未针对人体结构进行专门设计。本文引入 SMPL-X 先验与双分支 SDS。
3. **DreamHuman (Kolotouros et al., 2023)**：基于 NeRF 与 imGHUM 的文本到可动画 3D 人体，生成质量高但计算成本高。本文用 3DGS 替代以提升效率。
4. **TADA (Liao et al., 2023)**：通过位移场变形 SMPL-X 并优化 UV 贴图，采用两阶段管道，难以高效捕获细粒度细节。本文端到端优化 3DGS。
5. **HumanNorm (Huang et al., 2023)**：同期工作，微调文本到深度/法线模型施加结构约束，但两者预测未对齐且使用 DMTet 表示。本文统一模型同步去噪 RGB 与深度，使用 3DGS。
6. **HyperHuman (Liu et al., 2023)**：提出结构扩散模型，但侧重图像生成而非 3D 优化。本文将其思想延伸到 SDS 框架中。

## 局限性与未来方向
- **局限性**：
  1. 现有文本到图像模型在手部、足部生成上能力有限，导致这些部位有时无法忠实渲染。
  2. 初始化依赖 SMPL-X，可能限制对非标准人体形态（如极端姿势、特殊体型）的泛化。
- **未来方向**：
  1. 改进手部/足部生成质量，可能结合专用子模型或增强扩散先验。
  2. 探索更灵活的人体先验（如可变形状参数）以适应多样化生成需求。
  3. 扩展到其他动画角色或动物生成。

## 研究启发与可借鉴点
1. **结构先验在 3DGS 初始化中的应用**：将参数化人体模型（SMPL-X）网格采样作为高斯初始化，可有效约束初始分布，提升优化稳定性与细节生成能力。可迁移到 animal/character generation。
2. **双分支 SDS 联合优化外观与几何**：同步去噪 RGB 与深度，利用多模态扩散先验同时指导纹理与结构学习，避免了单一模态监督的局限。可推广到服装、配饰等细粒度生成。
3. **退火负提示引导降低 CFG 尺度依赖**：通过分解 SDS 为生成/分类器分数，并退火负提示权重，能在低 CFG 尺度下获得逼真效果，减少过饱和伪影。适用于其他文本到 3D 任务。
4. **尺寸条件剪枝消除浮动伪影**：针对 SDS 高方差导致的模糊伪影，设计基于高斯缩放因子的剪枝策略，而非依赖梯度阈值。该思想可用于其他基于 3DGS 的生成任务。
5. **端到端高效管道**：在单张 A100 上 1 小时内完成高质量 3D 人体生成，证明 3DGS 在文本到 3D 中的效率优势，为后续实时生成应用奠定基础。

## 关键术语表
- **Score Distillation Sampling (SDS)**：一种将预训练 2D 扩散模型先验蒸馏到 3D 表示中的优化技术，通过随机噪声估计梯度更新 3D 参数。
- **3D Gaussian Splatting (3DGS)**：一种显式神经表示方法，用一组各向同性高斯函数表征场景，支持快速可微渲染与实时可视化。
- **SMPL-X**：一个参数化人体模型，包含身体、手部和面部的拓扑结构，通过形状、姿态、表情参数生成 3D 网格。
- **Classifier-Free Guidance (CFG)**：在扩散模型推理中，通过结合条件与无条件预测来增强文本对齐程度的技术，通常通过调节尺度参数控制。
- **Negative Prompt Guidance**：在文本到图像生成中，使用负面描述词引导模型避免生成特定属性，本文将其应用于 3D 优化以减少伪影。
- **V-prediction**：扩散模型的一种训练目标，预测噪声与数据的加权组合（$\mathbf{v}=\alpha_t\epsilon-\sigma_t\mathbf{x}$），有时比直接预测噪声更稳定。
- **Alpha-blending**：3DGS 渲染过程中，沿视线顺序将高斯不透明度叠加以计算最终像素颜色的操作。
- **Spherical Harmonics (SH)**：用于表示高斯颜色随视角变化的球面函数，本文使用 0 阶 SH 近似均匀颜色。

## 可复现要素
- **数据集**：深度图标注使用 LAION-5B 数据集，通过 MiDaS 生成深度标签。未提及公开的基准测试集。
- **代码开源**：项目页面（https://alvinliu0.github.io/projects/HumanGaussian）可能提供代码，论文基于 ThreeStudio 框架开发。
- **权重开源**：未明确说明，但使用了预训练的 SD 2.0 模型。
- **关键超参数**：
  - 高斯初始化数量：100k
  - 训练迭代次数：3600 步
  - 密度控制区间：300–2100 步（每 300 步一次）
  - 仅剪枝阶段：2400–3300 步（每 300 步一次），缩放阈值 0.008
  - 学习率：位置 5e-5，缩放 1e-3，旋转 1e-2，颜色 1.25e-2，不透明度 1e-2
  - CFG 尺度 τ₁=7.5，负提示权重 τ₂ 从 1.0 衰减至 0（timestep 200）
  - RGB 与深度损失权重 λ₁=λ₂=0.5
  - 训练分辨率：1024，batch size=8
  - 硬件：单张 NVIDIA A100 (40GB)
