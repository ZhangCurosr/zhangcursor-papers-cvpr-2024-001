---
title: "DiffInDScene-Diffusion-based-High-Quality-3D-Indoor-Scene-Ge"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ju_DiffInDScene_Diffusion-based_High-Quality_3D_Indoor_Scene_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:41:37"
field: "3D视觉生成"
keywords: ["3D室内场景生成", "扩散模型", "TSDF", "稀疏卷积", "级联生成", "多视图立体refine"]
innovations: ["稀疏级联扩散节省两个数量级计算开销", "随机TSDF融合算法保留生成多样性", "多尺度VQGAN occupancy latent引导几何生成"]
benchmarks: ["3D-FRONT", "ScanNet", "NeuralRecon"]
---

# 论文速读：DiffInDScene: Diffusion-based High-Quality 3D Indoor Scene Generation

## 一句话总结
本文提出DiffInDScene，首个基于扩散模型的高质量房间级3D室内场景生成框架，通过稀疏级联扩散与随机TSDF融合算法，实现从零生成完整室内几何结构，并能显著提升多视图立体(MVS)重建质量。

## 研究问题与动机
- **核心问题**：现有扩散模型在图像和物体级3D生成表现优异，但房间级3D场景生成因计算成本过高而尚未被应用。
- **现有方法不足**：
  1. Text2Room等方法依赖2D图像生成再迭代重建，结果碎片化且扭曲严重；
  2. 传统3D扩散模型仅关注物体级生成，扩展到场景级面临指数级资源消耗；
  3. 3D重建方法（如NeuralRecon）在迭代融合过程中丢失大量网格细节。
- **关键洞察**：InstantNGP研究表明，普通3D场景中仅有约2.57%体素包含有效信息，利用稀疏性可大幅降低计算开销。

## 核心贡献（创新点）
1. **稀疏级联扩散框架**：设计多阶段稀疏扩散管道，仅在被占用体素上进行去噪，节省两个数量级的计算与内存开销（训练时仅占用约16GB显存，而非24GB）。
2. **多尺度Patch-VQGAN occupancy编码**：提出层级occupancy embedding生成方案，通过向量量化提供特征引导，使diffusion过程能持续优化结构合理性。
3. **随机TSDF融合算法**：借鉴KinectFusion增量对齐思想，提出随机选择而非平均融合策略，保留样本分布方差，解决局部裁剪生成的不一致问题。
4. **通用 refine 模块**：可将DiffInDScene作为后处理模块，提升NeuralRecon等MVS方法的重建质量，用户研究中甚至超越ScanNet地面真值网格。

## 方法详解
### 3阶段级联扩散流程
1. **Stage 1**：生成最低分辨率occupancy latent $z^{(1)}$，输入高斯噪声+边界框mask $M_{z_T^{(1)}}$，通过扩散模型$\mathcal{D}_1$去噪得到$z_0^{(1)}$。
2. **Stage 2**：条件生成更高reso的$z^{(2)}$，以$z^{(1)}$为条件，通过$\mathcal{D}_2$去噪。
3. **Stage 3**：最终TSDF体积生成，条件为$z^{(1)}, z^{(2)}$，对TSDF体素进行局部裁剪训练、随机融合推理。

### 稀疏扩散模型
- 采用DDPM范式，但$\epsilon_\theta$仅在occupied voxels上操作（使用TorchSparse稀疏卷积）。
- 损失函数：$L_{diff} = E[M \odot ||\epsilon - \epsilon_\theta(v_t, y, M, t)||_2^2]$。

### 多尺度VQGAN设计
- 编码器$E$将TSDF$x$编码为$(z^{(1)}, z^{(2)})$；
- 解码器$G_1, G_2$将latent recon为occupancy mask；
- 训练损失：$L = L_{rec} + \lambda_1 L_{vq} + \lambda_2 L_{GAN}$，其中$L_{rec}$包含BCE(occupancy) + $L_1$(TSDF值)。

### 随机TSDF融合
- 推理时将全局空间划分为$K$个重叠3D crop；
- 每个timestep，对voxel p，从覆盖它的crops中**均匀随机选择一个**更新全局TSDF：$x_t(p) = x_t^k(p), k \sim U(\mathcal{G}(p))$；
- 避免平均融合导致的分布方差衰减，保持生成质量与全局一致性。

## 实验与结果
### 数据集
- **生成任务**：3D-FRONT数据集（6813间 furnished houses），选取5913间尺寸$<512\times512\times128$、voxel size 0.04m。
- **Refine任务**：ScanNet数据集（1613场景），使用NeuralRecon作为MVS基线。

### 关键结果
| 指标 | Text2Room | Text2Room+Poisson | Ours |
|------|-----------|-------------------|------|
| Aspect Ratio mean↑ | 0.416 | 0.443 | **0.473** |
| Circularity mean↑ | 0.674 | 0.709 | **0.781** |
| Shape Regularity mean↑ | 0.716 | 0.730 | **0.816** |
| Completeness↑(用户) | 2.532 | 3.228 | **4.856** |
| Perceptual↑(用户) | 2.472 | 2.812 | **4.836** |

### Refine任务（ScanNet）
- Normal error <30° ratio：**59.27%** vs NeuralRecon 52.88%、GT 51.67%；
- 用户研究中NR+Ours总得分87.14，超过GT的85.62（细节/紧凑平面/锐利边缘均显著领先）。

### 资源对比
- Sparse: 0.008 TFLOPs、161.5M参数、batch=1时仅需11.8GB显存；
- Dense: 3.290 TFLOPs，batch=2即OOM，节省约**400倍**计算量。

## 相关工作脉络
1. **Text2Room [15]**：最相关工作，通过2D文本到图像模型生成图像序列再重建，本文直接3D生成，解决碎片化问题。
2. **DiffusionSDF [37]** / **Neural Wavelet Diffusion [16]**：对象级3D扩散生成，本文扩展至房间级场景。
3. **NeuralRecon [41]**：实时单目视频3D重建SOTA，本文以其输出作为occupancy条件进行refine。
4. **KinectFusion [17]**：经典TSDF融合算法，启发本文随机融合策略的设计灵感。
5. **DreamFusion [27]** / **Text2Mesh**：文本到3D生成，本文专注室内场景几何生成而非外观纹理。
6. **Patch-VQGAN [7]**：用于latent diffusion的视频生成，本文借鉴其multi-scale quantization思想设计occupancy encoder。

## 局限性与未来方向
- **天花板移除限制可视化**：文中大量展示需移除天花板，说明当前方法对封闭空间（如房间顶面）生成能力待验证。
- **未处理动态/可移动物体**：仅生成固定几何结构，家具等可配置物体未纳入生成流程。
- **条件生成能力有限**：目前主要为无条件生成，Sketch-to-Scene仅在补充材料展示，文本条件生成未深入。
- **分辨率上限**：当前最高$512\times512\times128$，更大场景需进一步扩展。
- **纹理生成依赖外部模块**：纹理由DreamSpace单独生成，未端到端联合优化。

## 研究启发与可借鉴点
1. **稀疏性利用范式**：任何3D扩散模型均可借鉴"仅在被占用区域去噪"的思路，大幅降低显存与算力需求。
2. **随机融合替代平均融合**：在局部生成+全局整合的场景中，随机选择优于简单平均，可保留生成分布的多样性。
3. **级联latent diffusion + VQGAN结构**：多层occupancy embedding可作为通用先验引导，该设计可迁移至其他3D生成任务。
4. **Refine作为post-processing模块**：将生成模型用于改进现有重建结果，是一条值得探索的实用路径。
5. **用户研究设计**：结合客观mesh质量指标与主观用户评分（40-51人），评估更全面可信。

## 关键术语表
- **TSDF (Truncated Signed Distance Function)**：截断符号距离函数，用于表示3D表面几何的隐式场，正值在表面外、负值在表面内、零点在表面上。
- **Diffusion Model**：扩散模型，通过逐步添加噪声再学习去噪的生成模型，DDPM为经典范式。
- **Sparse Convolution**：稀疏卷积，仅在非零体素上执行的3D卷积，由TorchSparse等库高效实现。
- **Patch-VQGAN**：基于patch的向量量化GAN，将连续latent离散化到codebook，用于压缩3Doccupancy表示。
- **Stochastic TSDF Fusion**：随机TSDF融合，在重叠区域随机选择一个crop的TSDF值更新全局，避免平均导致的方差衰减。
- **NeuralRecon**：实时连贯3D重建方法，从单目视频恢复TSDF体积，本文以其作为refine任务输入。
- **Aspect Ratio / Circularity / Shape Regularity**：网格质量三维度量，分别评估三角形长宽比、圆度和形状规则性。

## 可复现要素
- **数据集**：3D-FRONT（阿里巴巴开源）、ScanNet（公开，需申请）。
- **代码**：论文未提供开源链接（CVPR 2024常见情况，需关注作者homepage后续更新）。
- **关键超参**：$T=1000$ timestep、crop size $96\times96\times96$、voxel size 0.04m、max scene size $512\times512\times128$。
- **训练环境**：RTX 3090 GPU（24GB显存）。
