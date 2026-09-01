---
title: "Gaussian-Head-Avatar-Ultra-High-fidelity-Head-Avatar-via-Dyn"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Gaussian_Head_Avatar_Ultra_High-fidelity_Head_Avatar_via_Dynamic_Gaussians_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:44:09"
field: "3D 头像重建与神经渲染"
keywords: ["3D Gaussian Splatting", "Head Avatar", "Dynamic Deformation", "SDF Initialization", "Sparse View Reconstruction", "High-fidelity Rendering"]
innovations: ["全学习 MLP 变形场替代 LBS 建模夸张表情", "SDF+DMTet 几何引导初始化提升 3D 高斯训练稳定性", "距离相关权重解耦表情与姿态控制"]
benchmarks: ["NeRSemble", "HAvatar"]
---

# 论文速读：Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians

## 一句话总结
本文提出 Gaussian Head Avatar，一种基于可控 3D 高斯表示的高保真头像建模方法，通过全学习的表情条件变形场与几何引导初始化策略，在稀疏视角（16 相机）设置下实现 2K 分辨率的超高保真动态头像渲染，显著优于现有 NeRF-based 方法。

## 研究问题与动机
- **NeRF 在 2K 高分辨率下难以保留像素级细节**：现有 NeRF-based 头像方法在 2K 分辨率下仍难以准确合成皱纹、牙齿、眼睛等高频细节。
- **LBS 变形对夸张表情建模能力不足**：传统头像方法依赖线性混合蒙皮（LBS）+ blendshape（如 FLAME），无法准确表达夸张和细粒度表情。
- **3D 高斯表示的训练收敛困难**：作为离散化表征，3D 高斯的梯度无法在整个空间传播，随机初始化或简单模板初始化均易导致训练失败。
- **FLAME 模板无法覆盖长发和肩部**：直接用 FLAME 网格初始化高斯位置会丢失长发、肩部等非面部区域的高质量重建。

## 核心贡献（创新点）
1. **提出 Gaussian Head Avatar，首次将 3D Gaussian Splatting 应用于头像建模**：区别于 NeRF 的隐式体素表示，该方法利用显式离散高斯点进行可微分光栅化渲染，在 2K 分辨率下实现超高保真头像合成。
2. **设计全学习 MLP 变形场替代 LBS**：与 NeRSemble/INSTA 等依赖 3DMM 模板或 LBS 的方法本质不同，本文通过两个独立 MLP（表情/姿态）直接预测高斯点的位移，能准确建模夸张和细粒度表情。
3. **几何引导初始化策略（SDF + DMTet）**：不同于随机初始化或 FLAME 模板初始化，本文先用隐式 SDF 场 + DMTet 提取中性网格来稳定初始化高斯位置，约 10 分钟完成，显著改善收敛性。
4. **基于距离的表达式/姿态解耦权重函数**：通过高斯点到 3D landmarks 的距离动态分配 λ_exp 和 λ_pose，使面部区域主要由表情系数控制、颈部等非面部区域由姿态控制，避免单纯使用全局条件导致的抖动。

## 方法详解
**动态生成器框架**：构建中性高斯模型 $\{X_0, F_0, Q_0, S_0, A_0\}$（位置、特征向量、旋转、缩放、透明度），通过 MLP 基动态生成器 $\Phi$ 输入表情系数 $\theta$ 和姿态 $\beta$，输出动态属性 $\{X, C, Q, S, A\}$（公式 2）。

**位置变形（公式 3）**：分别用 $f_{def}^{exp}$ 和 $f_{def}^{pose}$ 预测表情位移和姿态位移，通过距离相关权重 $\lambda_{exp}(x)$ 和 $\lambda_{pose}(x)$ 加权融合，实现面部与头颈部的解耦控制。

**颜色预测（公式 4）**：不预设中性颜色，而是由 $f_{col}^{exp}$ 和 $f_{col}^{pose}$ 直接从特征向量 $F_0$ 预测表情依赖和姿态依赖的颜色分量。

**属性变化（公式 5）**：旋转 $Q$、缩放 $S$、透明度 $A$ 同样由两个 MLP 预测偏移量，捕捉表情引起的外观细微波动。

**训练损失（公式 8）**：总损失 = L1 重建损失 + $\lambda_{vgg}=0.1$ 的 VGG 感知损失 + $\lambda_{lr}=1$ 的低分辨率 L1 损失（约束 32 通道特征图的前 3 个通道为 RGB）。渲染流程为 512 分辨率特征图 → 超分辨率网络 Ψ → 2048 分辨率 RGB 图像。

**几何引导初始化（Sec 4.3）**：先训练隐式 SDF 网络 $f_{sdf}$（公式 9）及颜色/变形 MLP，通过 DMTet 提取中性网格；联合优化 3D landmarks 位姿与变形 MLP，使用 RGB 损失、silhouette loss（IOU）、landmark 损失、offset 正则、Laplacian 平滑等（公式 14）；最终将网格顶点赋给 $X_0$ 和 $F_0$，继承变形/颜色 MLP，旋转/缩放/透明度采用 Gaussian Splatting 原始初始化策略。

## 实验与结果
**数据集**：12 组数据（10 组 NeRSemble，2 组 HAvatar），16 相机约 120° 分布，2K 分辨率视频（NeRSemble 每身份 2500–3000 帧；HAvatar 每身份 3000 帧，8 相机 4K 后裁至 2K）。

**自 reenactment 定量结果（表 1）**：
| 方法 | PSNR ↑ | SSIM ↑ | LPIPS(512) ↓ | LPIPS(2K) ↓ | FID(2K) ↓ |
|---|---|---|---|---|---|
| NeRFBlendShape | 25.91 | 0.836 | 0.123 | 0.229 | 54.80 |
| NeRFace | 27.14 | 0.849 | 0.147 | 0.234 | 65.11 |
| HAvatar | 27.19 | 0.883 | 0.064 | 0.209 | 31.06 |
| Ours (w/o SR) | 27.82 | 0.887 | 0.080 | 0.202 | 45.50 |
| **Ours (full)** | **27.70** | **0.883** | **0.056** | **0.098** | **18.50** |

本文在 LPIPS(512) 和 FID(2K) 上大幅领先（LPIPS 从 0.064→0.056，FID 从 31.06→18.50），表明高频细节重建能力显著优于 SOTA。

**跨身份 reenactment**（表 2）：PSNR 27.58，SSIM 0.882，LPIPS 0.059，优于所有对比方法。

**新视角合成**（3D 一致性评估）：使用 8 相机训练、8 相机验证，本文方法在 PSNR/SSIM/LPIPS 上均最优。

**消融实验（表 3）**：
- FLAME-Init：PSNR 28.73，SSIM 0.875，LPIPS 0.123（长发/肩部模糊）
- Mesh-Deform（LBS 替代全学习变形）：PSNR 28.83，SSIM 0.874，LPIPS 0.116（夸张表情失真）
- **Ours（完整）**：PSNR 28.94，SSIM 0.876，LPIPS 0.108（最佳）

## 相关工作脉络
1. **NeRF-based 头像方法**（NeRFace [16]、HAvatar [71]、NeRFBlendShape [17]）：本文在表示层面替代 NeRF 为 3D Gaussian Splatting，利用其可微分光栅化在高频细节和渲染速度上的优势。
2. **INSTA [76]**：该工作采用 3DMM 网格+LBS 控制 NeRF 变形，本文与之本质区别在于放弃 LBS，改用全学习 MLP 变形场，能更准确建模 3DMM 模板无法捕捉的复杂/夸张表情。
3. **4D Gaussian 动态场景工作**（Dynamic 3D Gaussians [40]、4D Gaussian Splatting [59]、Deformable 3D Gaussians [66] 等）：这些方法可重建动态场景但**不支持动画控制**，本文首次将 3D Gaussian 应用于可动画头像建模。
4. **Implicit SDF-based 头像**（iMAvatar [73]、PointAvatar [74]）：本文与 iMAvatar 均利用隐式 SDF，但本文仅在初始化阶段使用 SDF 引导，主训练阶段采用显式 3D 高斯表示以获得更好的渲染质量。
5. **Gaussian Splatting [26]**：本文直接扩展自 Kerbl et al. 的静态 3D Gaussian Splatting，核心创新在于引入表达式条件动态生成器和几何引导初始化策略。

## 局限性与未来方向
- **口腔内部区域（舌头、牙齿）模糊**：论文自述因缺乏专门的跟踪方法导致口腔内部细节重建不佳。
- **长发区域偶有模糊**：虽然初始化策略改善了长发重建，但在极端表情下仍可能出现模糊。
- **未来方向**：论文建议引入专用跟踪方法来改善口腔内部重建；同时指出 3D Gaussian 表示有望成为头像重建的主流方向，暗示可扩展至全身 avatar 或更多动态人体建模场景。

## 研究启发与可借鉴点
1. **几何引导初始化策略的通用性**：利用 SDF+DMTet 提取网格来初始化离散点表示的思路，可迁移至其他基于点/高斯的动态场景建模任务，有效解决训练不收敛问题。
2. **距离相关权重解耦设计**：基于点到 landmarks 距离的 $\lambda_{exp}/\lambda_{pose}$ 权重函数设计巧妙且简洁，可推广至其他需要区分"局部变形区域"与"刚性运动区域"的人体动画建模任务。
3. **超分辨率网络与可微分渲染的结合**：先在低分辨率（512）下进行 Gaussian 渲染，再经超分辨率网络生成 2K 图像，这一设计在保证细节恢复的同时降低了训练显存开销，值得在其他高分辨率 NeRF/Gaussian 任务中参考。
4. **全学习变形场替代 LBS 的可行性验证**：本文证明了 MLP 变形场在头像动画中的有效性，后续可将此思路推广至全身 avatar（如结合 SMPL 的人体建模）。
5. **伦理风险意识**：论文明确讨论了方法可能被用于生成虚假肖像视频的风险，提醒研究者在推进技术的同时需关注深度伪造的伦理治理。

## 关键术语表
**3D Gaussian Splatting**：一种基于离散 3D 高斯椭球的显式场景表示与可微分光栅化渲染技术，相比 NeRF 具有更高的渲染速度和细节重建能力。

**Signed Distance Function (SDF)**：隐式几何表示方法，通过符号距离场描述点到物体表面的有符号距离，零水平集即为物体表面。

**Deep Marching Tetrahedra (DMTet)**：一种可微分的网格提取算法，将 SDF 值嵌入四面体网格中通过插值提取等值面，生成高质量三角网格。

**Linear Blend Skinning (LBS)**：传统角色动画中常用的蒙皮变形技术，通过骨骼权重线性混合顶点位移，对夸张表情的建模能力有限。

**FLAME**：一种参数化 3D 人脸形状与表情模型，广泛用于头像重建研究，但拓扑固定无法表达长发等复杂结构。

**NeRSemble**：多视角头部头像重建数据集，包含 10 个身份的 16 相机同步视频，是本文的主要实验基准。

**HAvatar**：另一个多视角头像数据集，提供 8 相机 4K 分辨率视频，本文额外使用了其中的 2 个身份数据。

**Expression Coefficients (θ)**：通过 BFM 拟合得到的表情参数，用于驱动头像的动态变形，是本文动态生成器的核心条件输入。

## 可复现要素
- **数据集**：NeRSemble（公开）+ HAvatar（部分公开）；数据预处理需背景去除 + BFM 拟合获取 68 个 3D landmarks 与表情系数。
- **代码/权重**：项目页面 https://yuelangx.github.io/gaussianheadavatar，论文未明确声明代码开源状态。
- **关键超参**：$\lambda_{exp}$ 距离阈值 $t_1=0.15, t_2=0.25$（head 长度归一化为 1）；损失权重 $\lambda_{vgg}=0.1, \lambda_{lr}=1, \lambda_{sil}=0.1, \lambda_{def}=1, \lambda_{offset}=0.01, \lambda_{lmk}=0.1, \lambda_{lap}=100$。
- **初始化时间**：约 10 分钟完成 SDF 引导的初始化流程。
- **分辨率**：训练时渲染 512×512 特征图，经超分辨率网络生成 2048×2048（2K）RGB 图像。
