---
title: "Artist-Friendly-Relightable-and-Animatable-Neural-Heads"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xu_Artist-Friendly_Relightable_and_Animatable_Neural_Heads_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:02:32"
field: "神经渲染与数字人脸重建"
keywords: ["Neural Head Avatar", "Relighting", "Animatable Neural Rendering", "Mixture of Volumetric Primitives", "Nearfield Illumination", "OLAT Capture"]
innovations: ["引入逐基元局部光照/视角方向条件实现近场重打光与近场视角效果", "纯几何驱动的表达编码器支持推理期 unseen expression 艺术控制", "仅用 32 LED 灯条的低成本 OLAT 动态采集方案"]
benchmarks: ["Novel Light Directions", "Novel Performances", "Novel Light + Performances (held-out)", "TRAvatar* (reimplemented baseline)"]
---

# 论文速读：Artist-Friendly-Relightable-and-Animatable-Neural-Heads

## 一句话总结
本文提出了一种可同时支持**动态表情动画**与**任意光照重打**的神经网络头像方法，在 MVP（混合体积基元）架构基础上引入逐基元局部光照/视角方向条件，仅需稀疏 LED 灯条即可实现近场照明与近场视角效果，训练数据无需昂贵的光效舞台。

## 研究问题与动机
- **现有方法无法同时满足"动画化"与"重打光"**：动态神经头像（如 NeRSemble、MVP）支持运动回放但无法重打光；静态重打光 NeRF（如 ReNeRF、NeLF）支持任意光照但不能驱动新表情。
- **已有方案对硬件要求高**：传统 OLAT（单光条件）采集依赖密集光效舞台（light stage），成本高；TRAvatar 等方法只能使用固定基底插值，无法外推新光照方向或建模近场效应。
- **近场光照与近场视角尚未在动态神经头像中实现**：已有神经头像要么假设远处平行光，要么无法同时支持近场点光源和可变焦距相机运动（如 dolly zoom）。
- **艺术家友好性不足**：现有方法通常将外观信息"烘焙"进编码向量，推理时难以进行 unseen expression 生成或艺术性控制。

## 核心贡献（创新点）
1. **逐基元局部光照与视角方向条件**：在 MVP 外观分支中引入每个体素基元的局部光照方向 $\mathbf{l}_k$ 和视角方向 $\mathbf{v}_k$，使模型能处理近场照明与近场视角（如 dolly zoom），与 MVP 仅用全局远场方向形成本质区别。
2. **低成本的 OLAT 数据采集方案**：仅需 10 台相机 + 32 根 LED 灯条（照相机/远场光照设置），无需密集光效舞台，通过与 ReNeRF 相同的硬件即完成动态性能采集。
3. **纯几何驱动的表情编码器**：移除原 MVP 中的纹理分支，仅使用 tracked 3D 网格作为输入，使得推理时可以通过未见过的 artist-controlled blendshape 参数驱动神经头像，而非只能回放训练集表情。
4. **引入抠像损失（matting loss）**：针对黑色背景场景，用 MODNet 提取的 mask 与累积密度计算 MAE 损失，避免背景区域出现漂浮基元，这是原 MVP 所没有的改进。

## 方法详解
- **整体架构**：基于 MVP，包含三个可训练组件——Mesh Encoder（将 tracked 网格映射为全局表达码 $z \in \mathbb{R}^{256}$，经 1-layer MLP reshape 为 $8 \times 8 \times 256$ 特征图）、Geometry Branch（与 MVP 完全相同，输出基元的位置/旋转/缩放）和 Appearance Branch（RelMVP，核心创新部分）。
- **逐基元局部方向计算**：先通过 Geometry Branch 得到基元的空间变换 $(\mathbf{t}_k, \mathbf{R}_k)$，再计算：
  $$\mathbf{l}_k = \mathbf{p}_{olat} - \mathbf{R}_k \cdot \mathbf{t}_k, \quad \mathbf{v}_k = \mathbf{p}_{cam} - \mathbf{R}_k \cdot \mathbf{t}_k$$
  分别表示从基元中心指向光源/相机的局部方向。
- **多尺度条件注入**：将每个基元的 6 通道（v/l 拼合）信息在 UV 空间以全分辨率存储，每个卷积层前对 I 进行双线性下采样并与当前特征拼接；浅层接近全局方向，深层允许各基元 specialization。
- **透明度独立分支**：$\alpha$ 与光照/视角无关，单独用一个相同架构但不依赖 I 的分支预测，RGB 输出经 ReLU，中间层用 LeakyReLU。
- **损失函数**：
  $$\mathcal{L} = 1.0\mathcal{L}_{\text{pho}} + 10.0\mathcal{L}_{\text{geo}} + 0.01\mathcal{L}_{\text{vol}} + 0.001\mathcal{L}_{\text{kld}} + 0.1\mathcal{L}_{\text{mat}}$$
  其中 $\mathcal{L}_{\text{mat}} = \text{MAE}(\mathcal{M}, \tilde{\alpha}(\Theta))$，$\mathcal{M}$ 由 MODNet 从输入图提取。

## 实验与结果
- **数据集**：3 位受试者，每位约 1800 帧（去除全亮帧后），每帧含 10 路多视角图像 + 1 个 OLAT 光源位置 + 1 个 tracked 3D 面部网格。
- **对比基线**：TRAvatar*（作者复现的带光照分支的 TRAvatar，因无公开代码）。
- **关键数值**（新光照方向 / 新表演 / 两者均未见）：

  | 受试者 | 设置 | Ours PSNR | TRAvatar* PSNR |
  |---|---|---|---|
  | S1 | 新光照 | **32.19** | 29.68 (+2.51) |
  | S2 | 新光照 | **32.20** | 30.13 (+2.07) |
  | S3 | 新光照 | **33.73** | 31.21 (+2.52) |
  | S1 | 新表演 | **39.75** | 32.37 (+7.38) |
  | S2 | 新表演 | **29.52** | 28.86 (+0.66) |
  | S3 | 新表演 | **31.28** | 28.88 (+2.40) |
  | S1 | 两者均新 | **29.11** | 28.68 (+0.43) |
  | S3 | 两者均新 | **30.76** | 30.65 (+0.11) |

  最优结果：S1 新表演条件下 PSNR=39.75，显著优于所有基线；SSIM 与 LPIPS 也全面领先。
- **定性效果**：成功展示任意环境贴图重打光、近场点光源、dolly zoom 近场视角、novel expression 合成等。
- **训练设置**：A6000 GPU，lr=1e-4，batch=12，200K 迭代（约 2 天），图像下采样至 1024×768。

## 相关工作脉络
1. **MVP [25]**：动态神经头像基线，使用混合体积基元高效渲染，但仅支持固定光照；本文在此基础上扩展外观分支以支持重打光。
2. **ReNeRF [50]**：静态场景的 OLAT 重打光方法，使用 32 LED 灯条；本文沿用其采集硬件方案，但扩展到动态人脸。
3. **TRAvatar [51]**：同期工作，给 MVP 添加线性光照分支，但假设固定基底且不支持近场效果；本文通过逐基元方向条件克服了这一限制。
4. **DRAM [2]**：基于 VAE 的网格级重打光模型，需tracked mesh + average texture；本文直接用神经体素渲染，避免了薄结构（头发、眼睛）的网格表示缺陷。
5. **NeRF/NeRSemble 系列**：支持动态但无重打光能力；本文填补了"动态+重打光"的空白。
6. **Light Stage OLAT 传统方法 [6,41,42]**：需密集光源阵列；本文仅用 32 根 LED 条，大幅降低采集成本。

## 局限性与未来方向
- 仅覆盖**头部正面**（约前半球），无法重建完整 360° 神经头像，边缘（头发、颈部）有伪影；受限于物理采集设备。
- **极端表情的外推**受限，需更多表情多样性训练数据（当前训练帧数仅为原 MVP 的 1/10）。
- **快速运动时运动模糊**会在训练数据中出现，导致重建模糊。
- **注视方向与头发运动不可控**，因输入为网格表达，缺乏眼动和发丝动力学建模。
- 未来可通过扩展采集范围、增加表情变体、引入显式发丝/眼球模型来改善。

## 研究启发与可借鉴点
1. **逐基元/逐体素局部方向条件**是一种通用设计模式，可迁移到任何基于 volumetric primitive 或 voxel grid 的动态神经渲染任务中，用于建模近场效应。
2. **纯几何驱动表达编码**（移除纹理分支）是一个简洁有效的推理期可控性设计：只要保证训练时网格 tracking 质量，即可在 inference 自由替换 blendshape 参数。
3. **ODLAT 采集时序编排**（相邻帧光照尽可能不同）提升了数据效率，这一策略可推广到其他多光照神经渲染的数据采集。
4. **Matting loss 替代原 MVP 的背景模型**是对纯色背景场景的有效简化，值得在类似 setting 下复用。
5. 本方法可作为**下游应用的基础模块**：如数字人直播、电影级虚拟制作、VR 化身等，艺术控制接口友好。

## 关键术语表
**MVP (Mixture of Volumetric Primitives)**：将场景表示为一组带颜色和不透明度的几何基元的混合体，通过传统体素渲染实现高效神经渲染。
**OLAT (One-Light-At-A-Time)**：每次只开启一根灯条的采集方式，用于分解场景光照以支持后续重打光。
**RelMVP**：本文提出的 Relightable MVP，在原 MVP 外观分支中引入逐基元局部光照/视角方向条件的新架构。
**Nearfield Lighting/View**：近场照明/视角，指光源或相机距离被摄体较近、产生明显空间衰减和透视变化的情况。
**Dolly Zoom**：希区柯克式变焦效果，相机推进同时缩小焦距，产生背景压缩感；本文通过逐基元局部视角方向实现了该效果。
**Tracked 3D Face Mesh**：通过 landmark-based 优化获得的每帧 3D 面部网格，用作表情驱动的唯一输入。
**Matting Loss**：基于 MODNet 提取的 alpha mask 与渲染累积透明度计算的 MAE 损失，用于抑制背景伪影。

## 可复现要素
- **数据集**：自建采集，使用 10 相机 + 32 LED 灯条；**未公开**。
- **代码/权重**：论文未提及开源；Baseline MVP [25] 与 ReNeRF [50] 有公开代码可供参考。
- **关键超参**：lr=1e-4，batch=12，200K 迭代，图像分辨率 1024×768，$N^2=16384$ 基元，$M=8$ 体素分辨率/基元，RGB 输出通道 48（= $3 \times M_z = 3 \times 8$），loss 权重 $\lambda_{\text{pho}}=1.0, \lambda_{\text{geo}}=10.0, \lambda_{\text{vol}}=0.01, \lambda_{\text{kld}}=0.001, \lambda_{\text{mat}}=0.1$。
- **训练硬件**：A6000 GPU，约 2 天。
