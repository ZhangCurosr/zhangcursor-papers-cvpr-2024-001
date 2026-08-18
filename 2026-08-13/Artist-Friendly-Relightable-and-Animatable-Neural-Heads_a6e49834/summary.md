---
title: "Artist-Friendly-Relightable-and-Animatable-Neural-Heads"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Artist-Friendly_Relightable_and_Animatable_Neural_Heads_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:02:43"
field: "神经渲染与数字人"
keywords: ["Neural Avatar", "Relightable Rendering", "Mixture of Volumetric Primitives", "Near-field Lighting", "Dynamic NeRF", "Facial Animation"]
innovations: ["在MVP外观分支中引入每原语局部光照/视角方向，实现近场光照与视角建模", "去纹理编码的Mesh Encoder，使z仅驱动几何变形以支持未见表情的艺术化驱动"]
benchmarks: ["PSNR", "SSIM", "LPIPS", "MAE"]
---

# 论文速读：Artist-Friendly-Relightable-and-Animatable-Neural-Heads

## 一句话总结
本文提出 RelMVP，在 MVP（Mixture of Volumetric Primitives）架构基础上，通过在外观分支引入**每个原语级别的局部光照和视角方向**（per-primitive local light/view directions），首次实现了同时支持**动态表情控制**与**任意光源重光照**（包括近场照明）的神经头像模型，且仅需廉价的稀疏 LED 灯条阵列采集数据。

## 研究问题与动机
- **动态与重光照的联合建模缺失**：现有神经头像工作要么专注于动态动画（MVP、NeRSemble 等），要么仅支持静态场景的重光照（ReNeRF、NeLF 等），两者同时支持的可行方案仍非常有限。
- **近场光照与视角建模困难**：已有方法（如 TRAvatar）依赖固定基底的线性光照分支，无法插值/外推新光照方向，也无法建模近场光照与近场视角效应。
- **采集成本过高**：传统光stage（light stage）需要密集布置的大量光源，成本高昂；作者希望用低成本硬件（32 根可独立控制的 LED 灯条）实现同等甚至更强的重光照能力。
- **艺术家友好性不足**：多数动态神经头像仅支持表演回放，缺乏对未见表情的艺术化驱动能力；且现有方法将外观烘焙进潜变量 z，限制了推理时的表情操控灵活性。

## 核心贡献（创新点）
1. **首次实现同时 animatable + relightable 的神经头像架构**：将 MVP 的动态头像能力与基于 OLAT 序列的重光照能力相结合，支持任意远场环境贴图与近场点光源的动态重光照。
2. **每原语局部光照/视角条件化设计**：在外观分支中为每个体素原语（primitive）计算局部光照方向 $\mathbf{l}_k$ 和视角方向 $\mathbf{v}_k$，并在全网络各层concatenate，使模型同时支持近场照明和近场视角（含变焦效果）。
3. **去纹理编码的 Mesh Encoder 设计**：从原始 Deep Appearance Model encoder 中移除纹理分支，z 仅驱动几何变形而非外观，使艺术家能在推理时用全新未见表情驱动神经头像。
4. **低成本采集方案**：采用 ReNeRF 的稀疏 LED 灯条阵列（10 相机 + 32 根灯条）替代昂贵光stage，以 interleaved OLAT/full-on 序列采集动态表演，显著降低数据采集门槛。

## 方法详解

### 整体架构
基于 MVP（Lombardi et al., 2021），三个主要可训练组件：
- **Mesh Encoder**：将输入人脸网格映射到全局表达潜向量 $\mathbf{z} \in \mathbb{R}^{256}$，经 1-layer MLP 变换为 $8\times8\times256$ 特征图后送入几何与外观分支。
- **Geometry Branch**（沿用 MVP 原文）：仅依赖 $\mathbf{z}$，输出 $N^2=16384$ 个原语的变换参数 $(\mathbf{t}_k, \mathbf{R}_k, \mathbf{s}_k)$。
- **Illumination-Modulated Appearance Branch（核心创新）**：除 $\mathbf{z}$ 外，额外接收每原语局部光照方向 $\{\mathbf{l}_k\}$ 和视角方向 $\{\mathbf{v}_k\}$，输出每个原语的 RGB$\alpha$ 体素网格 $\mathbf{C}_k \in \mathbb{R}^{4\times8^3}$。

### 局部方向计算
给定 OLAT 光源三维位置 $\mathbf{p}_{olat}$ 和相机位置 $\mathbf{p}_{cam}$，第 $k$ 个原语的局部方向为：
$$\mathbf{l}_k = \mathbf{p}_{olat} - \mathbf{R}_k \cdot \mathbf{t}_k, \quad \mathbf{v}_k = \mathbf{p}_{cam} - \mathbf{R}_k \cdot \mathbf{t}_k$$
两者拼接为 6-channel UV-space 图像 $\mathbf{I} \in \mathbb{R}^{6\times(N\cdot M)\times(N\cdot M)}$。

### 外观分支设计
- 输入：表达式特征图 $\mathbf{z}' \in \mathbb{R}^{256\times8\times8}$ + 下采样后的 $\mathbf{I}$。
- 7 层 transpose convolution（kernel=4, stride=2, padding=1），分辨率每层×2，最终达到 $1024\times1024$（即 $N\cdot M=1024$）。
- 关键技巧：$\mathbf{I}$ 在每一卷积层被双线性下采样后 concat 到中间特征层——浅层近似全局方向，深层保留每原语特异性方向，从而同时支持远场/近场光照。
- 透明通道 $\alpha$ 由独立分支预测（不依赖光照/视角），RGB 输出经 ReLU 激活。

### 损失函数
$$\mathcal{L} = \lambda_{\text{pho}}\mathcal{L}_{\text{pho}} + \lambda_{\text{geo}}\mathcal{L}_{\text{geo}} + \lambda_{\text{vol}}\mathcal{L}_{\text{vol}} + \lambda_{\text{kld}}\mathcal{L}_{\text{kld}} + \lambda_{\text{mat}}\mathcal{L}_{\text{mat}}$$
其中 matting 损失为新增项：$\mathcal{L}_{\text{mat}} = \text{MAE}(\mathcal{M}, \tilde{\alpha}(\Theta))$，$\mathcal{M}$ 由 MODNet 提取。权重：$\lambda_{\text{pho}}=1.0, \lambda_{\text{geo}}=10.0, \lambda_{\text{vol}}=0.01, \lambda_{\text{kld}}=0.001, \lambda_{\text{mat}}=0.1$。

### 数据采集策略
- 序列模式：$F, O_1, O_2, F, O_3, O_4, \ldots$（F=全亮帧，$O_i$=第 i 根灯条单独亮），24fps，相邻 OLAT 帧光源方向尽可能不同以最大化数据效率。
- 每名受试者约 2700 张/相机，平均 ~1800 帧用于训练（去除全亮帧后）。
- 面部跟踪：仅在 full-on 帧上用 landmark-based 3D 人脸跟踪方法优化 actor-specific blendshape 参数，再线性插值到 OLAT 帧。

## 实验与结果

### 数据集与基线
- **数据集**：3 名受试者（S1–S3），使用 10 相机 + 32 LED 灯条的 sparse capture setup。
- **基线**：仅 reimplemented 了 TRAvatar（同期工作，记为 TRAvatar*），因同时支持动态+重光照且开源的代码极少。
- **验证集划分**：held-out 光照方向 / held-out 表演 / 两者均有 held-out。

### 定量结果（PSNR↑ / MAE↓ / SSIM↑ / LPIPS↓，越高越好）

| 场景 | 方法 | PSNR | MAE | SSIM | LPIPS |
|------|------|------|-----|------|-------|
| Held-out 光照，S1 | ours | **32.19** | **2.88** | **0.899** | **0.278** |
| 同 S1 | TRAvatar* | 29.68 | 4.03 | 0.870 | 0.326 |
| Held-out 表演，S1 | ours | **39.75** | **2.65** | **0.873** | **0.343** |
| 同 S1 | TRAvatar* | 32.37 | 3.24 | 0.854 | 0.390 |
| 光照+表演 held-out，S1 | ours | **31.28** | **3.27** | **0.863** | **0.338** |
| 同 S1 | TRAvatar* | 28.86 | 4.12 | 0.869 | 0.323 |

- 方法在**所有三项指标（PSNR、SSIM、LPIPS）**及三种 held-out 设置下均稳定优于 TRAvatar*。
- 最大 PSNR 提升：held-out 表演条件下 +7.38 dB（S1）。

### 定性结果亮点
- 支持任意远场环境贴图重光照（Fig.4, Fig.5）。
- 成功渲染**近场点光源**照明（Fig.6）与**dolly-zoom**近场视角效果（Fig.7）。
- 支持未见表情的艺术化驱动（Fig.3 bottom）。

## 相关工作脉络
- **MVP（Lombardi et al., 2021）**：动态神经头像的基线表示，本工作在此基础上扩展外观分支以支持重光照，本质区别在于 MVP 仅支持固定光照下的 animatable 渲染。
- **TRAvatar（Yang et al., 2023）**：同期工作，亦在 MVP 上加线性光照分支；但依赖固定基底（需极密集basis才能达到高保真），无法插值/外推新光照方向，也不能建模近场效应——本文方法从根本上规避了这些限制。
- **ReNeRF（Xu et al., 2023）**：静态场景的重光照 NeRF，引入 OLAT MLP + spherical codebook；本文将其思想扩展到动态场景，且无需 dense light stage。
- **DRAM（Bi et al., 2021）**：基于 VAE 的深度可重光照外观模型，适用于 mesh 驱动的面部；本文处理的是完整 volumetric 头像，且支持近场光照。
- **NeRFs for dynamic scenes（Neural Volumes, NeRSemble, Nerfies, D-NeRF 等）**：支持动态但受限于性能回放，无重光照能力；本文填补了这一空白。
- **Light Stage 方法（Debevec et al., Sun et al.）**：需要密集光源的静态重光照采集方案；本文使用廉价 LED 灯条阵列实现动态+重光照联合采集。

## 局限性与未来方向
- **非 360° 完整头像**：当前仅重建正面头部，边界处（发梢、颈部）存在伪影；作者认为受限于物理采集设备而非方法本身。
- **极端表情泛化有限**：训练帧数仅为原始 MVP 的约 1/10，增加表情多样性可缓解。
- **快速运动导致模糊**：若训练数据中存在运动模糊，重建结果也会模糊。
- **凝视与头发不可控**：输入为 mesh，无法驱动视线变化与头发运动。
- **未来方向**：扩展至全身/360° 头像、增加训练数据覆盖更多表情、与生成式模型结合实现更丰富的艺术化驱动。

## 研究启发与可借鉴点
- **Per-primitive local direction conditioning** 的设计思路可迁移到其他 volumetric primitive 或 NeRF 变体中，用于建模近场效应（如近场相机运动、近场阴影）。
- **Interleaved OLAT/full-on 采集策略**：在单帧中交替 OLAT 与全亮帧，既保证重光照训练数据多样性，又通过全亮帧辅助网格跟踪——这种"一箭双雕"的数据策略值得借鉴。
- **去纹理编码驱动纯几何**：将 expression latent z 仅用于驱动几何变形而非外观烘焙，保留了推理时新表情的艺术化可控性，这一设计理念可扩展到其他 avatar 建模任务。
- **Matting loss 替代背景模型**：在纯黑背景数据下，用简单的 matting loss（基于累积密度）替代 MVP 原始的背景建模，实现更简洁高效的分离。
- **低成本硬件替代高端设备**：用 32 LED 灯条 + 10 相机替代传统光stage，在保持甚至提升效果的同时大幅降低采集成本，对实际部署有直接参考价值。

## 关键术语表
- **MVP（Mixture of Volumetric Primitives）**：将场景表示为一组带 RGBα 信息的 3D 体素原语集合，通过 guide mesh 组织，支持高效神经渲染。
- **OLAT（One-Light-At-a-Time）**：每次仅点亮一个光源的采集方式，用于分解光照与外观，是图像级重光照的基础。
- **RelMVP**：本文提出的可重光照动态神经头像架构，即在 MVP 外观分支中引入每原语局部光照/视角条件。
- **Local light/view directions**：相对于每个原语中心计算的局部光照方向和视角方向，使近场照明与视角成为可能。
- **Guide mesh**：由几何分支预测的粗粒度参考网格，用于初始化并约束体素原语的空间布局。
- **Dolly zoom**：摄像机推进的同时缩小焦距（增大 FOV）的镜头运动效果，本文通过 per-primitive local view directions 实现了该效果。
- **Latent expression code z**：从输入面部网格编码得到的低维潜变量，用于驱动几何变形，不含外观信息。
- **TRAvatar**：同期工作，在 MVP 上加线性光照分支以支持重光照，但受限于固定基底无法泛化到新光照方向。

## 可复现要素
- **数据集**：作者未公开数据集；使用自有采集的 3 名受试者数据（10 相机 + 32 LED 灯条，约 2700 张/相机/人）。
- **代码/权重**：论文未提及开源计划。
- **关键超参**：$N^2=16384$ 个原语，每原语 $M=8$ 分辨率体素网格；appearance branch 7 层 transpose conv（kernel=4, stride=2）；lr=0.0001（Adam）；batch size=12；训练 200,000 轮（~2 天/A6000 GPU）；输入分辨率降至 1024×768。
