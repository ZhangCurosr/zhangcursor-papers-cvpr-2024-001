---
title: "HEAL-SWIN-A-Vision-Transformer-On-The-Sphere"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Carlsson_HEAL-SWIN_A_Vision_Transformer_On_The_Sphere_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:45:08"
---

# 论文速读：HEAL-SWIN-A-Vision-Transformer-On-The-Sphere

## 一句话总结
提出 HEAL-SWIN，将天体物理领域的 HEALPix 等面积嵌套网格与 SWIN Transformer 结合，直接在无畸变的高分辨率球面数据上进行窗口化自注意力计算。首次将自动驾驶鱼眼图像视为原生球面信号，在语义分割与深度估计任务上显著优于传统平面基线及现有球面模型。

## 研究问题与动机
- 高分辨率广角鱼眼图像在自动驾驶中日益普及，但将其投影至平面矩形网格会引入严重的空间畸变与投影损失，影响安全关键任务的精度。
- 现有球面模型多依赖 Driscoll–Healy 网格与 SO(3) Fourier 变换，采样在极区过密、计算复杂度随带宽三次方增长，且无法高效处理仅覆盖半球的鱼眼数据。
- 缺乏一种原生支持高分辨率、部分球面覆盖、无需 Fourier 域运算且计算开销可控的球面视觉 Transformer。

## 核心贡献（创新点）
1. 构建 HEAL-SWIN Transformer，将 HEALPix 的四叉树嵌套结构直接映射至 SWIN 的 patching、windowing 与 merging 操作，实现高效的球面局部自注意力计算。与依赖 Fourier 变换或高频带宽的球面 CNN 不同，本文在前向域直接操作离散像素列表，彻底规避了 SO(3) 变换的三次方算力瓶颈。
2. 设计 Grid Shifting 与 Spiral Shifting 两种适应 HEALPix 的窗口移位策略，有效解决半球覆盖数据的边界冲突与全局信息分布问题。与现有将点云投影至球面后做刚性旋转移位的方法不同，本文的移位完全基于 HEALPix 网格索引的一维重排，无需额外几何旋转且计算代价极低。
3. 首次将汽车场景鱼眼图像作为无畸变球面信号进行端到端深度学习，在合成与真实数据集的语义分割、单目深度估计任务上全面超越平面 SWIN 基线。与常规先做畸变校正再输入平面网络的处理范式不同，本文证明原生球面表示能更准确地保留 3D 几何先验，直接提升下游障碍物检测与避障所需的点云质量。
4. 在 Stanford 2D-3D-S 室内鱼眼数据集上验证球面 Transformer 的通用性，超越 HexRUNet、Spin-SphCNN 等同类参数规模模型。与早期基于二十面体网格或图卷积的球面分割方法不同，本文利用 HEALPix 的层级均匀性实现了更稳定的长程依赖建模。

## 方法详解
- **一维 SWIN-UNet 变体**：输入数据以 HEALPix 嵌套排序（nested ordering）的一维列表形式输入，编码器采用 patch merging 下采样，解码器对称使用 patch expansion 上采样，并通过跳跃连接拼合特征。
- **Patching & Windowing**：将 $n_{\text{patch}} = 4^k$ 个连续 HEALPix 像素合并为一个 patch，再将 $n_{\text{win}} = 4^k$ 个 patch 组合为 attention window。利用 HEALPix 等面积、同质化的特性，确保每个 patch/window 在球面上覆盖的立体角一致，下采样/上采样同样以 $4^k$ 为单位聚合或展开连续列表段。
- **Shifting 策略**：
  - *Grid Shifting*：沿基础四边形网格的轴向平移半个 window 大小。在基础四边形交界面处会发生窗口碰撞，需将溢出像素重排至其他窗口，并通过 attention mask 屏蔽跨区域像素对的注意力。
  - *Spiral Shifting*：将嵌套排序临时转为环状排序（ring ordering），沿等纬度环滚动 $n_{\text{shift}} = \sqrt{n_{\text{win}}}/2$ 步后再转回，仅在极点与半球边界产生边界效应，无内部冲突。消融实验表明 Spiral 略优于 Grid。
- **Relative Position Bias**：在每个基础四边形内部近似为规则矩形网格，据此计算窗口内像素对的相对坐标 $(x,y)$，查表得偏置 $B_{ij} = \hat{B}_{x(i)-x(j)+\sqrt{n_{\text{win}}},\ y(i)-y(j)+\sqrt{n_{\text{win}}}}$，全窗口共享同一可学习偏置表，绝对位置嵌入经实验验证无收益。
- **半球覆盖处理**：HEALPix 原始 12 个基础四边形中仅取前 8 个（base pixels）对应鱼眼视角的半球视野，通过直接截断列表索引实现，避免全空间零填充导致的算力浪费。
- **损失函数**：语义分割采用与 [50] 一致的加权交叉熵；深度估计采用 $L_2$ 损失，训练与评估时掩码天空区域，并将深度值标准化为零均值单位方差，所有重采样均使用 nearest neighbor 以保持高对比度边缘。

## 实验与结果
- **数据集**：WoodScape（真实汽车鱼眼）、SynWoodScape（CARLA 合成，含 Large 与 Large+AD 两子集）、Stanford 2D-3D-S（室内 RGB-D 鱼眼）。
- **基线**：平面 SWIN-T（patch=2×2, window=8×8, 12层, ~41M参数）、Gauge CNN、UGSCNN、HexRUNet、SphCNN、Spin-SphCNN。
- **语义分割 mIoU（3次运行平均）**：
  - Large SynWoodScape：HEAL-SWIN **0.947** vs SWIN 0.918（+2.9%）
  - Large+AD SynWoodScape：HEAL-SWIN **0.841** vs SWIN 0.809（+3.2%，最强提升）
  - WoodScape（真实）：HEAL-SWIN **0.628** vs SWIN 0.617（+1.1%）
- **深度估计**：在 SynWoodScape 上将预测深度反投影为球面点云，以 Chamfer Distance 评估 3D 几何一致性，HEAL-SWIN 显著优于 SWIN，表明球面表示学到了更准确的三维结构。
- **室内球面分割**：HEAL-SWIN mIoU **44.3** / mAcc 61.9，超越 HexRUNet (43.3/58.6) 等同类球面模型，成为该任务新 SOTA。
- **推理效率**：在等价分辨率下 HEAL-SWIN 与 SWIN 单像素延迟基本持平（297±26 ns vs 296±39 ns）；低分辨率时 HEALPix 连续内存布局带来更小方差与更快推理。
- **数据规模消融**：随训练集比例增大，HEAL-SWIN 相对 SWIN 的性能增益持续扩大，说明球面表示更利于从大数据中抽取几何先验。

## 相关工作脉络
- **Fourier-domain Spherical CNNs**（SphCNN, Spin-SphCNN 等）：依赖 Driscoll–Healy 网格与 SO(3) 球谐变换，计算量随带宽三次方增长且难以处理半球星面；HEAL-SWIN 完全脱离 Fourier 域，直接在像素列表上操作。
- **Graph-based Spherical CNNs**（DeepSphere 及其 HEALPix 变体）：基于图邻接关系建模，无法利用 HEALPix 的天然四叉树层级进行高效的窗口化局部注意力；HEAL-SWIN 将 SWIN 的窗口移位思想原生嵌入网格索引。
- **Spherical Transformers**（Spherical ViT, Climformer 等）：早期工作多采用均匀球面网格提取 patch 但缺乏局部移位机制，扩展至高阶分辨率时显存与算力急剧上升；HEAL-SWIN 继承 SWIN 的层级设计，以极小计算开销支持高分辨率。
- **Point Cloud / LiDAR Spherical Transformers**（SWPT, Spherical Transformer for LiDAR）：将三维点云投影至球面并进行刚性旋转实现移位；HEAL-SWIN 直接处理原生球面/鱼眼图像，无需额外投影步骤且避免点云稀疏性带来的表征损失。
- **Distortion-aware Flat CNNs**（变形卷积核、球面多面体展开等）：在平面网格上做畸变校正或核形变，本质上仍是平面对球面的
