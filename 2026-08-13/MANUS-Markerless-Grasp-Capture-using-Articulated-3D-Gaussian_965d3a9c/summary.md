---
title: "MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussian"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:01:28"
field: "手-物交互三维重建与接触估计"
keywords: ["hand-object grasp capture", "3D Gaussian splatting", "contact estimation", "markerless motion capture", "multi-view reconstruction", "articulated hand modeling"]
innovations: ["提出基于骨骼驱动的 articulated 3D Gaussian 手部表示，支持高效接触地图计算", "构建 50+ 相机 7M+ 帧多视角抓握数据集 MANUS-Grasps，并提供湿油漆接触 ground truth 评估协议", "引入网格权重蒙皮替代 MANO 权重直接迁移，有效减少形变伪影"]
benchmarks: ["MANUS-Grasps (w/ wet-paint ground truth)", "InterHand2.6M (visual quality eval)"]
---

# 论文速读：MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussians

## 一句话总结
本文提出 MANUS，一种基于 articulated 3D Gaussians 的无标记手-物抓握捕捉方法，通过多视角 RGB 视频精准估计手与物体的接触关系，并构建包含 50+ 相机、7M+ 帧的 MANUS-Grasps 数据集以支持此类神经场方法的研究。

## 研究问题与动机
- 现有手-物抓握捕捉方法（基于骨架、网格或参数化模型如 MANO）无法准确建模手部形状，导致接触估计不精确；隐式神经表示虽能提升质量，但接触计算依赖昂贵的采样。
- 现有抓握数据集存在三类缺陷：依赖特殊硬件（热像仪、标记点）、RGB 视角数少且自遮挡严重、缺乏真实接触标注，难以支撑高保真神经场训练。
- 密集多视角（360°）可大幅削弱自遮挡影响，是精准接触建模的关键，但当前数据集普遍缺乏足够的相机数量。
- 接触质量评估缺乏可靠的 ground truth，现有方法（如 ContactDB 的热成像）受热耗散影响，难以精确反映真实接触区域。

## 核心贡献（创新点）
- **MANUS-Hand**：提出基于 3D Gaussian Splatting 的无模板可 articulation 手部表示，相比隐式场（如 LiveHand、TAVA）支持更高效的 contact map 计算。
- **MANUS 抓握捕捉框架**：将手部与物体均以 3D Gaussians 表示，通过简单拼接实现瞬时/累积接触估计，无需昂贵采样。
- **MANUS-Grasps 数据集**：构建包含 53 台 RGB 相机、7M+ 帧、30+ 日常场景的真实多视角数据集，解决神经场方法对视角数量的需求。
- **湿油漆接触评估协议**：利用物体表面湿漆转移到手上的物理现象，自动生成接触 ground truth，为模板-based 方法提供公平对比基准。

## 方法详解
**MANUS-Hand 表示**：
- 由 21 根骨骼、26 DoF 的骨架驱动，采用 3D Gaussian 作为形变单元。
- 初始化：高斯均值分布在每根骨骼中点附近，标准差匹配骨骼长度；协方差、透明度、SH 系数按 3DGS 协议初始化。
- 皮肤蒙皮权重：受 Fast-SNARF 启发，在 canonical 空间构建网格，用最近邻分配权重，再通过三线性插值查询高斯权重，避免直接使用 MANO 权重导致的训练偏移伪影。
- 正向运动学 + 线性混合蒙皮（LBS）将 canonical 高斯变换到 posed 空间；协方差通过旋转矩阵变换；SH 系数在 canonical 空间查询。
- 损失函数：$L_h = \alpha \mathcal{L}_1 + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{perc} + \delta \mathcal{L}_{iso}$，其中各向同性正则器 $\mathcal{L}_{iso} = (\frac{\min_s}{\max_s} - s)^2$（$s=0.4$）防止极端各向异性高斯引发接触伪影。

**接触估计**：
- 手-物高斯拼接：$G_f = \{G_o, G_h\}$，直接利用 Gaussian 的显式位置计算 3D 距离。
- 接触地图：$C = \{d(G_h, G_o), \text{ if } d < \tau; 0, \text{ otherwise}\}$，阈值 $\tau=0.004$。
- 瞬时接触：单帧距离判断；累积接触：逐帧累加。

**MANUS-Grasps 数据采集**：
- 53 台 RGB 相机（1280×720，120 FPS），立方体布局，3ms 内同步误差。
- 15 个评估序列在物体表面涂绿色湿漆，抓握后油漆残留在手上作为接触 ground truth。
- 提供 2D/3D 关节、分割掩码（SAM + Instant-NGP），及经过 1C Filter 平滑的时间一致轨迹。

## 实验与结果
- **数据集**：MANUS-Grasps（7M+ 帧，50+ 相机，3 名被试，30+ 场景），InterHand2.6M（用于 MANUS-Hand 视觉质量评估）。
- **接触评估基线**：MANO fitting、HARP、MANUS（三者均将网格细分至 49,000 顶点）。
- **主要结果**（Table 2，湿油漆 ground truth，IoU / F1）：

| 方法 | Subject1 mIoU | Subject2 mIoU | Subject3 mIoU |
|------|-------------|-------------|-------------|
| MANO | 0.161 | 0.135 | 0.208 |
| HARP | 0.173 | 0.148 | 0.224 |
| **MANUS** | **0.206** | **0.152** | **0.275** |

- MANUS 在三个被试上 mIoU 分别提升 +19%、+3%、+23%（相对 HARP），F1 同步领先。
- 视角数消融（Table 4）：5 视角→30+ 视角，Subject1 mIoU 从 0.147 升至 0.206，验证密集视角的必要性。
- 视觉质量（Table 3，InterHand2.6M）：PSNR 26.32 / SSIM 0.9872，低于 LiveHand（31.16），但因未建模环境光遮蔽和阴影所致，核心目标仍是几何精度。

## 相关工作脉络
- **MANO / HARP**：参数化手部模型（MANO）和单目个性化重建（HARP），依赖低维参数空间，形状拟合存在固有偏差，接触估计易过分割。
- **LiveHand / TAVA**：基于隐式场的动态人体/手部重建，渲染质量高但 contact map 计算需密集采样，效率低。
- **ContactDB / ContactPose**：带接触标注的数据集，但分别依赖热成像（受热耗散影响）和 RGB-D（视角有限），场景/物体种类受限。
- **HOnnotate / DexYCB**：提供多视角手-物数据，但前者视角仅 1-5 个，后者仅 8 视角，不足以支撑 neural field 训练。
- **3D Gaussian Splatting (Kerbl et al.)**：静态场景的高效渲染表示；本文首次将其扩展至可 articulation 的手部模型，并引入网格权重蒙皮策略。
- **Dynamic 3D Gaussians (4DGS, Luiten et al.)**：支持动态场景但不提供骨架驱动控制；本文通过骨骼驱动实现可控手部形变。

## 局限性与未来方向
- 仅建模单手把手持静态物体，未处理双手协作或工具间接抓握。
- 未考虑皮肤拉伸引起的 pose-dependent 非线性形变，手指接触区域可能存在几何偏差。
- 长时间Manipulation（>单次抓握）未覆盖，动态抓取过程建模受限。
- 采集设备（53 台高速相机+LED 照明）成本高、可及性低，限制大规模复现。
- 接触评估指标（IoU/F1）仍有提升空间，湿油漆方法依赖特定实验条件。

## 研究启发与可借鉴点
- **密集多视角是接触估计的核心先决条件**：本文系统验证了视角数与接触精度的正相关关系，为后续方法设计提供了明确的采集策略指引。
- **网格权重蒙皮替代参数化权重**：针对 Gaussian 训练中位置偏移问题，用 grid-based 权重+三线性插值代替直接复制 MANO 权重，可有效减少伪影，此技巧可迁移至其他基于 Gaussian 的形变建模任务。
- **湿油漆物理接触 ground truth 设计**：提供了一种无需额外标注的接触验证方案，为未来接触估计benchmark 的设计提供了可复用的实验范式。
- **各向同性正则化对接触质量的影响**：$\mathcal{L}_{iso}$ 阻止极端扁平高斯产生，对接触地图的几何一致性至关重要，值得在后续接触估计工作中复用。
- **隐式 vs 显式表示的 trade-off**：本文明确区分了"视觉渲染质量"与"接触几何精度"两个目标，提示后续工作应根据任务目标选择表征形式，而非一味追求 photorealism。

## 关键术语表
- **MANUS-Hand**：基于 3D Gaussian Splatting 的无模板可驱动手部表征，支持高效形变与接触估计。
- **Articulated 3D Gaussians**：将 3D Gaussian 原语绑定至骨骼骨架，通过 LBS 实现可控形变的显式神经表示。
- **MANUS-Grasps**：包含 53 台相机、7M+ 帧、15 个湿油漆接触 ground truth 的多视角手-物抓握数据集。
- **Instantaneous Contact Map**：单帧时刻手-物高斯间距离小于阈值 $\tau$ 的接触集合。
- **Accumulated Contact Map**：整个抓握序列中所有帧瞬时接触的累加结果。
- **Grid Weights**：在 canonical 空间网格上分配的蒙皮权重，通过三线性插值获取高斯权重，避免 MANO 权重直接迁移导致的训练伪影。
- **Isotropic Regularizer**：约束优化后高斯各向同性程度的损失项，防止极端扁平方差引发接触伪影。

## 可复现要素
- **数据集**：MANUS-Grasps 计划公开（论文作者声明 dataset publicly available），含 camera poses、segmentation masks、estimated contacts；InterHand2.6M 为公开基准。
- **代码/权重**：论文未明确提供开源链接，仅注明项目主页 ivl.cs.brown.edu/research/manus，截至论文发表时代码未同步开源。
- **关键超参**：接触阈值 $\tau = 0.004$；各向同性正则化参数 $s = 0.4$；MANO 网格细分至 49,000 顶点用于公平对比；相机帧率 120 FPS，分辨率 1280×720。
