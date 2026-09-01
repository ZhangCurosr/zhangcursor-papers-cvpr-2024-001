---
title: "MANUS: Markerless Grasp Capture using Articulated 3D Gaussians"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:41:57"
field: "手物交互与接触估计"
keywords: ["hand-object grasp capture", "3D Gaussian splatting", "contact estimation", "markerless motion capture", "multi-view RGB", "articulated hand model", "neural rendering"]
innovations: ["将 3D Gaussian splatting 扩展为可形变铰链式手部表示 MANUS-Hand，支持高效正运动学驱动与接触计算", "提出基于网格加权皮肤绑定的 3D Gaussian 绑定策略，避免模板权重迁移导致的伪影", "构建 50+ 视角 7M+ 帧真实多视角抓取数据集 MANUS-Grasps，并以湿颜料转移作为自动接触 ground truth"]
benchmarks: ["MANUS-Grasps (wet-paint contact evaluation)", "InterHand2.6M (visual quality)"]
---

# 论文速读：MANUS: Markerless Grasp Capture using Articulated 3D Gaussians

## 一句话总结
本文提出 MANUS，一种基于可形变 3D Gaussian 表示的无标记手-物抓取捕捉方法，通过多视角 RGB 图像同时高精度重建手部形状、外观及手-物接触区域；同时发布大规模多视角数据集 MANUS-Grasps（50+ 相机、7M+ 帧），并创新性利用湿颜料转移建立接触真实标注。

## 研究问题与动机
- **手-物接触估计不准确**：现有无标记抓取捕捉方法依赖骨架、网格或参数化模型（如 MANO），其低维表征无法精确刻画手部细节形状，导致接触估计误差大（Figure 1 可视化对比）。
- **隐式神经表示采样成本高**：articulated neural implicit representations（如 LiveHand、Lisa）虽能高精度建模形状，但接触估算需密集三维采样，效率低下。
- **缺乏支持神经场方法的多视角数据**：现有抓取数据集要么相机视角少（1–9 视角）、存在严重遮挡，要么依赖.marker/特殊传感器/热成像，难以支撑基于像素对齐的 3D Gaussian 优化。
- **接触质量缺乏可靠的无标注评估基准**：现有数据集缺少 ground-truth 接触标注，而热成像等方法受热消散影响亦不精确。

## 核心贡献（创新点）
- **MANUS-Hand：模板自由的可形变 3D Gaussian 手部表示**——将 3D Gaussian splatting 扩展到铰链结构手部（21 骨、26 DOF），通过网格加权皮肤绑定实现高效正/逆运动学驱动，与 TAVA/LiveHand 等隐式方法相比支持更高效的接触计算。
- **MANUS 抓取捕捉框架**——手部和物体均以 3D Gaussian 拼接表示，直接基于高斯中心距离计算瞬时/累积接触图，无需隐式场的 volumetric sampling。
- **MANUS-Grasps 大规模多视角数据集**——53 台 RGB 相机以 120 FPS 采集，覆盖 30+ 场景、3 位被试、超 7M 帧，并提供相机位姿、分割 mask 及基于湿颜料转移的接触 ground truth。
- **湿颜料接触评估基准**——首次在真实多视角数据集中引入 paint transfer 作为接触质量的自动可分割 ground truth，避免人工标注成本。
- **系统性 ablation 验证多视角密度必要性**——Table 4 证明随相机视角从 30+ 降至 5，接触 mIoU/F1 显著下降，量化证实遮挡消解对接触估计的关键作用。

## 方法详解
- **MANUS-Hand 表示与初始化**：以 21 骨手部骨架为驱动，每个骨骼中点附近按正态分布初始化 3D Gaussian 质心，标准差匹配骨长；协方差、透明度、SH 系数按 3DGS 协议初始化。
- **网格加权皮肤绑定**（Grid-based skinning）：抛弃直接复制 MANO 权重的方式（训练中易出现权重失配和伪影），在 canonical 空间构造均匀网格体素，用最近邻法分配 skinning weights，再对查询高斯做三线性插值得 $W$，变换矩阵 $T_g = W T_b$，位置 $\mu_p = T_g \mu$，协方差 $\Sigma_p = R_g \Sigma R_g^T$。
- **外观渲染**：canonical 空间中优化 SH 系数 $\phi_g$； posed 视角下将 view direction $\nu_p^g$ 反变换至 canonical 空间 $\nu_c^g = T_g^{-1}\nu_p^g$ 查询颜色，经可微 rasterizer $\mathcal{R}$ 合成图像 $\mathcal{I} = \mathcal{R}(\mu_p, \nu_c, \Sigma_p, \alpha, \phi)$。
- **优化损失**：$\mathcal{L}_h = \alpha \mathcal{L}_1 + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{perc} + \delta \mathcal{L}_{iso}$，其中各向异性正则 $\mathcal{L}_{iso} = (\frac{\min_s}{\max_s} - 0.4)^2$ 抑制异常扁塌高斯以减少接触渲染伪影。
- **物体表示**：静态 3D Gaussian 场景建模，结合 object mask 裁剪漂浮 outlier 高斯，保持几何一致性。
- **接触估计**： posed 空间中对于每对手部高斯 $G_h$ 与物体高斯 $G_o$，若欧式距离 $d(G_h, G_o) < \tau$（实验中 $\tau = 0.004$）则计入接触，得到瞬时接触图 $C$；累积接触图 $C_{acc}$ 对序列各帧 $C$ 累加。接触距离值映射为渲染颜色输出 contact map。

## 实验与结果
- **数据集**：
  - 训练/评估集：MANUS-Grasps（53 相机、120 FPS、1280×720、3 位被试、30+ 场景、15 条 wet-paint 评估序列、超 7M 帧）。
  - 手模视觉质量评估：InterHand2.6M（2 位被试，取 ROM07-RT-Finger-Occlusions 测试段，75% 训练/25% 评估）。
- **MANUS-Hand 视觉质量对比**（Table 3，PSNR/SSIM/LPIPS）：
  - LiveHand：PSNR 31.16 / SSIM 0.9818 / LPIPS 0.0278 / Test time 0.022s
  - TAVA：PSNR 22.85 / SSIM 0.983 / LPIPS 0.099 / Test time 11.00s
  - Ours：PSNR 26.32 / SSIM 0.9872 / LPIPS 0.068 / Test time 0.049s
  - 结论：虽不及 LiveHand 峰值（原因：未建模环境遮蔽与阴影），但在 SSIM 上超越两者，显著优于 TAVA；测试速度远快于 TAVA。
- **接触质量对比**（Table 2，wet-paint ground truth，mIoU / F1）：
  - 被试 1：MANO mIoU 0.161 / F1 0.270；HARP mIoU 0.173 / F1 0.289；MANUS mIoU 0.206 / F1 0.335
  - 被试 2：MANO mIoU 0.135 / F1 0.228；HARP mIoU 0.148 / F1 0.247；MANUS mIoU 0.152 / F1 0.251
  - 被试 3：MANO mIoU 0.208 / F1 0.338；HARP mIoU 0.224 / F1 0.361；MANUS mIoU 0.275 / F1 0.424
  - 结论：MANUS 在所有被试上均优于 MANO 与 HARP，被试 3 提升最显著（mIoU 相对 MANO +32%，相对 HARP +23%）。
- **多视角消融**（Table 4）：相机数从 30+ 降至 5 时，被试 1 mIoU 从 0.206 → 0.147（−29%），被试 3 mIoU 从 0.275 → 0.214（−22%），证明密集多视角是接触精度的关键。
- **定性结果**：Figure 5/6 显示 MANUS 接触区域贴合真实 paint residue，MANO/HARP 存在明显过度分割。

## 相关工作脉络
- **MANO / TESA 等参数化手模型**：低维参数空间难以表达个体手部细微形状，接触估算易因模型-真实几何错位而产生虚假接触；MANUS 以 3D Gaussian 摆脱模板约束。
- **LiveHand / Lisa 等 articulated neural implicit representations**：隐式场精度高但接触需 volumetric sampling，速度慢且不适合直接求接触图；MANUS 显式 Gaussian 拼接使接触估算仅需最近邻距离查询。
- **HARP（单目个性化手重建）**：使用 MANO 模板，接触估计精度受限；本文在其基础上通过多视角密集监督显著提升接触 IoU。
- **ContactDB / ContactPose**：分别依赖热成像与 RGB-D，前者受热消散影响、后者视角有限；MANUS 引入湿颜料自动分割方案提供无设备依赖的 ground truth。
- **H2O / HOnnotate / DexYCB 等数据集**：视角数 1–9、多为单/双手无接触标注或需 mocap/传感器；MANUS-Grasps 首次提供 50+ 视角 + 接触 ground truth 的大规模数据。
- **Dynamic 3D Gaussians（4DGS、Dynamic D3GS）**：面向自由变形场景建模，无骨骼约束，不适合具运动学结构的手；本文引入可驱动 articulation 的 3D Gaussian 框架。

## 局限性与未来方向
- **单只手 + 静态物体假设**：未处理双手交互、工具间接抓取以及被摄物体自身形变/滑动导致的接触变化。
- **未建模皮肤拉伸等非线性形变**：骨骼驱动 skinning 在极端屈曲姿态下可能出现几何失真，影响接触估计精度。
- **长时操纵未覆盖**：当前 pipeline 针对短时抓取序列，长时间连续操纵场景有待探索。
- **采集系统门槛高**：53 台同步相机阵列成本高、部署复杂，限制了方法在普通实验室的推广；作者承诺开源数据集以缓解。
- **接触评估指标仍有提升空间**：当前 IoU/F1 基于 2D 投影渲染比较，未充分建模接触面法向等几何属性。

## 研究启发与可借鉴点
- **网格加权皮肤绑定（Grid-based skinning）替代模板权重迁移**：对任意基于骨骼的显式/半显式形变场（如 3DGS-driven human/hand）均可复用，有效避免训练中权重漂移导致的伪影。
- **各向异性正则项 $\mathcal{L}_{iso}$ 促进接触渲染稳定性**：对任何以 Gaussian 做几何逼近的任务均可参考，抑制极端扁塌形状带来的边界伪影。
- **湿颜料 contact ground truth 的自动化方案**：无需人工标注即可在真实场景中获取接触 label，可迁移至足-地、夹爪-物体等广义接触评估任务。
- **多视角密度对接触估计的定量影响**：Table 4 的消融范式可为后续多视角系统设计的相机布置提供直接的工程指导准则。
- **手-物 Gaussian 拼接范式**：将两类可微分 Gaussian 场景简单 concat 即可联合渲染与接触计算，思路可推广至多对象 manipulation 场景。

## 关键术语表
- **3D Gaussian Splatting (3DGS)**：以一组带位置、协方差、不透明度和球谐系数的 3D 高斯原语表示辐射场，通过 tile-based 可微光栅化实现实时渲染的神经表示方法。
- **MANUS-Hand**：本文提出的模板自由、可驱动的铰链式 3D Gaussian 手部表示，支持任意 posed 状态下的高保真形状与外观重建。
- **Grid-based skinning weights**：在 canonical 网格体素上分配骨骼权重并通过三线性插值获取高斯绑定权重的方法，避免 MANO 权重直接迁移导致的训练伪影。
- **Instantaneous contact map**：特定时间步下手部高斯与物体高斯在 3D 空间距离小于阈值 $\tau$ 的接触集合。
- **Accumulated contact map**：整个抓取序列各帧瞬时接触图的累积叠加，反映完整接触历史。
- **Wet-paint transfer ground truth**：在被抓物体表面涂抹湿润颜料后抓取，颜料残留在手上作为真实接触区域的自动可分割 ground truth。
- **Isotropic regularizer $\mathcal{L}_{iso}$**：惩罚高斯最小/最大缩放比例偏离设定目标值（0.4）的正则项，用于避免接触渲染出现尖刺伪影。
- **MANUS-Grasps**：本文发布的 53 相机、120 FPS、超 7M 帧多视角手-物抓取数据集，包含 30+ 日常场景、3 位被试及湿颜料接触标注。

## 可复现要素
- **数据集**：MANUS-Grasps 已声明公开（论文附 ivl.cs.brown.edu/research/manus）；InterHand2.6M 为开源基准。
- **代码/权重**：论文给出 ivl.cs.brown.edu/research/manus，代码开源信息待核实；MANUS-Hand 需基于各被试序列单独优化。
- **关键超参**：接触阈值 $\tau = 0.004$；各向异性正则目标 $s = 0.4$；骨骼数 21、DOF 26；初始高斯分布均值在骨骼中点、标准差匹配骨长。
- **渲染/损失权重 $\alpha, \beta, \gamma, \delta$**：论文正文未给出具体数值，详见 supplementary。
- **相机参数**：53 台 RGB 相机，1280×720@120 FPS，帧间不同步误差 ≤ 3 ms；内参/外参通过 COLMAP +  Fiducial markers 标定。
- **姿态估计**：AlphaPose 提取 2D 关节 → 3D 三角化 → IK 拟合；时序平滑使用 1'C Filter；手/物分割使用 SAM + Instant-NGP 二值 mask。
