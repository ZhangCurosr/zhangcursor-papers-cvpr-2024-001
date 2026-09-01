---
title: "HOI-M-sup-3-sup-Capture-Multiple-Humans-and-Objects-Interact"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhang_HOI-M3_Capture_Multiple_Humans_and_Objects_Interaction_within_Contextual_Environment_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:45:33"
field: "多人多物体交互感知与生成"
keywords: ["Human-Object Interaction", "Multiple Human-Object Interaction", "Motion Capture", "Diffusion Generation", "Multi-view Reconstruction", "Inertial Tracking", "3D Pose Estimation"]
innovations: ["首个多人多物3D交互大规模数据集HOI-M³（199序列/181M帧/42视角）", "惯性辅助多物体联合优化追踪管线（mask/offscreen/collision/smooth四约束）", "单目一次性多人HOI捕获与条件扩散生成基线"]
benchmarks: ["HOI-M³", "PCK_rel", "PCK_abs", "Chamfer_o", "V2V", "FID", "Pene"]
---

# 论文速读：HOI-M³: Capture Multiple Humans and Objects Interaction within Contextual Environment

## 一句话总结
本文提出了首个大规模多人多物体交互数据集 HOI-M³（199个序列、181M帧、42视角、31人×90物体），并以此为基础构建了单目多人HOI捕获与扩散模型驱动的无结构交互生成两个下游任务的强基线。

## 研究问题与动机
- **数据匮乏**：现有HOI数据集（如GRAB、BEHAVE、InterCap）均聚焦单人-单物交互，缺乏真实世界中多人多物共存的3D标注数据。
- **场景局限**：HSI（Human-Scene Interaction）数据集（如PiGraphs、PROX、HUMANISE）多将环境视为静态CAD模型，忽略动态可移动物体；且仅建模单人与静态场景的交互。
- **方法瓶颈**：现有单目HOI捕获方法（如PHOSA、CHORE）依赖弱投影相机模型，无法准确估计多实例根深度，且在匹配阶段无法处理多人与多物的联合推理。
- **生成空白**：当前Motion Generation工作（如Human Motion Diffusion Model、InterFusion）仅支持单人运动或单物生成，尚无模型能生成多人多物的社交交互序列。

## 核心贡献（创新点）
1. **首个多人多物真实3D交互数据集**：HOI-M³覆盖5种日常场景（卧室、餐厅、客厅、健身室、办公室），每序列至少2人+5物，是已公开数据集中规模与场景丰富度之最。
2. **惯性辅助的多物体鲁棒追踪管线**：通过嵌入物体内置IMU + 42视角刚性优化（mask约束、offscreen损失、碰撞惩罚、平滑正则）实现高精度的物体6DoF轨迹估计，解决密集遮挡下的漂移问题。
3. **一次性单目多人HOI捕获方法**：设计中心热图+人体网格图+物体网格图的多任务预测头，在一个前向传播中同时回归多个人体的SMPL参数与多个物体的6D位姿，避免传统多阶段级联误差累积。
4. **面向多人HOI的扩散生成基线**：将条件DDPM扩展至500维联合表征（5人×88维+10物×6维），以PointNet提取的物体几何与预设人数/物数为条件，实现无结构社交交互序列生成，填补该任务空白。

## 方法详解
### 数据集采集
- **硬件**：42台Z CAM电影级相机（4K@60fps），每个预扫描物体内部嵌入IMU。
- **标定**：受试者起跳时IMU信号峰值与RGB帧对齐实现时统；三视图人工标注点三角化求解IMU-RGB刚性偏移。
- **标注**：SAM生成初始分割掩码→专业标注员校正首帧→自动广播全序列。

### 惯性辅助多物体追踪（公式2–5）
对每个物体优化平移 $T_t$ 与旋转偏移 $R_t^{\text{off}}$，目标函数：
$$\min_{R,T} \lambda_{\text{mask}} E_{\text{mask}} + \lambda_{\text{offscreen}} E_{\text{offscreen}} + \lambda_{\text{collision}} E_{\text{collision}} + \lambda_{\text{smt}} E_{\text{smt}}$$
- **Mask约束**（Eq.3）：可微渲染下42视角Human-Object掩码L2差异。
- **Offscreen损失**（Eq.4）：惩罚超出摄像机视锥（近/远平面、上下左右边界）的顶点投影。
- **碰撞约束**：借鉴[47,60]，对人与物穿透情形施加惩罚。
- **平滑约束**（Eq.5）：约束相邻帧旋转增量与纯IMU旋转增量之差，抑制IMU漂移导致的抖动感。

### 单目一次捕获（图3）
- **Human-Object Center Heatmap**：高斯分布编码人体根关节与物体mask中心。
- **Human Mesh Map**：特征位置回归SMPL参数$(\theta, \beta)$与归一化根深度 $\hat{Z}=Z\cdot w/f$。
- **Object Mesh Map**：6D旋转表示+物体绝对深度图。
- **联合损失**（Eq.7）：$L_{\text{sum}} = \lambda_\theta L_\theta + \lambda_\beta L_\beta + \lambda_{\text{obj}} L_{\text{obj}} + \lambda_{3D} L_{3D} + \lambda_{2D} L_{2D} + \lambda_{hm} L_{hm} + \lambda_{\text{depth}} L_{\text{depth}}$。

### 扩散生成（图4，公式8–10）
- **表征**：$x=[x_1,...,x_N]\in\mathbb{R}^{500}$，每人88维（pose 72+shape 10+root 3+orient 3），每物6维（trans 3+pose 3）。
- **条件输入**：PointNet提取的物体几何全局特征 + MLP嵌入的人数/物数。
- **训练目标**（Eq.10）：$\mathcal{L}=\mathbb{E}_{x_0,n}\|\hat{x}_\theta(x_n,n)-x_0\|_1$，即去噪网络直接预测原始数据。

## 实验与结果
### 数据集规模（表1）
| 维度 | HOI-M³ | 次优NeuralDome | 提升倍数 |
|---|---|---|---|
| 录制时长 | **20 h** | 4.3 h | 4.7× |
| 帧数 | **180.5 M** | 71 M | 2.5× |
| 分辨率 | 3840×2160 @ 60fps | 3840×2160 @ 60fps | 同等 |
| 物体数 | **90** | 23 | 3.9× |
| 多人多物标记 | ✓ | ✗ | — |

### 单目捕获评测（表2，HOI-M³测试集）
- **PCK_rel**：Ours **68.5** vs PHOSA 43.9 / CHORE 10.4（相对提升+56%/+558%）。
- **PCK_abs**：Ours 5.9（两基线因弱投影模型无法计算）。
- **Chamfer_o**：Ours **235.0** vs PHOSA 1454.3 / CHORE 465.8（降低84%/49%）。
- **V2V**：Ours **297.8** vs PHOSA 691.4 / CHORE 340.2。
- **关键结论**：现有单HOI方法在多实例场景中根深度估计严重失准，本文一次性预测多实例显著领先。

### 生成评测（表3，20次采样平均）
- **FID**（Separated：人/物分开评估）：人 16.50 ± 0.04 / 物 10.61 ± 0.06；Joint 36.91 ± 0.09。
- **Pene**（人体穿透率）：Separated 1.45% / 3.89%；Joint 9.27%。
- **定性**：给定物体几何与人数/物数，模型可合成语义一致的客厅交互序列（图6）。

## 相关工作脉络
1. **单人-单物HOI捕获**：PHOSA [65]、CHORE [52]、StackFlow [19]、I'M HOI [72] 均围绕单人与单个可动物体建模；本文首次扩展到多人×多物联合估计。
2. **手势/抓握数据**：GRAB [45]（120物体、51抓取姿态）、BEHAVE [4]（20物体）均为单人设定；本文规模与多主体性是质的跨越。
3. **Human-Scene Interaction**：PROX [14]、SAMP [15]、HUMANISE [50]、CIRCLE [3] 将人嵌入静态重建场景；本文强调**动态可移动物体**与**多人社交行为**。
4. **惯性辅助捕获**：GraviCap [8]、CHAIRS [22]、NeuralDome [63] 使用IMU但仅针对单人；本文将其推广至每个可动物体的联合优化。
5. **运动生成**：Human Motion Diffusion Model [46]、Hoi-Diff [37]、InterFusion [11] 聚焦单人或文本驱动单人HOI；本文提出首个**多人多物无结构扩散生成**框架。
6. **多视角重建**：NeuralDome [63]（76相机）提供密集多视角基准；本文以42相机+IMU实现成本更低的规模化采集，并开放单目下游任务。

## 局限性与未来方向
- **室内限定**：42相机 dome 设置难以部署到户外 wild 场景，泛化到非结构化环境仍具挑战。
- **场景覆盖有限**：人力成本高昂，仅涵盖5类日常房间，缺乏室外、公共交通、自然光照变化等场景。
- **固定光照与背景**：数据采集在受控照明条件下进行，背景变化少，限制模型在真实开放环境的泛化能力。
- **后续方向**：扩展至室外户外采集；引入语义/语言条件生成复杂社交叙事；结合物理仿真提升交互合理性。

## 研究启发与可借鉴点
1. **IMU嵌入可动物体**：低成本高精度物体位姿先验思路可直接迁移至RoboTching、AR/VR物体追踪等任务，无需依赖密集摄像头阵列。
2. **一次性多实例预测头**：Center Heatmap + Mesh Map 联合回归设计避免了级联误差，可作为多目标3D感知任务的通用范式（如多车辆-行人联合估计）。
3. **扩散模型中的固定长度填充策略**：将最大5人×10物硬编码为500维向量并通过MLP注入人数/物数条件，是一种简洁的变长社交序列生成方案，可借鉴至多智能体行为合成。
4. **四约束联合优化**（mask+offscreen+collision+smooth）在目标检测缺失场景下仍能稳定跟踪，适用于弱监督/半监督物体追踪研究。
5. **跨数据集联合训练**：在BEHAVE + InterCap + HOI-M³上预训练再在HOI-M³上评测的策略，为小样本新数据集的迁移学习提供了参考路径。

## 关键术语表
- **HOI (Human-Object Interaction)**：人与可动物体之间的空间与运动耦合关系建模。
- **SMPL**：Skinned Multi-Person Linear模型，6090顶点/24关节的参数化人体形状-姿态表示。
- **6D Rotation Representation**：用两个3维向量（列向量）表示旋转矩阵，避免欧拉角万向节锁与角度不连续性。
- **PCK / PCK_abs**：Percentage of Correct Keypoints；rel为根对齐后误差，abs为绝对相机坐标系误差。
- **Chamfer Distance / V2V**：点云间双向最近邻距离均值与平均顶点对顶点距离，评估物体网格重建精度。
- **FID (Fréchet Inception Distance)**：生成分布与真实分布之间的Fréchet距离，越小表示生成质量越高。
- **Pene (Penetration)**：人体signed distance function为正的物体顶点比例，量化物理穿透程度。
- **DDPM (Denoising Diffusion Probabilistic Model)**：通过前向加噪-反向去噪的马尔可夫链学习数据分布的生成模型。

## 可复现要素
- **数据集**：HOI-M³（199序列、181M帧）将公开；代码与预训练模型将在项目页发布（https://juzezhang.github.io/HOIM3_ProjectPage/）。
- **关键超参**：相机数42、分辨率3840×2160@60fps、SMPL N=6090/K=24、扩散表征维度500（5人×88+10物×6）、去噪步数N（论文未明确给出具体值，参见附录）。
- **依赖工具**：ViTPose [57]、Easymocap [2]、SAM [26]、Polycam [39]、RealityCapture 标定。
