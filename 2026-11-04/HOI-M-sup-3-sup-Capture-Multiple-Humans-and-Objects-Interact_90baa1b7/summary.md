---
title: "HOI-M-sup-3-sup-Capture-Multiple-Humans-and-Objects-Interact"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_HOI-M3_Capture_Multiple_Humans_and_Objects_Interaction_within_Contextual_Environment_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:02:11"
field: "多主体人体-物体交互感知与生成"
keywords: ["Human-Object Interaction", "Motion Capture", "Diffusion Model", "Multi-person Multi-object", "3D Reconstruction", "Dataset"]
innovations: ["首个多人多物3D MoCap数据集HOI-M³（181M帧/42视角/90物体）", "IMU辅助多目标联合优化跟踪管线", "单目一步式多人多物捕捉与未结构化扩散生成基线"]
benchmarks: ["HOI-M³ Capture Benchmark (PCK_rel/PCK_abs/Chamfer/V2V)", "HOI-M³ Generation Benchmark (FID/Pene)"]
---

# 论文速读：HOI-M³: Capture Multiple Humans and Objects Interaction within Contextual Environment

## 一句话总结
本文提出了首个面向多主体多物体交互（Multiple huMans and Multiple objects, HOI-M³）的大规模3D动作捕捉数据集，包含181M帧、42视角、199个序列，并以此为基础设计了单目多HOI捕捉与未结构化多HOI生成两项新任务及强基线方法，填补了该领域的数据与方法空白。

## 研究问题与动机
- **数据稀缺限制多主体交互建模**：现有HOI MoCap数据集主要关注单人与单一物体的交互，缺乏多主体、多物体同时交互的高质量标注数据，导致数据驱动的HOI建模难以扩展到真实社交场景。
- **密集遮挡挑战精确捕捉**：日常情境下的多人多物交互存在严重遮挡问题，需要穹顶式密集相机阵列（dome-like dense cameras）及物体挂载IMU等多模态输入才能获得准确轨迹，现有方法难以应对。
- **现有方法局限于单对象场景**：已有数据驱动的MoCap进展（如PHOSA、CHORE等）仅适用于单-person scenarios，无法建模多人与多物体间的复杂空间与物理关系。
- **缺少面向生成任务的数据基准**：当前生成模型（如MoGen）仅针对单人运动或单物体设计，尚无支持多人多物交互生成的数据集与评估标准。

## 核心贡献（创新点）
1. **首个多主体多物体交互3D MoCap数据集 HOI-M³**：与已有HOI数据集聚焦单人单物不同，本数据集每序列至少包含2人5物，覆盖90类日常物体与31名受试者，规模最大且模态最丰富。
2. **基于IMU辅助的鲁棒多目标跟踪管线**：通过在预扫描物体中嵌入IMU并结合42视角RGB，采用联合优化框架（mask约束+offscreen损失+碰撞惩罚+平滑约束）实现高精度6D位姿估计，显著优于无IMU方案。
3. **单目多HOI一步式捕捉基线**：与现有两阶段/单对象方法（如PHOSA、CHORE）不同，本文设计单前向网络同时预测人-物中心heatmap、人体SMPL网格、物体6D姿态，支持一般场景下的多人多物3D重建。
4. **首个未结构化多人多物交互生成模型**：将扩散模型扩展至高维（500维）联合表示，以物体几何（PointNet提取）与数量作为条件，生成语义一致的多人多物互动序列，填补生成领域空白。

## 方法详解
### 数据采集系统
- 部署42台Z CAM电影级相机（4K分辨率，60fps）形成穹顶式密集视角；
- 对90种日常物体进行Polycam预扫描获取高精度mesh模板；
- 在每个物体内部嵌入IMU传感器，通过受试者起跳动作实现RGB-IMU时间同步，并使用三角测量校准IMU-RGB刚体偏移。

### 多人运动捕捉
- 使用ViTPose检测2D人体关键点，构建跨视角亲和矩阵进行多视图匹配；
- 通过三角化重建3D关键点轨迹，并用Easymocap将SMPL参数模型（N=6090顶点，K=24关节）拟合到3D关键点。

### IMU辅助多物体跟踪
- 物体每帧状态由平移 $T_t \in \mathbb{R}^3$ 和旋转偏移 $R_t^{\text{off}}$ 表征，IMU提供可靠旋转先验 $R_t^{\text{IMU}}$；
- 优化目标（公式2）联合四类能量项：
  - **Mask约束** $E_{\text{mask}}$（公式3）：基于42视角可微渲染与SAM分割mask的L2距离；
  - **Offscreen损失** $E_{\text{offscreen}}$（公式4）：惩罚超出视锥体的投影坐标；
  - **碰撞约束** $E_{\text{collision}}$：防止人与物/物与物穿透；
  - **平滑约束** $E_{\text{smt}}$（公式5）：保持IMU旋转信号的时序平滑性。

### 单目多HOI捕捉
- 输入单帧图像，一次性回归三类maps：人-物中心heatmap（Gaussian分布）、人体mesh map（SMPL参数θ/β+根深度）、物体mesh map（6D旋转+R/T+深度）；
- 总损失函数（公式7）加权组合：$\ell_1$ SMPL参数损失、6D物体pose损失、3D/2D关键点损失、heatmap MSE、深度损失。

### 多HOI生成（扩散模型）
- 联合表示维度500：前440维对应最多5人（每人24×3姿态+10形状+3平移+3方位角），后60维对应最多10物体（各3平移+3姿态角）；
- 条件注入：用MLP编码人数/物数，PointNet提取物体几何全局特征，二者拼接后输入去噪网络；
- 训练损失为标准DDPM重构损失（公式10）：$\mathcal{L} = \mathbb{E}_{x_0,n} ||\hat{x}_\theta(x_n, n) - x_0||_1$。

## 实验与结果
### 数据集统计（对比Table 1）
| 指标 | HOI-M³ | 次优NeuralDome | 提升倍数 |
|------|--------|----------------|----------|
| 帧数(M) | 181 | 71 | 2.55× |
| 录制时长(h) | 20 | 4.3 | 4.65× |
| 物体种类 | 90 | 23 | 3.91× |
| 多主体标记 | ✓ | ✗ | 首次 |

### 单目多HOI捕捉（Table 2）
- **Human PCK_rel**：Ours 68.5 vs PHOSA 43.9 / CHORE 10.4（分别提升 **+56%** / **+558%**）；
- **Object Chamfer**：Ours 235.0 vs PHOSA 1454.3 / CHORE 465.8（分别降低 **-84%** / **-50%**）；
- **Object V2V**：Ours 297.8 vs PHOSA 691.4 / CHORE 340.2（分别降低 **-57%** / **-12%**）；
- 单目绝对深度PCK_abs达5.9（Matched子集3.3），而基线因弱投影模型无法计算该指标。

### 多HOI生成（Table 3）
- **FID（Joint）**：36.906 ± 0.087；
- **Separate FID**：people 16.502 / objects 10.609；
- **Pene（穿透率）**：Joint 9.265%（Separated: people 1.452% / objects 3.887%）。

## 相关工作脉络
1. **NeuralDome [63]**：42视角密集相机但仅支持单人-单物交互；HOI-M³扩展至多人多物且含IMU信号。
2. **PHOSA [65] / CHORE [52]**：SOTA单物HOI单目捕捉方法，依赖预设物体数量且无法处理多人场景；本文一步式方法同时输出多人多物结果。
3. **BEHAVE [4] / InterCap [18]**：公开HOI数据集但均为单人对单物；HOI-M³在规模（帧数↑2.5×）与任务类型（多主体）上形成代际跨越。
4. **HUMANISE [50] / CIRCLE [3]**：聚焦静态场景中的单人交互（HSI），忽略动态多物；HOI-M³强调contextual environment中动态对象与社交行为。
5. **MoGen系列 [9,37,38,46]**：扩散模型应用于单人运动生成；本文将其扩展至高维联合表示（500维），支持多人多物未结构化生成。
6. **GRAB [45] / D3D-HOI [56]**：单人手部-物体交互数据集，侧重细粒度抓握；HOI-M³关注全身多主体协同交互的宏观行为模式。

## 局限性与未来方向
- **室内场景限制**：受限于穹顶相机硬件成本，当前仅覆盖5种室内房间（卧室、餐厅、客厅、健身房、办公室），户外wild场景尚未探索。
- **场景多样性不足**：数据集仅包含5类日常场景且背景变化少，光照条件固定，泛化至其他环境的能力有限。
- **人力成本高**：每序列需专业标注员配合SAM进行初始帧标注，难以快速扩展至大规模开放场景。
- **生成任务处于初步阶段**：FID与Pene指标仍有优化空间，未来可引入文本条件、物理约束或交互语义标签以提升生成可控性。
- **潜在扩展方向**：室外多点云采集、动态光照建模、基于语言描述的细粒度交互生成、虚实融合的沉浸式应用。

## 研究启发与可借鉴点
1. **IMU+密集RGB的混合捕捉范式可迁移**：物体挂载低成本IMU（如Xsens）结合视觉mask约束，能有效缓解遮挡与漂移，适用于任何需要高精度6D追踪的动态物体场景（如机器人抓取、AR交互）。
2. **一步式多实例预测架构**：将多目标center heatmap与网格回归合并到单前向网络，避免级联误差累积，该设计可推广至多目标姿态估计、多人体重建等任务。
3. **扩散模型高维联合表示技巧**：将人数/物数作为MLP embedding、用PointNet编码几何条件，实现了变长实例的统一生成框架，对多智能体仿真、多人运动生成具有参考价值。
4. **标准化评测协议建立**：引入PCK_abs、Chamfer、V2V、FID、Pene等多维度指标，为后续工作提供了可复现的比较基准，建议在本团队相关方向沿用类似评估体系。
5. **开放数据驱动新任务**：首次提出"unstructured generation"概念（无需预定义交互模板），提示未来可用生成模型探索更自由的社交行为合成路径。

## 关键术语表
- **HOI (Human-Object Interaction)**：人与物体之间的物理或语义交互，涵盖抓取、搬运、操作等行为，是本研究的核心建模对象。
- **MoCap (Motion Capture)**：通过标记点、惯性传感器或视觉方法记录人体或物体三维运动的技术，本文指基于密集相机+IMU的全程无标记捕捉。
- **SMPL (Skinned Multi-Person Linear model)**：一种参数化人体形状与姿态模型，用24个关节角度（θ∈R^{24×3}）和10个形状系数（β∈R^{10}）表示Mesh，本文用于人体重建。
- **6D Rotation Representation**：用R∈R^{3×3}列向量形式表示三维旋转（避开万向锁），比欧拉角更易优化，本文用于物体姿态回归。
- **DDPM (Denoising Diffusion Probabilistic Models)**：通过前向加噪与反向去噪学习数据分布的生成模型，本文将其扩展至500维多人多物联合空间。
- **PointNet**：直接处理点云的全局特征提取网络，本文用于从预扫描object mesh提取几何embedding作为扩散模型条件输入。
- **PCK / PCK_abs**：Percentage of Correct Keypoints，衡量3D关键点预测精度；PCK_abs为不含根对齐的绝对坐标版本，更严格评估定位能力。
- **Pene (Penetration Rate)**：生成序列中人体表面顶点穿透物体表面的比例（%），用于量化物理合理性，越低越好。

## 可复现要素
- **数据集**：HOI-M³（199序列，181M帧，42视角，90物体，31受试者）——论文声明将公开发布（https://juzezhang.github.io/HOIM3_ProjectPage/）
- **代码**：单目捕捉pipeline与扩散生成模型代码——论文声明将开源
- **预训练权重**：论文声明将提供pre-trained models
- **关键超参**：
  - 相机：42台Z CAM，4K@60fps
  - 优化权重：λ_mask, λ_offscreen, λ_collision, λ_smt（论文未给出具体数值，见Appendix）
  - 扩散步数N：论文未明确，常规DDPM取1000步
  - 输入分辨率：512×512（捕捉任务）
  - 物体模板：Polycam扫描，类别数90
