---
title: "DiffusionRegPose-Enhancing-Multi-Person-Pose-Estimation-usin"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:41:42"
field: "多目标姿态估计"
keywords: ["multi-person pose estimation", "diffusion model", "end-to-end regression", "crowded scene", "keypoint completion"]
innovations: ["首次将单步端到端关键点回归转换为扩散去噪采样过程", "引入人检-姿态联合交互注意力（H2K）双向提升检测与去噪鲁棒性", "基于 MLE 的概率不可见关键点补全以缓解遮挡歧义"]
benchmarks: ["MS COCO val2017", "CrowdPose test set"]
---

# 论文速读：DiffusionRegPose-Enhancing-Multi-Person-Pose-Estimation-using-a-Diffusion-Based-End-to-End-Regression

## 一句话总结
论文提出 DiffusionRegPose，首次将单步端到端关键点回归框架转换为基于扩散的去噪采样过程，通过概率化不可见关键点补全与人检-姿态联合交互注意力，有效缓解拥挤/遮挡场景下的姿态歧义，在 CrowdPose 上 AP 提升 4.0%。

## 研究问题与动机
1. 现有单步确定性回归方法在拥挤或遮挡场景下容易漏检或误检关键点，因其无法对姿态歧义进行概率推理。
2. Heatmap 类方法在关节坐标外推（如人物截断到图像外）上表现弱于直接回归方法。
3. 现有扩散姿态工作（如 DiffusionPose）为自顶向下范式，需额外人体检测器，并非真正端到端。
4. 3D/时序扩散姿态方法高度依赖上游 2D 关键点质量，难以直接解决 2D 多人姿态本身的歧义问题。

## 核心贡献（创新点）
1. **将单步端到端回归转换为扩散采样过程**：首次把多人姿态估计由确定性去噪回归变为可推理多模态姿态分布的扩散采样。与 PETR/ED-Pose 等确定性单步方法本质不同，本文通过多步去噪显式建模姿态后验分布。
2. **人检-姿态联合交互注意力（H2K）**：引入 Human-to-Keypoint 特征扩展与跨注意力，使人检框与去噪关键点双向对齐，相比 ED-Pose/GroupPose 无显式交互设计，本文在拥挤场景同时提升检测 AP 与关键点 AP。
3. **基于 MLE 的概率不可见关键点补全**：对遮挡导致的不可见关键点用可见点的高斯分布做最大似然估计，给出"合理"初始坐标而非默认零值。与 Dataset 中默认 zero-pad 设定相比，能避免多关键点混淆并缓解大偏差，显著改善训练曲线。

## 方法详解
**整体框架**：DiffusionRegPose 由 Keypoint Completion → Forward Diffusion Process → Model Forward Process → Reverse Diffusion Process 四个环节组成。

1. **固定候选数与 Padding**：每张图像的候选姿态 pad 到固定数量 $N_i$（默认 100）。训练阶段 GT 经 padding 作为扩散初始状态 $y_0$。

2. **前向扩散（加噪）**：
   - $y_t = \sqrt{\gamma_t}(\zeta \cdot y_0) + \sqrt{1-\gamma_t}\epsilon$，$\epsilon \sim \mathcal{N}(0,I)$，其中 $\zeta$ 为信号缩放系数（默认 5.0，高于图像生成常用的 1.0）。
   - 对加噪后的 $y_t$ 做 Keypoint Self-Attention (K-SA)，得到 $Q_{CK}$。

3. **模型前向（特征与交互）**：
   - 图像经 Backbone 与 Transformer Encoder 得到 token $F$。
   - Human Detection Decoder $D_H$ 得到 $F_H$，再经 Human-to-Keypoint 扩展得到 $F_{H2K}$。
   - Keypoint Cross-Attention (K-CA) 将 $Q_{CK}$ 与 $F_{H2K}$ 做交叉注意力，输出粗关键点 $cKpts$ 与粗人体框 $cBox$。
   - 扩散解码器 D 回归出 $y_t', b_t, c_t$，损失采用 focal loss（分类）+ L1（框回归 + 关键点回归）。

4. **反向扩散（去噪采样）**：推理时采用 DDIM 采样策略，从 Gaussian 噪声开始，逐步去噪 $T$ 步得到最终姿态与检测框。

5. **不可见关键点补全（KC）**：
   - 把图像中部分可见的人体关键点坐标整理成矩阵 $M$，按可见/不可见分区 $Y=[Y_I; Y_V]$。
   - 计算 $Y_V$ 的均值 $\mu$ 与协方差 $\Sigma$，用 Cholesky 分解 $\Sigma^{-1}=LL^\top$，以最小二乘求解 $Y_I$，得到 $Y_{MLE}$ 作为扩散初始化的合理坐标。

## 实验与结果
**数据集**：MS COCO val2017、CrowdPose test set。
**Backbone**：ResNet-50；**训练**：80 epoch、batch=8、AdamW、lr=2e-4（30/65 epoch 各衰减 0.1）。

- **COCO**：AP 72.5，AP50=89.8，AP75=79.5，APb=46.5。相比 ED-Pose（71.6）提升 0.9%，相比 GroupPose（72.0）提升 0.5%；优于 PETR（68.8）3.7%。即便使用更弱的 ResNet-50，仍超过采用 HRNet-w32 的自底向上方法至少 2.9% AP。
- **CrowdPose**：AP 72.7，APb=63.1。相比 ED-Pose（69.9）提升 2.8%；相对 SimpleBaseline（60.8）提升 11.9%。按拥挤度细分：AP_E 提升 1.6%，AP_M 提升 2.7%，AP_H 提升 **4.0%**；检测 AP 提升 2.9%，体现对遮挡场景的显著鲁棒性。
- **消融**：信号缩放 $\zeta=5$ 最优；Empty padding 与 GT repeat 相当；$N_i=100$ 优于 50/200。

## 相关工作脉络
1. **PETR / QueryPose / ED-Pose / GroupPose**：均为一步端到端回归或 DETR 类框架；本文与之不同的核心在于引入扩散去噪过程以建模姿态多模态歧义，并在 decoder 中显式建立人检与关键点的双向注意力。
2. **DiffusionPose（自顶向下）**：在 heatmap 上做扩散去噪，但需额外人体检测器，非端到端；本文是真正的端到端单步 pipeline。
3. **Diff-Pose / Diffupose / D3DP**：面向 3D 或多假设姿态估计；本文聚焦 2D 多人姿态的 2D 歧义去噪，且无需 3D/时序先验。
4. **DiffusionDet**：将物体检测建模为去噪过程，启发本文把姿态估计同样视为从噪声到目标的去噪回归。
5. **Top-down 方法（SimpleBaseline / HRFormer-B）**：依赖强人体检测器；本文在 AP_b 弱于它们的条件下仍逼近其姿态性能，证明去噪扩散对遮挡场景的补偿能力。

## 局限性与未来方向
1. 人体检测 AP 仍明显低于专业 top-down 检测器，表明检测头仍有提升空间。
2. 扩散多步采样带来额外推理开销，在实时性要求严格的场景下需优化或蒸馏。
3. 不可见关键点用全局可见点的高斯统计补全，可能忽略个体姿态特异性（如动作极端情形）。
4. 仅验证于 2D 场景，向 3D、视频时序或域自适应推广尚待探索。

## 研究启发与可借鉴点
1. **扩散去噪用作端对端回归的"歧义推理器"**：可将确定性 one-stage 检测/回归头改造为多步去噪，适用于任何存在多解/遮挡歧义的任务。
2. **人检-细粒度预测的联合注意力交互**：H2K 扩展模块可迁移至目标跟踪、实例分割等需要实例框与细粒度属性对齐的场景。
3. **概率化缺失项补全（MLE + Cholesky）**：该无参数补全策略通用性强，可应用于缺失特征、缺失关节、缺失像素等任务。
4. **信号缩放系数适配任务密度**：姿态/检测等稀疏任务的 $\zeta$ 宜高于图像生成，提示 diffusion hyperparameter 应按下游任务稀疏度调参。

## 关键术语表
**One-stage end-to-end**：无需两阶段检测或分组后处理，直接从图像回归出全部人体姿态与检测框的端到端范式。
**Diffusion probabilistic model (DDPM)**：通过学习去噪过程来建模数据分布，前向逐步加噪、反向逐步去噪的生成模型。
**Key-point completion via MLE**：基于可见关键点的高斯统计，对不可见关键点做最大似然估计的合理坐标补全方法。
**K-SA / K-CA**：Keypoint Self-Attention / Cross-Attention，分别在加噪关键点内部、关键点与人检特征之间建立注意力交互。
**Signal scale $\zeta$**：控制前向扩散中信噪比比例的超参，决定 GT 初始信号被保留的程度。
**DDIM sampling**：去噪扩散隐式模型的非 Markov 采样策略，可在更少步数下近似真实去噪轨迹。

## 可复现要素
- **数据集**：MS COCO、CrowdPose（均为公开数据集）。
- **代码**：论文声明代码将在 https://github.com/cici203/DiffusionRegPose 开源。
- **权重**：论文未提及。
- **关键超参**：Backbone ResNet-50；80 epoch；batch=8；AdamW lr=2e-4，30/65 epoch 各 ×0.1；$N_i=100$；signal scale $\zeta=5$；padding 默认 empty pose；推理步数 $T$ 见 supplement（本文正文未列明具体数值）。
