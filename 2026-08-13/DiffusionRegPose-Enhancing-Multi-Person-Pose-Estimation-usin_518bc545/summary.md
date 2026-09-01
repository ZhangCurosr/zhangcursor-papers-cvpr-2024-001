---
title: "DiffusionRegPose-Enhancing-Multi-Person-Pose-Estimation-usin"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:41:38"
field: "多人体姿态估计"
keywords: ["多人体姿态估计", "扩散模型", "端到端回归", "拥挤场景", "关键点去噪", "姿态歧义建模"]
innovations: ["将一阶段端到端关键点回归转化为基于扩散的去噪采样过程，首次用于多人体姿态估计", "提出姿态去噪与人检测之间的双向注意力交互机制，实现 mutual benefit", "提出概率性不可见关键点补全方法，基于高斯分布 MLE 改善遮挡场景学习"]
benchmarks: ["COCO val2017", "CrowdPose test"]
---

# 论文速读：DiffusionRegPose-Enhancing-Multi-Person-Pose-Estimation-usin

## 一句话总结
本文提出了 **DiffusionRegPose**，将一阶段端到端关键点回归模型转化为基于扩散模型的采样过程，通过逐步去噪推理姿态不确定性，有效缓解拥挤/遮挡场景下的漏检与误检问题。在 CrowdPose 上 AP 提升达 **4.0%**（高拥挤场景），COCO 上 AP 达到 **72.5%**。

## 研究问题与动机
- **核心问题**：现有确定性一阶段回归方法在拥挤、遮挡场景中难以处理姿态歧义性，导致漏检和误检频发。
- **现有工作不足**：端到端方法（PETR、QueryPose、ED-Pose）依赖单次回归，无法建模多模态姿态分布；已有的扩散姿态方法（DiffPose、DiffusionPose）依赖外部人体检测器或 2D→3D 转换，非真正端到端。
- **动机**：受 DiffusionDet 启发，将噪声姿态逐步去噪恢复为目标姿态，同时利用姿态去噪与人检测之间的注意力交互实现 mutual benefit。

## 核心贡献（创新点）
- **首次将一阶段端到端关键点回归模型转化为扩散采样过程**：区别于 ED-Pose 等直接回归方法，DiffusionRegPose 以 DDIM 采样方式逐步去噪，能建模姿态后验分布。
- **提出姿态去噪与人检测之间的注意力交互机制**：通过 K-SA 与 K-CA 模块实现两类信息的交叉关注，相互促进，提升拥挤场景下的检测精度与去噪鲁棒性。
- **提出概率性不可见关键点补全方法**：利用可见关键点的均值和协方差矩阵，通过最大似然估计（MLE）+ 最小二乘求解补全遮挡关键点坐标，改善训练初始分布。
- ** CrowdPose 上高拥挤场景 AP 显著提升 4.0%**，同时人体检测 AP 提升 2.9%，验证了方法在极端遮挡场景的有效性。

## 方法详解
- **骨干网络与特征提取**：使用 ResNet-50 作为 backbone，多尺度特征送入 Deformable Attention Encoder 得到 token $F$。
- **正向扩散过程**：对 GT 姿态集合 $y_0$ 加入高斯噪声，引入尺度参数 $\zeta$ 调节信噪比，得到带噪姿态 $y_t = \sqrt{\gamma_t}(\zeta \cdot y_0) + \sqrt{1-\gamma_t}\epsilon$。
- **K-SA 模块**：对带噪姿态 $y_t$ 执行关键点自注意力，生成 query $Q_{CK}$，建模关键点间相关性。
- **人-关键点 Token 扩展**：通过 Human-Detection Decoder $D_H$ 和 Human-to-Keypoint 扩展模块 $D_{H2K}$ 获得 coarse keypoint token $F_{H2K}$，引导去噪过程。
- **K-CA 模块**：交叉注意力 $CA(Q_{K_{SA}}, K_{F_{H2K}}, V_{F_{H2K}})$ 融合姿态 query 与人体检测特征，输出粗粒度关键点 $cKpts$ 和粗人体框 $cBox$。
- **扩散解码器 D**：回归最终关键点坐标 $y'_t$、人体框 $b_t$ 及分类 $c_t$，损失函数包括 focal loss（分类）、L1 loss（框回归和关键点回归）。
- **反向扩散采样（推理）**：从纯高斯噪声出发，采用 DDIM 迭代去噪 T 步逐步恢复姿态，输出 $N_i$ 个候选姿态。
- **关键点补全**：对遮挡不可见关键点，基于可见关键点的 $\mu$ 和 $\Sigma$，使用 MLE + Cholesky 分解的最小二乘解计算补全坐标 $Y_I'$，使初始输入具有合理人体结构先验。
- **Padding 策略**：将 GT 填充至固定数量 $N_i$（默认 100），采用空姿态 padding（全零坐标）效果最优。

## 实验与结果
- **数据集**：MS COCO val2017（17 关键点）、CrowdPose test（14 关键点，含拥挤度划分 E/M/H）。
- **基线**：
  - 端到端一阶段方法：PETR（68.8 AP）、QueryPose（68.7）、ED-Pose（71.6）、GroupPose（72.0）
  - 非端到端：Top-down（SimpleBaseline 70.4，DiffusionPose 75.9）和 Bottom-up（SWAHR 68.9，LOGO-CAP 69.6）
- **COCO 结果**：DiffusionRegPose（ResNet-50）AP = **72.5%**，$\mathrm{AP}_{50}=89.8\%$，$\mathrm{AP}_{75}=79.5\%$，均优于所有端到端方法；超越 GroupPose 0.5 AP。
- **CrowdPose 结果**：AP = **72.7%**，$\mathrm{AP}_{50}=91.1\%$，$\mathrm{AP}_{75}=79.3\%$，超越 ED-Pose（69.9）2.8 AP；其中 $\mathrm{AP}_H$（高拥挤）达 **64.9%**，比 ED-Pose 提升 **4.0%**。
- **人体检测 AP**：COCO 上 $\mathrm{AP}_b=46.5\%$，CrowdPose 上 $\mathrm{AP}_b=63.1\%$（较 ED-Pose 提升 2.9%）。
- **消融结论**：信号尺度 $\zeta=5.0$ 最优；空姿态 padding 效果最佳；$N_i=100$ 时表现最优，过多 query（200）引入噪声反而下降。

## 相关工作脉络
- **PETR / QueryPose / ED-Pose / GroupPose**：端到端一阶段 pose 估计的代表工作，以 set prediction 方式直接回归姿态；本文与它们的本质区别在于将回归过程替换为扩散去噪采样，能显式建模姿态歧义性。
- **DiffusionDet（Chen et al., 2022）**：将扩散模型用于目标检测去噪；本文受其启发，将其思想迁移至多人生理姿态估计领域。
- **DiffusionPose（Qiu et al., 2023）**：Top-down 范式，利用扩散模型从噪声 heatmap 生成关键点；本文与之不同在于无需额外人体检测器，实现真正端到端。
- **DiffPose / Diffupose / D3DP**：面向 3D 姿态估计的扩散方法，依赖 2D 检测结果；本文专注于 2D 多人体姿态估计，且不依赖预检测器。
- **SimpleBaseline / HRNet / HRFormer**：Top-down 非端到端方法，使用 HRNet-w48 等更强骨干；本文以轻量 ResNet-50 即超越或接近部分 Top-down 方法。

## 局限性与未来方向
- **推理需要多步迭代**：T 步去噪过程导致推理延迟高于纯一次回归方法，难以满足极低延迟场景需求。
- **仅针对 2D 姿态估计**：未扩展至 3D 姿态估计或视频序列任务，潜力待挖掘。
- **固定 $N_i$ padding 策略的泛化性**：不同场景下实例数量差异大，固定 padding 策略（空姿态/复制）可能不够灵活。
- **论文暗示可迁移至 3D**：在 Introduction 中提到灵感来自 3D 姿态扩散方法，但未给出 3D 实验验证。

## 研究启发与可借鉴点
- **扩散采样替代确定性回归的思路**：可将此范式迁移到其他结构化预测任务（如关节点检测、人手姿态、动物姿态），用去噪过程替代单次回归以处理歧义性。
- **概率性关键点补全策略**：基于可见关键点高斯分布的 MLE 补全方法可复用到其他存在遮挡的结构化推理任务中，作为数据预处理技巧。
- **检测-预测交互注意力的 mutual benefit 设计**：K-SA 和 K-CA 的双向交互机制可作为通用组件嵌入到其他多任务端到端框架。
- **信噪比超参 $\zeta$ 的调优经验**：目标检测/姿态估计等稀疏任务相比图像生成需要更高 $\zeta$ 值（5~10），这一发现可用于指导其他扩散检测任务的超参选择。

## 关键术语表
- **DiffusionRegPose**：本文提出的方法名，将一阶段端到端关键点回归转化为基于扩散的去噪采样过程。
- **Keypoint Completion（关键点补全）**：利用可见关键点的统计分布（均值和协方差），通过 MLE 和最小二乘法补全被遮挡不可见关键点的坐标。
- **K-SA（Keypoint Self-Attention）**：关键点自注意力模块，在扩散过程中建模带噪关键点之间的内部关联。
- **K-CA（Keypoint Cross-Attention）**：关键点交叉注意力模块，实现去噪姿态 query 与人体检测 token 之间的信息交互。
- **DDIM（Denoising Diffusion Implicit Models）**：一种加速扩散采样的确定性采样策略，被用于本文推理阶段的去噪过程。
- **Signal Scale $\zeta$**：扩散过程中调节信号与噪声比例的关键超参，稀疏任务（检测/姿态）需取较大值（5~10）。
- **AP（Average Precision）**：基于 OKS（Object Keypoint Similarity）指标计算的平均精度，多人体姿态估计的核心评估指标。
- **OKS（Object Keypoint Similarity）**：评估预测关键点与 GT 之间相似度的度量，考虑了关键点标尺归一化，是 COCO/CrowdPose 的标准评估指标。

## 可复现要素
- **数据集**：MS COCO val2017、CrowdPose test；均公开可用。
- **代码**：已开源，GitHub 地址：https://github.com/cici203/DiffusionRegPose
- **权重**：论文未明确说明是否开源，代码仓库链接已给出。
- **关键超参**：Backbone = ResNet-50；optimizer = AdamW（weight decay $1\times10^{-4}$）；epochs = 80；batch size = 8；初始 lr = $2\times10^{-4}$；lr 在第 30 和 65  epoch 各衰减 0.1 倍；实例 query 数 $N_i=100$；信号尺度 $\zeta=5.0$；DDIM 采样步数 T 论文未明确给出具体数值（supplementary material 中应有详细说明）。
