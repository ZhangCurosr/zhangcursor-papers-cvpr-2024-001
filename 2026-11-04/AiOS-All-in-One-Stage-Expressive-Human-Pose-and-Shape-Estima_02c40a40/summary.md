---
title: "AiOS-All-in-One-Stage-Expressive-Human-Pose-and-Shape-Estima"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:56:27"
field: "3D人体姿态与形状估计"
keywords: ["Expressive Human Pose and Shape Estimation", "One-Stage Method", "DETR", "SMPL-X", "3D Human Mesh Recovery", "Multi-person Estimation"]
innovations: ["首个无需检测器裁剪的端到端一阶段EHPS框架", "渐进式Set Prediction结合Global/Local双token表征", "跨人/同人分层注意力掩码机制提升拥挤场景性能"]
benchmarks: ["AGORA", "UBody", "EHF", "ARCTIC", "EgoBody", "BEDLAM"]
---

# 论文速读：AiOS: All-in-One-Stage Expressive Human Pose and Shape Estimation

## 一句话总结
本文提出 **AiOS**（All-in-One-Stage），首个无需额外人体检测步骤、端到端直接在整图上完成多人的表达性人体姿态与形状估计（EHPS）的一阶段方法，基于 DETR 架构引入人形 token 和关节相关 token 的渐进式预测，在 AGORA、UBody、EHF 等多个基准上达到 SOTA。

## 研究问题与动机
- 现有 EHPS 方法普遍采用 **两阶段或多阶段范式**：先用现成检测器裁剪人体，再用独立模型回归身体、手部、面部，导致性能受限。
- **裁剪带来上下文丢失**：裁剪后的人体图像缺少周围场景信息，且在人群拥挤场景中难以区分个体，容易产生干扰和拼接伪影。
- 现有 **one-stage HPS 方法**（如 ROMP、BEV）仅针对全身人体（SMPL）回归，依赖单一的全局特征，无法满足手部、面部等局部精细结构的需求。
- 已有 one-stage EHPS 方法（如 OSX、SMPLer-X）仍需外部提供边界框进行裁剪，在真实场景中精度大幅下降。

## 核心贡献（创新点）
- **首个端到端一阶段 EHPS 框架**：完全摒弃外部检测器与裁剪步骤，直接在整图上完成多人多部位联合估计，与现有两阶段方法有本质区别。
- **渐进式 Set Prediction 设计**：将 EHPS 建模为渐进式集合预测问题，分三阶段（body-location → body-refinement → whole-body-refinement）逐级细化人体位置与特征，与 ROMP/BEV 的全局特征单阶段估计不同。
- **Human-as-Tokens + Joint-related Tokens 双层表征**：引入人体 bounding box token 捕获全局特征，同时引入关节 token 捕获局部细节特征，实现全局与局部特征的融合。
- **跨人/同人注意力掩码机制**：通过精细设计的自注意力掩码，允许跨人交互建模人际关系，同时限制关节 token 仅在同一个体的内部交互以降低复杂度，与全连接 attention 方案相比效果更好。
- **强基线验证**：即便仅使用 naive AiOS 基线也能超越多数 SOTA，完整 AiOS 在 AGORA 上 NMVE 降低 9%、EHF 上 PVE 降低 30%，并可为两阶段方法提供更优的边界框。

## 方法详解
**整体架构**：基于 DETR，包含 ResNet-50 骨干网络、Transformer Encoder、以及多层 Deformable Decoder。

**Encoder 阶段**：
- 多尺度特征图展平并拼接，加入位置编码 $PE$，得到图像特征 token $T_{img} \in \mathbb{R}^{M \times D}$。
- 通过 Transformer Encoder 编码后，使用 FFN 分类筛选 top-$M_h = 900$ 个候选人体 token 作为 $T_{body}$，作为后续解码器输入。

**Naive AiOS（基础版）**：
- **Body-location Decoder（2层）**：从 $T_{body}$ 生成身体位置查询 $Q_{bl}$，通过 self-attention 更新后，以 $Q_{bl}$ 引导 cross-attention 从图像特征中聚合人体区域信息；回归 body box，使用 $L_{box}$ 监督，并过滤保留 top-100 token。
- **Whole-body-location Decoder（4层）**：通过可学习嵌入 $E_{bl}$ 将 body token 扩展为 hand/face location tokens，拼接为 $T_{full}$，并扩展对应 query $Q_{full}$；使用自注意力探索 inter-part 和 inter-human 关系，使用掩码确保只有本人的 hand/face box 可与本人 body box 交互；最终回归 body、hand、face 位置及 SMPL-X 参数，使用 $L_{smplx}$（含 $L_{param}$、$L_{kp3d}$、$L_{kp2d}$）监督。

**完整 AiOS**（在原基础上的改进）：
- **Body-refinement Decoder（2层）**：在原有 body token 基础上，引入 17 个可学习嵌入 $E_{bj}$ 扩展出 body joint tokens $T_{bj}$，形成细粒度 token 集 $T_{bd}$；在约束同一个人内部 joint attention 的前提下进行 self-attention；同时回归 body/hand/face box（$L_{box}$）和 2D joint 位置（$L_{j2d}$），并在本阶段对 body 参数施加 $L_{smplx}$ 监督。
- **Whole-body-refinement Decoder（2层）**：进一步扩展出手部和面部的 joint tokens，形成完整 token 集 $T_{wd}$；回归所有 box（$L_{box}$）、所有 joint（$L_{j2d}$）和完整的 SMPL-X 参数（$L_{smplx}$）。

**损失函数**：总体损失为各阶段各项损失的加权和，包括 $L_{box}$、$L_{cls}$、$L_{j2d}$、$L_{smplx}$ 等。

## 实验与结果
**训练数据集**：AGORA、BEDLAM、COCO（多人场景）；UBody、ARCTIC、EgoBody（单人场景）。

**测试数据集**：AGORA、UBody、EHF、ARCTIC、EgoBody、BEDLAM。

**评估指标**：PA-MPVPE、MPVPE、NMVE、NMJE、F-score、Precision、Recall 等，单位为 mm。

**主要结果**：
- **AGORA SMPL-X**：AiOS 取得 NMVE = 97.8 mm（SOTA），较 SMPLer-X 降低约 9%；$AiOS_{0.3}$ 取得 NMJE = 96.0 mm。
- **AGORA SMPL**：AiOS 相对 BEV 的 NMVE 降低 43%（108.3 → 61.2 mm），NMJE 降低 40%（113.2 → 68.0 mm）。
- **UBody**：AiOS 取得 PA-PVE = 32.5 mm（All）、PVE = 58.6 mm，超越 SMPLer-X。
- **EHF**（零样本泛化）：AiOS 取得 PVE = 45.4 mm，较 SMPLer-X 降低约 30%。
- **ARCTIC**：PVE 降低约 10%。
- **EgoBody**：PVE 降低约 3%。
- **AiOS 检测框质量**：将 AiOS 产生的 box 用于 OSX/SMPLer-X，使两者 MVE 进一步提升，证明定位质量的优越性。

**消融实验**：
- Joint token 的引入显著提升了各部位精度，尤其对全身整体提升明显。
- 渐进式 SMPL-X 监督（第 2/3 阶段而非全部阶段）优于仅在最后阶段监督或全程监督。
- 原始的 attention 掩码设计（同 person 内 joint attention）优于 full attention 和仅 inter-human 限制。

## 相关工作脉络
- **ROMP**（one-stage HPS）：首次尝试从整帧恢复多人人体网格，但仅基于单一全局特征向量，无法处理精细的 EHPS 任务。
- **BEV / TRACE**：引入 BEV 或时序信息辅助 HPS，但仍局限于 SMPL 级身体估计，缺乏手部和面部。
- **OSX**：首个 one-stage EHPS 方法，去除了 part expert 但依然依赖边界框裁剪，对检测框噪声敏感。
- **SMPLer-X**：利用大规模视觉模型和数据的 one-stage EHPS，仍需要 bounding box 作为输入，本文证明去除裁剪可获得更强泛化。
- **HybrIK-X / H4W / CLIFF**：Top-down 两阶段方法的代表，依赖高精度边界框和裁剪，在嘈杂场景下表现下降。
- **DINO / ED-Pose / DAB-DETR**：本文架构借鉴来源，将 DETR 系列的目标检测思想推广至 3D 人体网格估计。

## 局限性与未来方向
- 模型性能仍有较大提升空间，尤其是在训练数据中加入更多含多人的真实场景数据。
- 当前方法未涉及时序信息，难以直接应用于视频或跟踪任务。
- 对于低分辨率图像下手部估计的优化尚未充分探索。
- 可扩展至更多维度的感知任务，如多人 3D 跟踪和 3D 定位。

## 研究启发与可借鉴点
- **渐进式 Set Prediction 范式**：将复杂的 3D 人体估计任务分解为多个渐进阶段（定位→细化→回归），每一阶段引入更细粒度的 token，这一设计思路可迁移至其他多目标、多部位估计任务。
- **Joint-related Token 引入局部特征**：用可学习的 joint embedding 扩展 global location token 以捕获局部上下文，对类似任务的局部-全局特征融合具有参考价值。
- **Attention 掩码的精细化设计**：通过区分 inter-human 和 intra-human attention 路径，既保留了人际关系建模能力，又控制了计算复杂度，为多目标 Transformer 设计提供了新思路。
- **检测框质量的间接验证**：将自家检测框提供给两阶段 SOTA 方法使用并取得提升，这种"检测器即服务"的思路可用于评估自身定位性能。
- **团队结合点**：可将渐进式 token 扩展思路与手势估计、表情估计等细粒度任务结合，或与追踪、3D 重建等下游任务做统一框架设计。

## 关键术语表
- **EHPS（Expressive Human Pose and Shape Estimation）**：表达性人体姿态与形状估计，指从单目图像中联合估计人体全身、手部和面部的 3D 网格参数。
- **SMPL-X**：可扩展的人体参数化模型，同时编码身体姿态、形状、手部姿态和面部表情。
- **DETR（Detection Transformer）**：基于 Transformer 的端到端目标检测框架，将检测视为直接集合预测问题。
- **Naive AiOS vs. Full AiOS**：Naive 版本仅含 body-location 和 whole-body-location 解码器；Full 版本额外引入了 body-refinement 解码器和 joint tokens。
- **PA-MPVPE / NMVE**：Procrustes 对齐后的顶点位置误差；归一化顶点误差（综合考虑了检测和重建精度）。
- **Progressive Set Prediction**：渐进式集合预测，指分阶段逐步细化检测结果（从粗定位到细粒度回归）的预测范式。
- **Deformable Decoder**：可变形解码器，通过可学习的参考点和采样机制高效地从图像特征中提取局部区域特征。

## 可复现要素
- **数据集**：AGORA、BEDLAM、COCO、UBody、ARCTIC、EgoBody、EHF（部分数据集公开，部分需申请）。
- **代码与权重**：论文主页 https://ttxskk.github.io/AiOS/，代码开源信息论文未明确说明。
- **训练配置**：16×V100 GPU，batch size=32；先在 AGORA+BEDLAM+COCO 上训练 60 epoch，再在所有数据集上 fine-tune 50 epoch。
- **骨干网络**：ResNet-50。
- **关键超参**：候选人体 token 数 $M_h = 900$，过滤后保留 $M_b = 100$。
