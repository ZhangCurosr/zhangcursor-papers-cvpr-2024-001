---
title: "AiOS-All-in-One-Stage-Expressive-Human-Pose-and-Shape-Estima"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:00:10"
---

# 论文速读：AiOS: All-in-One-Stage Expressive Human Pose and Shape Estimation

## 一句话总结
本文提出了首个无需外部检测器与图像裁剪的**全一体化单阶段（All-in-One-Stage）**表达性人体姿态与形状估计（EHPS）框架 AiOS；该方法基于 DETR 架构将多人网格恢复建模为渐进式集合预测问题，通过多粒度 Token 设计在全帧上联合回归身体、手部与面部 SMPL-X 参数，在多个基准上刷新 SOTA。

## 研究问题与动机
- 现有 EHPS 方法普遍采用两阶段范式：先用独立检测器获取边界框并裁剪图像，再分别用不同子网络回归身体、手、脸参数，导致全局上下文丢失、裁剪引入背景干扰、且部位间/人物间缺乏关联建模。
- 已有的单阶段 EHPS 方法（如 OSX、SMPLer-X）虽取消了部分专家网络，但仍必须依赖边界框进行图像裁剪，在真实噪声框或无框条件下性能显著退化。
- 传统全帧单阶段 HPS 方法（如 ROMP、BEV）仅将人体压缩为单一全局特征向量，无法满足 EHPS 对精细手部姿态与面部表情局部特征的高分辨率需求。
- 拥挤与强遮挡场景下，通用检测器难以准确区分多个人体实例，直接裁剪极易将他人肢体混入图像，造成定位与回归误差累积。

## 核心贡献（创新点）
1. **首个端到端单阶段 EHPS 框架**：彻底摒弃外部检测器与图像裁剪，直接在完整帧上联合恢复多人完整 SMPL-X 网格；与依赖 GT 框或两阶段流程的基线有本质区别。
2. **“Human-as-Tokens”渐进式集合预测机制**：将每位人体抽象为身体/手部/面部位置 Token 与对应关节 Token 的集合，通过分层解码器逐步从全局粗定位推进至局部细粒度特征聚合；与仅依赖中心热图或单一向量的表征方式不同。
3. **混合粒度注意力掩码设计**：限制关节 Token 仅在个体内部交互以降低 $O(N^2)$ 复杂度，同时允许定位 Token 跨人共享上下文，显著增强拥挤场景下的多目标区分能力；与全自由交互或纯单人体注意力方案均有本质差异。

## 方法详解
- **骨干与特征编码**：使用 ResNet-50 提取多尺度特征图 $F_{img}$，沿空间维度展平拼接后叠加位置编码 $PE$，得到图像特征 Token $T_{img} \in \mathbb{R}^{M \times D}$；经 Transformer Encoder 得到增强特征 $T'_{img}$，并通过 FFN 分类筛选 Top $M_h=900$ 个候选人体定位 Token $T_{bl}$。
- **Naive AiOS（基础架构）**：前两层 Body-location Decoder 结合身体位置查询 $Q_{bl}$ 进行自注意力更新，再以条件交叉注意力从 $T'_{img}$ 聚合全局身体特征；后四层 Whole-body-location Decoder 将身体 Token 广播扩展出左手、右手、面部位置 Token，拼接为 $T_{full}$，联合回归各部位边界框并输出 SMPL-X 参数。
- **Full AiOS（完整架构）**：在 Naive 基础上插入 Body-refinement Decoder 与 Whole-body-refinement Decoder。前者利用可学习嵌入 $E_{bj}$ 扩展出 17 个身体关节 Token $T_{bj}$，构成细粒度 Token 集 $T_{bd}=[T_{bl}, T_{bj}, T_{lhl}, T_{rhl}, T_{fl}]$，在第 2 阶段同步回归身体参数；后者进一步扩展手、脸关节 Token，形成完整集 $T_{wd}=[T_{bl}, T_{bj}, T_{lhl}, T_{lhj}, T_{rhl}, T_{rhj}, T_{fl}, T_{fj}]$，在第 3 阶段精调局部特征并完成最终参数回归。第二阶段结束后将 Token 数量下采样至 $M_b=100$ 以降低计算开销。
- **渐进监督策略**：第 1 阶段仅施加定位损失；第 2 阶段监督身体 SMPL-X 参数与身体/手/脸粗定位；第 3 阶段监督完整全身参数。消融表明过早施加几何损失会干扰定位收敛。
- **损失函数**：总体损失为各阶段加权和，包含边界框回归损失 $L_{box}$、人体存在分类损失 $L_{cls}$、2D 关节投影损失 $L_{j2d}$ 以及 SMPL-X 参数损失 $L_{smplx}$（含姿态 $\theta$、形状 $\beta$、表情 $\psi$ 子项）。

## 实验与结果
- **训练与评测设置**：训练集含 AGORA、BEDLAM、COCO、UBody、ARCTIC、EgoBody；测试集含 AGORA、UBody、EHF、ARCTIC、EgoBody、BEDLAM。使用 16×V100，batch size 32，60 epoch 预训练 + 50 epoch 微调。
- **AGORA SMPL-X**：AiOS 取得 NMVE 97.8 mm、NMJE 96.0 mm，较当前 SOTA SMPLer-X 的 NMVE 降低约 **9%**；仅以自身预测框驱动 OSX/SMPLer-X 时两者性能仍提升，证明定位质量高且两阶段方法对框敏感。
- **AGORA SMPL**：$\mathrm{AiOS}_{0.5}$ 取得 NMVE 61.2 mm、NMJE 68.0 mm，较一阶段 HPS 方法 BEV（108.3 mm）提升 **43%**。
- **单场景基准**：UBody 上 PVE 58.6 mm（较 SOTA 降低约 **30%**）；EHF 上 PVE 45.4 mm（降低约 **10%**）；EgoBody 上 PVE 降低约 **3%**。所有对比均未使用 GT 边界框。
- **消融结论**：Full AiOS 较 Naive 稳定提升；渐进式 SMPL-X 监督（第 2、3 阶段）优于全阶段监督或仅第 3 阶段监督；限定关节 Token 仅同主体内自注意力（Ours）显著优于全自由交互（Full）或仅限制跨人（Inter-human Only）。

## 相关工作脉络
- **多阶段 EHPS**（PIXIE、H4W 早期、OSX 前作）：依赖独立检测器与裁剪图像分别训练身体/手/脸专家；本文打破该范式，改为端到端全帧统一回归，避免裁剪带来的信息损失与装配伪影。
- **全帧单阶段 HPS**（ROMP、BEV、TRACE）：基于中心热图提取单一全局向量回归 SMPL；本文指出其对 EHPS 局部细节（手、脸）建模不足，引入多粒度 Token 与关节
