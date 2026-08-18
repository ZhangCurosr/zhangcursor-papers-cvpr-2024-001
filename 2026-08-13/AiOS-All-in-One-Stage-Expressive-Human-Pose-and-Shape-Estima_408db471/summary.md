---
title: "AiOS-All-in-One-Stage-Expressive-Human-Pose-and-Shape-Estima"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:00:38"
---

# 论文速读：AiOS: All-in-One-Stage Expressive Human Pose and Shape Estimation

## 一句话总结
本文提出了首个端到端全单阶段（All-in-One-Stage）的表达性人体姿态与形状估计框架 AiOS，彻底摒弃外接检测器与图像裁剪，通过基于 DETR 的渐进式集合预测机制，从原始全帧图像直接联合回归多人的 SMPL-X 全身网格参数，在多项主流基准上刷新 SOTA。

## 研究问题与动机
1. **裁剪操作丢失全局上下文与位置信息**：现有 EHPS 主流方法依赖两步/多阶段范式，先用通用检测器获取边界框再裁剪图像，裁剪会丢弃人体周围的环境线索与空间位置信息，不利于遮挡与密集人群场景的恢复。
2. **检测框噪声导致性能显著退化**：两步法对边界框质量高度敏感，真实场景中检测框的偏差或漏检会直接引发肢体缺失与重建伪影（如 RoboSMPLX 所示），而使用 GT 框评测存在脱离实际应用的偏差。
3. **现有单阶段方法仅限身体部分**：ROMP、BEV 等单阶段网格恢复方法仅针对 SMPL 身体估计，依赖单一中心 heatmap 提取全局特征，无法捕捉手部精细姿态与面部表情所需的细粒度局部特征。
4. **缺乏跨人与跨部位关联建模**：分部位独立回归割裂了人与人之间的空间关系以及同一人体内身体-手-脸的内在关联，易在关节连接处产生不自然的错位与伪影。

## 核心贡献（创新点）
1. **首个无需检测器的全单阶段 EHPS 框架**：将多人表达性人体网格恢复重构为基于 DETR 的渐进式集合预测问题，端到端完成定位与回归，从根本上消除裁剪依赖与误差传播链。
2. **“Human-as-Tokens” 分级查询与特征聚合**：创新性地提出将人体表示为边界框 token 与关节 token 的组合，通过不同位置查询与监督信号分别捕获全局身体上下文与细粒度局部器官特征，填补单阶段方法在局部细节上的短板。
3. **渐进式解码与定制化注意力掩码**：设计三级解码器流水线（Body-location → Body-refinement → Whole-body-refinement），并引入严格的 attention mask（限制跨人关节注意力、保持同主体内 token 互联），在强遮挡与拥挤场景下显著提升多目标分离与回归稳定性。
4. **高质量定位先验可反哺两步法**：AiOS 生成的边界框精度足够高，可直接替换 OSX、SMPLer-X 等两步法的外接检测框，带来额外的性能增益，验证了一阶段联合优化的双重价值。

## 方法详解
- **骨干与编码器**：采用 ResNet-50 提取多尺度特征图，经空间展平与位置编码
