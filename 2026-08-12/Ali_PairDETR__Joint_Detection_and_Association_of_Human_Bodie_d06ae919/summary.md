---
title: "PairDETR : Joint Detection and Association of Human Bodies and Faces"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ali_PairDETR__Joint_Detection_and_Association_of_Human_Bodies_and_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:15:16"
---

# 论文速读：PairDETR : Joint Detection and Association of Human Bodies and Faces

## 一句话总结
本文提出 PairDETR，一种基于 Deformable DETR 的端到端目标对检测与关联框架。通过引入“相对预期”虚拟节点将 NP-hard 的图匹配问题近似为多项式时间的二分匹配，在 CrowdHuman 和 CityPersons 上以可比检测精度（AP）达成了当前最优的面部-人体关联性能（mMR⁻²），彻底摆脱了传统非可微后处理流水线。

## 研究问题与动机
- **核心问题**：如何在保持 DETR 端到端训练优势的同时，联合检测多个目标并建立它们之间的归属关联（如人体与人脸同属一人）。
- **现有方法不足1**：传统检测器（Faster R-CNN、YOLO 等）依赖锚框与 NMS 等非可微后处理，难以直接扩展到多对象关联任务。
- **现有方法不足2**：直接扩展 DETR 的集合预测到图预测会因 GT 节点类型不一致（完整对 vs 单目标）导致边容量混用，演变为 NP-hard 的 FixedCost MaxFlow 问题，无法在多项式时间内求解。
- **现有方法不足3**：已有联合方法（如 BFJ、BPJ）采用“独立检测+嵌入匹配/Head Hook”的流水线，缺乏联合优化；HOTR 等方法假设输出恒为交互对，无法处理仅单目标可见的遮挡场景。

## 核心贡献（创新点）
- **端到端图预测框架**：将 DETR 的查询机制从单目标扩展至对象对，在不引入 Relation Net 或显式后处理的情况下实现检测与关联的联合优化。
- **近似二分匹配策略**：通过引入“相对预期类”节点统一匹配边容量为 2，将 NP-hard 固定代价流问题多项式化，使 Hungarian 算法可直接用于图匹配。
- **自适应相对点机制**：针对 Deformable Attention 动态切换采样参考中心（Face-Body 对取 Face 点，Head-Body 对取 Body 点），显著提升遮挡场景下的定位与关联精度。
- **弱标注友好设计**：证明可用几何规则生成的 Body-relative expected face 替代真实 Head 标注，在几乎不损失性能的前提下降低对稀缺标注的依赖。

## 方法详解
- **基础架构**：以 Deformable DETR 为骨干（ResNet-50 + Encoder-Decoder Transformer），每个 Query 输出一个配对（Body + Face/Head），匹配类别总数为 3（Face、Body、Head-relative-Face），若存在无 Body 场景则扩展至 4 类。
- **NP-hard 建模**：真实 GT 图中节点分为类型 A（单目标 $\mu$ 或 $\nu$）与类型 B（对 $\mu$-$\nu$），预测边与 GT 边的容量不同（1 vs 2），直接求解 FixedCost MaxFlow 是 NP-hard。
- **近似匹配代价函数**：当某部件不可见时，添加“相对预期节点”$\hat{\mu}$ 或 $\hat{\nu}$ 统一容量。匹配代价定义为：
  $C_{\mathrm{pair}} = f_1 C_\mu + (1-f_1) C_{\hat{\mu}} + f_2 C_\nu + (1-f_2) C_{\hat{\nu}}$
  其中 $f_1,f_2 \in \{0,1\}$ 指示是否为原始节点。代价沿用 Deformable DETR 的 $\ell_1$ + GIoU 组合，随后使用 Hungarian 算法求解线性分配。
- **两阶段训练**：Stage 1 使用 CO
