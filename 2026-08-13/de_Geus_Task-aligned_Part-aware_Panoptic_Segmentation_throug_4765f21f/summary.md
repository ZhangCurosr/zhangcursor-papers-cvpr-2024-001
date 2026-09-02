---
title: "Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/de_Geus_Task-aligned_Part-aware_Panoptic_Segmentation_through_Joint_Object-Part_Representations_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:16:06"
field: "图像分割"
keywords: ["panoptic segmentation", "part-aware segmentation", "mask classification", "object-part representation", "scene understanding"]
innovations: ["使用共享查询联合预测对象级和部件级分割，使学习目标与PPS任务目标对齐", "在JOPS head中仅预测与对象类别兼容的部件，从源头消除对象-部件不兼容并简化学习任务"]
benchmarks: ["Pascal-PP", "Cityscapes-PP"]
---

# 论文速读：Task-aligned Part-aware Panoptic Segmentation through Joint Object-Part Representations

## 一句话总结
论文提出了 TAPPS（Task-Aligned Part-aware Panoptic Segmentation），通过一组共享查询联合预测对象级分割和对象内部的部件级分割，使学习目标与 PPS 任务目标对齐，在 Pascal-PP 和 Cityscapes-PP 上取得新的 SOTA。

## 研究问题与动机
- **学习目标与任务目标不对齐**：现有 SOTA 方法（如 Panoptic-PartFormer [18,19]）使用独立的查询集分别预测对象级和部件级分割，部件级预测是"对象无关"的语义分割，而非每个对象实例内的部件分割，需要通过后处理规则将部件分配给对象。
- **特征表示冲突**：网络同时学习"区分不同对象实例"和"将同类部件分组在一起"两个矛盾目标，损害了 thing 实例的可分离性。
- **预测不兼容**：分开预测可能导致对象与部件类别不匹配（如 car 对象出现 human-arm 部件），需要丢弃冲突预测的后处理流程。
- **信息利用不充分**：对象级与部件级信息本可互补，但分开查询无法在共享表征中编码这种互补关系。

## 核心贡献（创新点）
- **共享查询联合表示**：提出 TAPPS，用一组共享查询同时表征对象及其包含的部件，使每个查询显式预测一个对象实例及其内部所有兼容部件，学习目标与 PPS 任务目标完全对齐。
- **兼容部件约束**：在 JOPS head 中，基于预预测的对象类别，仅对与该类别兼容的部件类别生成部件查询和掩码，从源头消除对象-部件不兼容问题，简化部件分割学习任务。
- **实例可分离性提升**：以 object-instance-aware 方式联合学习对象和部件，显著提升了 thing 类别的 PQ^Th 指标（Pascal-PP +3.6 / Cityscapes-PP +1.8）。
- **SOTA 结果**：在 Pascal-PP 上 PartPQ 达 60.4（COCO 预训练），相比最强基线提升 +6.3；在 Cityscapes-PP 上 PartPQ 达 64.8，提升 +1.7。

## 方法详解
- **整体架构**：基于 Mask2Former [3] 的 mask classification 元架构，输入图像经 backbone（ResNet-50 / Swin-B）提取多尺度特征，经 pixel decoder（Semantic FPN）得到高分辨率特征 F ∈ R^{E×H×W}；N^q 个可学习查询 Q^0 经 transformer decoder 处理后得到 processed queries Q。
- **JOPS Head（Joint Object and Part Segmentation）**：每个查询 Q_i 并行预测：
  1. **对象类别**：经 single FC 层输出各类别 logits，取最高分作为 ĉ_obj_i。
  2. **对象掩码**：经 3-layer MLP 生成 mask query Q^m_i，与特征 F 做矩阵乘并 sigmoid 激活，得 M̂_obj_i ∈ [0,1]^{H×W}。
  3. **部件掩码**：先经 adaptation MLP（1-2 层）调整查询，再通过 N^pc 个独立 FC 层生成 per-object 部件查询 Q^{pt}_i ∈ R^{N^{pt}×E}；随后根据已预测的 ĉ_obj_i，筛选出与之兼容的 N^c 个部件类别对应的查询 Q^{pt,c}_i，与 F 相乘并 sigmoid 得到兼容部件掩码 M̂_pt_i ∈ [0,1]^{N^c×H×W}。
- **训练损失**：对象级使用交叉熵（类别）+ Dice + CE（掩码）组成 L_obj；部件级同样使用 Dice + CE 组成 L_pt；总损失 L = λ_obj·L_obj + λ_pt·L_pt，实验取 λ_obj = λ_pt = 1。匈牙利匹配基于对象级类别和掩码进行。
- **推理时无需后处理**：由于每个查询显式绑定一个对象及其兼容部件，自动满足 PPS 的 part-whole 约束。

## 实验与结果
- **数据集**：
  - Pascal-PP：59 个对象级类别（39 stuff + 20 thing），57 个部件类别；训练集 4998 张，验证集 5105 张。另评估 Pascal-PP-107（107 个非背景部件类别）以增加挑战性。
  - Cityscapes-PP：19 个对象级类别（11 stuff + 8 thing），23 个部件类别；训练集 2975 张，验证集 500 张。
- **评估指标**：PartPQ（主指标）、PartSQ^Pt（对象内有部件类的 mIoU）、PQ^Th / PQ^St（对象级 panoptic quality）。
- **主要结果（Pascal-PP，COCO 预训练）**：
  - TAPPS（Swin-B）：PartPQ^Pt = **72.2**，PartPQ^NoPt = 56.3，PartPQ^All = **60.4**，PartSQ^Pt = 78.1，PQ^All = 63.0；较 Panoptic-PartFormer++ 提升显著。
  - TAPPS（RN-50）：PartPQ^Pt = **67.2**，PartPQ^All = 54.7，PartSQ^Pt = 75.1，PQ^All = 57.7。
- **主要结果（Cityscapes-PP，COCO 预训练）**：
  - TAPPS（Swin-B）：PartPQ^Pt = **53.0**，PartPQ^All = **64.8**，PartSQ^Pt = 68.0，PQ^All = 68.0。
  - TAPPS（RN-50）：PartPQ^All = 61.3。
- **与 JPPF 对比**：在 Pascal-PP 上 TAPPS（RN-50）PartPQ^Pt = 59.6，大幅超越 JPPF（EfficientNet-B5）的 48.3；Cityscapes-PP 上与 JPPF 相当但 Backbone 更轻量。
- **消融**：仅在训练时约束兼容部件可提升 PartSQ^Pt（+1.4），测试时约束使冲突率从 66% 降至 0%；1-2 层 adaptation MLP 足够；固定部件查询优于动态查询（PartSQ^Pt 75.1 vs 73.1）。

## 相关工作脉络
- **PPS 任务奠基**：De Geus et al. [5]（CVPR2021）首次定义 PPS 任务及 PartPQ 指标，提出基于规则的部件-对象匹配后处理策略。
- **多任务融合方法**：JPPF [10]（ICPRAM2022）使用共享编码器 + 独立语义/实例/部件头，提出改进的规则融合策略；TAPPS 以端到端方式替代规则后处理。
- **Transformer-based SOTA**：Panoptic-PartFormer [18]（ECCV2022）和 Panoptic-PartFormer++ [19]（arXiv2023）使用独立 learnable queries 分别处理 thing/stuff/parts；TAPPS 指出其学习目标与任务目标不对齐的根本缺陷，并以联合查询机制超越。
- **级联范式**：ViRReq [39]（CVPR2023）以 cascading 方式按需分割部件，需多个网络；TAPPS 在单网络内完成端到端联合预测，推理更高效。
- **统一分割框架**：OneFormer [11]（CVPR2023）、K-Net [50]（NeurIPS2021）等探索统一多任务分割，TAPPS 在 PPS 这一特定任务上针对对象-部件层级关系进行了专门设计。
- **实例感知部件分割**：人体解析方向（如 Self-Correction [16]、Holistic Parsing [17]）关注 instance-aware part segmentation 但不覆盖 background stuff；TAPPS 完整覆盖 PPS 的双层抽象（对象 + 部件 + 背景）。

## 局限性与未来方向
- **部件类别固定**：当前方法依赖预定义的部件类别层级，不支持开放词汇（open-vocabulary）部件识别。
- **数据集规模限制**：Cityscapes-PP 仅 2975 张训练图且场景单一（街道），提升空间受限；更复杂场景下的泛化性有待验证。
- **动态部件查询实验不佳**：消融显示固定部件查询优于动态查询，说明对 PPS 任务而言，固定兼容约束比动态分配更有效，但未深入探索混合策略。
- **未来方向（作者提及）**：扩展到更多抽象层次（如子部件层级）、支持更灵活的类别层次结构、开放词汇部件分割等。

## 研究启发与可借鉴点
- **学习目标与任务目标对齐原则**：TAPPS 的核心洞察——当任务要求 X 属于 Y 的从属关系时，应在表征层面强制这种绑定而非事后匹配——可推广至其他层级分割/检测任务（如场景图生成、层级实体识别）。
- **兼容约束简化学习任务**：仅预测与对象类别兼容的部件子类，避免模型学习"空掩码"，这一思路可用于任何具有预定义类别层次的结构化预测任务。
- **固定查询 vs 动态查询的取舍**：消融表明对结构化层级预测，固定查询（per-class）配合兼容掩码比动态分配更稳定，为后续设计提供反面教训参考。
- **与 Mask2Former 的即插即用性**：TAPPS 完全基于公开代码 Mask2Former [3] 构建，JOPS head 可视为一个轻量的后处理模块，易于迁移到其他 mask classification 基线上。
- **实例可分离性的间接提升**：联合学习部件信息意外提升了 thing 实例的 PQ，说明部件级监督可作为辅助信号改善实例分割，值得在多任务学习中进一步探索。

## 关键术语表
- **Part-aware Panoptic Segmentation (PPS)**：同时在对象级（thing instances + stuff regions）和部件级（对象内部的语义部件）进行分割，并要求部件显式链接到其父对象的多层级分割任务。
- **PartPQ (Part-aware Panoptic Quality)**：PPS 任务的标准评估指标，综合衡量对象级识别/分割质量和对象内部件级分割质量。
- **Mask Classification Framework**：以 learnable queries 为核心、通过 bipartite matching 将查询匹配到 ground-truth 对象、联合预测类别和像素级掩码的统一分割范式（Mask2Former 为其代表）。
- **JOPS Head (Joint Object and Part Segmentation Head)**：TAPPS 提出的核心模块，对每个共享查询同时输出对象类别、对象掩码和兼容部件掩码。
- **Object-instance-aware**：部件分割在每个对象实例内部独立进行，而非跨对象聚合为统一的部件语义图。
- **PartSQ^Pt**：仅针对有部件类别的对象级类别，计算 TP 预测内部件级别的 mIoU，反映对象内部部件分割精度。
- **Pascal-PP / Cityscapes-PP**：两个主流 PPS 基准数据集，前者来自 Pascal VOC 扩展（59 object classes, 57 part classes），后者来自 Cityscapes 街道场景扩展（19 classes, 23 part classes）。
- **Bipartite Matching**：在训练期间将预测查询与 ground-truth 对象建立一对一匹配的最优分配算法，用于监督学习和避免重复预测。

## 可复现要素
- **数据集**：Pascal-PP [1,5,6,29] 和 Cityscapes-PP [4,5]，均为公开基准（需申请访问）。
- **代码**：已开源，见 https://tue-mps.github.io/tapps/，基于 Mask2Former 公开代码构建。
- **权重**：论文未提供预训练权重下载链接，但提供了完整训练配置。
- **关键超参**：batch size=16，4×A100 GPU；AdamW，weight decay=0.05，初始 lr=10⁻⁴，polynomial decay power=0.9；λ_obj=λ_pt=1；N^q 与 Mask2Former 默认一致（论文未明确给出具体数值，参见 supplementary）；Cityscapes-PP 训练 90k iters，Pascal-PP ImageNet 预训练 60k iters / COCO 预训练 10k iters。
