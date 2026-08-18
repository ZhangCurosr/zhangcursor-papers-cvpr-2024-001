---
title: "PairDETR : Joint Detection and Association of Human Bodies and Faces"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ali_PairDETR__Joint_Detection_and_Association_of_Human_Bodies_and_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:00:55"
field: "行人检测与关联"
keywords: ["目标检测", "端到端关联", "DETR", "图预测", "人体检测", "近似二分匹配", "CrowdHuman", "CityPersons"]
innovations: ["提出近似二分匹配损失将NP-hard配对匹配问题映射至多项式时间求解", "在DETR框架内实现无需后处理的端到端body-face联合检测与关联", "用几何近似替代缺失标注几乎无损地降低标注依赖"]
benchmarks: ["CrowdHuman", "CityPersons"]
---

# 论文速读：PairDETR: Joint Detection and Association of Human Bodies and Faces

## 一句话总结
论文提出了 PairDETR，一种基于 DETR 架构的端到端人体与人脸联合检测和关联方法，通过近似二分匹配损失解决了图预测中的 NP-hard 问题，在 CrowdHuman 和 CityPersons 数据集上均达到最优的人体-人脸关联性能。

## 研究问题与动机
1. **现有检测方法孤立处理目标**：常见的人体检测模型只分别检测身体、人脸或头部，不考虑目标之间的关系。
2. **后处理依赖人工设计**：主流方案先独立检测再使用非可微的后处理（如 NMS+重叠匹配）来关联，无法端到端训练。
3. **直接扩展 DETR 到图预测是 NP-hard**：当预测目标是"对（pair）"而真值中部分节点是单目标时，二分匹配图不再二分，固定代价最大流问题是 NP-hard，无法在多项式时间内求解。
4. **小目标与遮挡场景的挑战**：crowd 场景下存在大量部分可见、遮挡的人脸/身体，需要模型具备鲁棒的关联能力。

## 核心贡献（创新点）
1. **提出 PairDETR 端到端联合检测与关联框架**：在 Deformable DETR 基础上扩展为图预测模型，每个查询独立预测一个（身体，人脸/头部）对，无需任何手设计的后处理。
2. **引入近似二分匹配损失以多项式时间求解 NP-hard 配对匹配问题**：通过"相对估计类"节点（如用头部标注近似缺失的人脸）统一边容量，将固定代价最大流映射回二分匹配，利用匈牙利算法求解。
3. **在 CrowdHuman 和 CityPersons 上均达到 SOTA 关联性能**：CrowdHuman mMR⁻² 达 42.9%（较 BFJ 提升 33.3%），同时保持与仅训练检测的模型相当甚至更优的 AP。
4. **证明头部标注非必需，可用几何近似替代**：用身体顶部居中方块（宽度比 α=0.3）生成的"预期人脸"近似替换真实头部标注，mMR⁻² 仅下降 0.1%，大幅降低标注成本。

## 方法详解
**架构基础**：采用 Deformable DETR，ResNet-50 骨干提取特征，编码器-解码器 Transformer 预测固定数量的 pair。

**近似二分匹配代价设计**：
- 真实节点：body（μ）和 face（ν）同时存在时，边容量为 2，代价为两个 box 的匹配代价之和。
- 当 ν 缺失（人脸不可见）时，引入相对估计类节点 $\hat{\nu}$（使用头部标注或几何近似），代价函数为：
$$C_{\text{pair}} = f_1 \cdot C_\mu + (1-f_1) \cdot C_{\hat{\mu}} + f_2 \cdot C_\nu + (1-f_2) \cdot C_{\hat{\nu}}$$
其中 $f_1, f_2 \in \{0,1\}$ 表示对应类别节点是否为原始真实节点。
- 统一所有边容量为 2 后，问题转化为标准最大二分匹配，可用匈牙利算法在 $\mathcal{O}(VE)$ 内求解。

**类别设计**：因模型始终预测 pair，需新增类别区分"真实人脸"与"相对估计人脸"，总类别数为 3（或 4，若身体也可能缺失）。

**两阶段训练**：Stage 1 在 COCO 权重上仅在 CrowdHuman 上微调检测（无关联）；Stage 2 冻结骨干，加入关联匹配器训练。

**Adaptive Relative Points**：对于（body, face）对使用人脸的相对点采样，对于（body, head）对使用身体的相对点采样，显著提升 deformable attention 效果。

**损失函数**：对 pair 中每个 box 独立计算分类交叉熵 + L1 + GIoU 损失，总计四路 loss。

## 实验与结果
**数据集**：
- **CrowdHuman**：15k 训练图（340k person 标注，含 head/body），4.3k 验证图，平均每图 23 个 person。
- **CityPersons**：3k 训练图（~19k bbox），500 验证图，平均每图 7 个 annotation。

**评估指标**：mMR⁻²（log-average miss-matching rate，越小越好）和 AP（越大越好），按可见度分为 reasonable/bare/partial/heavy/hard 五个子集。

**主要结果**：

| 数据集 | 指标 | PairDETR | 最佳基线 | 提升 |
|---|---|---|---|---|
| CrowdHuman | mMR⁻² (all) | **42.9** | 52.5（FPN+BFJ） | ↓33.3% |
| CrowdHuman | AP body | 87.1 | 88.7（FPN+BFJ） | -1.6 |
| CrowdHuman | AP face | 72.6 | 71.1（FPN+POS） | +1.5 |
| CityPersons | mMR⁻² (reasonable) | **22.22** | 26.4（BPJDet-L） | ↓16% |
| CityPersons | AP face | **70.2** | 68（FPN+BFJ） | +2.2 |
| CityPersons | AP body | 84.1 | 84.4（FPN+BFJ） | -0.3 |

**大框（>96×96）对比**：PairDETR mMR⁻²=41.31 vs BPJDet-L 53.81，AP body=93.14 vs 90.17。

**消融结果**：
- 加入关联后 AP body +0.6%，AP face +0.4%。
- 头部近似方案（Head Approximation）显著优于 Basic/MinCost MaxFlow/Body Approx。
- Adaptive relative points 最佳（mMR⁻²=42.9），固定 single reference point 次之。
- 用几何近似替代头部标注：mMR⁻² 从 42.9→42.8，几乎无损失。

**统计显著性**：多 seed 实验 std < 0.5，结果稳定。

## 相关工作脉络
1. **DETR (Carion et al., ECCV2020)**：开创端到端集合预测目标检测，本文以此为基础扩展为 pair 预测。
2. **Deformable DETR (Zhu et al., ICLR2021)**：引入可变形注意力解决收敛慢和小目标检测弱的问题，本文以此作为骨干架构。
3. **BFJ (Wan et al., ICCV2021)**：先分别检测 body 和 face，再用 embedding loss + head box hook 做关联，非端到端；本文消除所有后处理。
4. **BPJDet (Zhou et al., ICME2023)**：扩展 YOLO head 支持 body-part 检测及中心偏移，需额外后处理且 backbone 不同难以公平比较。
5. **HOTR (Kim et al., CVPR2021)**：也将 DETR 扩展到图预测（human-object interaction），但假设输出始终为 pair；本文允许缺失关联（仅有 body 无 face）。
6. **PETR (Shi et al., CVPR2022)**：扩展 DETR 用于 body parts 关联以解决姿态估计，使用多个 pose query；本文每 query 独立预测一个 pair，思路更简洁。

## 局限性与未来方向
1. **依赖缺失部分的近似标注或生成**：虽然可用几何近似替代，但仍需额外的（身体或头部）标注信息来处理遮挡情况。
2. **Deformable DETR 每 query 仅支持单一相对点**：难以直接扩展到更复杂的图结构（如 multi-node association），改用标准 DETR 则面临收敛慢和小目标弱的问题。
3. **类别数随对象类型指数增长**：n 类对象的完整组合需 $2^n$ 个类别来区分推理时的所有情况，扩展至多目标关联时复杂度急剧上升。
4. **未探索 RT-DETR 等实时架构**：作者指出 RT-DETR 可缓解小目标问题，但超出本文范围，留待未来工作。

## 研究启发与可借鉴点
1. **近似二分匹配策略可迁移到其他 pair 预测任务**：将 NP-hard 固定代价流问题通过"补充相对估计节点"映射回多项式时间二分匹配的思路，适用于任何存在部分缺失关联的对象对预测问题（如 hand-object、eye-mouth 等）。
2. **几何近似替代标注的策略具有实用价值**：用简单规则（身体顶部居中方块，α=0.3）生成的伪标注几乎不损失性能，可降低对新数据集的标注依赖。
3. **两阶段训练策略值得借鉴**：先在预训练权重上训练纯检测 stage，再冻结骨干加入关联 matcher 微调，既稳定又高效，可作为 DETR 类模型扩展新能力的通用范式。
4. **Adaptive Reference Point 设计**：根据 pair 类型动态选择相对点（face-body 用 face 点，head-body 用 body 点），是提升 deformable attention 效果的有效技巧，可迁移至其他多目标关节点检测任务。
5. **多 seed 统计显著性报告**：论文报告了均值±标准差，增强了结果可信度，可作为实验报告的参考规范。

## 关键术语表
- **mMR⁻²（log-average miss-matching rate）**：评估关联质量的指标，综合不同 IoU 阈值下的 miss-matching rate 的对数平均，值越低表示关联性能越好。
- **Approximated Bipartite Matching**：通过将一般图的固定代价最大流问题近似映射回二分匹配，从而用匈牙利算法在多项式时间内求解 NP-hard 配对匹配问题。
- **Relatively Estimated Class（相对估计类）**：当某类别节点缺失时引入的虚拟节点（如用头部近似不可见的人脸），使其代价 criteria 与原节点相近，以维持二分匹配结构。
- **Deformable DETR**：DETR 的改进版本，引入可变形注意力机制，支持稀疏采样点，加速收敛并改善小目标检测。
- **Adaptive Relative Points**：根据预测 pair 的类型（face-body 或 head-body）动态选择 deformable attention 的参考点，使采样位置自适应。
- **Hungarian Algorithm（匈牙利算法）**：求解二分图最小权完美匹配的经典算法，时间复杂度 O(V³)，用于本文的近似匹配损失计算。
- **FixedCost MaxFlow**：图中部分边的代价与流量无关（固定代价）的最大流问题，本文证明其为 NP-hard，是引入近似的核心动机。
- **COCO initial weights**：在 COCO 数据集上预训练的模型权重，本文用于两阶段训练的第一阶段初始化。

## 可复现要素
- **数据集**：CrowdHuman 和 CityPersons 均为公开数据集；BFJ 提供的补充人脸标注也被使用。
- **代码**：训练代码和预训练权重已开源，GitHub: https://github.com/mts-ai/pairdetr
- **关键超参**：初始学习率 4×10⁻⁵，epoch 40 时 LR ×0.1，共训练 50 epoch；AdamW 优化器，weight decay=0.0001；batch size=1/GPU，6 GPU 训练。
- **输入尺寸**：CrowdHuman 最长边 1400（COCO transform padding）；CityPersons 2048×1024。
- **两阶段训练**：Stage 1 COCO 预训练权重微调检测；Stage 2 冻结骨干加关联 matcher。
- **几何近似参数**：α=0.3（预期人脸宽度/身体宽度比）。
- **Random seed**：多 seed 实验 std < 0.5，具体 seed 值论文未详列。
