---
title: "Identifying-Important-Group-of-Pixels-using-Interactions"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sumiyasu_Identifying_Important_Group_of_Pixels_using_Interactions_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:59:32"
field: "计算机视觉可解释性"
keywords: ["可解释人工智能", "Shapley值", "交互项", "Vision Transformer", "像素可视化", "模型解释"]
innovations: ["提出MoXI方法，通过自上下文Shapley值和交互项联合贪心选择重要像素组", "从博弈论角度证明简单贪心策略的理论正当性并降低计算复杂度至二次级"]
benchmarks: ["ImageNet", "Insertion/Deletion Curve", "Common Corruption"]
---

# 论文速读：Identifying-Important-Group-of-Pixels-using-Interactions

## 一句话总结
本文提出 MoXI（Model eXplanation by Interactions），一种基于博弈论中 Shapley 值和交互项的高效像素组识别方法，通过在逐像素插入和删除两个视角下联合考虑像素的个体贡献与协同交互效应，将计算复杂度从指数级降至二次级，显著优于 Grad-CAM、Attention rollout 和 Shapley value 等现有可视化方法。

## 研究问题与动机
- **现有方法仅衡量像素个体贡献，忽略协同效应**：Grad-CAM、Attention rollout 和 Shapley value 均以特征图或注意力权重度量单个像素的贡献，无法捕捉像素间的合作协同（synergy）关系；例如在 ImageNet 插入实验中，这些方法往往只高亮类物体（鸭子），而忽略了背景（海面）的集体贡献。
- **高 Shapley 值像素集合未必是最优信息集合**：单纯选取 Shapley 值最高的像素集合，忽略了像素间的信息重叠（redundancy），导致所选像素集虽个体贡献大但整体信息增益低。
- **交互项计算存在指数计算瓶颈**：直接计算 Shapley 值需要 O(|N|·2^|N|) 次前向传播，交互项更昂贵，限制了博弈论方法在实际模型解释中的可用性。
- **已有交互相关研究未解决高效像素组选择问题**：尽管已有工作将交互用于对抗迁移性分析、误分类理解等，但缺乏一个高效且直接面向像素组识别的博弈论框架。

## 核心贡献（创新点）
- **提出 MoXI，首次通过交互项联合选取像素组**：本文在像素插入和删除两种设定下，利用自上下文（self-context）Shapley 值和交互项联合指导贪心选择，区别于仅依赖单个 Shapley 值选取像素的 SHAP/Covert 等方法，MoXI 能同时选出物体和背景的互补区域。
- **理论上证明简单贪心策略的博弈论合理性**：通过游戏论推导证明，Greedy 步骤中的第二项优化等价于最大化 Shapley 值与交互项之和（插入）或带权交互项之和（删除），为简单算法提供了严谨的理论支撑，区别于以往经验性启发式方法。
- **将计算复杂度从指数级降至二次级**：引入自上下文（self-context）和全上下文（full-context）变体，使 Shapley 值和交互项的计算仅需 O(|N|²) 次前向传播，突破了博弈论方法的计算瓶颈。
- **系统验证了方法的有效性与一致性**：在 ImageNet 上，MoXI 仅用 4% 可见 patch 即达到 90% 准确率（基线最高仅 25%），并在多种常见噪声（高斯噪声、雾）、不同类别数和不同模型架构（ViT-T、DeiT-T、ResNet-18）上保持一致性优势。

## 方法详解
**任务定义**：给定图像 x 的像素索引集 N，目标是找到按贡献排序的 patch 序列 S₁, S₂, …, S_|N|。

**像素插入（Pixel Insertion）**：从空图像逐步加入 patch，最大化分类置信度 f(S)。

- 第 k 步选择 patch b_k 的决策规则（公式 7）：
  - b_k = argmax_{b∈N\S_{k-1}} [ φ⁽⁰⁾(b) + I⁽⁰⁾(S_{k-1}, b) ]
  - φ⁽⁰⁾(a) = f({a}) − f(∅) 为自上下文 Shapley 值（公式 5）
  - I⁽⁰⁾(a₁, a₂) = f({a₁,a₂}) − f({a₁}) − f({a₂}) + f(∅) 为自上下文交互项（公式 6）
- 关键洞察：即使某 patch 有大的 φ⁽⁰⁾，若与已选集合 S_{k-1} 存在大的负交互 I⁽⁰⁾(S_{k-1}, b)，也不应入选，因为信息冗余。

**像素删除（Pixel Deletion）**：从完整图像逐步移除 patch，最小化剩余图像的置信度。

- 定义基于删除的 Shapley 值 φ_d（公式 9）和全上下文交互项 I_d（公式 11）。
- 第 k 步选择 patch b_k 的决策规则（公式 13）：
  - b_k = argmax_{b∈N\S_{k-1}} [ φ_d⁽|N|⁾(b) + (|N|−|S_{k-1}|) · I_d⁽|N|−|S_{k-1}|⁾(S_{k-1}, b) ]
- 注意删除情形中交互项带有放大权重 (|N|−|S_{k-1}|)。

**ViT 适配**：针对 Vision Transformer，采用 attention masking 而非零填充实现 patch 遮挡，更准确地反映 mask 对模型的影响。

**计算复杂度**：MoXI 最坏情况仅需 O(|N|²) 次前向传播，相比原始 Shapley 值的 O(|N|·2^|N|) 大幅降低。

## 实验与结果
- **数据集**：ImageNet，选取每类各 1 张共 1000 张正确分类的测试图像。
- **模型**：ViT-T、DeiT-T（ViT 系列）和 ResNet-18（CNN 对比）。
- **基线方法**：Grad-CAM、Grad-CAM++、Attention rollout、Shapley value、MoXI(-)（不含交互项的版本）。
- **评价指标**：Insertion curve（逐步添加高贡献 patch 后准确率上升）、Deletion curve（逐步移除高贡献 patch 后准确率下降）、常见噪声鲁棒性、跨类别数一致性。
- **核心结果（ViT-T）**：
  - **Insertion**：仅 4% 可见 patch 时，MoXI 准确率达 **90%**，Grad-CAM 2%、Attention rollout 4%、Shapley value 25%。
  - **Deletion**：仅移除 10% patch 时，MoXI 将准确率降至 **16%**，Grad-CAM 和 Attention rollout 仅降至约 79%。
  - **Patch 数量更少**：MoXI 在插入和删除任务中均能用更少的 patch 达成目标。
  - **交叉验证**：DeiT-T 和 ResNet-18 上获得类似结论；常见噪声（高斯噪声、雾）下 MoXI 同样表现最强。
  - **一致性**：随类别数从 10 增至 1000，MoXI 的准确率几乎不下降，而 Attention rollout 显著下降。
  - **类特异性定位**：对非预测类（如 tiger cat）的 patch 定位同样有效，突出类判别性区域。

## 相关工作脉络
- **Grad-CAM / Grad-CAM++**：基于梯度加权特征图的可视化方法，度量个体像素贡献但依赖中间特征而非直接置信度；MoXI 直接从 logits/confidence 出发，更准确反映真实贡献。
- **Attention Rollout**：基于 Vision Transformer 注意力权重的可视化；MoXI 指出 attention weight 大小与真实贡献不一定对齐，且无法捕捉像素间协同效应。
- **SHAP / Covert et al. (ICLR 2023)**：将 Shapley value 用于像素贡献度量；MoXI 的关键区别在于引入交互项，解决了"高 Shapley 像素集合冗余"问题，并降低计算复杂度。
- **Zhang et al. (ICLR 2022) — Dropout 的博弈论解释**：指出 interactions 与 dropout 正则化的相似性；MoXI 与本文关系是独立应用，MoXI 聚焦高效像素组识别而非正则化分析。
- **Sumiyasu et al. (2022) — 误分类的博弈论理解**：前作研究了误分类图像中交互项的分布差异；MoXI 是同一理论框架在高效可视化方向的应用延伸。
- **Wang et al. (ICLR 2022) — 对抗迁移性与交互**：证明对抗样本可迁移性与交互负相关；MoXI 与本文定位不同，MoXI 面向正常分类的解释而非对抗鲁棒性。

## 局限性与未来方向
- **计算成本仍随 patch 数量平方增长**：O(|N|²) 对于高分辨率图像（如 224×224 对应 196 个 patch）尚可接受，但对极高分辨率或视频序列可能仍然昂贵。
- **仅适用于 Transformer 和 CNN**：目前主要在 ViT 和 ResNet 上验证，对新兴架构（如 Mamba、State Space Models）的适用性有待探索。
- **仅针对图像分类任务**：尚未扩展到目标检测、分割、视频理解等其他视觉任务。
- **采样近似精度有限**：Shapley value 使用 200 次随机采样近似，可能在高维情况下精度不足。
- **未来方向**：（1）扩展到视频/3D 数据的时序像素组识别；（2）与训练过程结合，利用交互项作为正则化提升模型鲁棒性；（3）探索更高效的交互近似算法（如多项式时间近似）。

## 研究启发与可借鉴点
- **自上下文（self-context）变体的设计思路**：将 Shapley 值和交互项限定在已选子集上计算，既保持博弈论理论严谨性又大幅降低复杂度，该思想可迁移至其他需要组合优化的可解释性场景。
- **贪心选择的博弈论正当化**：将看似朴素的贪心算法从 Shapley 值+交互的视角给出严格推导，这种"从理论导出简单算法"的研究范式对方法论论文写作有借鉴价值。
- **交互项作为信息冗余度量**：I⁽⁰⁾(a₁, a₂) = f({a₁,a₂}) − f({a₁}) − f({a₂}) + f(∅) 的物理意义直接对应信息重叠程度，该公式可作为特征选择、数据压缩等领域的通用工具。
- **一致性可解释性评估新范式**：通过改变模型训练类别数来检验解释方法的稳定性，提供了一种新颖的 XAI 评估维度，可应用于其他可视化方法的评价。
- **与可迁移方向的结合机会**：MoXI 的交互度量可用于分析模型在不同域/类别间的注意力转移，或辅助设计更高效的对比学习正负样本对。

## 关键术语表
- **Shapley 值（Shapley value）**：源自合作博弈论的归因方法，衡量单个参与者在所有可能联盟中的平均边际贡献。
- **交互项（Interaction）**：衡量两个及以上参与者共同合作时对总奖励产生的额外贡献（超出个体贡献之和的部分）。
- **自上下文变体（Self-context variant）**：将 Shapley 值和交互项的计算限定在当前已选子集范围内，而非全集，从而大幅降低计算量。
- **全上下文变体（Full-context variant）**：在像素删除设定下，以原始全集 N 为上下文的 Shapley 值和交互项定义。
- **插入曲线（Insertion curve）**：从全遮挡图像开始，按方法排序逐步添加高贡献 patch，记录准确率随添加比例的变化曲线。
- **删除曲线（Deletion curve）**：从完整图像开始，按方法排序逐步移除高贡献 patch，记录准确率随遮挡比例的变化曲线。
- **Vision Transformer（ViT）**：将图像分块为固定大小 patch 并输入 Transformer 进行视觉分类的模型架构。
- **SET-SUM 任务**：合成测试任务，标签为图像中数字类型之和，用于直观展示交互项在去冗余方面的必要性。

## 可复现要素
- **数据集**：ImageNet（公开），实验使用其中 1000 张每类各一张的正确分类样本。
- **代码开源**：是，仓库地址 https://github.com/KosukeSumiyasu/MoXI（论文声明）。
- **预训练模型**：ViT-T、DeiT-T（来自 TIMM 库）、ResNet-18（标准公开权重）。
- **关键超参**：patch 大小 16×16，共 14×14 个 patch；Shapley 值采样次数 200；masking 方式：feature patch deletion。
- **训练类别数一致性实验**：分别使用 10、20、100、1000 类数据训练模型进行评估。
