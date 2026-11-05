---
title: "Clustering-for-Protein-Representation-Learning"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Quan_Clustering_for_Protein_Representation_Learning_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:21:34"
---

# 论文速读：Clustering-for-Protein-Representation-Learning

## 一句话总结
本文提出一种迭代神经聚类框架，将蛋白质表示学习形式化为“球形初始化-跨注意力聚合-可学习提名”的循环过程，自动从全量氨基酸中筛选出决定结构与功能的关键节点，在 EC 编号预测、GO 术语预测、蛋白质折叠分类与酶反应分类四项基准上均取得 SOTA。

## 研究问题与动机
1. **核心问题**：如何从氨基酸序列与三维空间中学习紧凑且具备功能判别力的蛋白质表示。
2. **现有方法不足**：主流序列 CNN/LSTM、结构 GCN/GVP/IEConv 等方法通常平等聚合所有氨基酸节点，忽略了生物学中“少数关键氨基酸主导折叠与活性”的客观事实。
3. **动机来源**：胰岛素 A/B 链、ACTH 激素等案例表明，保留关键残基即可维持功能，移除非关键区域影响有限；不同任务可能依赖不同的关键子集，需一种数据驱动的选择机制。

## 核心贡献（创新点）
1. **提出迭代神经聚类表示学习框架**：首次将蛋白质表示学习拆解为 SCI-CRE-CN 三步循环，实现层次化降采样与关键组分自适应发现，区别于传统固定图池化或全局平均。
2. **1D/3D 融合的跨注意力簇表征提取（CRE）**：在簇内显式编码相对几何坐标、3D 方向、空间距离与序列序位，并以 medoid 为 query 进行 cross-attention 加权聚合，本质区别在于引入任务感知的位置-空间耦合权重。
3. **GCN 驱动的端到端簇提名机制（CN）**：通过图卷积对簇打分并按固定比例 $\omega$ 筛选 Top-$N_t$ 个 medoid 进入下一轮，取代硬阈值或经验剪枝，使聚类过程与下游分类损失联合优化。
4. **零预训练超越大规模蛋白语言模型**：在不使用 ESM-1b/ProtBERT-BFD 等自监督预训练的前提下，于多项任务上匹敌或反超依赖千万至十亿级序列数据的预训练模型，验证了几何聚类范式的效率。

## 方法详解
- **图构建**：蛋白质表示为 $\mathcal{P}=(\mathcal{V},\mathcal{E},\mathcal{Y})$，节点 $v_n$ 为氨基酸，边 $\mathcal{E}$ 覆盖空间邻接与序列邻接，初始特征维度 256。
- **SCI (Spherical Cluster Initialization)**：以候选节点 $v_n$ 为 medoid，在固定半径 $r$ 内收集邻居构成簇 $\mathcal{H}^n$；随迭代推进半径线性放大（第 1~4 轮分别为 $r, 2r, 3r, 4r$），邻接矩阵每轮重建，避免重复探索同一区域。
- **CRE (Cluster Representation Extraction)**：簇内节点特征经 MLP 编码为 $x_k^n = f(g_k^n, o_k^n, d_k^n, s_k, e_k)$，其中 $g$ 为相对坐标、$o$ 为 3D 方向、$d$ 为欧氏距离、$s$ 为序列序位、$e$ 为 128 维氨基酸嵌入。采用 cross-attention 计算 medoid 与成员的注意力 $\gamma_k^n = \frac{\exp(w[x_n, x_k^n])}{\sum_k \exp(w[x_n, x_k^n])}$，加权求和得簇表示 $\tilde{x}_n = \sum_k \gamma_k^n x_k^n$。堆叠 $B=2$ 个 CRE block 并加 skip connection，保证旋转不变性。
- **CN (Cluster Nomination)**：GCN 对簇打分 $\varphi_n = \sigma(W_1 \tilde{x}_n + \sum_m A_{n,m}(W_2 \tilde{x}_n - W_3 \tilde{x}_m))$，加权后 $\hat{X}^c = \Phi \odot X^c$，选取 Top-$N_t$ 个簇，其中 $N_t = \lfloor \omega \cdot N_{t-1} \rfloor$（默认 $\omega=40\%$）。被选 medoid 按原始序列顺序重连成图，作为下一轮输入。
- **分类头**：第 $T$ 轮输出的 $N_T$ 个节点特征经全连接层投影至标签维度，单标签任务用 Softmax+CE，多标签任务用 Sigmoid+BCE。全程端到端训练。

## 实验与结果
- **数据集与任务**：EC 编号预测（15,550/1,729/1,919，538 个二分类子任务）、GO 术语预测（29,898/3,322/3,415，MF/BP/CC 三子任务）、蛋白质折叠分类（12,312/736/718，Fold/Superfamily/Family 三场景）、酶反应分类（29,215/2,562/5,651）。
- **主要结果**：
  - **EC 预测**：$F_{max}$ 达 **0.866**，较 CDConv (+5.6%) 与 GearNet (+6.9%) 显著提升。
  - **GO 预测**：BP **0.474**、MF **0.675**、CC **0.483**，三项均领先次优方法。
  - **折叠分类**：Fold **63.1%**、Super. **81.2%**、Fam. **99.6%**，平均 **81.3%**，较 CDConv (+3.3%)、ProNet (+10.4%)、GearNet (+14.8%) 全面超越。
  - **酶反应分类**：准确率 **89.6%**，较 CDConv (+1.1%)、GearNet (+4.3%) 领先；推理耗时仅 **10.9 ms**（单 RTX 3090），远快于 GearNet（无法单卡训练）与 New IEConv（75.3 ms）。
- **对比预训练模型**：无需自监督预训练，在 Fold/GO/Reaction 等任务上仍优于 ESM-1b、ProtBERT-BFD、DeepFRI 等依赖大规模序列语料的方法。
- **结论**：四项任务全部 SOTA，证明迭代聚类能有效捕获功能性关键组件。

## 相关工作脉络
1. **序列/CNN 基方法** (ResNet/LSTM/Transformer [66])：仅利用 1D 序列，忽略空间拓扑；本文通过 SCI 的 3D 半径与方向向量显式补全几何信息。
2. **图神经网络基方法** (GCN [46], GAT [79], GraphQA [7])：平等聚合邻居；本文引入可微注意力与递进提名，实现“重要节点自适应保留”而非全局平均。
3. **几何建模方法** (GVP [44], IEConv [37], GearNet [94], CDConv [22])：侧重原子/氨基酸级卷积或对比预训练；本文独特定位是将聚类降采样作为核心归纳偏置，以任务监督直接驱动关键组分发现。
4. **蛋白语言模型** (ESM-1
