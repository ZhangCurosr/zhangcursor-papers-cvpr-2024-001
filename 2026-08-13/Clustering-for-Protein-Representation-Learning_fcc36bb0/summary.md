---
title: "Clustering-for-Protein-Representation-Learning"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Quan_Clustering_for_Protein_Representation_Learning_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:13:52"
---

# 论文速读：Clustering-for-Protein-Representation-Learning

## 一句话总结
本文提出一种迭代神经聚类框架，将蛋白质建模为图结构，通过融合一维序列与三维空间信息，自动发现决定蛋白质结构与功能的关键氨基酸残基；在EC编号预测、GO项预测、折叠分类与酶反应分类四项任务上均取得SOTA，且无需大规模自监督预训练。

## 研究问题与动机
- 现有蛋白质表征方法通常对所有氨基酸平等对待，忽略了生物学事实中仅少数“关键组分”（critical components）主导蛋白质折叠构象与功能活性。
- 序列基方法（1D CNN/Transformer）缺乏3D空间感知；结构基GNN/CNN虽引入坐标，但节点聚合方式难以自适应聚焦高信息量残基。
- 蛋白质表征需要将序列顺序约束与空间拓扑约束统一建模，并在保留关键功能基序的同时实现层次化信息压缩。
- 如何在端到端学习中让模型自动筛选并逐级聚焦最具判别力的氨基酸子集，是提升结构-功能映射精度的关键瓶颈。

## 核心贡献（创新点）
1. **提出迭代神经聚类蛋白质表征框架**：设计包含SCI、CRE、CN三步的可微循环流程，逐轮聚合候选簇并筛选medoid节点，实现蛋白质的层次化关键组分表征学习。与已有GNN/CNN方法平等的节点聚合方式不同，本文通过可学习的聚类-提名机制实现数据驱动的自适应降采样与关键残基聚焦。
2. **构建1D序列与3D几何融合的簇特征编码器**：在CRE步骤中显式拼接相对几何坐标、3D方向向量、空间距离、序列相对位置与氨基酸嵌入，并经交叉注意力加权聚合。与仅依赖欧氏距离或固定kernel的图池化方法不同，该设计同时捕获一级序列关系与三级空间构象，更符合蛋白质物理化学特性。
3. **提出基于GCN的层次化聚类提名策略**：利用GCN计算簇的提名得分并按比例筛选Top-k簇进入下一迭代，同时随迭代递增聚类半径以扩大感受野。与固定图池化或随机采样策略不同，该提名过程与下游分类任务联合优化，使筛选出的残基直接服务于功能预测。
4. **在四项标准任务上刷新SOTA并提供强可解释性验证**：全面超越CDConv、GearNet等主流方法，且无需预训练即优于ESM-1b等语言模型；可视化表明模型成功定位蛋白质环/螺旋末端的功能性基序，同家族蛋白聚类结果高度一致。与以往仅报告指标的方法不同，本文从生物结构视角验证了聚类过程确实发掘了具功能意义的子结构。

## 方法详解
- **图表示**：蛋白质 $\mathcal{P}=(\mathcal{V},\mathcal{E},\mathcal{Y})$，节点 $v_n$ 为氨基酸，边表示空间接触或序列邻接；节点特征 $x_n\in\mathbb{R}^{256}$ 含类型嵌入、方向、位置等。
- **迭代循环**（$T=4$ 轮，每轮堆叠 $B=2$ 个CRE块，采用残差连接）：
  1. **Spherical Cluster Initialization (SCI)**：以当前轮 $N_{t-1}$ 个medoid节点为中心，在固定半径 $r_t$（$r_t=t\cdot r$，即 $r,2r,3r,4r$）内收集空间与序列邻居组成簇 $\mathcal{H}^n$，重新计算邻接矩阵 $A$。
  2. **Cluster Representation Extraction (CRE)**：对簇内成员 $v_k^n$，构造 $x_k^n=f(g_k^n, o_k^n, d_k^n, s_k, e_k)$，其中 $g$ 为相对几何坐标，$o$ 为3D方向，$d$ 为空间距离，$s$ 为序列序，$e$ 为氨基酸one-hot嵌入。通过交叉注意力计算medoid与成员的权重 $\gamma_k^n=\frac{\exp(w[x_n,x_k^n])}{\sum_k\exp(w[x_n,x_k^n])}$，聚合得簇表征 $\tilde{x}_n=\sum_k \gamma_k^n x_k^n$。
  3. **Cluster Nomination (CN)**：将簇表征输入GCN计算得分 $\varphi_n=\sigma(W_1\tilde{x}_n+\sum_m A_{nm}(W_2\tilde{x}_n-W_3\tilde{x}_m))$。按得分选取Top-$N_t$ 个簇（$N_t=\lfloor\omega\cdot N_{t-1}\rfloor$，$\omega=0.4$），其medoid节点按原始序列顺序连边构成子图，作为 $t+1$ 轮输入。
- **训练协议**：末端接全连接分类头；单标签任务（Fold/Reaction）用Softmax+CE，多标签任务（EC/GO）用Sigmoid+BCE。整体端到端联合优化，并保证旋转不变性。

## 实验与结果
- **数据集
