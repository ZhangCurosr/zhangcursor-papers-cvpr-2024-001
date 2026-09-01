---
title: "Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:54:11"
field: "水下视觉与语义分割"
keywords: ["Marine Animal Segmentation", "SAM", "Domain Adaptation", "Connectivity Prediction", "Parameter-Efficient Fine-tuning"]
innovations: ["提出Dual-SAM双分支框架适配预训练SAM至水下场景", "设计MCP自动提示生成策略替代单点人工提示", "提出C³P连通性预测范式建模像素间结构化关系"]
benchmarks: ["MAS3K", "RMAS", "UFO120", "RUWI", "USOD10K"]
---

# 论文速读：Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine

## 一句话总结
本文提出 Dual-SAM，一种面向海洋动物分割（MAS）的双分支特征学习框架，通过引入多粒度耦合提示（MCP）和交又连接预测（C³P）范式，有效适配预训练 SAM 模型并显著提升在复杂水下场景中的分割性能。

## 研究问题与动机
- **水下环境特殊性**：水下图像存在光照变化、水体浑浊、色彩失真、相机与目标运动等问题，传统陆地图像分割方法直接迁移效果不佳。
- **SAM 水下适配不足**：SAM 在自然图像上预训练，缺乏水下先验知识；其单点提示不足以提供充分的先验引导。
- **解码器表达能力有限**：SAM 简单解码器难以捕捉海洋生物的复杂细节和精细边界。
- **像素连通性被忽视**：传统逐像素预测方法忽略离散像素间的结构连通性，导致边界不规则。

## 核心贡献（创新点）
- **Dual-SAM 双分支框架**：继承 SAM 能力并自适应融入水下场景先验，与直接微调 SAM 的方法相比，引入双分支 gamma 校正与适配器机制实现高效领域适配。
- **多粒度耦合提示（MCP）策略**：通过自生成提示注入综合水下先验信息，区别于 SAM 的单点提示依赖人工标注，实现无需额外标注的自动提示生成。
- **扩张融合注意力模块（DFAM）**：结合扩张卷积与通道注意力逐步融合 SAM 编码器多层特征，区别于 SAM 原始简单解码器，增强感受野与上下文感知。
- **十字交叉连接预测（C³P）范式**：将二进制掩码扩展为 8 通道连通性地图进行结构化预测，区别于传统逐像素标量预测，显式建模像素间互联关系。
- **伪标签互监督（PMS）**：双分支通过伪标签相互监督实现互补表征学习，区别于单分支方法，在训练过程中促进双路协同优化。

## 方法详解
- **Dual-SAM 编码器（DSE）**：采用 gamma 校正补偿水下光照：$I^{\beta} = \sqrt[\gamma]{I^{\alpha}}$，其中 $\gamma = \lg(0.5) - \lg(mean_I^{gray}/255)$。在 SAM 编码器的 MHSA 块中注入 LoRA 低秩可训练矩阵（Query 和 Value 分支），并在 FFN 中加入 Adapter，仅更新线性投影矩阵以节省参数量。
- **多粒度耦合提示（MCP）**：将原始图像 $I^{\alpha}$ 与校正图像 $I^{\beta}$ 拼接后进行 Patch Embed，经多层 Transformer 迭代生成特征 $I_i^{\omega}$，再以 DSE 输出特征作为 Query/Key、$I_i^{\omega}$ 作为 Value 进行多头交叉注意力（MHCA），生成耦合提示 $\mathcal{P}_i^{\omega}$，最终得到 $\mathcal{P}_i^{\alpha}$ 和 $\mathcal{P}_i^{\beta}$ 作为自生成提示注入特征。
- **扩张融合注意力模块（DFAM）**：融合 prompted 特征与金字塔特征，通过 $1\times1$ 卷积拼接后计算通道注意力权重 $W^g$，再用 dilation rate=2 的 $3\times3$ 卷积扩展感受野，构建特征金字塔结构。
- **十字交叉连接预测（C³P）**：将单通道掩码标签转换为 8 通道标签，表示中心像素与十字交叉方向像素的连通性。定义邻域像素集 $\Omega_{w,h}^{1}$（距离为1）和 $\Omega_{w,h}^{2}$（距离为2），直接预测连通性地图而非逐像素分类。
- **伪标签互监督（PMS）**：对双解码器输出进行阈值化得到伪标签 $\hat{P}_l^{\alpha/\beta}$，用于交叉监督另一分支，引入动态更新系数 $\mu = 0.1 \times e^{-5\times(1-t/T)^2}$ 控制监督强度。总损失为 $\mathcal{L} = \sum_{l=1}^{4}((\mathcal{L}_l^{\alpha} + \mathcal{L}_l^{\beta}) + \mu(\ddot{\mathcal{L}}_l^{\alpha} + \ddot{\mathcal{L}}_l^{\beta}))$。推理时采用互确认策略：$P_{w,h,c}=1 \cap P_{u,v,9-c}=1 \to P_{w,h}=1$。

## 实验与结果
- **数据集**：MAS3K（1769 训练/1141 测试）、RMAS（2514/500）、UFO120（1500/120）、RUWI（525/175）、USOD10K（9229/1026）。
- **评估指标**：mIoU、$S_\alpha$、$F_\beta^w$、$mE_\phi$、MAE。
- **主要结果**：
  - **MAS3K**：mIoU 0.789（vs. H2Former 0.748，提升 +4.1%）、$S_\alpha$ 0.884、$F_\beta^w$ 0.838、MAE 0.023，五项指标全优。
  - **RMAS**：mIoU 0.735、$S_\alpha$ 0.860、$F_\beta^w$ 0.812，超越 SAM-DA（mIoU 0.686）。
  - **UFO120**：mIoU 0.810，超越 H2Former 0.780（+3.0%）。
  - **RUWI**：mIoU 0.904，超越 H2Former 0.871（+3.3%）。
  - **USOD10K**：$S_\alpha$ 0.9238、maxF 0.9311、MAE 0.0185，全面超越 SAM-DA。
- **对比基线**：CNN 方法（SINet、PFNet、C2FNet 等）、Transformer 方法（SETR、TransUNet、H2Former）、SAM 适配方法（SAM-Ad、SAM-DA）。
- **结论**：在五个 MAS 数据集上均达到 SOTA，相对最佳基线提升 3-5%。

## 相关工作脉络
- **MASNet [12]**：基于 Siamese 结构的海洋动物分割方法，利用数据增强学习共享语义信息；Dual-SAM 在此基础上进一步引入 SAM 全局表征能力与双分支互监督机制。
- **H2Former [14]**：层次化混合 Transformer，在 MAS 数据集上表现优异；Dual-SAM 通过适配器更高效地适配 SAM 预训练权重，避免从头训练 ViT 的高数据需求。
- **SAM-Ad [6] / SAM-DA [27]**：分别通过 Adapter 和域适应适配 SAM；Dual-SAM 进一步引入 MCP 自动提示生成与 DFAM 融合结构，弥补单点提示不足。
- **Aquasam [53]**：唯一已知的 SAM 水下微调工作；Dual-SAM 采用双分支 gamma 校正与连接预测范式，提供更强的领域适配与结构建模能力。
- **ConnNet [25]**：提出像素连通性预测用于显著性分割；C³P 将其扩展到十字交叉范围，适应海洋生物多样形状与尺度。
- **LoRA [19] / Adapter [18]**：参数高效微调方法；本文将其组合应用于 SAM 编码器，以最小参数量实现领域适配。

## 局限性与未来方向
- 当前方法依赖 gamma 校正进行光照补偿，对于极端水下条件（如强散射、严重色偏）的泛化能力有待验证。
- 数据集规模相对较小（MAS3K 仅 3103 张），在更大规模水下数据上的验证不足。
- 双分支结构增加推理复杂度，实时性有待进一步评估与优化。
- 未来可将 C³P 范式推广至其他需要结构感知的密集预测任务，或结合时序信息用于视频级海洋动物分割。

## 研究启发与可借鉴点
- **参数高效适配策略**：LoRA + Adapter 组合应用于 SAM 编码器的做法可迁移至其他 Foundation Model 的领域适配任务。
- **自生成提示机制**：MCP 无需人工标注即能生成高质量提示，对遥感、医学等标注稀缺领域具有借鉴价值。
- **连接预测范式**：C³P 将掩码预测转化为连通性建模，思路可推广至边界敏感任务（如细胞分割、遥感地物提取）。
- **双分支互监督**：PMS 的对称学习机制为多视图/多模态融合提供新思路，可结合本团队在对比学习方向的工作。
- **gamma 校正预处理**：针对水下图像的轻量级光照补偿策略可作为通用预处理模块应用于其他水下视觉任务。

## 关键术语表
- **Marine Animal Segmentation (MAS)**：海洋动物分割，指从水下图像中精确分割出海洋生物个体的任务。
- **Segment Anything Model (SAM)**：Meta 提出的通用图像分割基础模型，具备零样本迁移能力，但预训练于自然图像。
- **Multi-level Coupled Prompt (MCP)**：多粒度耦合提示，通过自生成方式融合双分支特征生成的水下先验提示。
- **Dilated Fusion Attention Module (DFAM)**：扩张融合注意力模块，结合扩张卷积与通道注意力实现多尺度特征融合。
- **Criss-Cross Connectivity Prediction (C³P)**：十字交叉连接预测，将掩码扩展为 8 通道连通性地图的结构化预测范式。
- **Pseudo-label Mutual Supervision (PMS)**：伪标签互监督，双分支互相生成伪标签进行交叉监督的训练策略。
- **LoRA (Low-Rank Adaptation)**：低秩自适应，通过低秩矩阵注入实现大模型参数高效微调的技术。

## 可复现要素
- **数据集**：MAS3K、RMAS、UFO120、RUWI、USOD10K 均为公开数据集。
- **代码**：已开源，地址 https://github.com/Drchip61/DualSAM。
- **权重**：使用预训练 SAM-B 初始化编码器，其余随机初始化。
- **关键超参**：输入尺寸 512×512，batch size=8，学习率 0.001，weight decay=0.1，训练 50  epoch，每 20 轮 LR ×0.1，阈值 ξ=0.5，AdamW 优化器。
