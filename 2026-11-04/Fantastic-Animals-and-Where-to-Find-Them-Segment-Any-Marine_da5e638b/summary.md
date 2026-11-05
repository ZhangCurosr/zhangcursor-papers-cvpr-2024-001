---
title: "Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:51:17"
field: "水下视觉分割"
keywords: ["Marine Animal Segmentation", "Segment Anything Model", "Foundation Model Adaptation", "Connectivity Prediction", "Underwater Image Segmentation", "Parameter-Efficient Fine-tuning"]
innovations: ["提出多级耦合自动提示（MCP）替代 SAM 单点提示，注入海洋域先验", "设计十字交叉连通性预测（C³P）范式，从逐像素分类转向结构化连通性建模", "双解码器伪标签互监督（PMS）促进互补特征学习的训练策略"]
benchmarks: ["MAS3K", "RMAS", "UFO120", "RUWI", "USOD10K"]
---

# 论文速读：Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine

## 一句话总结
本文提出 Dual-SAM 框架，通过双分支编码器、多级耦合自动提示（MCP）、膨胀融合注意力（DFAM）和十字交叉连通性预测（C³P）四个核心组件，解决 SAM 模型直接迁移到海洋动物分割（MAS）任务时面临的域差异大、提示信息不足、上下文建模弱等问题，在五个 MAS 基准数据集上均达到 SOTA。

## 研究问题与动机
1. **海洋环境独特性**：水下图像存在光照变化、水体浑浊、颜色失真、相机/目标运动模糊等挑战，传统陆地分割方法难以直接适用。
2. **SAM 的域适配瓶颈**：SAM 基于十亿级自然图像预训练，缺乏海洋域先验；其简单解码器难以捕捉海洋生物的复杂细节与边界。
3. **单点提示信息匮乏**：SAM 原始的单点/框提示无法提供充分的语义引导，对具有伪装、模糊边界的海洋生物分割效果受限。
4. **像素级预测的结构丢失**：传统逐像素分类忽略离散像素间的连通性，导致预测掩码结构不连贯、边界不规则。

## 核心贡献（创新点）
1. **Dual-SAM 双分支编码框架**：继承 SAM 范式并引入 gamma 校正与双路特征提取，使模型同时从原始图像和光照补偿图像中学习互补的海洋特征。*区别于 vanilla SAM 仅单路输入，该方法显式建模海洋图像的光照失真问题。*
2. **多级耦合提示（MCP）策略**：通过 Patch Embed + Transformer 迭代生成自动提示，并以 DSE 输出作为 Query/Key、提示作为 Value 进行跨注意力融合。*区别于 SAM 手工或单点提示，MCP 自动生成多层级、耦合的上下文先验。*
3. **膨胀融合注意力模块（DFAM）**：将 SAM 简单解码器替换为带空洞卷积与通道注意力的特征金字塔结构，渐进式融合多级提示特征。*区别于 SAM 原生的轻量解码器，DFAM 增强了对复杂海洋结构的上下文感知。*
4. **十字交叉连通性预测（C³P）范式**：将单通道掩码转换为 8 通道连通性标签，预测像素间十字交叉方向的邻接关系而非逐像素分类。*区别于传统逐像素预测，C³P 显式建模结构化连通性，提升边界完整性。*
5. **伪标签互监督（PMS）**：双解码器互相生成阈值化伪标签进行交叉监督，并引入动态指数更新系数避免早期噪声干扰。*区别于单解码器方案，PMS 促进双分支的互补学习与协同优化。*

## 方法详解
**整体框架（Fig. 2）**：包含五大组件——Dual-SAM 编码器（DSE）、MCP、DFAM、C³P 和 PMS。

**DSE（双分支编码器）**：
- 输入海洋图像 $I^\alpha$，通过 gamma 校正得到 $I^\beta = \sqrt[\gamma]{I^\alpha}$，其中 $\gamma = \lg(0.5) - \lg(\text{mean}_I^{\text{gray}}/255)$。
- 将 SAM-B 编码器冻结，在 MHSA 的 Q/V 部分注入 LoRA 低秩适配器（$W^{down}, W^{up}$），在 FFN 中注入标准 Adapter。
- 前向传播：$Q_j = X_j W_q + (X_j W_q^{down})W_q^{up}$，$V_j = X_j W_v + (X_j W_v^{down})W_v^{up}$，$X_{j+1} = \psi(\text{FFN}(\phi(H_j))W^{down})W^{up} + H_j$。

**MCP（多级耦合提示）**：
- 拼接原始图与校正图：$I_0^\omega = \text{PatchEmb}([I^\alpha, I^\beta])$。
- 经 4 层 Transformer 迭代：$I_i^\omega = \text{Trans}(I_{i-1}^\omega), i=1,2,3,4$。
- 以 DSE 输出 $X_j^\alpha, X_j^\beta$ 为 Q/K，$I_i^\omega$ 为 V，做多头交叉注意力生成耦合提示 $\mathcal{P}_i^\alpha, \mathcal{P}_i^\beta$，再通过可学习权重 $g_i^{\alpha/\beta}$ 融合进特征。

**DFAM（膨胀融合注意力模块）**：
- 融合提示特征 $E_i$ 与上層特征 $G_i$，经 1×1 卷积和通道注意力（GAP → 降维 → ReLU → 升维 → Sigmoid）生成门控权重。
- 经 dilation=2 的 3×3 卷积上采样得到 $G_{i+1}$，构建特征金字塔。

**C³P（十字交叉连通性预测）**：
- 将 GT 单通道 mask 展开为 8 通道连通性标签，分别表示中心像素 $(w,h)$ 到各十字方向距离为 1 和 2 的邻域的连通关系。
- 训练损失为逐通道的 BCE：$\mathcal{L}_l^{\alpha/\beta} = -\sum_{w,h,c}[Y \ln P + (1-Y)\ln(1-P)]$。
- 推理时通过互确认规则 $P_{w,h,c}=1 \cap P_{u,v,9-c}=1 \rightarrow P_{w,h}=1$ 将连通图恢复为二值 mask。

**PMS（伪标签互监督）**：
- 对每层预测做阈值化（$\xi=0.5$）得到伪标签 $\hat{P}_l^{\alpha/\beta}$，再以 BCE 损失交叉监督另一分支。
- 引入动态系数 $\mu = 0.1 \times e^{-5(1-t/T)^2}$ 控制互监督权重随训练epoch递增。
- 总损失：$\mathcal{L} = \sum_{l=1}^4 [(\mathcal{L}_l^\alpha + \mathcal{L}_l^\beta) + \mu(\ddot{\mathcal{L}}_l^\alpha + \ddot{\mathcal{L}}_l^\beta)]$。

## 实验与结果
**数据集**：MAS3K（1769/1141）、RMAS（2514/500）、UFO120（1500/120）、RUWI（525/175）、USOD10K（9229/1026，用于泛化验证）。

**评估指标**：mIoU、$S_\alpha$、$F_\beta^w$、$mE_\phi$、MAE。

**主要结果**：
- **MAS3K**：Dual-SAM 全面领先，mIoU=**0.789**（↑3-5% vs 次优），$S_\alpha$=0.884，$F_\beta^w$=0.838，MAE=**0.023**。
- **RMAS**：mIoU=**0.735**，$S_\alpha$=0.860，$F_\beta^w$=0.812。
- **UFO120**：mIoU=**0.810**，$S_\alpha$=0.856，$F_\beta^w$=0.864。
- **RUWI**：mIoU=**0.904**，$S_\alpha$=0.903，$F_\beta^w$=0.939，MAE=**0.035**。
- **USOD10K（泛化）**：$S_\alpha$=0.9238，maxF=**0.9311**，MAE=**0.0185**，超越 TC-USOD（0.9215/0.9236/0.0201）。
- 相比 SAM-Ad / SAM-DA 等同类型 SAM 微调方法，提升约 3-4%。

## 相关工作脉络
1. **CNN-based MAS 方法（SINet, PFNet, MASNet, ECDNet 等）**：依赖局部特征提取，缺乏长程上下文建模能力；本文方法在 Transformer 基础上弥补此缺陷。
2. **Transformer-based 方法（SETR, TransUNet, H2Former）**：具备全局感受野但需大量数据；本文利用 SAM 预训练知识 + 参数高效微调，缓解数据瓶颈。
3. **SAM 微调方法（SAM-Ad, SAM-DA, Aquasam）**：仅通过 Adapter 注入域知识或使用单点提示；本文进一步设计 MCP + DFAM + C³P，系统性解决提示不足与解码器薄弱问题。
4. **连通性预测方法（ConnNet）**：仅考虑相邻像素连接；本文 C³P 扩展为十字交叉多距离（d=1,2）八方向建模，更适合海洋生物多样形态。
5. **水下显著目标检测（USOD10K 相关：SGL-KRN, TC-USOD 等）**：本文证明 Dual-SAM 可跨任务泛化至水下显著性检测，并在 USOD10K 上刷新 SOTA。

## 局限性与未来方向
1. **计算开销**：双分支结构 + MCP 迭代 Transformer 增加了推理延迟，未报告 FLOPs/参数量对比，实时部署潜力待验证。
2. **泛化边界**：仅在五个公开 MAS 数据集上验证，对复杂多物种、极端浑浊场景的鲁棒性未充分讨论。
3. **提示自动化假设**：MCP 完全自动生成本文未引入人工交互模式，在需要精确交互的场景下灵活性受限。
4. **域外泛化**：未测试在未见过的海洋生态域（如深海热液口、极地海域）上的零样本迁移性能。

## 研究启发与可借鉴点
1. **双分支 + gamma 校正的域适配策略**：将原始输入与光照补偿输入并行处理，可作为视觉 Foundation Model 适配光照畸变场景（如内窥镜、遥感低光照）的通用范式。
2. **自动多级提示生成机制（MCP）**：用 Transformer 迭代生成可学习的 soft prompt，替代手工提示或单点提示，可迁移至医学影像、遥感等需要上下文先验的分割任务。
3. **C³P 连通性预测范式**：将 mask 预测转化为结构化连通性预测，适用于任何边界不规则、形态多变的目标分割（如细胞、树叶、云雾）。
4. **PMS 互监督训练策略**：双解码器交叉伪标签监督 + 动态系数调度，可推广至半监督/自蒸馏设置，提升数据利用效率。
5. **参数高效微调的组合策略**：LoRA（Q/V）+ Adapter（FFN）同时使用，比单一 adapter 效果更好，可为其他 FM 微调提供超参参考。

## 关键术语表
**MAS（Marine Animal Segmentation）**：海洋动物分割，针对水下环境中海洋生物实例的像素级分割任务。
**SAM（Segment Anything Model）**：Meta 提出的通用图像分割基础模型，支持零样本迁移与多类型提示引导。
**MCP（Multi-level Coupled Prompt）**：多级耦合提示，通过 Transformer 迭代生成自动提示并跨注意力注入编码器特征的策略。
**DFAM（Dilated Fusion Attention Module）**：膨胀融合注意力模块，结合空洞卷积与通道注意力实现多尺度特征金字塔融合。
**C³P（Criss-Cross Connectivity Prediction）**：十字交叉连通性预测，将二值 mask 解码为 8 通道连通性图以显式建模像素间结构性关系。
**PMS（Pseudo-label Mutual Supervision）**：伪标签互监督，双解码器互相生成阈值化伪标签进行交叉蒸馏的正则化策略。
**LoRA（Low-Rank Adaptation）**：低秩适应，通过在预训练模型注意力矩阵中注入低秩增量矩阵实现参数高效微调。

## 可复现要素
- **数据集**：MAS3K、RMAS、UFO120、RUWI、USOD10K 均为公开数据集。
- **代码**：已开源，见 https://github.com/Drchip61/DualSAM。
- **权重**：SAM-B 预训练权重（官方提供），其余模块随机初始化后训练。
- **关键超参**：输入尺寸 512×512，batch size=8，初始 lr=0.001（AdamW），weight decay=0.1，每 20 epoch ×0.1 衰减，共 50 epoch；伪标签阈值 ξ=0.5；MCP 中 j=3×i。
