---
title: "Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhang_Fantastic_Animals_and_Where_to_Find_Them_Segment_Any_Marine_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:54:02"
field: "水下图像分割"
keywords: ["Marine Animal Segmentation", "Segment Anything Model", "Domain Adaptation", "Underwater Image Segmentation", "Prompt Learning", "Connectivity Prediction"]
innovations: ["提出多级耦合提示（MCP）策略，通过自生成提示解决 SAM 单点提示不足问题", "设计交错交叉连通性预测（C³P）范式，以结构化方式捕获像素间连通关系", "构建双分支互监督框架（PMS），协同增强水下场景的特征学习"]
benchmarks: ["MAS3K", "RMAS", "UFO120", "RUWI", "USOD10K"]
---

# 论文速读：Fantastic-Animals-and-Where-to-Find-Them-Segment-Any-Marine

## 一句话总结
本文提出 Dual-SAM 框架，通过双分支 SAM 编码器、多级耦合提示（MCP）、膨胀融合注意力模块（DFAM）和交错交叉连通性预测（C³P）机制，将 SAM 适配于海洋动物分割（MAS）任务；在五个公开数据集上均达到 SOTA 性能。

## 研究问题与动机
1. **领域先验缺失**：SAM 在自然图像上训练，缺乏水下环境先验知识（浑浊、光照变化、色彩失真等），直接迁移效果差。
2. **单点提示不足**：SAM 原始单位置提示（点/框）对水下复杂场景的引导能力有限，无法提供全面的先验信息。
3. **简单解码器局限**：SAM 解码器结构简单，难以捕捉海洋动物的复杂细节与边界。
4. **像素间连通性被忽略**：传统逐像素分类方法忽视了离散像素间的连通关系，导致分割边界不规则。

## 核心贡献（创新点）
1. **Dual-SAM 双分支特征学习框架**：引入 Gamma 校正补偿光照并构建双分支结构，使 SAM 能自适应融合水下先验知识，与 vanilla SAM 形成本质区别——后者未经过水下场景适配。
2. **多级耦合提示（MCP）策略**：将原始图像与校正图像拼接后通过 Transformer 迭代生成 auto-prompts，以 Multi-Head Cross-Attention 注入编码器，与 SAM 的单点提示机制形成鲜明对比，实现更全面的多层级先验引导。
3. **膨胀融合注意力模块（DFAM）**：结合膨胀卷积与通道注意力构建特征金字塔解码器，逐层融合提示特征；相比 SAM 原有简单解码器，显著增强上下文感知与感受野。
4. **交错交叉连通性预测（C³P）范式**：将单通道 mask 转化为 8 通道连通性标签，预测像素间的十字交叉连通关系；与 ConnNet 仅考虑相邻像素的方法相比，覆盖更广的连通范围，获得更完整的结构化分割表示。

## 方法详解
- **Dual-SAM Encoder（DSE）**：输入图像经 Gamma 校正得 $I^\beta = \sqrt[\gamma]{I^\alpha}$（$\gamma = \lg(0.5) - \lg(\text{mean}_I^{\text{gray}}/255)$），双分支图像并行输入 SAM-B 编码器；在 MHSA 的 Query 和 Value 中注入低秩适配器（LoRA），在 FFN 中注入标准 Adapter，仅训练这些新增参数，冻结 SAM 主干。
- **MCP 生成**：将原始图 $I^\alpha$ 与校正图 $I^\beta$ 拼接后经 Patch Embed 得 $I_0^\omega$，经 4 层 Transformer 迭代生成 $I_i^\omega$；再以编码器输出 $X_j^\alpha$（Query）和 $X_j^\beta$（Key）、$I_i^\omega$（Value）做 Cross-Attention，得到耦合提示 $\mathcal{P}_i^\alpha$ 与 $\mathcal{P}_i^\beta$，加权残差融合入特征。
- **DFAM**：融合提示特征 $E_i$ 与上级特征 $G_i$，经 $1\times1$ 卷积 + GELU 得中间特征，再通过通道注意力门控；经膨胀率=2 的 $3\times3$ 卷积后上采样进入下一层，构建特征金字塔。
- **C³P**：将 gt mask 扩展为 8 通道，分别编码中心像素到上下左右及四个对角方向距离为 1 和 2 的像素连通关系；双解码器分别输出 $\alpha/\beta$ 分支的连通图，损失为逐像素 BCE：$\mathcal{L}_l^{\alpha/\beta} = -\sum_{w,h,c}[Y\ln P + (1-Y)\ln(1-P)]$。
- **PMS 互监督**：两分支预测经阈值 $\xi=0.5$ 生成伪标签，互相作为监督信号；引入动态更新系数 $\mu = 0.1\times e^{-5(1-t/T)^2}$ 在训练初期给予较小权重；推理时采用互确认策略：若 $(w,h)$ 与其对称位置 $(u,v)$ 的连通预测均为正，则判定为前景。

## 实验与结果
- **数据集**：MAS3K（1769 训练 / 1141 测试）、RMAS（2514/500）、UFO120（1500/120）、RUWI（525/175）、USOD10K（9229/1026）。
- **评估指标**：mIoU、Sα、Fβ^w、mEφ、MAE。
- **主要结果**：
  - MAS3K：mIoU=**0.789**，Sα=0.884，Fβ^w=0.838，mEφ=0.933，MAE=0.023，全面提升超 **3–5%**，领先次优方法。
  - RMAS：mIoU=0.735，MAE=0.022。
  - UFO120：mIoU=**0.810**，Fβ^w=0.864，MAE=0.064。
  - RUWI：mIoU=**0.904**，Fβ^w=0.939，MAE=0.035。
  - USOD10K：Sα=0.9238，maxF=0.9311，MAE=0.0185，超越所有基线。
- 消融实验验证了 C³P、双分支、PMS、MCP、DFAM、Adapters 各项组件的有效性。

## 相关工作脉络
1. **CNN 基线（SINet/PFNet/C2FNet/MASNet）**：基于 CNN 的海底动物分割方法，缺乏长程上下文建模能力；本文方法基于 SAM+Transformer 架构，从根本上解决此问题。
2. **Transformer 基线（SETR/TransUNet/H2Former）**：利用 ViT 全局感知能力的分割网络；本文通过适配器高效微调预训练 SAM，避免从零训练 Transformer 所需的大规模数据。
3. **SAM 适配方法（SAM-Ad/SAM-DA/AquaSAM）**：均通过适配器或修改解码器迁移 SAM；本文进一步引入 MCP 多级提示与双分支互监督，解决单点提示不足的问题。
4. **连通性预测方法（ConnNet）**：仅考虑相邻像素间的直接连通；本文 C³P 扩展至十字交叉范围内的距离-1 和距离-2 像素，覆盖更广的结构信息。
5. **水下显著性检测（TC-USOD/D3Net/BTS-Net 等）**：关注显著目标定位而非精细分割；本文在 USOD10K 上的实验验证了方法的泛化能力。
6. **LoRA/Adapter 高效微调**：LoRA [19]、Adapter [18] 在 NLP/视觉中的应用；本文将其联合用于 SAM 编码器，仅增加少量可训练参数。

## 局限性与未来方向
- Gamma 校正为经验性简单操作，未采用更深度的水下图像增强方法（如物理模型驱动或深度学习增强）。
- 方法针对静态图像设计，未扩展到视频序列或时序分割任务。
- 未讨论多实例重叠严重场景下的分割性能。
- 消融实验中主要报告 MAS3K 数据，其他数据集趋势虽一致但细节披露有限。
- 未来可能扩展至多类别海洋生物分割及实时推理优化。

## 研究启发与可借鉴点
1. **双分支 + 互监督范式**：通过双路径预测相互生成伪标签进行监督，可有效缓解域偏移问题，可迁移至其他跨域分割任务（如医学影像、遥感）。
2. **SAM 适配的低成本方案**：仅需在 SAM 编码器中插入 LoRA + Adapter 即可实现高效微调，为其他领域适配 SAM 提供了轻量级范式参考。
3. **自生成提示替代手工提示**：MCP 通过跨注意力自动生成提示，摆脱了对人工标注点/框的依赖，可推广至无提示分割场景。
4. **连通性预测替代逐像素分类**：C³P 从向量/结构层面建模分割掩码，对边界模糊、形变剧烈的目标尤为有效，值得在其他分割任务中探索。

## 关键术语表
- **MAS（Marine Animal Segmentation）**：海洋动物分割，指从水下图像中精确分割出海洋生物区域的任务。
- **SAM（Segment Anything Model）**：Meta 提出的通用图像分割基础模型，支持零样本泛化，通过提示引导生成任意物体的分割掩码。
- **MCP（Multi-level Coupled Prompt）**：多级耦合提示，将原图与 Gamma 校正图拼接后经 Transformer 迭代生成自动提示，通过交叉注意力注入编码器。
- **DFAM（Dilated Fusion Attention Module）**：膨胀融合注意力模块，结合膨胀卷积与通道注意力，用于逐层融合 SAM 编码器的多级特征。
- **C³P（Criss-Cross Connectivity Prediction）**：交错交叉连通性预测，将单通道 mask 展开为 8 通道连通性标签，捕捉像素间十字交叉范围的结构性关系。
- **PMS（Pseudo-label Mutual Supervision）**：伪标签互监督，双分支通过阈值化对方预测生成伪标签，互相提供补充监督信号。
- **LoRA（Low-Rank Adaptation）**：低秩适配，通过在 Q/V 矩阵中注入低秩分解矩阵实现高效微调。
- **Adapter**：网络适配器，在 Transformer FFN 中插入压缩-激发两层结构，以少量参数注入领域特定信息。

## 可复现要素
- **数据集**：MAS3K、RMAS、UFO120、RUWI、USOD10K，均为公开数据集。
- **代码**：已开源，https://github.com/Drchip61/DualSAM。
- **模型权重**：SAM-B 预训练权重（官方提供），其余随机初始化后训练；论文未明确公开最终模型权重链接。
- **关键超参**：输入尺寸 $512\times512$，batch size=8，学习率=0.001（每 20 epoch ×0.1），weight decay=0.1，AdamW 优化器，训练 50 epochs，伪标签阈值 $\xi=0.5$，SAM 编码器第 $j=3\times i$ 层注入 MCP。
