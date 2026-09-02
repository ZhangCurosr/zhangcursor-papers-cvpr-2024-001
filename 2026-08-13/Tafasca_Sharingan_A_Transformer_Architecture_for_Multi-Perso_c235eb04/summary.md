---
title: "Sharingan: A Transformer Architecture for Multi-Person Gaze Following"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:18:33"
field: "多人员视线性（Multi-Person Gaze Following）"
keywords: ["gaze following", "multi-person vision", "transformer architecture", "dense prediction", "Conditional DPT", "vision transformer"]
innovations: ["将人物信息编码为单个 location-aware gaze token 并与图像 token 在 Transformer 中联合交互", "提出 Conditional DPT 多尺度条件化解码器用于生成 gaze heatmap", "首次在原任务设定下实现端到端多人员 gaze following，无需可学习对象嵌入和 Hungarian matching"]
benchmarks: ["GazeFollow", "VideoAttentionTarget", "ChildPlay"]
---

# 论文速读：Sharingan: A Transformer Architecture for Multi-Person Gaze Following

## 一句话总结
本文提出 Sharingan，一个基于 Transformer 的多人员视线性（gaze following）架构，通过将每个人编码为单个 gaze token 并与图像 token 联合处理，在一次前向推理中同时预测多人注视点，在 GazeFollow、VideoAttentionTarget 和 ChildPlay 三个数据集上均达到 SOTA。

## 研究问题与动机
- 传统双塔 CNN 架构需对每个目标人物单独做一次前向传播，处理同一场景中多人的注视预测时效率极低。
- 现有基于 Transformer 的多视角方法（如 Tonini et al. [38]、Tu et al. [39]）使用固定数量的可学习嵌入同时解码人物与注视目标，需要额外的匹配步骤才能与标注关联，难以用现有基准可靠量化评估，也无法直接集成到更大的行为理解系统中。
- 大多数工作聚焦于如何构建人物凝视表示与场景显著性图，却忽视了最终 heatmap 的解码机制；现有解码模块输入维度极低（如 7×7），限制了预测精度。
- 如何在保持原始任务 formulations（给定头图和头框，预测 2D 像素坐标）的前提下，设计一种能原生支持任意人数、单次前向推理的高效 Transformer 架构。

## 核心贡献（创新点）
1. **Gaze Token 表征**：将人物信息编码为单个 location-aware gaze token（head crop embedding + bounding box embedding 相加），与图像 token 在 Transformer 中联合交互，避免了以往视觉注意力图/视锥等冗余表征，且在 ablation 中性能最优。
2. **Conditional DPT 多尺度解码器**：借鉴 DPT 思想，从 Transformer encoder 的多个中间层提取图像和 gaze token 表示，在模拟的不同分辨率下逐步融合，形成轻量级的多尺度解码机制，输出更细粒度的 gaze heatmap 并能更好表达不确定性。
3. **首次在原任务设定下实现端到端多人 gaze following**：与 DETR 式 set-prediction 方法不同，本文不引入可学习对象嵌入和 Hungarian matching，而是直接沿用原 gaze following 的任务设定（head crop + head box → gaze heatmap），使结果可直接与 benchmark 对照。
4. **系统性实验与深入分析**：在 GazeFollow、VideoAttentionTarget、ChildPlay 上均达 SOTA；并深入分析了训练人数 Np^tr 的影响、heatmap vs 2D point 回归的差异、跨数据集泛化等关键问题。

## 方法详解
- **Image Tokens**：标准 ViT patch projection，将场景图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ 切分为 $N$ 个 patch token，附加位置编码，维度 $D$。
- **Gaze Token（单人）**：头图 $\mathbf{h}_{\text{crop}}$ 经 gaze backbone（预训练 ResNet-18）得到 gaze embedding $\mathbf{g}^{\text{emb}}$；该 embedding 经 MLP $\mathcal{O}_{\text{gv}}$ 预测归一化 2D 凝视向量 $\mathbf{g}_{\text{v}}$（用 angular loss 监督）；同时 $\mathbf{g}^{\text{emb}}$ 经线性投影 $\mathcal{P}_{\text{gaze}}$ 映射到 token 维度，再与 head bounding box $\mathbf{h}_{\text{bbox}}$ 经 $\mathcal{P}_{\text{bbox}}$ 投影得到的 embedding 相加，得到位置感知 gaze token：$\mathbf{x}^{\text{g}} = \mathbf{x}^{\text{emb}} + \mathbf{x}^{\text{bbox}}$。
- **Multi-person 扩展**：$N_p$ 人产生 $N_p$ 个 gaze token，拼接为 $\mathbf{x}^{\text{g}} = \mathbf{x}_1^{\text{g}} \oplus \ldots \oplus \mathbf{x}_{N_p}^{\text{g}}$，与图像 token 共同输入 Transformer。
- **Transformer Encoder**：标准 ViT-base（来自 Multimodal MAE 初始化），$L$ 层，输入 $\mathbf{x} = \mathbf{x}^{\text{img}} \oplus \mathbf{x}^{\text{g}}$，输出 $\mathbf{x}^{\text{out}}$。
- **Conditional DPT Decoder**：取 encoder 第 4、8、16、32 层的图像和 gaze token 表示，逐层在不同分辨率 $(H/k, W/k)$ 下进行条件化融合：将图像 feature map 复制 $N_p$ 份，分别与每人 gaze token 做 element-wise dot-product，再通过残差 convnet 逐级上采样与融合，最终以 conv head 输出尺寸 $(B, N_p, 1, H_{\text{hm}}, W_{\text{hm}})$ 的 gaze heatmap。
- **In-Out 分类**：将输入与输出 gaze token 拼接后通过 7 层 MLP，预测每人"注视是否在图像内"的二元标签。
- **Loss**：$\mathcal{L} = \lambda_{\text{reg}} \mathcal{L}_{\text{reg}} + \lambda_{\text{ang}} \mathcal{L}_{\text{ang}} + \lambda_{\text{io}} \mathcal{L}_{\text{io}}$，其中 $\lambda_{\text{reg}} = 1000$，$\lambda_{\text{ang}} = 3$；heatmap loss 为像素级 MSE，angular loss 为 $1 - \cos(\mathbf{g}_v^{gt}, \mathbf{g}_v^{pred})$。
- **训练策略**：GazeFollow 上训练 20 epoch，其余数据集冻结除 gaze decoder 和 In-Out classifier 外的参数 fine-tune 2 epoch；使用 AdamW，lr=3e-5，cosine annealing，辅以 SWA。

## 实验与结果
- **数据集**：GazeFollow（~130K 标注实例）、VideoAttentionTarget（164K 实例）、ChildPlay（257K 实例）。
- **评估指标**：AUC、Avg. Distance、Min. Distance、PLAH（Precision of Looking at People's Heads）、AP。
- **GazeFollow**：AUC=0.944，Avg. Dist.=0.113，Min. Dist.=0.057，PLAH=0.667；超过仅有多人能力的 Jin [16]（Avg. Dist. 0.126→0.113）及最强单模态方法 Gupta [12]（I+D+P，Avg. Dist. 0.114→0.113，且无需深度/姿态模态）。
- **VideoAttentionTarget**：Dist.=0.107，PLAH=0.738，AP=0.891；zero-shot 从 GazeFollow 迁移即达 Dist.=0.113，优于 Jin [16] 的 0.134。
- **ChildPlay**：Dist.=0.106，PLAH=0.600，AP=0.990。
- **Ablation**：Gaze Token 优于 Head Mask Embed、Head Crop Tokens、Gaze Cone Tokens；Conditional DPT 优于 Token-to-Heatmap MLP、Dot-Product、Up & Dot-Product；$N_p^{\text{tr}} \geq 2$ 时性能稳定，$N_p^{\text{tr}}=1$ 会显著损害多人推理能力。
- **Heatmap vs 2D Point**：2D point 在 Avg. Dist.（0.106）和 PLAH（0.683）上略优，但在 Min. Dist.（0.066 vs 0.057）和 RLAH（0.368 vs 0.571）上明显落后，说明多峰分布场景下 heatmap 更稳健。

## 相关工作脉络
- **Recasens [29]**：gaze following 任务提出者，双塔 CNN 架构（场景塔 + 人物塔），需逐人前向传播；本文在其任务定义基础上改进多人性效。
- **Chong [7]**：扩展支持 in-vs-out 预测，引入 heatmap 输出；本文沿用此设定但用 Transformer 替代双塔 CNN。
- **Fang [11] / Gupta [12]**：融合深度/姿态等多模态的单人生视线性方法，精度高但依赖额外模态且推理低效；本文仅用图像模态即达 comparable 甚至更优性能，且支持多人。
- **Jin [16]**：最早的卷积式多人 gaze following，但各人物的计算相互独立、忽略人物间交互；本文通过 Transformer attention 实现人物-场景联合推理。
- **Tonini [38] / Tu [39]**：基于 Transformer 的 set-prediction 多人方法，使用固定数量可学习嵌入同时预测 head box 和 gaze target，需后处理匹配；本文坚持原任务设定，无需 matching。
- **Tafasca [35]（ChildPlay 作者）**：结合深度的单人多人员视线性方法；在视频数据集上本文与其性能接近，但本文不依赖深度模态且原生支持多人。

## 局限性与未来方向
- **可解释性不足**：图像 token 与 gaze token 共享同一 Transformer 权重，难以理解模型如何组合两类信息；未来可探索解耦场景与人物处理、在架构中选择性融合的方案。
- **视频时序稳定性**：当前架构为帧级设计，未显式建模时序一致性，在视频场景下预测稳定性有待提升。
- **gaze backbone 依赖人脸检测器**：head crop 和 head box 依赖外部检测器（CrowdHuman 训练），检测误差会传导至 gaze 预测。
- **未来方向**：扩展整合深度、语义等多模态输入；支持更多输出类型（手势、交互关系）；应用于共享注意力和互相凝视等下游任务。

## 研究启发与可借鉴点
- **Gaze Token 设计**：将人物信息压缩为单个 token（embedding + bbox embedding 相加）而非视觉 mask 或 attention map，是兼顾效率与信息密度的有效思路，可迁移至其他"对象-场景"交互任务（如目标追踪、hand-object interaction）。
- **Conditional DPT 解码器**：将 DPT 的多尺度密集预测思想与条件化（per-person dot-product）结合，为 dense prediction 任务中的多条件融合提供了一个轻量且有效的范式。
- **训练人数无关性**：证明 $N_p^{\text{tr}} \geq 2$ 即可让模型学会处理任意人数推理，这一发现对多人理解任务（multi-person action recognition、crowd analysis）的训练策略设计有参考价值。
- **Heatmap vs Point 回归的系统分析**：本文对两种输出形式的细致对比（多峰分布下的表现差异、Avg. Dist. 指标的局限性）提醒研究者在 gaze 相关任务中应综合多指标评估，避免单一指标误导。
- **跨数据集 zero-shot 泛化验证**：仅用 GazeFollow 训练即在 VideoAttentionTarget 和 ChildPlay 上取得竞争性结果，表明良好架构设计可显著减少对大规模跨域标注的依赖。

## 关键术语表
- **Gaze Following（视线性）**：在图像中预测目标人物注视的 2D 像素坐标，无需可穿戴设备或 3D 头部姿态假设。
- **Gaze Token**：由 head crop 经 gaze backbone 提取的 embedding 与 bounding box embedding 相加得到的单一 token，用于在 Transformer 中承载人物特定信息。
- **Conditional DPT**：本文提出的多尺度 gaze 解码器，以不同 encoder 层的图像和 gaze token 为条件，在多个模拟分辨率下渐进融合输出 gaze heatmap。
- **In-vs-Out Prediction**：预测人物注视点是否落在图像框架内的二分类任务，作为辅助训练信号。
- **PLAH（Precision of Looking at People's Heads）**：预测与标注的注视点落在同一人头框内时的精确率，引入语义信息的评估指标。
- **Angular Loss**：监督 2D 凝视向量预测的损失，最大化预测向量与 GT 向量之间的余弦相似度。
- **ViT（Vision Transformer）**：将图像切分为 patch 并送入标准 Transformer 进行视觉表征学习的架构。
- **Multimodal MAE**：多模态掩码自编码器，本文用于初始化 ViT encoder 的预训练模型。

## 可复现要素
- **数据集**：GazeFollow、VideoAttentionTarget、ChildPlay（均为公开数据集）。
- **代码/权重**：论文声明"code, checkpoints, and data extractions will be made publicly available soon"，截至论文发表时暂未开源。
- **关键超参**：输入分辨率 224×224，heatmap 输出 64×64；gaze backbone 为 Gaze360 预训练的 ResNet-18；encoder 为 ViT-base（Multimodal MAE 初始化）；训练 20 epoch（GazeFollow），fine-tune 2 epoch；lr=3e-5，$\lambda_{\text{reg}}=1000$，$\lambda_{\text{ang}}=3$；AdamW + cosine annealing + SWA；训练时 $N_p^{\text{tr}}=2$。
