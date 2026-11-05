---
title: "Flexible-Biometrics-Recognition-Bridging-the-Multimodality-G"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Tiong_Flexible_Biometrics_Recognition_Bridging_the_Multimodality_Gap_through_Attention_Alignment_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:08:22"
---

# 论文速读：Flexible-Biometrics-Recognition-Bridging-the-Multimodality-G

## 一句话总结
本文提出 **Flexible Biometric Recognition (FBR)** 框架，通过融合面部、眶周图像与软生物属性，结合多模态融合注意力（MFA）与多模态提示微调（MPT）机制，学习模态不变的共享嵌入，在统一架构内同时高效支撑 intra-modality 与 cross-modality 身份识别任务。

## 研究问题与动机
1. **单一模态的固有缺陷**：纯人脸识别易受口罩、整容影响；纯眶周识别易受墨镜/眼镜遮挡干扰，独立部署鲁棒性不足。
2. **传统多模态系统的部署瓶颈**：需同时采集、存储多模态模板，计算与存储开销大，且实际场景中常面临部分模态缺失或仅存档单模态的问题。
3. **Intra-/Cross-modality 性能权衡**：现有跨模态方法（如 HA-ViT）侧重跨模态对齐，牺牲了单模态判别力；条件生物特征方法依赖显式 conditioning，灵活性受限。
4. **标准 Prompt Tuning 的局限**：VPT/SPT 主要针对单任务微调设计，缺乏对多模态间细粒度关联的逐层引导，难以直接适配面部-眶周-属性的融合对齐需求。

## 核心贡献（创新点）
1. **FBR 统一框架**：首次将面部、眶周与软生物属性纳入同一 Transformer 骨干，输出模态不变嵌入，支持训练时多模态融合、推理时按需单模态调用的灵活部署。
2. **MFA-ViT 骨干设计**：提出基于深度可分离卷积的融合自注意力模块（DWFC-MSA），在 ViT 层级显式建模三类模态的局部空间结构与跨模态依赖，避免简单拼接带来的特征冲突。
3. **MPT 逐层提示机制**：区别于仅作用于输入层的 VPT，MPT 在每个 MFA 块输入处动态更新提示向量，既促进跨模态特征交互，又保留各模态独特判别信息。
4. **联合损失与分类头优化**：联合大间隔 Softmax 损失（$\mathcal{L}_{\mathrm{LM}}$）与跨模态对比损失（$\mathcal{L}_{\mathrm{CL}}$）打破 intra/cross 性能 trade-off；证明使用 MPT 输出（PRM）作分类头显著优于标准 class token（CLS）。

## 方法详解
- **输入与编码**：面部 $I_f$ 与眶周 $I_p$ 图像尺寸 $112 \times 112$，切分为 14 个 $8 \times 8$ patch，经共享 ViT 投影得 $\mathbf{Z}_f, \mathbf{Z}_p \in \mathbb{R}^{1 \times d}$（$d=1024$）；47 维软属性 $I_a$ 经 Feature Tokenizer 映射为 $\mathbf{Z}_a \in \mathbb{R}^{1 \times d}$。
- **MFA 块结构**：$M=2$ 个 MFA 块，每块含 $N=4$ 层 $F_n$。每层由残差分支组成：① $3 \times 3$ DWS-Conv 提取局部空间特征；② DWFC-MSA（LayerNorm → $3 \times 3$ 深度卷积多头自注意力 → MLP）捕获模态内/间关联；③ LeakyReLU + 残差叠加。
- **MPT 逐层更新**：可学习提示 $\mathbf{P}_*$ 附加于 $\mathbf{T}_*$ 与 $\mathbf{Z}_*$ 之前。每层通过 $1 \times 1$ Conv + ReLU 计算 $\mathbf{P}'_{*,n+1} = \mathrm{ReLU}(\mathrm{Conv}([\mathbf{P}_{*,n}, \mathbf{P}_{*,n+1}]))$，再与当前层输入拼接送入 $F_n$，实现提示向量的递进式跨模态引导。
- **特征聚合**：舍弃标准 CLS token，将各 MFA 块的最终提示输出求和后平均池化得到联合嵌入 $\mathbf{J}_*$，作为模态不变的身份表征。
- **损失函数**：$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{LM}_f} + \mathcal{L}_{\mathrm{LM}_p} + \mathcal{L}_{\mathrm{CL}}$。$\mathcal{L}_{\mathrm{LM}}$ 为大间隔 softmax，强化同模态同类/异类边距；$\mathcal{L}_{\mathrm{CL}}$ 为三组跨模态对比损失（$f$-$p$, $f$-$a$, $p$-$a$），通过正样本相似度与负样本（同模态异人 + 跨模态异人）加权比值约束嵌入空间分布，温度系数 $\theta$ 分别设为 0.03（$f$-$p$）与 0.04（含属性项）。
- **灵活推理**：推理阶段 detach softmax，按输入模态提取 $\mathbf{J}_f$ 或 $\mathbf{J}_p$，与 Gallery 集计算余弦相似度完成匹配。

## 实验与结果
- **数据集**：训练集 VGGFace2 + MAAD-Face（1.49M 样本，9,131 身份，47 维属性，8:2 划分）；测试集 Ethnic, FaceScrub, IMDB, Cross-Modal DB。
- **评估协议**：Rank-1 准确率；所有模型统一作为特征提取器，cosine similarity 匹配。
- **主要结果**（Rank-1 %）：
  - **Ethnic**：$f$-$f$ 94.82, $p$-$p$ 89.98, $f$-$p$ 86.70, $p$-$f$ 89.07
  - **FaceScrub**：$f$-$f$ 95.71, $p$-$p$ 93.06, $f$-$p$ 90.38, $p$-$f$ 92.02
  - **IMDB**：$f$-$f$ 86.03, $p$-$p$ 80.53, $f$-$p$ 75.28, $p$-$f$ 76.36
  - **Cross-Modal DB**：$f$-$f$ 85.88, $p$-$p$ 76.54, $f$-$p$ 72.01, $p$-$f$ 75.96
- **最强提升**：相对次优基线 HA-ViT，FaceScrub $p$-$f$ 提升 **+3.85%**，Ethnic $p$-$f$ 提升 **+11.33%**；相较单模态 Baseline 跨模态任务从 ~0.3% 跃升至 72%~92%，验证模态对齐的有效性。
- **消融结论**：① 加入 $I_a$ 带来显著增益；② MPT 比无 prompt / VPT 表现更优，Grad-CAM 显示 MPT 更精准聚焦双眼共享
