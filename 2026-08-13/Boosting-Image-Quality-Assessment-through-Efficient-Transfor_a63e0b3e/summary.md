---
title: "Boosting-Image-Quality-Assessment-through-Efficient-Transfor"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:16"
---

# 论文速读：Boosting-Image-Quality-Assessment-through-Efficient-Transfor

## 一句话总结
本文提出 LoDa（LOcal Distortion Aware efficient transformer adaptation），通过将预训练 CNN 提取的多尺度局部失真特征以跨注意力机制注入冻结的大规模预训练 ViT 主干，以仅 9M 可训练参数高效完成图像质量评估（IQA）任务，在七个主流基准上全面刷新 SOTA 并验证了大规模基础模型的 Scaling Law 同样适用于低层 IQA 任务。

## 研究问题与动机
- **标注数据稀缺与模型规模化矛盾**：IQA 依赖复杂的人眼主观评分，数据采集成本高、规模有限，难以支撑直接全量微调或从头预训练千亿级视觉基础模型。
- **ViT 的局部归纳偏置缺失**：大规模预训练 ViT 擅长建模全局非局部依赖，但对图像局部结构、高频失真与边缘纹理的刻画较弱，而 IQA 同时高度依赖局部细节与全局语义。
- **现有高效适配方法未针对 IQA 特性设计**：Adapter、LoRA、VPT 等通用视觉适配技术主要面向高级语义任务，未显式补充 IQA 所需的局部失真先验，且在高维 ViT 空间直接交互会带来参数量爆炸。
- **Scaling Law 在低层任务的适用性待验证**：高维视觉语言模型的成功促使学界思考：在补充适当先验的前提下，更大预训练骨干是否也能在 IQA 这类低层任务上持续受益。

## 核心贡献（创新点）
1. **首次系统性验证大规模预训练 ViT 可通过高效适配赋能 IQA**：冻结 ViT 主干，仅训练轻量适配模块，以极少参数挖掘预训练知识，缓解数据稀缺瓶颈。与现有方法本质区别在于不依赖任务特定预训练或全量微调，而是走“通用大模型+局部先验注入”路线。
2. **提出多尺度局部失真提取与注入机制**：利用预训练 CNN 的强局部归纳偏置提取多尺度失真特征，并通过降维跨注意力让 ViT token 主动查询对齐，解决 ViT 16×16 patch 与 CNN 多尺度特征空间粒度不对齐的问题。与 MUSIQ 等直接拼接特征的方法本质不同，本文强调“查询式特征融合”与参数高效性。
3. **设计降维适配器式跨注意力（Down-projected MHCA）**：借鉴 NLP Adapter 思想，将高维 ViT token 与失真 token 投影至低维 $r$ 后再执行交叉注意力，并引入近零初始化的可学习缩放因子 $s^i$，在保证预训练分布不被破坏的同时将可训练参数控制在 9M。相比 LoRA/Adapter 等通用方案，该设计专为 IQA 的局部高频失真补偿而定制。

## 方法详解
- **整体架构**：输入图像并行送入冻结的预训练 CNN 与冻结的大规模预训练 ViT。CNN 分支负责抽取多尺度特征并聚合局部失真，ViT 分支提供全局语义表征。仅训练局部失真提取器、局部失真注入器及顶部单层回归头，ViT 与 CNN 权重全程冻结。
- **局部失真提取器（Eq.1）**：对 CNN 第 $j$ 阶段输出的多尺度特征 $F^j \in \mathbb{R}^{b \times c_j \times m_j \times n_j}$，经可训练卷积层 $\phi_j$ 与平均池化聚合失真信息，得 $\bar{F}^j \in \mathbb{R}^{b,c,m,n}$，展平拼接后得到多尺度失真 token $F_{msd} \in \mathbb{R}^{b, \sum_j(m \times n), c}$。
- **局部失真注入器（Eq.2-4）**：ViT 第 $i$ 层输出 $F_{vit}^i$ 作为 Query，$F_{msd}$ 作为 Key/Value。为避免 768 维直接计算带来的开销，两者先经可训练 MLP $f(\cdot)$ 降维至 $r$ 维，再执行多头交叉注意力：$\bar{F}_{msd}^i = \text{MHCA}(Q_i, K_i, V_i) + Q_i$，随后通过可学习标量 $s^i$（初始化接近 0）加权融合并升维回原维度：$\hat{F}_{vit}^i = F_{vit}^i + s^i \times \hat{F}_{msd}^i$。
- **回归头与损失（Eq.5）**：取 ViT 的 CLS token 输入单层回归头预测质量分数。采用直接基于 PLCC 构造的可微损失
