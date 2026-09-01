---
title: "FocSAM-Delving-Deeply-into-Focused-Objects-in-Segmenting-Any"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Huang_FocSAM_Delving_Deeply_into_Focused_Objects_in_Segmenting_Anything_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:58:47"
field: "交互式图像分割"
keywords: ["交互式分割", "Segment Anything Model", "动态窗口注意力", "动态ReLU", "高效推理"]
innovations: ["提出Focus Refiner模块动态精炼图像Embedding以解决SAM稳定性问题", "设计Dwin-MSA实现基于预测掩码的动态窗口注意力计算", "发明P-DyReLU将点击融合Query信息注入像素级动态激活"]
benchmarks: ["SBD", "DAVIS", "GrabCut", "Berkeley", "MVTec", "COD10K"]
---

# 论文速读：FocSAM-Delving-Deeply-into-Focused-Objects-in-Segmenting-Any

## 一句话总结
FocSAM针对SAM在挑战性样本上交互式分割不稳定的问题，通过引入Focus Refiner模块动态聚焦目标区域并深度融合初始点击信息，在保持低计算开销的同时使交互分割性能达到SOTA水平，CPU推理速度仅为SimpleClick的约5.6%。

## 研究问题与动机
1. **SAM Pipeline导致无法动态聚焦**：SAM在交互前一次性预处理图像生成全局Embedding，无法像传统zoom-in策略那样在交互过程中动态放大目标区域，难以应对伪装目标等挑战性场景。
2. **轻量级Decoder融合不足**：为追求实时性，SAM的Decoder仅通过少量Cross-Attention融合点击信息与图像Embedding，初始点击对分割结果的显著影响未能被充分利用。
3. **稳定性缺陷严重限制实际应用**：如图1所示，SAM在9次点击后IoU可达88.59，但第10次点击后骤降至12.78，这种不稳定性大幅降低了人工标注效率和用户体验。

## 核心贡献（创新点）
1. **提出FocSAM管道改造方案**：在SAM原有Pipeline基础上引入Focus Refiner，每对象交互一次动态调整图像Embedding，使模型能聚焦目标区域，与SAM本质区别在于打破了"预处理一次固定Embedding"的限制。
2. **设计Dynamic Window MSA（Dwin-MSA）**：基于上一轮预测掩码定位目标包围盒，动态选取与之相交的窗口进行注意力计算，避免背景冗余计算，同时采用Shift策略保证长距离交互，与Swin Transformer的固定窗口机制本质不同。
3. **发明Pixel-wise Dynamic ReLU（P-DyReLU）**：利用Decoder输出的点击融合Query Embedding与图像Embedding的点积相似度作为像素级激活参数，实现"增强目标相关、抑制无关"的动态激活，与标准ReLU及普通DyReLU相比引入了交互式语义引导。
4. **实现零样本交互式分割SOTA性能与极致效率的平衡**：在6个基准数据集上与SimpleClick-ViT-H性能持平，但CPU推理时间仅需其5.6%，当单图对象超10个时进一步降至1.2%。

## 方法详解
**整体架构**：FocSAM在SAM的Image Encoder + Prompt Encoder + Decoder框架上叠加Focus Refiner，Refiner在每对象的第K次交互（K=2）后被激活，输出精炼后的图像Embedding替代原始Embed Fed给后续交互。

**Focus Refiner核心设计**：
- 由12个Refine Block堆叠而成（6个Plain + 6个Shift交替）
- 每个Block接收图像Embedding F、上一轮预测掩码M^(K-1)、点击融合Query Embedding q_c^(K-1)

**Dwin-MSA（Dynamic Window Multi-head Self-Attention）**：
- 将图像Embedding划分为S×S窗口（默认S=16）
- 根据掩码生成包围盒，仅选取与包围盒相交的窗口进行计算
- 配合Shift策略交替使用Dwin和Shift Dwin，确保框内所有Patch充分交互
- 相比RoIAlign的优势：无需线性插值假设、自适应处理不同尺寸目标

**P-DyReLU（Pixel-wise Dynamic ReLU）**：
- 利用相似度 $\hat{f} \cdot q_f^\top$（像素级）和全局平均池化结果生成激活系数
- 系数经Channel-wise MLP变换后得到 $\bar{a}^0, \bar{b}^0, \bar{a}^1, \bar{b}^1$
- 最终激活：$\max\{\bar{a}^0 \odot \hat{f} + \bar{b}^0, \bar{a}^1 \odot \hat{f} + \bar{b}^1\}$
- 作用：基于点击信息动态调节每个像素的激活阈值，强化目标响应

**训练策略**：
- 两阶段训练：①先用COCO+LVIS微调SAM Decoder 320k步；②冻结Decoder训练Refiner 160k步
- 损失函数：NFL（Normalized Focal Loss）+ PTL（Point Loss，源自BRS）
- Image Encoder和Prompt Encoder全程冻结

## 实验与结果
**数据集**：训练集COCO + LVIS；测试集GrabCut、Berkeley、SBD、DAVIS、MVTec、COD10K（6个零样本基准）

**评估指标**：NoC@90（达到90% IoU所需点击次数，20次上限）、SPC（CPU每秒点击数）

**关键结果**（Table 1）：
| 方法 | SPC (SPC仅Decoder) | GrabCut | Berkeley | SBD | DAVIS | MVTec | COD10K | Mean |
|------|---------------------|---------|----------|-----|-------|-------|--------|------|
| SimpleClick-ViT-H | 6.99 | 1.50 | 1.75 | 4.70 | 4.78 | 10.56 | 9.13 | 5.40 |
| **FocSAM-ViT-H** | **0.39 (0.02)** | **1.32** | **1.47** | **4.69** | **4.77** | 11.14 | **8.91** | **5.38** |

- FocSAM在5/6数据集上达最优，Mean NoC@90仅5.38（vs SimpleClick 5.40）
- CPU推理速度提升约**17.9倍**（SPC从6.99降至0.39，仅为SimpleClick的5.6%）
- 多对象场景（>10个目标）下速度优势进一步放大至**83倍**（1.2%时间）

**稳定性验证**（Figure 4）：在SBD/MVTec/COD10K上统计连续点击的ΔIoU分布，FocSAM的退化样本显著少于SAM，证明Refiner有效缓解了SAM的不稳定性。

**消融实验**（Table 2）：
- Dwin-MSA和P-DyReLU单独贡献相当
- 两者结合在100次点击（NoC@95）的挑战性样本上增益更明显，验证互补性

## 相关工作脉络
1. **SAM [28]**：本文基础模型，采用"一次性Encoder预处理+轻量Decoder交互"范式，但缺乏动态聚焦能力和深度交互信息融合。
2. **SimpleClick [36]**：首个将大ViT引入交互式分割的方法，每次交互全模型推理，精度与FocSAM相当但速度慢17倍以上。
3. **InterFormer [23]**：提出特征复用减少冗余的Pipeline，FocSAM同样沿用了SAM式"预提取Embedding"的高效交互策略。
4. **f-BRS [46]**：传统Zoom-in方法的代表，通过放大目标区域提升分割精度，但无法直接适配Transformer的Embedding空间。
5. **DyReLU [6]**：P-DyReLU的理论基础，通过输入依赖的参数生成动态激活函数，本文将其扩展至像素级并引入点击引导。
6. **FocalClick [5]**：早期基于CNN的交互式分割SOTA，在MVTec/COD10K等挑战性数据集上表现优异，但推理速度无法与ViT-based方法相比。

## 局限性与未来方向
1. **Refiner仅在特定交互步激活**：当前设置K=2，即仅在第二次点击后激活Refiner，未探索多步动态激活策略。
2. **窗口大小固定为16**：Dwin-MSA的窗口尺寸未做自适应研究，对不同尺度目标可能非最优。
3. **仅支持点击提示**：虽然SAM本身支持Box/Mask提示，但本文未扩展Refiner至其他提示类型。
4. **训练依赖COCO+LVIS**：与SAM原生的SA-1B训练数据不同，可能限制在极长尾域上的泛化能力。

## 研究启发与可借鉴点
1. **Refiner式设计思路可迁移**：在冻结Backbone的前提下，通过轻量Refiner模块在交互过程中动态调整中间表示，这一思路可推广至其他Foundation Model的下游适配（如Grounding、Detection）。
2. **窗口动态选择策略**：Dwin-MSA的"基于预测掩码动态选窗"思想可直接迁移至目标检测、实例分割等任务中的RoI动态分配问题。
3. **点击引导的Pixel-wise激活**：P-DyReLU将交互式语义（点击）融入激活函数设计，为"如何利用用户反馈动态调制网络行为"提供了新思路。
4. **两阶段训练策略**：先微调Decoder再训练Refiner且冻结Decoder的做法，有效解决了Refiner依赖Decoder输出的训练不稳定问题，可作为Foundation Model微调的通用范式参考。
5. **复杂度与性能解耦设计**：FocSAM证明了通过巧妙利用已有Embedding而非重算，可在几乎不增加计算开销的前提下显著提升模型性能，对Efficient AI研究有参考价值。

## 关键术语表
**FocSAM**：本文提出的交互式分割方法，通过Focus Refiner动态聚焦目标区域并深度融合初始点击信息。
**Dwin-MSA（Dynamic Window Multi-head Self-Attention）**：基于预测掩码动态选择窗口的自注意力模块，仅对目标相关区域进行计算。
**P-DyReLU（Pixel-wise Dynamic ReLU）**：利用点击融合Query Embedding生成的像素级动态激活函数，实现目标增强与背景抑制。
**NoC@90**：Number of Clicks at 90% IoU，衡量达到90%分割精度所需的平均点击次数。
**SPC（Seconds Per Click）**：CPU上每次点击的平均推理耗时，衡量交互式分割的实时性。
**Focus Refiner**：FocSAM新增的核心模块，在交互过程中动态精炼图像Embedding。
**Zoom-in Strategy**：传统交互式分割中将目标区域放大以获取更高精度的一种策略。
**Interactive Segmentation**：通过用户点击/框选等提示逐步引导模型完成图像分割的任务形式。

## 可复现要素
- **训练数据集**：COCO + LVIS（公开）
- **测试数据集**：GrabCut、Berkeley、SBD、DAVIS、MVTec、COD10K（均公开）
- **代码开源**：https://github.com/YouHuang67/focsam
- **权重**：基于SAM ViT-Huge预训练权重（公开），Refiner部分需自行训练
- **关键超参**：
  - Refiner Block数：12（6 Plain + 6 Shift交替）
  - 窗口大小S：16
  - Embedding维度：256
  - 激活步K：2（第二次点击后启动Refiner）
  - 训练阶段：Decoder 320k步 + Refiner 160k步，Batch Size=16
  - 优化器/学习率：论文未提及详细值，见补充材料
