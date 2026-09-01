---
title: "FocSAM-Delving-Deeply-into-Focused-Objects-in-Segmenting-Any"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_FocSAM_Delving_Deeply_into_Focused_Objects_in_Segmenting_Anything_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:58:49"
field: "交互式图像分割"
keywords: ["interactive segmentation", "SAM", "FocSAM", "dynamic window attention", "pixel-wise dynamic ReLU", "object-focused refinement"]
innovations: ["提出聚焦细化器(Dwin-MSA+P-DyReLU)，在SAM流水线中动态增强目标相关图像嵌入", "Dwin-MSA按预测mask动态选取窗口计算自注意力，避免背景冗余", "P-DyReLU用query-patch相似度逐像素生成激活参数实现特征自适应选择"]
benchmarks: ["Grab-Cut", "Berkeley", "SBD", "DAVIS", "MVTec", "COD10K"]
---

# 论文速读：FocSAM-Delving-Deeply-into-Focused-Objects-in-Segmenting-Any

## 一句话总结
论文针对 SAM 交互式分割在难样本上稳定性差的问题，在 SAM 流水线基础上引入一个**聚焦细化器（focus refiner）**，通过动态窗口自注意力（Dwin-MSA）和像素级动态 ReLU（P-DyReLU）逐次迭代调整图像嵌入，使其持续聚焦目标对象；最终在多个基准上追平 SOTA SimpleClick 的分割质量，而 CPU 推理时间仅为 SimpleClick 的约 **5.6%**。

## 研究问题与动机
1. **SAM 不支持交互中动态"放大聚焦"**：预处理阶段一次性生成全局图像嵌入，无法在交互过程中像图像级 zoom-in [46] 那样动态裁剪并重新聚焦目标区域。
2. **轻量解码器难以深度融合少量关键点击信息**：为追求实时性，SAM 解码器只通过简单 cross-attention 融合点击，导致早期有效点击对最终 mask 的影响被稀释。
3. **难样本上性能极不稳定**：如图 1 所示，在第 9 次点击后 IoU=88.59，但再点一次反而骤降至 12.78——这种"越改越差"现象严重制约实际可用性。
4. **现有强方法计算代价过高**：SimpleClick 等全模型逐次推理的方法虽精度高，但 CPU 上每点击需数秒，难以用于在线标注场景。

## 核心贡献（创新点）
1. **提出聚焦细化器（focus refiner）架构**：在 SAM 流水线中插入一个"仅对目标相关嵌入做微调"的额外模块，每对象交互一次即替换原图像嵌入；与 SAM 原流水线的本质区别在于**嵌入可随交互动态更新而非一次性固定**。
2. **动态窗口多头自注意力（Dwin-MSA）**：基于前次预测 mask 的包围盒动态选取相交窗口，只在目标区域内计算自注意力，避免背景冗余计算；与 Swin 固定窗口注意力（[38]）的本质区别在于**窗口位置由 mask 决定且随交互迭代变化**。
3. **像素级动态 ReLU（P-DyReLU）**：利用 SAM 解码器输出的 click-fused query 嵌入与 patch 嵌入的点积相似度逐像素生成激活系数，增强目标相关嵌入并抑制无关嵌入；与原始 DyReLU（[6]）本质区别在于**参数 θ(x) 由交互信号（query）驱动而非纯输入驱动**。
4. **端到端验证在 6 个数据集上的 SOTA 性价比**：NoC@90 追平 SimpleClick-ViT-H，同时 SPC 降至 5.6%，本质上是**用极小计算代价换取大幅稳定性提升**。

## 方法详解
### 整体流程
在 SAM 原始流水线之上，FocSAM 新增一个**focus refiner**，每次针对一个对象的第 K 次交互（K≥2）时被触发一次：
- 输入：当前图像嵌入 $F \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times 256}$、上一轮预测 mask $M^{(K-1)}$、click-fused query 嵌入 $q_c^{(K-1)}$。
- 输出：细化后的图像嵌入 $F_r^{(K)}$，替换后续交互中的 $F$。

Refiner 由 **12 个 refine block** 堆叠（6 plain + 6 shift 交替），所有 block 共享同一 mask $M^{(K-1)}$。

### Dwin-MSA（动态窗口自注意力模块）
1. **窗口划分**：将 $F$ 划分为 $S \times S$ 窗口（论文取 $S=16$），得到 $\bar{F} \in \mathbb{R}^{L \times S \times S \times 256}$。
2. **动态选窗**：由 $M^{(K-1)}$ 得到包围盒，选取所有与盒相交的窗口，得到 $F_W \in \mathbb{R}^{M \times S \times S \times 256}$；未选中窗口冻结不变。
3. **Shift 策略**：plain/shift 交替实现远距离 patch-to-patch 信息交换（借鉴 [38,39]）。
4. **MSA 内部计算**：
   - Query 与窗口内 patch 做 cross-attention：$q_f = \text{Attn}(q_c, f, f)$。
   - 窗口内 self-attention：$\hat{f} = \text{Attn}(f, f, f)$。
   - P-DyReLU 激活：$\hat{f}_q = \text{P-DyReLU}(\hat{f}; q_f)$。
   - 残差输出：$f_q = f + \text{DeformConv}(f + \hat{f}_q)$，$q_q = q_f$。

### P-DyReLU（像素级动态 ReLU）
参考 SAM 解码器中 query·image 点积的思想，构造逐像素激活参数：
$$
a^0 = b^0 = \text{Expand}(\hat{f} \cdot q_f^\top), \quad
a^1 = b^1 = \text{Expand}(\text{AvgPool}(\hat{f}))
$$
其中 $\hat{f} \cdot q_f^\top$ 本质是**每个 patch 与 query 的未归一化相似度**（越高说明该 patch 属于目标的可能性越大）。再经 channel-wise MLP 变换尺度与偏置：
$$
\bar{a}^0 = \text{MLP}(a^0; W_a^0), \ldots
$$
最终激活：
$$
\text{P-DyReLU}(\hat{f}; q_f) = \max\{\bar{a}^0 \odot \hat{f} + \bar{b}^0,\; \bar{a}^1 \odot \hat{f} + \bar{b}^1\}
$$
**物理意义**：高相似度的 patch 得到较强正增益，低相似度 patch 被抑制，实现自适应特征选择。

### 训练策略
- **损失函数**：normalized focal loss（NFL，来自 RITM [29]）+ 辅助 point loss（PTL，来自 BRS [25]）：
  $$\text{PTL} = \sum_i (M(x_i,y_i) - z_i)^2$$
- **两阶段训练**：① 冻结编码器，微调 SAM 解码器 320k iter（batch=16）；② 冻结解码器，训练 refiner 160k iter（同 batch）。
- **预提取嵌入**：用 SAM 图像编码器预先算好 COCO+LVIS 所有图像的嵌入并缓存，避免训练时重复编码。
- **点击模拟**：沿 InterFormer [23] 的 click simulation 策略生成训练样本。

## 实验与结果
### 数据集
训练：COCO [33] + LVIS [16]；评测（zero-shot）：Grab-Cut [45]、Berkeley [26]、SBD [19]、DAVIS [43]、MVTec [2]、COD10K [10]。

### 评估指标
- **NoC@90**：达到 IoU≥90% 平均所需点击数（上限 20 次）。
- **SPC（Seconds Per Click）**：CPU 上每次点击平均推理时间（含 refiner，decoder-only 时间另括号标注）。

### 主要结果（Table 1）
| 方法 | GrabCut | Berkeley | SBD | DAVIS | MVTec | COD10K | **Mean NoC@90** | **SPC (s)** |
|---|---|---|---|---|---|---|---|---|
| SimpleClick-ViT-H | 1.50 | 1.75 | 4.70 | 4.78 | 10.56 | 9.13 | 5.40 | 6.99 |
| SAM-ViT-H | 1.88 | 2.09 | 7.62 | 5.19 | 13.97 | 10.36 | 6.85 | 0.35 |
| **FocSAM-ViT-H (Ours)** | **1.32** | **1.47** | **4.69** | **4.77** | 11.14 | **8.91** | **5.38** | **0.39** |

- **FocSAM 在 6 个数据集中 5 个取得最佳 NoC@90**，均值 5.38 略优于 SimpleClick-ViT-H 的 5.40。
- **SPC=0.39s**，仅为 SimpleClick-ViT-H（6.99s）的 **5.6%**；若图像含 >10 个对象则进一步降至约 1.2%。
- MVTec 略逊于 SimpleClick（11.14 vs 10.56），但综合均值仍最优。

### 稳定性分析（Figure 4）
在 SBD/MVTec/COD10K 上模拟 100 次点击，统计连续点击间 ΔIoU（过滤掉劣化 <1% 的样本）。FocSAM 的 ΔIoU 分布整体右移，表明**额外点击造成性能骤降的概率显著低于 SAM**。

### Ablation（Table 2）
| Dwin-MSA | P-DyReLU | SBD NoC@90 | MVTec NoC@90 | COD10K NoC@90 |
|---|---|---|---|---|
| ✗ | ✗ | 7.62 | 13.97 | 10.36 |
| √ | ✗ | 4.75 | 11.29 | 9.26 |
| ✗ | √ | 4.76 | 11.48 | 9.33 |
| **√** | **√** | **4.69** | **11.14** | **8.91** |

- 两模块单独贡献相近（互补），组合后在难样本（100 次点击 NoC@95 指标）上提升更明显。

## 相关工作脉络
1. **f-BRS [46]**：早期反向传播细化方法，支持图像级 zoom-in，但依赖全图重推理；FocSAM 在**嵌入空间**做轻量聚焦，无需重跑编码器。
2. **RITM [29]**：引入 mask guidance 迭代训练；FocSAM 同样使用 NFL 损失，但增加了**动态嵌入聚焦**机制。
3. **SimpleClick [36]**：首个将 ViT 引入交互式分割的工作；FocSAM 保留 SAM 的编码器-解码器分离范式，用 refiner 弥补其聚焦能力不足，**以 5.6% 推理时间追平其精度**。
4. **InterFormer [23]**：提出特征复用减少冗余；FocSAM 沿袭其训练稳定策略（click simulation、两阶段训练），但额外引入**对象级动态聚焦**。
5. **Swin Transformer [38,39]**：固定窗口+shift 注意力；FocSAM 的 Dwin 将窗口改为**由 mask 动态确定**，实现目标区域自适应聚焦。
6. **DyReLU [6]**：输入依赖的动态激活；FocSAM 的 P-DyReLU 将驱动信号从输入扩展到**跨模态 query（点击信息）**，实现交互式特征筛选。

## 局限性与未来方向
1. **Refiner 仅在 SAM 流水线上验证**，未推广至 InterFormer 或其他 base model。
2. **仅支持 click 提示**，对 bounding box / coarse mask 等其他 prompt 形式的泛化未验证。
3. **需要冻结编码器重新训练解码器和 refiner**，工程部署时需额外 fine-tuning 步骤。
4. **Window size=16 固定**，对不同分辨率/不同尺寸目标可能非最优。
5. **未来方向**：① 扩展到多 prompt 类型（box、mask、文本）；② 设计更轻量的 refiner 用于移动端；③ 与 SAM-HQ、SAM2 等后续变体结合；④ 探索无监督/半监督下的动态聚焦。

## 研究启发与可借鉴点
1. **"预提取嵌入 + 轻量在线细化"范式**：先在离线阶段把重计算（ViT 编码）结果缓存，交互时只在该嵌入上做低成本修正——这一范式可迁移到任何基于 Foundation Model 的实时交互任务。
2. **Dwin 的动态窗口选择思想**：将"由预测结果决定计算区域"推广到其它视觉任务（如检测、跟踪）中，可实现**按任务难度自适应计算预算分配**。
3. **P-DyReLU 的 cross-modal 激活机制**：用 query embedding 驱动像素级激活参数，这一设计可复用到任何需要"条件化特征选择"的模块（如多模态融合、条件生成）。
4. **两阶段冻结训练策略**：先训 decoder 再训 refiner，解决 refiner 对 decoder 输出的梯度依赖问题——此策略可复用于任何引入额外模块但需保持 backbone 稳定的场景。
5. **稳定性指标 ΔIoU 分析**：论文首次系统性地用连续点击的 IoU 变化分布来刻画交互分割稳定性，这一评估视角值得后续工作广泛采用。

## 关键术语表
- **Interactive Segmentation**：通过少量人工提示（点击、框等）引导模型逐步精化分割 mask 的交互任务。
- **FocSAM**：本文提出的在 SAM 流水线上加入聚焦细化器的交互式分割方法。
- **Dwin-MSA（Dynamic Window Multi-head Self-Attention）**：根据预测 mask 动态选取相交窗口并在这些窗口内计算自注意力的高效注意力机制。
- **P-DyReLU（Pixel-wise Dynamic ReLU）**：利用 query 与 patch 的相似度逐像素生成 piecewise-linear 激活参数，实现目标相关嵌入增强与非目标嵌入抑制的激活函数。
- **Focus Refiner**：每次交互时对被选中的目标相关嵌入进行细化的模块栈，用更新后的嵌入替换原 SAM 图像嵌入。
- **NoC@90**：达到 IoU≥90% 所需的平均最少点击次数，衡量交互效率的核心指标。
- **SPC（Seconds Per Click）**：CPU 上每次点击的平均推理耗时，衡量实时性的核心指标。
- **Click Simulation**：训练时通过模拟点击位置（错误预测区域中心）生成训练样本的策略。

## 可复现要素
- **数据集**：训练 COCO [33] + LVIS [16]；测试 Grab-Cut、Berkeley、SBD、DAVIS、MVTec、COD10K（均为公开数据集）。
- **代码**：论文声明已开源 —— https://github.com/YouHuang67/focsam。
- **权重**：使用 SAM 官方预训练 ViT-Huge 作为 backbone，refiner 与 decoder 为训练所得（见 GitHub）。
- **关键超参**：
  - Image encoder 输入尺寸：1024×1024
  - Embedding 维度：256
  - Window size $S$：16
  - Refine blocks：12（6 plain + 6 shift）
  - Refine step $K$：2（第 2 次点击后启动 refiner）
  - 训练阶段①：解码器微调 320k iter，batch=16
  - 训练阶段②：refiner 训练 160k iter，batch=16
  - GPU：4× NVIDIA RTX 3090；CPU：双 Intel Xeon Silver
- **损失**：NFL（RITM）+ PTL（BRS）
- **优化器/学习率**：论文未在正文中明确提及，详见 supplementary materials。
