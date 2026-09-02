---
title: "Prompt-Driven Referring Image Segmentation with Instance Contrasting"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Shang_Prompt-Driven_Referring_Image_Segmentation_with_Instance_Contrasting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:17:01"
field: "视觉-语言理解与分割"
keywords: ["Referring Image Segmentation", "Prompt Learning", "CLIP", "SAM", "Cross-Modal Prompting", "Instance Contrastive Learning", "Vision-Language"]
innovations: ["提出 CMP 双向跨模态提示实现 CLIP 到像素级任务的适配", "设计 ICL 实例对比学习提升多实例判别与多描述鲁棒性", "通过 prompt learning 端到端桥接 CLIP 与 SAM 生成三类 prompt"]
benchmarks: ["RefCOCO", "RefCOCO+", "G-Ref"]
---

# 论文速读：Prompt-Driven Referring Image Segmentation with Instance Contrasting

## 一句话总结
提出 **Prompt-RIS** 框架，通过 prompt learning 将 CLIP 与 SAM 端到端桥接，利用 Cross-Modal Prompting（CMP）实现双向跨模态交互以适配像素级任务，并引入 Instance Contrastive Learning（ICL）提升实例判别力，在 RIS 三大基准上均超越 SOTA。

---

## 研究问题与动机

1. **CLIP 缺乏像素级能力**：CLIP 预训练于图像级对比学习任务，无法直接进行细粒度文本-像素对齐，难以适配 RIS 这类像素级任务。
2. **单向 prompt 交互不足**：现有 prompt learning 方法多为单向信息流（如 V2L 或 L2V），无法充分实现视觉与语言的跨模态交互，不利于捕捉文本与图像的细粒度对应关系。
3. **"mask-classification" 两阶段方法存在局限**：现有将 CLIP 适配分割的方法先由 MaskFormer/SAM 生成无类别掩码提议，再用 CLIP 分类前景，这种方式忽略了全局上下文和实例间关系，且无法充分利用语言描述的定位、属性、关系信息。
4. **SAM 缺少位置先验时分割质量差**：直接将 SAM 用于 RIS 时，因缺乏 point prompt 等位置先验，难以生成准确分割结果。

---

## 核心贡献（创新点）

1. **提出 Prompt-RIS 端到端框架**：通过 prompt learning 将 CLIP 和 SAM 桥接在一起，将两者的 rich knowledge 迁移至 RIS 任务，区别于传统两阶段 mask-classification 范式。
2. **提出 Cross-Modal Prompting（CMP）**：在 CLIP 编码的中间层对称地执行 V2L 和 L2V 双向提示，实现充分的跨模态交互和细粒度文本-像素对齐，而非仅在最终层做对比学习。
3. **提出 Instance Contrastive Learning（ICL）**：利用同一图像的多条描述（可能指向不同实例或相同实例）进行实例级对比学习，提升模型对不同实例的判别力和对同实例多样描述的鲁棒性。
4. **CLIP 为 SAM 生成三类 prompt**：将 prompt-tuned CLIP 生成的粗掩码、点提示和文本提示同时输入 SAM，实现跨模型的知识传递，避免了手工标注 point prompt 的成本。

---

## 方法详解

**整体架构**：Prompt-RIS 由 CLIP（ViT-B/16）和 SAM（ViT-B/16）组成。图像分别以 480×480（CLIP）和 1024×1024（SAM）输入， patch 数 $N_C = 30 \times 30$，$N_S = 64 \times 64$。

**Cross-Modal Prompting（CMP）**：在 CLIP 图像/文本编码器的每一中间层，引入可学习 token $T_V^i$ 和 $T_L^i$（数量 $n=16 \ll N_C$），通过 cross-attention 分别生成 V2L prompt $T_{V2L}^i$ 和 L2V prompt $T_{L2V}^i$，与另一模态特征拼接后输入下一层编码器：

$$
T_{V2L}^i = \text{CrossAttn}(T_V^i, F_{I_C}^i), \quad [ \_ ; F_T^{i+1} ] = \mathbf{E}_{T_{clip}}^{i+1}([T_{V2L}^i ; F_T^i])
$$
$$
T_{L2V}^i = \text{CrossAttn}(T_L^i, F_T^i), \quad [ \ldots ; c^{i+1} ; F_{I_C}^{i+1} ] = \mathbf{E}_{I_{clip}}^{i+1}([T_{L2V}^i ; c^i ; F_{I_C}^i])
$$

**CLIP-based Prompting**：编码结束后，通过 cross-attention 小解码器更新特征 $F_{I_C}' = \text{CrossAttn}(F_{I_C}, F_T)$，$F_T' = \text{CrossAttn}(F_T, F_{I_C})$。取 $F_T'$ 中 [EOS] 位置作为全局文本特征 $t_g$，计算 text-pixel response map $S_C' = F_{I_C}' \cdot t_g$，reshape 后得到低分辨率粗掩码。

**三类 prompt 生成**：
- **Mask prompt**：将 $S_C'$ 上采样并投影到 SAM prompt encoder，得到 $P_{mask}$。
- **Point prompt**：对 $S_C$ 做 Gumbel-softmax（straight-through estimator 保证可微），得到 hard one-hot point map $M$，与预计算的 SAM 位置嵌入 $E$ 加权求和得 $e_p$；推理时直接取响应最高位置。
- **Text prompt**：以 CLIP 文本特征 $F_T$ 为 query，SAM 图像特征为 key/value，经两层 self-/cross-attention transformer 投影为 $P_{text}$。

**SAM Decoding**：$S' = \mathbf{D}_{sam}([T_{iou}; T_{out}; P_{point}; P_{text}], (P_{mask} + F_{I_S}))$，再 $\times 4$ 上采样得最终预测 $S$。

**Instance Contrastive Learning（ICL）**：对同一图像采样 $b$ 条描述，得到 $b$ 个掩码预测，用 Dice 系数计算 overlap $O_{ij}$，构建对比损失：

$$
L_{icl} = \frac{1}{b^2} \sum_{i,j} -w_i(Y_{ij} \log(O_{ij}) + (1-Y_{ij}) \log(1-O_{ij}))
$$

其中 $Y_{ij}$ 指示是否描述同一实例，$w_i$ 为第 $i$ 个掩码与 GT 的 IoU（抑制错误预测的误导）。总损失：

$$
\mathcal{L} = \mathcal{L}_{clip\_seg} + \mathcal{L}_{sam\_seg} + \mathcal{L}_{icl}
$$

（$\mathcal{L}_{seg}$ 为 binary cross-entropy + Dice loss）

**训练策略**：前 20 epoch 单独训练 CLIP（CMP + ICL）生成粗掩码，后 30 epoch 端到端联合训练；三类 prompt 随机 dropout 以减少依赖。

---

## 实验与结果

**数据集**：RefCOCO、RefCOCO+、G-Ref（均基于 MS-COCO），含 19,994/19,992/26,711 张图像。

**评估指标**：oIoU、mIoU、P@X（X ∈ {0.5, 0.7, 0.9}）。

**主要结果（Table 1）**：

| 数据集 | 指标 | Ours | 上一 SOTA | 提升 |
|--------|------|------|-----------|------|
| RefCOCO val | oIoU | **76.36** | 74.75 (CGFormer) | +1.61 |
| RefCOCO testA | oIoU | **80.37** | 77.30 | +3.07 |
| RefCOCO+ val | oIoU | **67.06** | 64.54 | +2.52 |
| G-Ref val(G) | oIoU | **64.79** | 62.51 | +2.28 |
| G-Ref test(U) | oIoU | **69.01** | 65.09 | +3.92 |

mIoU 同样全面领先，RefCOCO val mIoU 达 **78.10**，G-Ref test(U) 达 **71.29**。

**泛化性能（Table 2）**：在 unseen 类别上，Ours 在三个数据集上的 mIoU 均显著优于 CGFormer（平均提升 >2.5 分），证明 CLIP 知识得到充分利用。

**消融实验（Table 3，RefCOCO val）**：
- Baseline（CLIP + cross-modal decoder）：mIoU 67.94
- + CMP：mIoU 73.10（+5.16）
- + SAM：mIoU 77.40（+4.30）
- + ICL：mIoU **78.10**（+0.70；P@0.9 提升 3.32）

**CMP 双向验证（Table 4）**：V2LP +4.50 mIoU，L2VP +1.59，CMP（双向）= +7.16，优于单一方向。

**Prompt 类型消融（Table 5）**：Mask prompt 贡献最大（+1.35 mIoU vs 无 mask），三类 prompt 均有正向贡献且模型不过度依赖单一类型。

**ICL 影响**：w/ ICL 的 ImgAcc=79.34、InstAcc=77.74，w/o ICL 为 77.61/76.02。

---

## 相关工作脉络

1. **CRIS [43] / ETRIS [44]**：CLIP-based RIS 方法，采用两阶段 mask-classification 范式（先 mask 提议再分类），本文通过 prompt 驱动端到端桥接 CLIP 与 SAM，避免分类瓶颈并保留 SAM 的精细分割能力。
2. **CGFormer [41]**：引入 mask classification 到 RIS 的 Transformer 方法，使用 class token 进行 cross-modal reasoning；本文不用 class token，而是通过 CMP 实现更充分的逐层跨模态交互。
3. **LAVT [46] / DMMI [12]**：早期 Transformer-based RIS 方法，直接在编码器阶段做 cross-modal fusion；本文利用大规模预训练模型（CLIP+SAM）而非从头训练编码器。
4. **CoOp / MaPLe / Co-CoOp**：单向 prompt learning（V2L 或 L2V）；本文提出双向 CMP，在 CLIP 每一编码层对称执行 V2L 和 L2V 提示。
5. **SAM [23]**：通用分割大模型，需人工/外部提供 prompt；本文让 CLIP 自动为 SAM 生成 mask/point/text 三类 prompt，实现端到端无人工交互。
6. **CMPC [14] / LSMC [15]**：经典 RIS 方法，构建多模态图或依赖句法解析；本文聚焦利用 VL 大模型预训练知识，走 prompt tuning 路线。

---

## 局限性与未来方向

1. **计算开销**：同时运行 CLIP 和 SAM 两个大模型（尤其 SAM ViT-B），推理成本较高，未讨论轻量级变体。
2. **SAM 参数冻结**：训练时 SAM 完全冻结，无法针对 RIS 任务做自适应微调，可能限制性能上限。
3. **ICL 依赖多描述**：ICL 需要同一图像有多条描述（$b=4$），在描述稀疏的场景下效果受限；若描述不足则需重复采样并随机 mask 部分词。
4. **泛化场景**：仅验证了 open-vocabulary 设置下的未见类别泛化，对域外分布（如非 COCO 风格图像）的鲁棒性未充分讨论。
5. **未来方向**：可探索 SAM 微调策略、引入更轻量的分割解码器、将框架扩展至视频 RIS 或 3D 实例分割等任务。

---

## 研究启发与可借鉴点

1. **双向 cross-modal prompting 机制可迁移**：CMP 在每一编码层对称执行 V2L/L2V 提示的设计，可推广至其他需要细粒度跨模态对齐的任务（如 referring video segmentation、visual grounding）。
2. **Gumbel-softmax + 预计算嵌入实现可微 prompt 生成**：将 mask 转化为 point prompt 时采用 straight-through estimator 技巧，解决了不可微操作阻碍端到端训练的问题，思路可用于其他 sparse prompt 生成场景。
3. **Instance Contrastive Learning 的损失设计具有通用性**：用 Dice 系数度量 mask overlap 并结合 GT IoU 加权（$w_i$）抑制错误预测，可用于其他多实例分割或多描述对齐任务。
4. **三类型 prompt（mask/point/text）联合输入 SAM**：证明了利用上游模型自动生成 SAM 多种 prompt 的可行性，可探索扩展至 box prompt 或更多 prompt 类型。
5. **训练时随机 dropout prompt 类型**：减少解码器对特定 prompt 类型的依赖，提高各 prompt 的有效性，是提升多源 prompt 融合模型鲁棒性的实用技巧。

---

## 关键术语表

**Referring Image Segmentation (RIS)**：根据自然语言描述在图像中精确分割出目标对象的任务，区别于语义分割可处理任意类别、属性和位置描述。

**Cross-Modal Prompting (CMP)**：在 CLIP 编码的每一中间层对称执行 Vision-to-Language 和 Language-to-Vision 双向提示，通过 learnable tokens + cross-attention 实现细粒度跨模态交互。

**Instance Contrastive Learning (ICL)**：对同一图像的多条描述（可能指向相同或不同实例）进行对比学习，拉近同实例描述对应 mask 的相似度、推远不同实例 mask 的重叠。

**Gumbel-Softmax (Straight-Through Estimator)**：用于将离散的 point 位置选择过程近似为可微操作，使 mask 到 point prompt 的转换可在反向传播中梯度流动。

**SAM (Segment Anything Model)**：Meta 提出的通用分割大模型，支持 mask/point/box/text 等多种 prompt 输入，本文冻结其参数仅利用其解码能力。

**oIoU / mIoU**：overall mean Intersection over Union，分别按图像级和样本级聚合计算掩码与 GT 的 IoU，是 RIS 主流评估指标。

**Coarse mask / Fine mask**：CLIP 直接生成的低分辨率初步掩码（coarse），经 SAM 解码器细化后的高分辨率最终掩码（fine）。

**Prompt-driven framework**：指不直接修改预训练模型权重，而是通过引入可学习 prompt tokens 并依赖下游模型（如 SAM）进行适配的轻量级微调范式。

---

## 可复现要素

- **数据集**：RefCOCO、RefCOCO+、G-Ref（均基于 MS-COCO，公开可获取）
- **代码/权重**：论文未提及开源声明
- **关键超参**：
  - 图像编码器：CLIP ViT-B/16 + SAM ViT-B/16
  - 输入分辨率：CLIP 端 480×480，SAM 端 1024×1024
  - CMP prompt token 数 $n = 16$
  - ICL 每图像采样描述数 $b = 4$，batch size $B = 64$（16 图像 × 4 描述）
  - 优化器：AdamW，初始 lr = 1e-4，polynomial decay power = 0.9
  - 训练阶段：前 20 epoch 仅训练 CLIP+CMP+ICL，后 30 epoch 端到端联合训练
  - Prompt dropout：训练时三类 prompt 随机 dropout

---
