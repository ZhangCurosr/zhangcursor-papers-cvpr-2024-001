---
title: "Open-Vocabulary-Segmentation-with-Semantic-Assisted-Calibrat"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_Open-Vocabulary_Segmentation_with_Semantic-Assisted_Calibration_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:40:54"
field: "开放词汇视觉分割"
keywords: ["open-vocabulary segmentation", "vision-language model", "CLIP", "semantic segmentation", "domain adaptation", "cross-modal alignment"]
innovations: ["通过语义注入校准 proposal embedding 防止类别坍缩", "上下文移位策略缓解 CLIP 的域偏移问题", "提出考虑类别语义关系的 SG-IoU 评估指标"]
benchmarks: ["ADE20K-150", "ADE20K-847", "Pascal VOC", "Pascal Context-59", "Pascal Context-459"]
---

# 论文速读：Open-Vocabulary-Segmentation-with-Semantic-Assisted-Calibration

## 一句话总结
论文提出 SCAN（语义辅助校准网络），通过将 CLIP 的全局语义先验融入 proposal embedding 并采用上下文移位策略，缓解开放词汇分割中 proposal embedding 坍缩到已见类别和域偏移问题，在所有主流基准上达到 SOTA。

## 研究问题与动机
1. **Proposal embedding 坍缩**：现有两阶段方法的分割模型 proposal embedding 倾向于拟合训练语义空间，对未见过概念不敏感，导致分类坍缩到 in-vocabulary。
2. **CLIP 的域偏移**：CLIP 作为额外分类器时，输入是经过裁剪和遮罩的子图像，与自然图像分布差异大，导致全局上下文缺失和错误的背景先验。
3. **评估指标缺陷**：现有 mIoU 指标未考虑类别间语义关系，如"chair"和"armchair"在 ADE20K-150 中视为不同类别，但在开放词汇场景下识别为父类别应被认可。

## 核心贡献（创新点）
1. **Semantic Integration Module (SIM)**：将冻结 CLIP 的全局语义先验融入 proposal embedding，防止过拟合已见类别——与直接使用 CLIP 分类不同，本文通过 cross-attention 将语义注入到学习到的 proposal 中。
2. **Contextual Shift (CS) 策略**：在 CLIP 前向传播中随机替换一定比例的背景 patch 为原始图像的 [CLS] embedding，缓解域偏移并补充全局上下文——区别于 OVSeg 的微调策略，本文无需额外训练数据即可适配。
3. **Semantic-Guided IoU (SG-IoU)**：提出新评估指标，将类别间的父子/同义词语义关系纳入 IoU 计算——传统 mIoU 的刚性匹配在此场景下不够合理。
4. **统一校准框架**：同时校准 in-vocabulary 和 out-of-vocabulary 语义空间，在 CLIP ViT-L/14 下 ADE-150 达 33.5%、PC-459 达 16.7%，均为当时 SOTA。

## 方法详解
**整体框架（两阶段）**：
- 使用 Mask2Former 生成 N 个 class-agnostic mask proposals $M_N$ 和对应的 proposal embeddings $F_N \in \mathbb{R}^{N \times C}$
- 结合 SIM 校准后的 proposal embedding 和 Contextual Shift 后的 CLIP 分类结果进行最终匹配

**Semantic Integration Module (SIM)**：
1. 从冻结 CLIP 提取多层空间 token $F_{HW}^i$ 和 [CLS] token $F_{CLS}^i$
2. 对选中的 CLIP 层做低频增强（LFE）：通过 FFT → 高斯滤波 → IFFT 抑制高频纹理噪声
   $$F_s^i = IFFT(RELU(Conv(FFT(F_{HW}^i) * g))) + F_{HW}^i$$
3. 将增强后的特征 $F_s$ 与 proposal embedding $F_N$ 做 cross-attention：
   $$F_N' = MHA(F_N, F_s, F_s)$$
4. 融合最终 [CLS] embedding（对齐到视觉-语言统一空间）：
   $$F_N^{final} = Trans.Enc.(F_N' + \gamma * F_{CLS}^{final})$$

**Contextual Shift (CS)**：
- 在 CLIP 前向传播过程中，对选中的浅层随机替换 $\alpha$% 的背景 patch 为原始图像对应层的 [CLS] embedding
- 更新后的前向过程：
  $$F_m^i = \mathcal{V}^i(F_m^{i-1} | (M_k, F_{CLS}^{i-1}, \alpha)) \text{ for } i \in idx$$
- 后续跟随 OVSeg 策略在 masked image dataset 上微调 CLIP

**训练损失**：分割损失 = Dice + Cross Entropy（权重5），分类损失 = Cross Entropy（权重2）

## 实验与结果
**训练集**：COCO-Stuff（171类）；**微调数据**：OVSeg 提出的 masked images dataset（来自 COCO Captions）

**测试基准**：ADE20K-150、ADE20K-847、Pascal VOC、Pascal Context-59、Pascal Context-459

**主要结果（mIoU）**：

| 方法 | 基座模型 | ADE-150 | ADE-847 | PC-59 | PC-459 | VOC |
|------|----------|---------|---------|-------|--------|-----|
| SAN [50] | CLIP ViT-L/14 | 32.1 | 12.4 | 57.7 | 15.7 | 94.6 |
| **SCAN (Ours)** | **CLIP ViT-L/14** | **33.5** | **14.0** | **59.3** | **16.7** | **97.0** |

- 相比 SAN，ADE-150 提升约 1.4%，PC-459 提升 1.0%
- ablation 表明 SIM (+1.4 ADE-150) 和 CS (+1.3 ADE-150) 均有贡献，组合后达 33.5

**SG-IoU 结果**：SCAN 在 ADE-150 达 34.2，ADE-847 达 14.6，PC-459 达 17.2

## 相关工作脉络
1. **LSeg / OpenSeg**（ALIGN + RN101/EN-B7）：早期两阶段方法，依赖预训练 ALIGN 模型，未使用 CLIP，性能有限。
2. **SimSeg**：使用 CLIP ViT-B/16 + learnable tokens 直接对齐，但未解决域偏移和 embedding 坍缩问题。
3. **OVSeg**：发现遮罩背景影响 CLIP 识别，提出在 masked image dataset 上微调 CLIP——本文在此基础上增加上下文移位，无需额外训练。
4. **SAN**：设计 side-adapter 网络解耦分割与分类，但仍是两阶段范式；SCAN 通过语义注入实现更本质的校准。
5. **MAFT**：学习 mask-aware CLIP 表示，依赖 learnable prompt，易过拟合已见类别。
6. **ODISE**：基于 diffusion model 的方法，计算成本高；SCAN 保持轻量两阶段架构。

## 局限性与未来方向
1. **SIM 依赖选层选择**：空间 token 需从多层（如12,18,24层）获取多粒度信息，最佳层组合需经验调参。
2. **低频增强超参敏感**：截止频率 $\sigma$ 需根据任务调整（文中使用 7,5,3）。
3. **替换比例限制**：CS 策略中替换比例过高（如40%）会损害局部判断，需在上下文补充与局部细节间权衡。
4. **SG-IoU 需手动构建类别关联矩阵**：扩展到大 vocab 数据集时需人工标注语义关系，可扩展性待验证。

## 研究启发与可借鉴点
1. **跨模态先验注入**：将冻结 VL 模型的 [CLS] token 融入 segmentation proposal 的思路可迁移至其他开放词汇视觉任务（检测、查询分割）。
2. **频域滤波抑制噪声**：SIM 中采用 FFT 低通滤波提取 CLIP 空间 token 的高层语义，为多模态特征融合提供了去噪新角度。
3. **上下文移位类比 Domain Adaptation**：CS 通过 patch 级替换实现隐式域适应，可启发其他需要适配预训练 VL 模型的场景。
4. **语义感知的评估指标设计**：SG-IoU 的思路（考虑类别层次关系）可用于重新评估其他开放词汇任务的性能。

## 关键术语表
**Open-Vocabulary Segmentation (OVS)**：给定任意文本查询进行语义分割的任务，模型无需在训练时见过所有类别。
**Semantic Integration Module (SIM)**：将 CLIP 的全局语义先验通过 cross-attention 注入到 proposal embedding，防止 overfitting。
**Contextual Shift (CS)**：在 CLIP 前向中随机替换背景 patch 为全局 [CLS] embedding，缓解域偏移。
**Low-Frequency Enhancement (LFE)**：在频域对 CLIP 空间 token 做高斯滤波，抑制纹理噪声保留高层语义。
**Semantic-Guided IoU (SG-IoU)**：考虑类别间父子/同义关系的 IoU 变体，更适合开放词汇场景评估。
**In-vocabulary collapse**：分割模型 proposal embedding 倾向于只预测训练集中出现的类别，对未见过概念不敏感。
**Domain bias**：CLIP 输入因裁剪/遮罩偏离自然图像分布，导致分类性能下降。

## 可复现要素
- **训练数据集**：COCO-Stuff（公开，171类）
- **微调数据集**：masked images dataset（来自 OVSeg，基于 COCO Captions 构建，非官方开源）
- **代码**：论文声明开源（Code is available here，链接在论文中）
- **权重**：基于 Detectron2 + Swin Transformer-Base + CLIP ViT-L/14（OpenCLIP）
- **关键超参**：batch size=32，120k iter，lr=6e-5，最短边 resize 到 640，LFE kernel=(7,5,3)，CS 替换层=(1,3,5,7,9)，替换比例=30%
