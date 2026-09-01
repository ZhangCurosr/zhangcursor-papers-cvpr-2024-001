---
title: "KeyPoint-Relative-Position-Encoding-for-Face-Recognition"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:39:38"
field: "人脸识别与几何鲁棒表示"
keywords: ["Keypoint RPE", "Vision Transformer", "Face Recognition", "Relative Position Encoding", "Affine Robustness", "Low-Quality Recognition"]
innovations: ["提出 Keypoint RPE，将关键点位置动态调制 ViT 的 relative position attention bias，显著提升对未见过仿射变换的鲁棒性", "设计 Absolute/Relative/Multihead 三种 f(P) 变体，其中 Relative f(P) 使 query-key 关系取决于 query 与关键点的相对位置，Unaligned IJB-C TAR 从 14.91% 跃升至 85.22%", "KP-RPE 计算开销仅为 iRPE 的约 1/10（+0.02~0.07 GFLOP），并成功推广至步态识别（Gait3D）"]
benchmarks: ["TinyFace", "IJB-S", "CFPFP", "IJB-C", "AgeDB", "Gait3D"]
---

# 论文速读：KeyPoint-Relative-Position-Encoding-for-Face-Recognition

## 一句话总结
本文提出 **KP-RPE（Keypoint Relative Position Encoding）**，将关键点（如面部 landmarks）信息融入 Vision Transformer 的相对位置编码（RPE）中，使模型的空间关系建模从"仅依赖像素距离"扩展为"依赖像素到关键点的相对位置"，从而显著提升 ViT 对尺度、平移、姿态等仿射变换的鲁棒性。在人脸（TinyFace、IJB-S）和步态（Gait3D）识别上均验证了有效性。

## 研究问题与动机
1. **对齐失败场景下的性能退化**：主流人脸识别模型高度依赖面部对齐（RetinaFace 等），低质量/低分辨率图像中 landmarks 检测不准会导致对齐失败，传统 ViT 使用 Absolute PE 时性能骤降（如 MNIST→AffNIST，Accuracy 从 98.1% 降至 77.27%）。
2. **已有 RPE 的局限性**：经典 RPE/iRPE 仅以像素间距离/方向作为注意力偏置，不感知图像的语义内容；即使图像发生缩放、平移，key-query 关系依然静态不变，无法适应仿射变换。
3. **低质量/非对齐数据集的评估空白**：现有低质量识别方法（如 AdaFace、A-SKD）主要面向质量自适应损失或蒸馏，未从根本上解决输入对齐失败的几何鲁棒性问题。

## 核心贡献（创新点）
1. **洞察 RPE 可提升 ViT 对未见过仿射变换的鲁棒性**：通过 MNIST/AffNIST toy example 证明，在 ViT 中加入 RPE 即能部分缓解对齐失效带来的性能下降。
2. **提出 KP-RPE**：首次将关键点引入 RPE，使 attention bias $\mathbf{B}_{ij}$ 成为"关键点感知的距离偏置"，让 query-key 关系随图像中关键点位置动态调整，与 iRPE（纯几何/无关键点依赖）形成本质区别。
3. **三种 $f(\mathbf{P})$ 变体设计**：从 Absolute、Relative 到 Multihead Relative 逐步增强，Relative $f(\mathbf{P})$ 使每个 query 的偏置由其与关键点的相对位置决定；Multihead 版本进一步区分各 attention head，带来最优性能。
4. **通用性与高效性**：在人脸（对齐、未对齐、低质量）和步态（Gait3D，使用 17 关节骨架关键点）两类任务上均验证；计算开销仅增加约 0.02–0.07 GFLOP，吞吐量下降 <17%，约为 iRPE 效率的 10 倍。

## 方法详解
- **背景**：ViT self-attention 中，$e_{ij}' = \frac{(\mathbf{Q}_i + \mathbf{R}_{ij}^Q)(\mathbf{K}_j + \mathbf{R}_{ij}^K)^T}{\sqrt{d_k}}$，RPE 通过可学习表 $\mathbf{R}[d(i,j)]$ 注入距离编码。
- **KP-RPE 核心公式**：
  $$e_{ij}' = \frac{\mathbf{Q}_i \mathbf{K}_j^T + \mathbf{B}_{ij}}{\sqrt{d_k}}, \quad \mathbf{B}_{ij} = f(\mathbf{P})[d(i,j)] \in \mathbb{R}^1$$
  其中 $d(i,j)$ 沿用 iRPE 的 quantized x-y product distance（编码方向信息），$f(\mathbf{P})$ 是由关键点 $\mathbf{P} \in \mathbb{R}^{N_L \times 2}$ 决定的注意力偏置表。
- **Relative $f(\mathbf{P})$ 构建**：
  1. 生成 patch 位置网格 $\mathbf{M} \in \mathbb{R}^{N \times 2}$；
  2. 广播计算与关键点的差值：$\mathbf{D} = \mathrm{Expand}(\mathbf{M}, 1) - \mathrm{Expand}(\mathbf{P}, 0)$，形状 $\mathbb{R}^{N \times N_L \times 2}$；
  3. 线性映射：$f(\mathbf{P}) = \mathbf{D}' \mathbf{W}_L \in \mathbb{R}^{N \times K}$，其中 $\mathbf{D}' \in \mathbb{R}^{N \times (2N_L)}$；
  4. 最终 $\mathbf{B}_{ij} = f(\mathbf{P})[i, d(i,j)]$，即对每个 query $i$，按距离 $d(i,j)$ 查表得到偏置。
- **Multihead 扩展**：$\mathbf{W}_L \in \mathbb{R}^{(2N_L) \times N_d \cdot H \cdot K}$，区分 transformer depth $N_d$ 和各 head $H$，推理时 $f(\mathbf{P})$ 仅计算一次，开销极低。
- **关键点定义**：人脸用 5 个 facial landmarks（两眼、鼻尖、两口角）；步态用 17 个关节点；由 MobileNetV2-Backbone RetinaFace 实时预测，全程可微分（Differentiable Face Aligner）。

## 实验与结果
- **数据集**：训练 MS1MV2/MS1MV3/WebFace4M/WebFace12M；评估覆盖三类——高质量对齐（CFPFP、AgeDB、IJB-C）、故意未对齐（Raw CFPFP/IJB-C）、低质量（TinyFace、IJB-S）；步态使用 Gait3D。
- **消融 Tab.1（ViT-small）**：ViT+KP-RPE 在 Unaligned CFPFP Verification 达 **93.56%**（vs ViT 72.81%、iRPE 77.91%）；TinyFace Rank-1 **69.88%**；IJB-S Rank-1 **63.44%**，且 Aligned AgeDB/IJB-C 性能不降反升（TAR@0.01% 从 92.22%→94.20%）。
- **变体消融 Tab.2**：Relative $f(\mathbf{P})$ 较 Absolute 在 Unaligned 大幅提升（IJB-C TAR 从 14.91%→85.22%）；Multihead 再获小幅增益。
- **SoTA 对比 Tab.4（ViT-Base + AdaFace loss）**：WebFace4M 上 TinyFace Rank-1 **75.80%** / IJB-S Rank-1 **72.78%**，分别超越 AdaFace+ViT+iRPE（74.92%/71.93%）；WebFace12M 进一步提升至 **76.18%** / **72.94%**。
- **步态 Tab.5（Gait3D + SwinGait-2D）**：Rank-1 从 67.1%→**68.2%**，mAP 从 58.76→**60.81**。
- **最强结果**：WebFace12M + ViT+KP-RPE 在 TinyFace 达 **76.18% Rank-1**；Unaligned CFPFP Verification 达 **93.56%**。

## 相关工作脉络
1. **RPE 系列（Shaw et al. 2018 / iRPE Wu et al. 2021）**：iRPE 使用 product distance 编码方向，但未引入图像内容/关键点；KP-RPE 在此基础上使偏置表受关键点位置调制。
2. **低质量人脸识别（AdaFace Huang et al. 2022 / A-SKD）**：通过质量自适应 loss 削弱难样本影响，但未解决对齐失败时的几何鲁棒性；KP-RPE 从表示学习层面补充此能力。
3. **ViT 位置编码（Abs-PE / Standalone SA）**：Abs-PE 对仿射变换完全敏感；KP-RPE 证明在 ViT 中引入关键点条件化 RPE 是更优路径。
4. **关键点辅助视觉任务（Keypoint-based pooling / Pose estimation）**：此前关键点主要用于检测/骨架任务，KP-RPE 首次将其嵌入 Transformer self-attention 偏置中。
5. **非对齐人脸基准（TinyFace / IJB-S）**：本文系统性对比了多种 SoTA 在这些"对齐易失败"数据集上的表现，凸显 KP-RPE 的实用价值。
6. **步态识别（SwinGait-2D / Gait3D）**：验证 KP-RPE 可迁移至任意具有拓扑结构的关键点序列（17 关节骨架），体现方法通用性。

## 局限性与未来方向
1. **依赖外部关键点检测器**：需额外运行 landmark 预测（如 RetinaFace-MobileNet），在关键点多义或缺失（非固定拓扑目标）场景受限。
2. **仅适用于有固定拓扑的对象**：人脸/人体骨架等对象适用，但对通用物体识别、无明确关键点任务推广性未知。
3. **未来方向**：作者建议探索 keypoint 的**自发现机制**（self-discovery），减少对预训练检测器的依赖，进一步提升灵活性与泛化范围。

## 研究启发与可借鉴点
1. **"关键点条件化 RPE"可迁移至其他视觉 Transformer**：凡具有稳定拓扑先验的任务（人体姿态、手部识别、器官分割）均可借鉴此设计，将几何先验注入 attention。
2. **低成本增强仿射鲁棒性的范式**：不重新训练大模型，仅通过替换/增强 PE 模块即可显著提升对未见过仿射变换的泛化，适合工业部署中对齐不稳定场景。
3. **消融设计值得学习**：Absolute → Relative → Multihead 的渐进消融清晰展示了"让 RPE 感知关键点相对位置"的核心价值，Rel 变体带来最大增益（Unaligned IJB-C TAR 从 14.91%→85.22%），是后续研究者判断有效性的参考基线。
4. **可结合团队方向**：若团队关注低资源对齐/无监督对齐场景，KP-RPE 可作为即插即用的位置编码模块嵌入现有 ViT backbone；其 Multihead 版本开销极低（<0.1M 参数），适合端侧部署优化。

## 关键术语表
- **KP-RPE（Keypoint Relative Position Encoding）**：将图像关键点位置作为条件，动态调制 ViT self-attention 中相对位置偏置表，使 query-key 关系感知语义关键点的新位置编码方式。
- **RPE（Relative Position Encoding）**：用可学习表替代绝对坐标，以 query-key 距离（或方向距离）作为 attention 偏置的位置编码方法。
- **iRPE（Improvised RPE）**：引入 quantized x-y product distance 编码方向信息的 RPE 变体，被 KP-RPE 作为距离度量基础。
- **Differentiable Face Aligner**：基于 MobileNetV2-Backbone RetinaFace 的可微分关键点预测器，支持端到端梯度传播。
- **AdaFace**：质量自适应 margin loss，降低低质量/难样本对训练的影响，本文统一使用该 loss 进行公平对比。
- **PartialFC**：用于大规模分类器训练的采样加速技术，WebFace4M 实验中用于降低显存开销。
- **TAR@FAR=0.01%**：在固定假阳性率 0.01% 下的真阳性率，IJB-C 等 benchmark 的标准评估指标。
- **AffNIST**：对 MNIST 施加未见仿射变换的 toy benchmark，用于直观演示不同位置编码的鲁棒性差异。

## 可复现要素
- **数据集**：MS1MV2/MS1MV3、WebFace4M、WebFace12M、CFPFP、AgeDB、IJB-C、IJB-S、TinyFace、Gait3D 均为公开数据集（部分需申请）； keypoints 使用公开 RetinaFace。
- **代码**：论文声明"Code and pre-trained models are available"，但具体仓库地址未在正文给出，需在论文 supplement 或作者主页查找。
- **关键超参**：输入分辨率 112×112；AdamW 优化器；Cosine LR scheduler；训练 loss 为 AdaFace；ViT-small 用于消融，ViT-Base 用于 SoTA 对比；WebFace4M 上采用 PartialFC。
