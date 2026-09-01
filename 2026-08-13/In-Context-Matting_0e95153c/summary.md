---
title: "In-Context-Matting"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Guo_In-Context_Matting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:59:41"
field: "图像分割与抠图"
keywords: ["image matting", "in-context learning", "diffusion model", "foreground extraction", "cross-image matching"]
innovations: ["提出in-context matting新任务，单参考图指导批量自动抠图", "基于Stable Diffusion的Inter+Intra双路径相似度匹配机制", "构建ICM-57首个in-context matting评测数据集"]
benchmarks: ["ICM-57", "AIM-500"]
---

# 论文速读：In-Context-Matting

## 一句话总结
本文提出了"in-context matting"新任务，仅需一张参考图像及少量提示（mask/scribble/point），即可自动对同类别的一批目标图像生成高质量alpha matte；基于预训练Stable Diffusion构建的IconMatting模型在Accuracy与Automation之间取得了良好平衡。

## 研究问题与动机
1. **辅助输入型抠图（Auxiliary input-based matting）效率低下**：trimap、scribbles、mask等辅助输入需逐图提供，用户交互成本高，难以规模化应用。
2. **全自动抠图（Automatic matting）泛化性有限**：现有方法多局限于特定类别（人、动物、显著性目标），在自然场景中缺乏对任意感兴趣前景的针对性提取能力。
3. **精度与效率之间存在显著Gap**：现有两类方法分别偏向高精度低自动化、或高自动化低精度，缺乏兼顾两者的统一范式。
4. **核心挑战：跨图前景精准匹配**：给定参考图的前景区域，如何在目标图中准确定位并泛化到相同/同类前景的完整区域（含半透明细节）。

## 核心贡献（创新点）
1. **提出In-Context Matting新任务**：仅用一张参考图+提示即可对批量目标图自动完成抠图，在保持自动化的同时获得类似辅助输入型方法的精度与灵活性。
2. **设计IconMatting模型**：基于Stable Diffusion的特征提取能力，结合inter-similarity（参考-目标跨图匹配）与intra-similarity（目标图内自注意力补充），实现精准的前景区域定位。
3. **构建ICM-57评测数据集**：涵盖57组真实世界图像（同类别/同实例不同视角），为in-context matting任务提供首个系统评测基准。
4. **参考提示扩展机制（Reference-Prompt Extension）**：将point/scribble等稀疏提示通过参考图的自注意力图扩展为更完整的RoI mask，增强in-context query的信息量。

## 方法详解
**整体架构**：IconMatting由三部分构成——Feature Extractor、In-Context Similarity Module、Matting Head。

**Feature Extractor**：
-  backbone选用Stable Diffusion v2-1-v的U-Net（冻结权重，条件输入为空字符串，扩散步t=0）。
-  从第5、8、11层提取多尺度特征{F_l}，从第5层提取inter-features用于跨图匹配；同时保留目标图的自注意力图{A_l}用于intra-similarity。

**In-Context Similarity（核心模块）**：
- **Inter-similarity**：从参考图RoI区域提取in-context query {Q_k}（Q_k = R(k)，R = F_inter^r ⊙ M_RoI），计算与目标图inter-features的余弦相似度S_k = softmax(Q_k · F_inter^t / √d)，取平均得S，得到跨图相似度图（稀疏但精准）。
- **Intra-similarity**：利用目标图自注意力图填补S中的空洞——S'_l = A_l ⊙ S，通过注意力传播将稀疏匹配扩展至完整前景区域。

**Matting Head**：
- 融合多尺度in-context特征与原始图像细节，采用ViTMatte风格的detail decoder（含conv+norm+activation的融合块与上采样拼接），最终输出alpha matte。

**损失函数**：
- 抠图任务：ℓ₁ loss + Laplacian loss + Gradient loss。
- 分割任务：仅ℓ₁ loss。
- 借鉴HIM策略，仅从置信区域反向传播，降低边缘标注噪声影响。

## 实验与结果
**数据集**：
- **ICM-57**（新建）：57组图像，含人类、动物、植物及日常物体，覆盖同类别和同实例两种场景。
- **AIM-500**：重新组织用于评测自动抠图与in-context方法。
- 训练集：RM-1K + Open Images子集，共15,000张图像、450个context group。

**评估指标**：SAD、MSE（×10⁻³）、Grad、Conn（越低越好），每方法测试3轮取平均。

**主要结果（ICM-57）**：
| 方法 | MSE | SAD | GRAD | CONN |
|---|---|---|---|---|
| SegGPT | 0.0198 | 38.81 | 28.61 | 18.61 |
| SEEM | 0.0292 | 64.28 | 37.54 | 23.64 |
| VitMatte（trimap逐图） | 0.0030 | 16.16 | 14.28 | 11.14 |
| **Ours（1 mask/组）** | **0.0081** | **19.12** | **18.65** | **11.21** |

- ICM-57上MSE 0.0081，SAD 19.12，性能接近VitMatte（0.0030/16.16），显著优于SegGPT/SEEM。
- AIM-500上MSE 0.0062，SAD 18.65，优于AIM（0.0183/48.09）和LF（0.0667/191.74），与MGMiW（0.0030/16.72）相近。

**与交互抠图对比**：即使MatAny/MAM每图都接收交互提示，IconMatting仅用一组参考提示仍达更好或相当效果（Table 4）。

**消融结论**：
- 去除intra-similarity后MSE从0.0081升至0.0099，证明跨图稀疏匹配需自注意力补充。
- 移除inter与intra后MSE骤升至0.0315，丧失目标定位能力。
- 参考图数量从1增至3时性能持续提升，4时趋缓（Table 6）。

## 相关工作脉络
1. **Auxiliary input-based matting**（Trimat/Scribble/Mask引导）：如Deep Image Matting [38]、MGM [42]、MGMiW [23]——精度高但需逐图标注；本文以单参考图替代逐图输入。
2. **Automatic matting**：AIM [10]、LF [44]、ViTMatte [40]——零交互但类别受限；本文在此基础上引入参考指导实现类别自适应。
3. **In-context learning in vision**：SegGPT [35]（in-context segmentation）、Painter [34]、Prompt Diffusion [6]——已有分割/生成任务，但未涉及半透明alpha估计；本文首次将in-context学习引入matting。
4. **Diffusion model for correspondence**：DIFT [33]证明Stable Diffusion具备点-点匹配能力；本文将其扩展至区域-区域匹配并处理半透明细节。
5. **Interactive matting with SAM**：MatAny [41]、MAM [13]——基于SAM的逐图交互抠图；本文仅需参考图一次交互即覆盖整组，大幅减少用户成本。

## 局限性与未来方向
1. **参考图质量敏感**：若参考图RoI标注不完整或类别不匹配，匹配效果下降；对提示噪声的鲁棒性未充分评估。
2. **训练数据规模有限**：依赖RM-1K和Open Images子集，缺乏大规模in-context matting专用训练数据，限制了复杂场景泛化。
3. **单参考图场景下极端形变/遮挡处理**：表6显示增加参考图数量仍有小幅提升，说明单参考信息存在瓶颈。
4. **视频扩展仅初步验证**：Section 4.5展示了视频逐帧应用的可行性，但未做时序一致性优化（如光流约束、时序平滑）。
5. **Stable Diffusion推理开销大**：基于SD的feature extraction计算成本高，实时应用受限。

## 研究启发与可借鉴点
1. **扩散模型特征用于判别性任务**：利用预训练Stable Diffusion的emergent correspondence能力作为feature extractor，避免从头训练；可迁移至其他区域匹配/跨图定位任务。
2. **Inter + Intra双路径相似度融合**：跨图匹配（稀疏精准）+ 图内注意力传播（稠密补全）的组合策略，值得借鉴于few-shot segmentation、domain adaptation等场景。
3. **Reference-Prompt Extension机制**：用self-attention map将稀疏提示（point/scribble）扩展为密集RoI，是一种轻量高效的提示增强手段，可通用到其他视觉提示任务。
4. **置信区域损失截断策略**：借鉴HIM [31]仅在 confidently annotated region 回传梯度，缓解标注噪声——对任何依赖弱监督/弱标注的训练均有参考价值。
5. **单一参考指导批量目标的新范式**："一次标注，批量推理"思路可推广至object detection、semantic segmentation等需逐图标注的任务，具有显著的实用价值。

## 关键术语表
- **In-Context Matting**：新任务设定，仅用一张参考图（含提示）即可对一批同前景目标图像自动预测alpha matte。
- **Alpha Matte（α matte）**：表示图像每个像素前景透明度的单通道图，值域[0,1]。
- **Inter-similarity**：跨图相似度，通过reference RoI特征与target特征匹配获取目标区域定位。
- **Intra-similarity**：图内相似度，利用目标图自注意力图扩展inter-similarity的稀疏覆盖。
- **In-context Query**：从参考图RoI区域提取的特征序列，作为跨图匹配的query。
- **RoI Map**：Region of Interest map，用户提供的mask/ scribble/ point形式的提示输入。
- **ICM-57**：专为in-context matting构建的测试数据集，含57组真实图像。
- **ViTMatte**：基于预训练Vision Transformer的自动抠图方法，本文matting head参考其detail decoder设计。

## 可复现要素
- **数据集**：ICM-57和训练集（RM-1K + Open Images子集）通过代码仓库获取；AIM-500为公开数据集。
- **代码**：开源，地址 https://github.com/tiny-smart/incontext-matting。
- **模型权重**：基于Stable Diffusion v2-1-v预训练权重微调；论文未单独提供预训练权重链接。
- **关键超参**：学习率0.0004，batch size 8，输入尺寸768×768，训练20,000次，AdamW优化器，无额外数据增强，扩散步t=0，提取U-Net第5/8/11层特征。
