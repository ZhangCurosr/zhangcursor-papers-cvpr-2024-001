---
title: "CorrMatch-Label-Propagation-via-Correlation-Matching-for-Sem"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sun_CorrMatch_Label_Propagation_via_Correlation_Matching_for_Semi-Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:14:48"
field: "半监督视觉分割"
keywords: ["半监督语义分割", "标签传播", "相关性图", "伪标签", "一致性正则化"]
innovations: ["提出像素传播策略，利用相关性图将全局相似度信息注入预测 logits", "提出区域传播策略，从相关性图提取类无关形状掩码并填充伪标签", "设计动态阈值更新机制，自适应调整高置信度筛选标准"]
benchmarks: ["Pascal VOC 2012", "Cityscapes"]
---

# 论文速读：CorrMatch-Label-Propagation-via-Correlation-Matching-for-Sem

## 一句话总结
本文提出了一种简单高效的半监督语义分割框架 CorrMatch，通过相关性图（correlation map）建模像素对之间的相似性，设计了像素传播和区域传播两种标签传播策略，显著提升了伪标签质量并扩大了高置信度区域，在 Pascal VOC 2012 和 Cityscapes 上取得了新的 SOTA 性能。

## 研究问题与动机
- **现有方法依赖复杂训练策略**：Mean Teacher、自训练等方法通常需要多个网络、多训练阶段或多强数据增强分支，训练流程复杂。
- **固定阈值筛选伪标签效率低下**：传统方法通过固定阈值挑选高置信度像素作为伪标签，阈值过严会限制无标签数据利用，过松则引入噪声，难以兼顾精度与覆盖率。
- **忽略了相关性图的双重价值**：相关性图不仅能量化像素对的语义相似性（便于同类别聚类），每行还隐式编码了类别无关的物体形状信息，已有工作未充分利用这些特性。
- **伪标签质量提升空间大**：仅依赖分类预测的高置信度像素作为伪标签，未能借助空间结构和形状先验对低置信度区域进行合理填补。

## 核心贡献（创新点）
- **揭示相关性图的双重优势**：指出相关性图既能建模全局像素对相似性，又能提取类无关形状信息，为半监督分割提供了新的监督信号来源；与已有工作的本质区别在于从"相似度度量"转向"结构化形状先验"的利用。
- **像素传播策略（Pixel Propagation）**：通过相关性图将全局相似性信息传播到预测 logits 中，增强模型对成对相似性的感知；区别于传统仅依赖 Hard Pseudo Label 的做法，引入了基于相关性匹配的监督损失。
- **区域传播策略（Region Propagation）**：从相关性图中提取二值化形状掩码，结合高置信度区域的多数投票类别进行标签填充；与阈值扩展方法的本质区别在于：不单纯扩大高置信区，而是利用形状信息确保语义结构一致性。
- **动态阈值策略**：提出基于 EMA 的动态阈值更新机制，自动适应训练进程；相比固定阈值或手动调参，具有更强的鲁棒性和自适应能力。

## 方法详解
**整体框架**：基于 DeepLabV3+ 的单阶段弱到强一致性正则化框架，包含有监督损失和无监督损失两部分。

**相关性图构建**：
- 从编码器后通过线性层提取特征 $w_1, w_2 \in \mathbb{R}^{D \times HW}$
- 计算相关性图：$\mathcal{C} = \text{Softmax}(w_1^\top \cdot w_2) / \sqrt{D}$，得到 $\mathbb{R}^{HW \times HW}$ 的全局成对相似度矩阵

**像素传播策略**：
- 将相关性图传播到预测 logits：$\mathbf{z}_i^u = f_1(\hat{\mathcal{F}}(x_i^u)) \cdot \mathcal{C}$，其中 $f_1$ 为双线性插值
- 计算相关性损失：$\mathcal{L}_u^c = \frac{1}{N}\sum_i \ell_c(\mathbf{z}_i^u, \mathcal{F}(x_i^w)) \odot \mathcal{M}_i$，$\mathcal{M}_i$ 为高置信度二值掩码

**区域传播策略**：
- 对相关性图每行归一化后二值化得到形状掩码 $\hat{\mathbf{c}}$，过滤掉相似度较低的位置
- 计算 $\hat{\mathbf{c}}$ 与高置信度区域 $\mathcal{M}_i$ 的交集，统计各类别出现次数 $G(l)$，选取占比最高的类别 $k^*$
- 当重叠比例 $r_1 > \tau$ 且类别占比 $r_2 > \tau$ 时，将 $k^*$ 填充到 $\hat{\mathbf{c}}=1$ 的区域，并扩展高置信度区域 $\mathcal{M}_i = \mathcal{M}_i \cup \hat{\mathbf{c}}$
- 采用随机采样策略（默认采样128个形状）以平衡计算效率

**动态阈值更新**：
- 初始化阈值 $\tau = 0.85$
- 使用 EMA 更新：$\Delta\tau = \frac{1}{|L|}\sum_{l \in L} \mathbb{1}(\mathcal{F}(x_i^w) = l) \odot \max(\hat{\mathcal{F}}(x_i^w))$

**总损失函数**：
- 有监督损失：$\mathcal{L}_s = \frac{1}{2}(\mathcal{L}_s^h + \mathcal{L}_s^c)$
- 无监督损失：$\mathcal{L}_u = \lambda_1\mathcal{L}_u^h + \lambda_2\mathcal{L}_u^s + \lambda_3\mathcal{L}_u^c$，默认权重 $[\lambda_1, \lambda_2, \lambda_3] = [0.5, 0.25, 0.25]$
- 总损失：$\mathcal{L} = \frac{1}{2}(\mathcal{L}_s + \mathcal{L}_u)$

## 实验与结果
**数据集**：Pascal VOC 2012（1464张标注图像）、aug Pascal VOC 2012（10582张）、Cityscapes（2975训练/500验证）

**评估基线**：UniMatch、ST++、U²PL、PS-MT、PCR、CCVC、AugSeg 等 SOTA 方法

**主要结果**：
- **Pascal VOC 2012（经典设置，321×321）**：
  - 1/16 split（92张）：**76.4% mIoU**，较 UniMatch 提升 +1.2%
  - 1/8 split（183张）：78.5%，提升 +1.3%
  - 1/4 split（366张）：79.4%，提升 +0.6%
  - Full split（1464张）：**81.8% mIoU**，提升 +0.6%
- **Pascal VOC 2012（aug设置，513×513）**：
  - 1/16 split：81.3% mIoU，较 AugSeg 提升 +2.0%
- **Cityscapes（513×513）**：
  - 1/16 split：76.6%，较 UniMatch 提升 +0.7%
  - 1/8 split：77.9%，提升 +0.6%
  - 1/4 split：78.4%，提升 +0.2%
  - 1/2 split：79.1%，提升 +0.9%

**消融实验关键发现**：
- 像素传播带来 +1.4%/+0.4%/+0.8% 提升（92/366/1464 split）
- 区域传播进一步带来 +0.6%/+0.5%/+0.5% 提升
- 动态阈值较固定阈值带来 +0.9%/+1.0% 提升（全量标注时）
- 随机采样优于均匀采样，128个采样点达到最佳性价比

## 相关工作脉络
- **Mean Teacher 系列（PS-MT、ELN、AEL）**：依赖额外教师网络或 VAT 扰动，训练复杂；CorrMatch 无需多网络，单阶段即可完成。
- **自训练方法（ST++、PseudoSeg）**：通常需要多训练阶段；CorrMatch 在单阶段框架内实现同等甚至更优效果。
- **FixMatch 系（UniMatch、CPS）**：依赖多强数据增强分支；CorrMatch 仅用弱到强一致性即可取得更好结果。
- **对比学习系（PC²Seg、U²PL）**：在特征空间做对比约束；CorrMatch 直接在预测空间利用相关性图传播，更直接高效。
- **动态阈值方法（FreeMatch、FlexMatch）**：用于分类任务的阈值自适应；本文将其适配到分割任务并结合形状先验，提出适合多类别场景的动态策略。

## 局限性与未来方向
- **计算开销**：相关性图构建涉及 $O((HW)^2)$ 的矩阵运算，尽管推理阶段无额外负担，但训练阶段计算成本较高，尤其在大分辨率图像上。
- **采样策略依赖**：区域传播采用随机采样128个形状，采样数量和质量对性能有影响，可能存在次优选择。
- **特征提取位置限制**：实验表明 backbone 特征效果最佳，但未深入探索不同网络架构下的迁移性。
- **未来方向**：可探索更高效的相似度计算方法（如低秩近似）、自适应采样策略、以及将该框架迁移到更具挑战性的场景（如远程传感、医学图像分割）。

## 研究启发与可借鉴点
- **相关性图的形状挖掘**：将相关性图的行向量视为类别无关的形状先验，这一视角可迁移到其他视觉任务（如实例分割、对象检测）的伪标签优化。
- **单一信号的多重利用**：同一相关性图同时服务于像素级相似度和区域级形状提取，避免了多模块设计的复杂度，值得在其他半监督框架中借鉴。
- **动态阈值与传播策略的协同**：动态阈值不独立优化，而是与传播策略配合使用，这种联合设计思路值得推广到阈值敏感的其他学习方法中。
- **训练无额外开销**：相关性图构建仅在训练阶段进行，推理零负担，这一设计原则对其他引入辅助模块的方法具有参考价值。

## 关键术语表
**Correlation Map（相关性图）**：通过特征向量对的内积和 Softmax 得到的成对相似度矩阵，用于刻画图像内任意两位置的语义关联程度。
**Pixel Propagation（像素传播）**：将相关性图乘入预测 logits，使模型预测融入全局相似度信息的策略。
**Region Propagation（区域传播）**：从相关性图中提取二值化形状掩码，与高置信度区域结合后填充类别标签的策略。
**Pseudo Label（伪标签）**：模型对无标签图像的高置信度预测，作为监督信号用于无监督损失计算。
**Weak-to-Strong Consistency（弱到强一致性）**：对无标签图像进行弱增强和强增强，约束两者预测一致性的正则化手段。
**Dynamic Threshold（动态阈值）**：基于 EMA 随训练进程自动调整的置信度阈值，替代固定阈值策略。
**mIoU（mean Intersection-over-Union）**：语义分割常用评估指标，计算各类别预测与真实掩码交并比的均值。

## 可复现要素
- **数据集**：Pascal VOC 2012（公开）、aug Pascal VOC 2012（含 SBD 粗标注，公开）、Cityscapes（公开）
- **代码**：已开源，地址 https://github.com/BBBBchan/CorrMatch
- **关键超参数**：
  - 初始阈值 τ = 0.85
  - 损失权重 [λ₁, λ₂, λ₃] = [0.5, 0.25, 0.25]
  - 形状采样数量 = 128（随机采样）
  - Pascal VOC 训练分辨率：321×321 或 513×513
  - Cityscapes 训练分辨率：801×801
  - 优化器：SGD，初始学习率 VOC 0.001 / Cityscapes 0.005
  - 训练轮数：VOC 80 epochs / Cityscapes 240 epochs
  - Batch size：16
  - GPU：4× A40（Cityscapes）
