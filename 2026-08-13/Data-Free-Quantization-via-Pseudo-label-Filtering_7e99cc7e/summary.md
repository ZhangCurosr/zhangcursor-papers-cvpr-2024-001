---
title: "Data-Free-Quantization-via-Pseudo-label-Filtering"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Data-Free_Quantization_via_Pseudo-label_Filtering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:16:53"
field: "模型压缩与量化"
keywords: ["Data-Free Quantization", "Model Compression", "Pseudo-label", "Self-Entropy", "Knowledge Distillation", "Synthetic Data Evaluation"]
innovations: ["首次提出基于自熵的合成数据可靠性评估机制", "设计主/辅多重伪标签差异化利用高/低可靠性样本"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet"]
---

# 论文速读：Data-Free-Quantization-via-Pseudo-label-Filtering

## 一句话总结
论文提出了首个在无数据量化中对合成数据进行可靠性评估的方法，利用自熵指标将合成数据划分为高/低可靠性样本，并设计多重伪标签（主伪标签+辅助伪标签）来差异化利用两类数据，显著提升量化后模型精度。

## 研究问题与动机
1. **量化性能损失依赖原始数据**：模型量化可大幅降低计算与存储开销，但量化引起的精度下降通常需要原始训练数据进行校正，这在数据隐私受限场景下不可行。
2. **现有DFQ方法忽略合成数据质量问题**：现有无数据量化（DFQ）方法通过BN层统计信息生成合成数据，但未考虑合成数据与原始数据之间的分布差异，直接用于训练可能引入误导性监督信号。
3. **缺乏合成数据可靠性评估机制**：现有方法没有对合成数据进行事前评估和分类，无法区分高质量与低质量样本，导致训练效率低下。
4. **不同可靠性样本需差异化利用**：高可靠性样本可提供强监督信号，低可靠性样本若直接使用会误导训练，但完全丢弃又会浪费潜在信息。

## 核心贡献（创新点）
1. **首次提出合成数据可靠性评估**：使用自熵（self-entropy）作为评估指标，量化预训练模型对合成数据的预测置信度，与现有DFQ方法仅关注数据生成质量不同。
2. **多重伪标签设计**：为高可靠性样本分配主伪标签（hard pseudo-label），为低可靠性样本分配辅助伪标签（primary + secondary label），与直接使用单一标签的方法本质不同，能有效避免低可靠性样本误导训练。
3. **动态可靠性阈值机制**：提出随训练epoch递减的动态阈值 $t$，在训练初期放宽高可靠性标准以快速收敛，后期严格标准以提升监督质量，这是现有方法未涉及的训练策略。
4. **伪标签与知识蒸馏融合训练框架**：将多重伪标签交叉熵损失与预测相似度损失（KL散度+中间特征MSE损失）相结合，形成统一的伪标签蒸馏训练框架。

## 方法详解
**1. 合成数据生成（Reliability Filtering前提）**
- 利用预训练模型BN层的均值 $\mu_l^p$ 和方差 $\sigma_l^p$ 作为先验，从随机初始化数据中优化合成数据分布：
  - BN匹配损失：$\mathcal{L}_{BN} = \sum_{l=1}^{L} (||\mu_l^p - \mu_l^s||_2^2 + ||\sigma_l^p - \sigma_l^s||_2^2)$
  - 总生成损失：$\mathcal{L}_{DATA} = \mathcal{L}_{BN} + \gamma \cdot \mathcal{L}_{IL}$，其中 $\mathcal{L}_{IL}$ 为inception loss

**2. 自熵可靠性评估**
- 使用预训练模型对合成数据的预测分布计算自熵：$\mathcal{H}_{self}(\hat{x}_i) = -\frac{1}{\log N_c} \sum_{c=1}^{N_c} (\mathbb{P}(\hat{x}_i, c) \cdot \log(\mathbb{P}(\hat{x}_i, c)))$
- 低自熵表示高置信度/高可靠性预测，高自熵表示低可靠性
- 动态阈值分类：$\hat{X}^h = \{\hat{x} | \mathcal{H}_{self}(\hat{x}) \leq t\}$，$\hat{X}^l = \{\hat{x} | \mathcal{H}_{self}(\hat{x}) > t\}$
- 阈值动态衰减：$t = T_u - f_t(epoch)(T_u - T_l)$，其中 $f_t(epoch) = \frac{epoch}{E}$

**3. 多重伪标签设计**
- **高可靠性样本**：主伪标签 $\hat{y}_i^m = \arg\max_c \mathbb{P}(\hat{x}_i^h, c)$，损失 $\mathcal{L}_{CE}^h = \frac{1}{N_h}\sum \text{CE}(P_q(\hat{x}_i^h), \hat{y}_i^m)$
- **低可靠性样本**：辅助伪标签包含主标签 $\hat{y}_i^p$ 和次标签 $\hat{y}_i^s$（第二高概率类别），加权损失：
  $\mathcal{L}_{CE}^l = \frac{1}{N_l}\sum (\lambda_i^p \cdot \text{CE}(P_q(\hat{x}_i^l), \hat{y}_i^p) + \lambda_i^s \cdot \text{CE}(P_q(\hat{x}_i^l), \hat{y}_i^s))$
  其中权重 $\lambda_i^p = \frac{\mathbb{P}(\hat{x}_i^l, p)}{\mathbb{P}(\hat{x}_i^l, p) + \mathbb{P}(\hat{x}_i^l, s)}$
- 总交叉熵损失：$\mathcal{L}_{CE}^{total} = \mathcal{L}_{CE}^h + \beta \cdot \mathcal{L}_{CE}^l$，$\beta < 1$

**4. 伪标签蒸馏训练**
- 预测相似度损失：$\mathcal{L}_P = \mathcal{L}_{CE}^{total} + \mu \cdot \mathcal{L}_{MSE}$
- MSE损失用于对齐中间层特征：$\mathcal{L}_{MSE} = \sum_{k=1}^{L} \frac{1}{N}\sum_{i=1}^{N} (f_k(\hat{x}_i) - f_k^q(\hat{x}_i))^2$
- KL散度损失：$\mathcal{L}_{KL} = \frac{1}{N}\sum_{i=1}^{N} P(\hat{x}_i) \cdot \log\frac{P(\hat{x}_i)}{P_q(\hat{x}_i)}$
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{KL} + \tau \mathcal{L}_P$

## 实验与结果
**数据集与模型**：CIFAR-10、CIFAR-100、ImageNet (ILSVRC2012)；ResNet-20/18/50、MobileNet-V1

**关键结果**：
- **CIFAR-10/100 (ResNet-20)**：
  - CIFAR-10: 4-bit 92.47%（超IntraQ +0.98%，超HAST +0.11%），3-bit 88.04%
  - CIFAR-100: 4-bit 66.94%（超IntraQ +1.96%，超HAST +0.26%），3-bit 57.03%（超IntraQ +8.78%）
- **ImageNet (ResNet-18)**：5-bit 70.35%，4-bit 67.02%（超HAST +0.11%，超AdaDFQ +0.49%）
- **ImageNet (MobileNet-V1)**：5-bit 68.52%（超HAST +0.92%），4-bit 57.70%（超HAST +1.81%）
- **ImageNet (ResNet-50)**：5-bit 76.08%，4-bit 68.97%（超AdaSG +0.39%）

**消融实验**：
- 多重伪标签有效：同时使用 $\mathcal{L}_{CE}^h$ 和 $\mathcal{L}_{CE}^l$ 达到最优（N₇）
- MSE特征对齐损失带来稳定提升
- 高可靠性样本训练效率优于低可靠性样本（N₅ > N₆）
- 兼容性验证：将 $\mathcal{L}_{CE}^{total}$ 与GDFQ结合提升0.43%，与IntraQ结合提升0.51%

**最强结果**：CIFAR-100 3-bit下超IntraQ 8.78%，MobileNet-V1 4-bit下超HAST 1.81%

## 相关工作脉络
1. **ZeroQ (CVPR'20)**：通过BN统计匹配优化合成数据，属于DBA方法，但未评估合成数据可靠性，本文在其蒸馏框架上增加可靠性过滤。
2. **GDFQ (ECCV'20)**：生成器基方法（GBA），训练复杂生成器，本文无需训练生成器，直接评估DBA生成的合成数据。
3. **IntraQ (CVPR'22)**：生成类内异构合成数据增强多样性，但未区分样本质量；本文方法可与其结合，在IntraQ基础上提升0.51%。
4. **HAST (ICCV'23)**：增加难样本比例以提升合成数据质量，侧重数据生成策略；本文侧重数据评估与差异化利用。
5. **Qimera (NeurIPS'21)**：利用叠加潜表征生成边界支持样本，属于GBA；本文聚焦DBA框架下的数据质量评估。
6. **AdaDFQ (CVPR'23)、AdaSG (AAAI'23)**：自适应无数据量化方法；本文在多处超越或接近其精度，核心优势在于可靠性评估机制。

## 局限性与未来方向
1. **自熵评估的局限性**：仅依赖预训练模型的预测熵，对于预训练模型本身置信度偏差较大的场景可能评估不准确。
2. **动态阈值的敏感性**：阈值上下界 $T_l, T_u$ 需手动设定，不同数据集和模型可能需要不同配置。
3. **未探索其他评估指标**：自熵仅反映预测不确定性，未考虑合成数据的分布多样性或特征判别性。
4. **仅验证图像分类任务**：未扩展到目标检测、语义分割等其他视觉任务。
5. **潜在改进方向**：结合多尺度自熵、探索更鲁棒的可靠性度量、扩展到更多模型架构（如Vision Transformer）、探索与生成器基方法的结合。

## 研究启发与可借鉴点
1. **自熵作为模型置信度代理**：在无数据场景下，可用预训练模型预测分布的自熵快速评估合成数据质量，这一思想可迁移至其他无数据压缩任务（如剪枝、蒸馏）。
2. **可靠性感知的差异化训练策略**：对样本进行质量分级并分配不同强度的监督信号，这一模式适用于半监督学习、自训练等场景。
3. **动态阈值的课程学习思想**：训练初期放宽标准快速收敛，后期严格标准精化性能，这一策略可与多种训练框架结合。
4. **主+辅双重标签设计**：为低置信度样本同时保留主标签和次标签，并基于预测概率自适应加权，可有效缓解噪声标签问题，适用于开放世界量化场景。
5. **生成与评估解耦**：先独立生成合成数据，再评估利用，这一模块化设计允许与不同生成方法（GBA/DBA）灵活组合。

## 关键术语表
**Data-Free Quantization (DFQ)**：无数据量化，指在没有原始训练数据的情况下，仅依靠预训练模型内部信息完成模型量化的方法。
**Self-Entropy (自熵)**：模型对单样本预测分布的熵值，低自熵表示高置信度预测，本文用作合成数据可靠性评估指标。
**Generator-Based Approach (GBA)**：生成器基方法，通过训练GAN/VAE等生成器产生合成数据用于量化训练。
**Distill-Based Approach (DBA)**：蒸馏基方法，将合成数据视为可训练参数，通过反向传播优化使其匹配预训练模型统计分布。
**Major Pseudo-label (主伪标签)**：分配给高可靠性样本的硬标签，即预训练模型最高概率预测类别。
**Auxiliary Pseudo-label (辅助伪标签)**：分配给低可靠性样本的双重软标签，包含主标签和次标签（第二高概率类别）。
**Knowledge Distillation (知识蒸馏)**：利用预训练教师模型指导量化学生模型训练，本文作为基础训练框架。
**Inception Loss (IL)**：利用预训练模型的中间层激活和分类输出约束合成数据的辅助损失，提升合成数据质量。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、ImageNet (ILSVRC2012)，均为公开数据集
- **代码/权重**：论文未明确说明开源；预训练模型来自pytorchcv库
- **关键超参**：
  - 合成数据量：5,120张（与IntarQ、HAST一致）
  - 数据优化迭代：1000次
  - CIFAR-10/100：$\gamma=10, T_l=0.2, T_u=0.5, \beta=0.3, \mu=100, \tau=1$
  - ImageNet：$\gamma=0.1, T_l=0.1, T_u=0.4, \beta=0.5, \mu=4000, \tau=1$
  - 优化器：SGD，weight decay=$10^{-4}$，momentum=0.9
  - 初始学习率：CIFAR为0.001，ImageNet为$10^{-5}$
  - 全部层量化（含第一层和最后一层）
