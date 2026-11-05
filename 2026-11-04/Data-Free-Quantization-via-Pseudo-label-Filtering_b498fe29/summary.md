---
title: "Data-Free-Quantization-via-Pseudo-label-Filtering"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Data-Free_Quantization_via_Pseudo-label_Filtering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:23:05"
field: "模型压缩与量化"
keywords: ["Data-Free Quantization", "模型量化", "伪标签", "知识蒸馏", "合成数据评估", "自熵"]
innovations: ["首次提出量化前合成数据可靠性评估机制，使用自熵度量样本可靠性", "设计主/辅双重伪标签，差异化利用高/低可靠性合成数据", "动态可靠性阈值策略，随训练进程逐步收紧样本筛选标准"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet ILSVRC2012"]
---

# 论文速读：Data-Free-Quantization-via-Pseudo-label-Filtering

## 一句话总结
本文首次提出在量化前**评估合成数据可靠性**的思想，利用预训练网络的**自熵**作为评估指标将合成数据划分为高/低可靠性样本，并分别赋予**主伪标签**和**辅助伪标签**进行差异化监督训练，在 CIFAR-10/100 和 ImageNet 上均取得了优于现有数据自由量化（DFQ）方法的精度。

## 研究问题与动机
1. **量化精度损失与数据依赖矛盾**：模型量化能显著降低计算和存储开销，但为弥补精度损失通常需要原始训练数据，在数据隐私受限场景下不可行。
2. **合成数据与原始数据存在本质差异**：现有 DFQ 方法通过 BN 层统计量生成合成数据，但合成数据无法完全匹配原始数据分布，已有可能导致低可靠性样本误导量化训练，而**所有现有方法均未对此进行区分处理**。
3. **合成数据缺乏可靠评估指标**：如何在无标签环境下对合成数据进行量化前的质量评估，是提升 DFQ 性能的关键空白。
4. **不同可靠性样本的差异化利用**：高可靠性样本可提供强监督信号，低可靠性样本则可能带来噪声，如何合理融合两者信息是方法设计的核心挑战。

## 核心贡献（创新点）
1. **首次提出量化前合成数据可靠性评估框架**：与 ZeroQ、IntraQ、HAST 等直接利用合成数据训练的 DBA 方法不同，本文在训练量化模型前先对合成数据进行可靠性甄别，这是区别于已有工作的本质创新。
2. **引入自熵（Self-Entropy）作为合成数据评估指标**：利用预训练网络在合成数据上的预测不确定性衡量可靠性，low self-entropy → 高可靠性，low self-entropy 比传统 KL 散度更直接反映模型预测置信度。
3. **设计主/辅双重伪标签机制**：高可靠性样本分配主伪标签（最大概率类别），低可靠性样本同时赋予主+辅伪标签（前两名概率类别），以软化监督避免低可靠性样本误导，与 HAST 的"构造难样本"思路形成互补。
4. **动态可靠性阈值策略**：阈值 $t$ 随训练 epoch 递减（从 $T_u$ 到 $T_l$），初期放宽高可靠性要求以快速收敛，后期收紧以保证监督质量，这一 curriculum 设计增强了训练稳定性。

## 方法详解
方法分为三个核心模块：

### 3.1 可靠性过滤（Reliability Filtering）
- **合成数据生成**：基于 BN 层统计对齐，优化随机初始数据 $\hat{\mathbf{x}}$，损失函数为：
  $$\mathcal{L}_{\text{DATA}} = \mathcal{L}_{\text{BN}} + \gamma \cdot \mathcal{L}_{\text{IL}}, \quad \mathcal{L}_{\text{BN}} = \sum_{l=1}^{L} (\|\boldsymbol{\mu}_l^p - \boldsymbol{\mu}_l^s\|_2^2 + \|\boldsymbol{\sigma}_l^p - \boldsymbol{\sigma}_l^s\|_2^2)$$
- **自熵评估**：用预训练网络对合成数据 $\hat{\mathbf{x}}_i$ 的预测分布计算：
  $$\mathcal{H}_{\text{self}}(\hat{\mathbf{x}}_i) = -\frac{1}{\log N_c} \sum_{c=1}^{N_c} \mathbb{P}(\hat{\mathbf{x}}_i, c) \cdot \log \mathbb{P}(\hat{\mathbf{x}}_i, c)$$
- **动态阈值划分**：$\hat{X}^h = \{\hat{\mathbf{x}} \mid \mathcal{H}_{\text{self}} \leq t\}$，$\hat{X}^l = \{\hat{\mathbf{x}} \mid \mathcal{H}_{\text{self}} > t\}$，其中 $t = T_u - f_t(\text{epoch})(T_u - T_l)$ 随 epoch 线性下降。

### 3.2 多重伪标签（Multiple Pseudo-Labels）
- **高可靠性样本**：主伪标签 $\hat{y}_i^m = \arg\max_c \mathbb{P}(\hat{\mathbf{x}}_i^h, c)$，损失 $\mathcal{L}_{\text{CE}}^h = \frac{1}{N_h}\sum \text{CE}(P_q(\hat{\mathbf{x}}_i^h), \hat{y}_i^m)$
- **低可靠性样本**：主标签 $\hat{y}_i^p$（最高概率类别）+ 辅标签 $\hat{y}_i^s$（次高概率类别），按预测概率比例加权：
  $$\lambda_i^p = \frac{\mathbb{P}(\hat{\mathbf{x}}_i^l, p)}{\mathbb{P}(\hat{\mathbf{x}}_i^l, p) + \mathbb{P}(\hat{\mathbf{x}}_i^l, s)}, \quad \lambda_i^s = \frac{\mathbb{P}(\hat{\mathbf{x}}_i^l, s)}{\mathbb{P}(\hat{\mathbf{x}}_i^l, p) + \mathbb{P}(\hat{\mathbf{x}}_i^l, s)}$$
  $$\mathcal{L}_{\text{CE}}^l = \frac{1}{N_l}\sum (\lambda_i^p \cdot \text{CE}(P_q(\hat{\mathbf{x}}_i^l), \hat{y}_i^p) + \lambda_i^s \cdot \text{CE}(P_q(\hat{\mathbf{x}}_i^l), \hat{y}_i^s))$$
- **总交叉熵损失**：$\mathcal{L}_{\text{CE}}^{\text{total}} = \mathcal{L}_{\text{CE}}^h + \beta \cdot \mathcal{L}_{\text{CE}}^l$（$\beta < 1$，降低低可靠性样本权重）

### 3.3 伪标签训练（Pseudo-Label Training）
- 以知识蒸馏为框架，总损失为：
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KL}} + \tau \cdot \mathcal{L}_{\text{P}}, \quad \mathcal{L}_{\text{P}} = \mathcal{L}_{\text{CE}}^{\text{total}} + \mu \cdot \mathcal{L}_{\text{MSE}}$$
- $\mathcal{L}_{\text{MSE}}$ 约束量化模型与全精度模型中间层特征的相似性：$\mathcal{L}_{\text{MSE}} = \sum_k \frac{1}{N}\sum_i (f_k(\hat{\mathbf{x}}_i) - f_k^q(\hat{\mathbf{x}}_i))^2$
- $\mathcal{L}_{\text{KL}}$ 为输出概率分布的 KL 散度：$\mathcal{L}_{\text{KL}} = \frac{1}{N}\sum_i P(\hat{\mathbf{x}}_i) \cdot \log\frac{P(\hat{\mathbf{x}}_i)}{P_q(\hat{\mathbf{x}}_i)}$

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100（ResNet-20）、ImageNet ILSVRC2012（ResNet-18/50、MobileNet-V1）
- **合成数据规模**：5,120 张，优化 1,000 次迭代（与 IntraQ、HAST 一致）
- **CIFAR-10 ResNet-20（4-bit）**：Top-1 **92.47%**，超越 IntraQ +0.98%、HAST +0.11%；3-bit 为 **88.04%**（0.30% 低于 HAST 的 88.34%）
- **CIFAR-100 ResNet-20（4-bit）**：Top-1 **66.94%**，超越 IntraQ +1.96%、AdaDFQ +0.13%；3-bit 为 **57.03%**，超越 IntraQ +8.78%
- **ImageNet ResNet-18（W4A4）**：Top-1 **67.02%**，超越 HAST +0.11%、AdaDFQ +0.49%、Qimera +3.18%；W5A5 为 **70.35%**
- **ImageNet MobileNet-V1（W4A4）**：Top-1 超越 HAST **1.81%**
- **ImageNet ResNet-50（W4A4）**：Top-1 **68.97%**，超越 AdaSG +0.39%
- **消融实验**：$\mathcal{L}_{\text{CE}}^h$、$\mathcal{L}_{\text{CE}}^l$、$\mathcal{L}_{\text{MSE}}$ 三项均有贡献；所提伪标签训练可与 GDFQ、IntraQ 等基线组合，带来额外提升（CIFAR-100 上 GDFQ+本文 +0.43%，IntraQ+本文 +0.51%）

## 相关工作脉络
1. **ZeroQ**（CVPR'20）：首个 DBA 方法，通过 BN 统计匹配生成合成数据，未考虑合成数据可靠性差异；本文在其框架上增加了可靠性评估模块。
2. **IntraQ**（CVPR'22）：引入类内多样性生成合成数据；本文与其本质区别在于 IntraQ 只关注数据生成多样性，而本文关注生成后数据的可靠性格局分化。
3. **HAST**（ICCV'23）：主动构造难样本以提升合成数据质量；本文与 HAST 形成互补——HAST 提升数据质量，本文提升数据利用效率。
4. **GDFQ**（ECCV'20）：GBA 代表，训练 GAN 生成合成数据；本文不依赖生成网络，通过 BN 统计直接优化，计算开销更低。
5. **AdaDFQ**（CVPR'23）：自适应 DFQ，引入零和博弈框架；本文从样本可靠性评估切入，两条技术路线相互独立。
6. **SQuant**（ICLR'22）：纯后训练量化（PTQ），不涉及合成数据，但可通过本方法的可靠性过滤思想改进其校准数据质量评估。

## 局限性与未来方向
1. **自熵评估的局限性**：自熵仅反映预训练网络对单样本的预测不确定性，未能建模样本间关系或流形结构，可能遗漏部分重要信息。
2. **动态阈值的设计依赖人工调参**：$T_l$、$T_u$ 的选取对最终性能有影响（论文中 CIFAR 和 ImageNet 使用不同设置），缺乏自适应确定机制。
3. **合成数据量固定为 5,120 张**：大规模数据集（如 ImageNet）上合成数据量相对较少，可能限制性能上限。
4. **未扩展到 INT8 以下的超低比特量化**：实验主要集中在 W4A4/W5A5，对 W2A2 等极低比特场景未做验证。
5. **仅验证了图像分类任务**：在目标检测、分割等下游任务上的泛化性有待探索。

## 研究启发与可借鉴点
1. **"先评估、后利用"范式**：在合成数据驱动方法（不仅是 DFQ，还包括自训练、半监督学习）中引入可靠性评估环节，可作为通用提升策略迁移。
2. **主辅伪标签机制**：对低置信度样本采用多标签软化监督而非丢弃，既利用了全部数据又降低了噪声风险，可借鉴于域适应、半监督学习中的伪标签筛选。
3. **动态阈值 Curriculum**：随训练进程逐步收紧可靠性标准，这种渐进式筛选思想可与课程学习（Curriculum Learning）结合，提升训练稳定性。
4. **与 GBA/DBA 方法的即插即用兼容性**：本文伪标签训练可与 GDFQ、IntraQ 等现有方法组合使用（Table 6 已验证），提示可靠性评估模块可作为 DFQ 的通用增强插件。
5. **中间特征 MSE 蒸馏**：在量化训练中同时约束输出概率（KL/CE）和中间特征（MSE），比仅用 KL 散度效果更稳定，这一设计值得在模型压缩的其他场景复用。

## 关键术语表
**Data-Free Quantization (DFQ)**：无需原始训练数据，通过预训练模型内部信息（如 BN 统计）合成数据进行量化的模型压缩方法。
**Self-Entropy**：模型自身预测概率分布的熵，用于衡量模型对特定样本的预测不确定性；熵越低表示预测越可靠。
**Generator-Based Approach (GBA)**：训练生成网络（如 GAN/VAE）来合成训练数据的 DFQ 方法，如 GDFQ、Qimera。
**Distill-Based Approach (DBA)**：直接迭代优化合成数据（而非训练生成器）使其匹配预训练模型分布的 DFQ 方法，如 ZeroQ、IntraQ、HAST。
**Major Pseudo-label**：分配给高可靠性合成样本的主标签，取预训练网络预测概率最高的类别，提供强监督信号。
**Auxiliary Pseudo-label**：分配给低可靠性合成样本的辅助标签，包含主标签和次标签，以软监督方式减少误导性。
**Knowledge Distillation**：利用全精度教师网络指导量化学生网络训练的技术，通过 KL 散度和特征 MSE 约束缩小两者差异。
**Batch Normalization Statistics Alignment**：通过最小化合成数据与预训练模型 BN 层均值/方差的差异来生成贴近原始数据分布的合成数据。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、ImageNet ILSVRC2012（ImageNet 需申请访问，其余公开）
- **代码**：论文未提及代码开源状态
- **权重**：全精度预训练模型来自 pytorchcv 库
- **关键超参**：合成数据量 5,120 张，迭代 1,000 次；$\gamma=10$（CIFAR）/0.1（ImageNet）；$T_l=0.2, T_u=0.5$（CIFAR）/0.1, 0.4（ImageNet）；$\beta=0.3$（CIFAR）/0.5（ImageNet）；$\mu=100$（CIFAR）/4000（ImageNet）；$\tau=1$
- **训练设备**：Nvidia GTX 3090 Ti，CUDA 11.4，Ubuntu 18.04，PyTorch
