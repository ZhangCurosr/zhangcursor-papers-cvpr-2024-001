---
title: "Data-Free-Quantization-via-Pseudo-label-Filtering"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Fan_Data-Free_Quantization_via_Pseudo-label_Filtering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:16:42"
field: "模型量化与压缩"
keywords: ["Data-Free Quantization", "Model Compression", "Pseudo-label", "Self-entropy", "Knowledge Distillation", "Neural Network Quantization"]
innovations: ["首次在校准前利用自熵评估合成数据可靠性并按质量分桶", "设计主/辅助多伪标签机制差异化利用高/低可靠合成样本", "动态阈值可靠性过滤策略随训练逐步收紧监督标准"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet (ILSVRC2012)"]
---

# 论文速读：Data-Free Quantization via Pseudo-label Filtering

## 一句话总结
论文提出了一种基于伪标签过滤的数据无关量化（Data-Free Quantization, DFQ）方法，首次在校准阶段对预训练模型生成的合成数据可靠性进行评估，利用自熵指标将合成数据划分为高/低可靠样本并分配主/辅助伪标签，从而在无需原始训练数据的情况下有效减少低比特量化的精度损失。

## 研究问题与动机
- **原始训练数据缺失问题**：传统量化的校准/微调需要原始训练数据，但在数据隐私限制场景下不可获取，亟需数据无关方案。
- **合成数据与真实数据存在差异**：现有DFQ方法（GBA/DBA）生成的合成数据与原始训练数据分布不同，但现有方法均直接使用该数据，未考虑这一差异对量化性能的负面影响。
- **缺乏对合成数据可靠性的评估机制**：现有工作没有可靠的指标来衡量合成数据对预训练模型的"可学习程度"，导致低质量样本可能误导量化训练。
- **伪标签分配策略不合理**：对所有合成样本一律使用单一标签会导致低置信度样本产生错误监督信号，需要区分对待。

## 核心贡献（创新点）
1. **首次引入合成数据可靠性评估**：在量化训练前对预训练模型生成的合成数据进行评估，这是DFQ领域首次系统考虑合成数据质量差异的工作，区别于ZeroQ/HAST等直接训练的方法。
2. **设计基于自熵的可靠性度量指标**：提出使用自熵（self-entropy）作为评估合成数据可靠性的新指标，自熵低=高置信预测=高可靠，可定量区分高质量与低质量合成样本。
3. **提出多伪标签（Multiple Pseudo-labels）分配机制**：对高可靠样本分配主伪标签（Major），对低可靠样本分配辅助伪标签（主标签+次标签），避免低可靠样本误导训练，区别于传统单标签伪标签方法。
4. **动态阈值可靠性过滤策略**：设计随训练epoch递减的动态阈值$T_l \to T_u$，训练初期放宽高可靠样本要求以加速收敛，后期收紧以提高监督质量。

## 方法详解
**1. 合成数据生成（BN匹配）**：利用预训练模型BN层的均值$\mu^p_l$和方差$\sigma^p_l$作为先验，通过最小化BN统计差异$\mathcal{L}_{BN}$生成合成数据，同时引入标签预测交叉熵$\mathcal{L}_{IL}$提升预训练模型对生成数据的预测置信度，总损失$\mathcal{L}_{DATA}=\mathcal{L}_{BN}+\gamma\cdot\mathcal{L}_{IL}$。

**2. 自熵可靠性评估**：对每个合成样本$\hat{x}_i$，计算预训练模型的自熵：
$$\mathcal{H}_{self}(\hat{x}_i) = -\frac{1}{\log N_c}\sum_{c=1}^{N_c}\left(\mathbb{P}(\hat{x}_i,c)\cdot\log\left(\mathbb{P}(\hat{x}_i,c)\right)\right)$$
$\mathcal{H}_{self}$越低表示预测越可靠。

**3. 动态阈值分割**：设定动态阈值$t=T_u-f_t(epoch)(T_u-T_l)$，将合成数据分为$\hat{X}^h$（高可靠，$\mathcal{H}_{self}\leq t$）和$\hat{X}^l$（低可靠，$\mathcal{H}_{self}>t$）。

**4. 多伪标签设计**：
- **高可靠样本**：主伪标签$\hat{y}_i^m=\arg\max_c\mathbb{P}(\hat{x}_i^h,c)$，计算$\mathcal{L}_{CE}^h$。
- **低可靠样本**：主标签$\hat{y}_i^p$（最高概率类别）+ 次标签$\hat{y}_i^s$（次高概率类别），按概率比分配权重$\lambda_i^p,\lambda_i^s$，计算$\mathcal{L}_{CE}^l$。
- 总CE损失：$\mathcal{L}_{CE}^{total}=\mathcal{L}_{CE}^h+\beta\cdot\mathcal{L}_{CE}^l$（$\beta<1$降低低可靠样本权重）。

**5. 知识蒸馏训练框架**：结合$\mathcal{L}_{CE}^{total}$与中间层MSE特征对齐$\mathcal{L}_{MSE}$，再加KL散度$\mathcal{L}_{KL}$，总损失为$\mathcal{L}_{total}=\mathcal{L}_{KL}+\tau\mathcal{L}_{P}$，其中$\mathcal{L}_{P}=\mathcal{L}_{CE}^{total}+\mu\cdot\mathcal{L}_{MSE}$。

## 实验与结果
- **数据集与模型**：CIFAR-10/100（ResNet-20）、ImageNet（ResNet-18/50、MobileNet-V1），合成数据5120张，优化1000次迭代。
- **CIFAR-10（ResNet-20）**：W4A4达到92.47%（+0.98% vs IntraQ，+0.11% vs HAST），W3A3达到88.04%（+1.36% vs HAST）。
- **CIFAR-100（ResNet-20）**：W4A4达到66.94%（+1.96% vs IntraQ，+0.26% vs HAST），W3A3达到57.03%（+8.78% vs IntraQ，+4.27% vs AdaSG）。
- **ImageNet（ResNet-18）**：W5A5达70.35%，W4A4达67.02%，均为SOTA。
- **ImageNet（MobileNet-V1）**：W4A4达57.70%（原文Table 3显示HAST为57.70%），W5A5达68.52%，较HAST分别提升1.81%（4-bit）和0.92%（5-bit）。
- **ImageNet（ResNet-50）**：W4A4达68.97%，超越AdaSG（68.58%）和AdaDFQ（68.38%）。
- **消融实验**：$\mathcal{L}_{CE}^h$和$\mathcal{L}_{CE}^l$各自贡献显著，MSE特征对齐带来稳定提升，组合后达到最佳性能。
- **通用性验证**：将本方法的多伪标签训练模块与GDFQ、IntraQ结合，分别提升0.43%和0.51%（CIFAR-100）、0.76%和0.32%（ImageNet），证明其可扩展性。

## 相关工作脉络
- **D-FQ（Weight Equalization）**：仅优化量化器参数（权重均衡+偏置校正），不利用合成数据，精度提升有限。
- **ZeroQ（BN统计匹配）**：DBA开创性工作，用BN层统计生成合成数据，但未考虑合成数据质量差异。
- **GDFQ/Qimera/ARC等（GBA）**：训练生成器网络（GAN/VAE）产生合成数据，质量较好但计算成本高、训练复杂。
- **IntraQ（类内异构合成）**：DBA方法，生成具有类内多样性的合成图像，同样未评估合成数据可靠性。
- **HAST（困难样本增强）**：通过生成困难样本提升训练效果，侧重于数据质量而非可靠性评估与筛选。
- **AdaSG/AdaDFQ**：最新GBA方法，分别从零和博弈视角和自适应角度改进合成数据生成，但同样忽略合成数据可靠性评估这一关键环节。
- **本文定位**：不同于上述方法"直接生成并使用合成数据"的思路，本文核心创新在于"生成前先评估→按可靠性分桶→差异化伪标签训练"，是对现有DFQ范式的补充而非替代。

## 局限性与未来方向
- **自熵指标的普适性待验证**：自熵仅基于单模型预测分布，在不同任务（如检测、分割）和多模态场景下的适用性未验证。
- **阈值动态策略依赖手动调节**：$T_l$和$T_u$边界需人工设定，不同模型和数据集的最佳范围不同。
- **仅验证了图像分类任务**：未扩展到目标检测、分割等其他视觉任务。
- **合成数据量固定为5120**：对不同复杂度模型和数据集的泛化性有待进一步研究。
- **未来方向**：探索更鲁棒的可靠性评估指标（如结合预训练模型的不确定性估计）；将方法推广至其他量化形式（如混合精度量化、权重量化）；结合主动学习思想自适应选择最优合成数据子集。

## 研究启发与可借鉴点
- **可靠性评估先于数据利用**：在数据驱动的训练流程中，先对合成/标注数据进行质量评估再差异化使用，这一设计思路可迁移至联邦学习、域自适应等领域的数据筛选问题。
- **动态阈值策略**：训练中动态调整评估标准（从宽松到严格）的思路可用于课程学习（Curriculum Learning）相关研究，值得在其他自蒸馏/伪标签场景中借鉴。
- **多伪标签加权机制**：基于预测概率分配主次标签并动态加权的设计，可与半监督学习中的置信度加权策略结合，探索更鲁棒的标签分配方案。
- **与现有DBA方法兼容性强**：本文的伪标签训练模块可作为"后处理插件"与GDFQ/IntraQ等生成方法结合，获得额外性能提升，证明了方法论层面的可复用价值。
- **特征对齐（MSE）与输出对齐（KL/CE）的组合**：同时约束中间特征和输出分布的蒸馏策略，可在模型压缩、神经网络剪枝等场景下复用。

## 关键术语表
- **Data-Free Quantization (DFQ)**：无需原始训练数据，仅依靠预训练模型内部信息完成量化的模型压缩技术。
- **Self-entropy**：模型自身预测分布的熵，衡量模型对某个样本预测的不确定性，低自熵代表高置信预测。
- **Generator-Based Approach (GBA)**：通过训练生成器网络（如GAN）生成合成数据进行量化的方法。
- **Distill-Based Approach (DBA)**：利用预训练模型反向传播直接优化合成数据（无需生成器）进行量化的方法。
- **Major Pseudo-label**：分配给高可靠合成样本的主伪标签，取预训练模型预测概率最高的类别。
- **Auxiliary Pseudo-label**：分配给低可靠合成样本的辅助伪标签，包含主标签和次标签，按概率比加权。
- **Knowledge Distillation**：用预训练大模型（教师）指导小规模/量化模型（学生）学习的方法。
- **Dynamic Threshold**：随训练进程递减的可靠性判断阈值，训练初期宽松、后期严格。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、ImageNet（ILSVRC2012），均为公开数据集。
- **代码/权重**：论文未提及代码开源情况；预训练模型来自pytorchcv库。
- **关键超参**：合成数据量5120张，生成优化1000次迭代；$\gamma=10$（CIFAR）/0.1（ImageNet）；$T_l,T_u$分别为0.2/0.5（CIFAR）和0.1/0.4（ImageNet）；$\beta=0.3$（CIFAR）/0.5（ImageNet）；$\mu=100$（CIFAR）/4000（ImageNet）；$\tau=1$；SGD优化器，weight decay=$10^{-4}$，momentum=0.9；初始学习率0.001（CIFAR）/1e-5（ImageNet）。
