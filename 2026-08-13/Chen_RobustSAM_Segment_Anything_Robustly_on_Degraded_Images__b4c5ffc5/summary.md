---
title: "RobustSAM: Segment Anything Robustly on Degraded Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:54"
field: "退化图像鲁棒分割"
keywords: ["鲁棒分割", "Segment Anything Model", "退化图像", "零样本分割", "频域特征抑制"]
innovations: ["冻结SAM主干并注入AMFG+AOTG双模块实现15类退化鲁棒性", "首次在SAM特征空间引入傅里叶退化抑制机制", "提出Robust-Seg 688K退化数据集作为首个系统化评测基准"]
benchmarks: ["MSRA10K", "LVIS", "NDD20", "STREETS", "FSS-1000", "COCO", "BDD-100k", "LIS"]
---

# 论文速读：RobustSAM: Segment Anything Robustly on Degraded Images

## 一句话总结
RobustSAM 在冻结原始 SAM 的前提下，仅新增两个轻量级抗退化模块（AMFG 与 AOTG）及一致性损失，使 SAM 在 15 种合成退化图像上仍能保持高质量零样本分割，同时额外贡献了 688K 规模的 Robust-Seg 基准数据集。

## 研究问题与动机
- **SAM 在低质量图像上显著衰退**：现有研究（Huang et al. 2023；Qiao et al. 2023；Wang et al. 2023）表明，低光照、噪声、模糊、恶劣天气和压缩伪影等退化会大幅降低 SAM 的分割精度，直接影响依赖 SAM 掩码的下游任务。
- **先验恢复方法不匹配分割需求**：通用图像复原算法（如 All-in-One、AirNet、IPT）以人眼视觉质量为优化目标，并未针对 SAM 等分割模型的特征敏感性进行训练，因此复原后的图像不一定带来分割提升。
- **直接微调 SAM 损害零样本能力**：对整个 SAM 或 decoder 进行端到端微调会导致灾难性遗忘（Kirkpatrick et al. 2017），在退化数据上表现提升的同时，干净图像的零样本泛化性能急剧下降。
- **已有鲁棒分割方法聚焦单类退化**：QualNet、URIE、FIFO 等仅处理单一退化类型，缺乏对多类退化组合的鲁棒性，且未考虑 Foundation Model（SAM）的零样本特性。

## 核心贡献（创新点）
1. **提出 RobustSAM 框架**：在冻结原始 SAM 全部参数的条件下，仅注入两个轻量抗退化模块，以极小参数增量（+403 MB vs. SAM 的 1250 MB）实现对 15 类退化的鲁棒分割。*与已有工作的本质区别：不同于直接微调 SAM（会破坏零样本能力），本方法通过一致性损失保留干净图像上的原始特征分布。*
2. **设计 AMFG（抗退化掩码特征生成）模块**：结合 Instance Normalization（去除退化风格）、Batch Normalization（保留细节）与 Squeeze-and-Excitation 通道注意力，并在频域通过傅里叶变换抑制退化幅度分量，从而提取与退化无关的分割特征。*与已有工作的本质区别：首次将频域退化抑制引入 SAM 的特征空间，而非仅在空域做复原。*
3. **设计 AOTG（抗退化输出 Token 生成）模块**：用轻量化 IN + MLP 对 SAM 输出 Token 进行退化无关化精炼，配合 Token 一致性损失约束。*与已有工作的本质区别：AOTG 仅作用于分类边界 Token（纹理信息少），避免了 AMFG 级别的重计算开销。*
4. **引入 Robust-Seg 数据集**：整合 7 个现有数据集的 43K 标注图像，施加 15 种合成退化，构建 688K 图像-掩码对，作为首个面向退化鲁棒性的大规模 SAM 评测基准。*与已有工作的本质区别：现有数据集（如 COCO、LVIS）仅提供干净图像，缺乏系统化的退化评估维度。*
5. **验证 SAM 先验任务的增强效果**：在单图像去 haze（SOTS 数据集）与去模糊（GoPro 数据集）任务中，以 RobustSAM 替换 SAM 作为结构先验，PSNR 分别提升 +1.48 dB 与 +1.86 dB。*与已有工作的本质区别：首次证明提升 SAM 鲁棒性可直接反哺依赖 SAM 掩码的下游复原任务。*

## 方法详解

**整体思路**：训练时，干净图像同时过原始 SAM（灰线部分，参数冻结）得到清晰特征；退化图像过 RobustSAM（紫线部分），经 AMFG 和 AOTG 模块精炼后，通过一致性损失向清晰特征对齐，并用分割损失监督掩码预测。推理时只使用 RobustSAM。

**AMFG 模块（Figure 3 左侧）**：
- 输入特征 $F_{CFD}$ 与 $F_{MFD}$ 分别经 Instance Normalization（IN）和 Batch Normalization（BN）并行处理，IN 去除退化风格，BN 防止细节丢失；
- 两路特征合并后经注意力机制生成注意力图，加权融合后再与原输入沿通道拼接；
- 引入 Squeeze-and-Excitation Channel（SEC）注意力进一步自适应精炼通道权重；
- **傅里叶退化抑制（Fourier Degradation Suppression）**：对增强特征做傅里叶变换，用 $1 \times 1$ 卷积在频域学习退化幅度分量的抑制，相位分量保持不变（保留结构信息），再做逆变换回空域；
- 精炼后特征 $\hat{F}_{CFD}$、$\hat{F}_{MFD}$ 经 Feature Fusion 模块融合为最终鲁棒掩码特征。

**AOTG 模块（Figure 3 右侧）**：
- 对 Robust Output Token $T_{RO}$ 使用多层 IN + 单层 MLP 进行轻量化退化无关化精炼，输出 $\hat{T}_{RO}$；
- 精炼 Token 与鲁棒掩码特征结合生成最终分割掩码。

**Robust Output Token（ROT）**：将 SAM 原始输出 Token 可训练化为 ROT，替代冻结的原始 Token，使其能吸收退化不变信息。

**损失函数**：
$$\mathcal{L}_{\text{Overall}} = \mathcal{L}_{\text{MFC}} + \lambda_1 \mathcal{L}_{\text{TC}} + \lambda_2 \mathcal{L}_{\text{Seg}}$$
其中：
- **Mask Feature Consistency Loss**（式 1）：$\mathcal{L}_{\text{MFC}} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2$，约束精炼特征与干净图像下原始 SAM 提取的对应特征一致；
- **Token Consistency Loss**（式 2）：$\mathcal{L}_{\text{TC}} = \|\hat{T}_{RO} - T_{OC}\|_2$，约束精炼 Token 与干净图像下原始 SAM 输出 Token 一致；
- **Segmentation Loss**（式 4）：$\mathcal{L}_{\text{Seg}} = \mathcal{L}_{\text{Dice}}(P, G) + \lambda_3 \mathcal{L}_{\text{Focal}}(P, G)$，其中 $P$ 为预测掩码，$G$ 为 ground truth；
- $\lambda_1, \lambda_2, \lambda_3$ 为损失权重超参。

**训练配置**：冻结 SAM 全部参数，学习率 0.0005，40 个 epoch（130K 迭代），8 × NVIDIA A100 GPU，batch size 8，仅使用点提示训练。

## 实验与结果

**数据集**：训练/验证用 MSRA10K、ThinObject-5k、LVIS 的训练集及各自退化增强；零样本测试用 NDD20、STREETS、FSS-1000、COCO（均为合成退化），以及含真实世界退化（低光照、模糊、雨、雪）的 BDD-100k 与 LIS。

**评价指标**：IoU、Dice、Pixel Accuracy（PA）、AP（COCO 标准）。

**关键结果（点提示，MSRA10K 测试集，Table 2）**：
- 退化场景 IoU：**RobustSAM 0.8609** vs. SAM 0.8194（+4.15 pp）vs. HQ-SAM 0.8358（+2.51 pp）
- 干净场景 IoU：RobustSAM 0.8726 vs. SAM 0.8402（+3.24 pp）
- 平均 IoU：**RobustSAM 0.8616**（最佳）

**关键结果（框提示，LVIS 测试集，Table 3）**：
- 退化场景 IoU：**RobustSAM 0.7506** vs. SAM 0.7341（+1.65 pp）
- 平均 IoU：**RobustSAM 0.7511**（最佳）

**零样本测试（NDD20 + STREETS + FSS-1000，Table 4）**：
- 退化场景 IoU：**RobustSAM 0.8195** vs. SAM 0.7981（+2.14 pp）vs. HQ-SAM 0.8079
- 干净场景 IoU：**RobustSAM 0.8529** vs. SAM 0.8295（+2.34 pp）
- 平均 IoU：**RobustSAM 0.8216**（最佳）

**零样本 COCO 测试（框提示，Table 5）**：
- AP：**RobustSAM 0.5130** vs. SAM 0.5002（+1.28 pp）
- APL（大目标）：**RobustSAM 0.5518** vs. SAM 0.5243（+2.75 pp）

**真实退化测试（BDD-100k + LIS，Table 6）**：
- 点提示 IoU：**RobustSAM 0.3717** vs. SAM 0.3056（+6.61 pp）
- 点提示 Dice：**RobustSAM 0.8926** vs. SAM 0.3837（+50.89 pp，显著差异源于指标分布）
- 框提示 IoU：**RobustSAM 0.8958** vs. SAM 0.8808（+1.50 pp）

**SAM 先验任务提升（Table 8）**：
- 去 haze（SOTS）：PSNR 从 21.677 → **23.159**（+1.48 dB），SSIM 0.8451 → **0.8685**
- 去模糊（GoPro）：PSNR 从 27.491 → **29.351**（+1.86 dB），SSIM 0.9066 → **0.9229**

**最强结果**：在真实退化 BDD-100k + LIS 上，点提示 IoU 达 0.3717（较 SAM 提升 6.61 pp），是全部测试中相对提升幅度最大的场景。

## 相关工作脉络
- **SAM（Kirillov et al. 2023）**：本文的基线模型；SAM 在干净图像上表现优异，但未针对退化场景优化。
- **HQ-SAM（Ke et al. 2023）**：提升 SAM 输出质量的高分辨率变体；与本文正交，本文关注退化鲁棒性。
- **AirNet（Li et al. 2022）/ IPT（Chen et al. 2021）**：通用图像复原方法，优化人眼视觉质量；本文指出其未针对分割下游任务优化。
- **URIE（Son et al. 2020）**：面向视觉识别的通用图像增强；与 AirNet+SAM 类似，仍属"先复原再分割"范式，本文证明其局限性。
- **QualNet（Kim et al. 2021）/ FIFO（Lee et al. 2022）**：面向单类退化（低质量/雾）的分割方法；本文方法统一处理 15 类退化且保留 SAM 零样本能力。
- **SAM 鲁棒性分析（Huang et al. 2023；Qiao et al. 2023；Wang et al. 2023）**：诊断性论文，指出 SAM 在退化下性能衰退，但无系统性解决方案；本文首次提出端到端鲁棒训练框架。
- **Degraded image segmentation（Endo et al. 2023；Guo et al. 2019）**：传统 CNN 分割方法对退化的改进；本文将其思路迁移至 Foundation Model 时代。

## 局限性与未来方向
- **合成退化到真实退化的泛化Gap**：训练数据仅含 15 种合成退化，虽然 BDD-100k/LIS 真实退化测试显示良好效果，但未覆盖所有真实场景（如水下、医学图像噪声）。
- **15 类退化的覆盖有限**：未包含所有可能退化类型（如运动模糊的具体变体、JPEG 不同质量级别、传感器噪声模型等）。
- **仅使用点提示**：训练和大部分评估仅基于点提示，框提示的零样本退化鲁棒性仍有较大提升空间。
- **一致性损失的超参敏感**：$\lambda_1, \lambda_2, \lambda_3$ 的选取对最终性能有影响，未给出系统的超参灵敏度分析。
- **未探索多模态提示的鲁棒性**：文本提示（text prompt）在退化条件下的表现未涉及，后续可扩展到多模态鲁棒性。

## 研究启发与可借鉴点
- **冻结主干 + 轻量适配器**的模式非常有效：冻结 1250 MB 预训练参数，仅训练 403 MB 新模块，既保留了预训练的零样本能力，又实现了 4+ pp 的退化性能提升，适合迁移到 SAM 的其他下游适配场景。
- **频域退化抑制思路**：将傅里叶变换引入特征精炼（抑制幅度分量、保留相位）为处理退化风格提供了一种新的特征解耦视角，可推广到去雾、去雨、超分等其他复原任务的特征空间。
- **一致性蒸馏而非分类监督**：用干净图像的原始特征作为教师信号（一致性损失），而非直接用 ground truth 监督新模块，是一种避免灾难性遗忘的有效策略，可与其它 Foundation Model 微调方法结合。
- **Simultaneous training with identity mapping**：15 种退化加 1 种恒等映射的训练策略，确保模型在干净图像上不降反升（MSRA10K 干净 IoU 0.8726 > SAM 的 0.8402），这一设计值得在其他鲁棒性工作中借鉴。
- **RobustSAM 作为下游任务的鲁棒先验**：本文证明了提升基础模型鲁棒性可反哺下游任务（去 haze/去模糊），这一"上游鲁棒 → 下游增益"的思路可用于评估分割 Foundation Model 的广泛适用性。

## 关键术语表
- **RobustSAM**：本文提出的抗退化 Segment Anything 模型，在冻结原始 SAM 基础上添加 AMFG/AOTG 两个轻量模块。
- **AMFG（Anti-Degradation Mask Feature Generation）**：抗退化掩码特征生成模块，结合 IN/BN 归一化与频域退化抑制提取退化不变特征。
- **AOTG（Anti-Degradation Output Token Generation）**：抗退化输出 Token 生成模块，用轻量 IN+MLP 精炼分类边界 Token。
- **ROT（Robust Output Token）**：可训练的 SAM 输出 Token 替代版本，区别于冻结的原始输出 Token。
- **$\mathcal{L}_{\text{MFC}}$（Mask Feature Consistency Loss）**：约束精炼特征与干净图像下原始 SAM 对应特征一致的 L2 损失。
- **$\mathcal{L}_{\text{TC}}$（Token Consistency Loss）**：约束精炼 Token 与干净图像下原始 SAM 输出 Token 一致的 L2 损失。
- **Robust-Seg**：本文构建的 688K 退化图像-掩码对数据集，用于训练和评测鲁棒分割。
- **Fourier Degradation Suppression**：在频域通过傅里叶变换抑制图像退化幅度分量、保留相位结构的核心技术。

## 可复现要素
- **数据集**：Robust-Seg（688K 图像-掩码对）由 7 个公开数据集合成：LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO；**论文未声明代码/权重开源**。
- **训练硬件**：8 × NVIDIA A100 GPU，batch size 8，学习率 0.0005，40 epoch / 130K 迭代，耗时约 30 小时。
- **关键超参**：$\lambda_1, \lambda_2, \lambda_3$（损失权重）、15 种合成退化类型（含恒等映射）。
- **测试协议**：点提示与框提示分别评测；测试集包括_seen_（MSRA10K、LVIS）与_zero-shot_（NDD20、STREETS、FSS-1000、COCO、BDD-100k、LIS）。
- **对比基线**：SAM、HQ-SAM、AirNet+SAM、URIE+SAM。
