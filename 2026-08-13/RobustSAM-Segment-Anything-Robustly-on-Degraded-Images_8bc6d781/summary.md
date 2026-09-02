---
title: "RobustSAM-Segment-Anything-Robustly-on-Degraded-Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:15:23"
field: "退化图像鲁棒分割"
keywords: ["Segment Anything Model", "退化图像分割", "零样本分割", "鲁棒性", "图像恢复", "频域抑制"]
innovations: ["提出RobustSAM框架，冻结原SAM参数仅微调少量模块以增强退化鲁棒性", "设计AMFG与AOTG模块，结合归一化、注意力与傅里叶退化抑制提取退化不变特征", "构建688K图像的Robust-Seg数据集并验证下游去雾/去模糊任务增益"]
benchmarks: ["MSRA10K", "LVIS", "NDD20", "STREETS", "FSS-1000", "COCO", "BDD-100k", "LIS"]
---

# 论文速读：RobustSAM-Segment-Anything-Robustly-on-Degraded-Images

## 一句话总结
本文提出RobustSAM，一种在保持SAM零样本泛化能力的同时增强其对各类图像退化鲁棒性的分割模型。通过设计抗退化Token生成模块和抗退化掩码特征生成模块，并构建包含688K图像-掩码对的Robust-Seg数据集，模型在多种合成与真实退化场景下显著优于基线方法。

## 研究问题与动机
- SAM虽在清晰图像上展现卓越零样本分割能力，但在低光照、噪声、模糊、恶劣天气等退化图像上性能大幅下降，且会直接影响下游任务（如去雾、去模糊）的效果。
- 现有图像恢复方法（如All-in-One、IPT、AirNet）以人眼视觉质量为导向，无法保证提升SAM等分割模型的性能。
- 直接微调SAM全参数或解码器会导致灾难性遗忘（catastrophic forgetting），严重损害零样本泛化能力。
- 缺乏专门针对退化图像的大规模分割数据集，难以系统训练和评估模型的退化鲁棒性。

## 核心贡献（创新点）
- 提出RobustSAM框架，在冻结原SAM参数的基础上仅增加少量可学习参数（403 MB），即可实现退化鲁棒分割，且支持点在/边界框提示。
- 设计抗退化掩码特征生成模块（AMFG）：结合实例归一化（IN）与批归一化（BN）双分支、注意力融合及傅里叶退化抑制，提取退化不变的特征表示。
- 设计抗退化输出Token生成模块（AOTG）：通过轻量级IN+MLP结构精炼输出Token，并微调原始输出Token（ROT），确保分类边界的稳定性。
- 构建Robust-Seg数据集：整合7个数据集共43K标注图像，施加15种合成退化，生成688K图像-掩码对，建立退化图像分割新基准。
- 验证方法在下游任务中的增益：将RobustSAM作为先验用于单图去雾（SOTS）和去模糊（GoPro），PSNR分别提升1.48 dB和1.86 dB。

## 方法详解
**模型架构：**
- 冻结原SAM图像编码器、提示编码器、掩码解码器，仅微调输出Token（称为Robust Output Token, ROT）、AMFG模块和AOTG模块。
- 退化图像经图像编码器提取特征后，通过ROB SAM层生成掩码特征 $F_{MFD}$、互补特征 $F_{CFD}$ 和输出Token $T_{RO}$。

**AMFG模块（抗退化掩码特征生成）：**
- 双分支归一化：IN分支去除退化风格属性；BN分支保留细节信息。
- 注意力机制动态加权两分支特征后与原特征拼接，辅以Squeeze-and-Excitation通道注意力（SEC）。
- 傅里叶退化抑制：将特征变换至频域，通过1×1卷积在幅度谱上过滤退化成分，保留相位以维持结构。

**AOTG模块（抗退化输出Token生成）：**
- 轻量级多层的IN + 单层MLP，精炼 $T_{RO}$ 为 $\hat{T}_{RO}$。

**一致性损失：**
- 掩码特征一致性损失：$\mathcal{L}_{MFC} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2$
- Token一致性损失：$\mathcal{L}_{TC} = \|\hat{T}_{RO} - T_{OC}\|_2$
- 分割损失：$\mathcal{L}_{Seg} = \mathcal{L}_{Dice}(P,G) + \lambda_3 \mathcal{L}_{Focal}(P,G)$
- 总损失：$\mathcal{L}_{Overall} = \mathcal{L}_{MFC} + \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{Seg}$

**训练：**
- 在8×A100 GPU上，学习率0.0005，训练40 epoch（130K次迭代），耗时约30小时。

## 实验与结果
**数据集与基线：**
- 训练集：LVIS、ThinObject-5k、MSRA10K的完整训练集。
- 验证集：MSRA10K、LVIS测试集；零样本测试：NDD20、STREETS、FSS-1000、COCO、BDD-100k、LIS。
- 基线：SAM、HQ-SAM、AirNet+SAM（两阶段恢复+分割）、URIE+SAM。

**关键结果（保留英文数据集名）：**
- MSRA10K（点提示，平均）：RobustSAM IoU=0.8616，PA=0.9641，较SAM（IoU=0.8207）提升+4.09%。
- LVIS（边界框，平均）：RobustSAM IoU=0.7511，PA=0.9328，较SAM（IoU=0.7346）提升+2.25%。
- 零样本（NDD20+STREETS+FSS-1000，点提示，平均）：RobustSAM IoU=0.8216，PA=0.9780，较SAM（IoU=0.8000）提升+2.16%。
- 零样本（COCO，边界框）：RobustSAM AP=0.5130，较SAM（AP=0.5002）提升+1.28pp。
- 真实退化（BDD-100k+LIS，点提示）：RobustSAM IoU=0.3717，Dice=0.8926，较SAM显著提升。
- 下游任务（表8）：去雾PSNR从21.677提升至23.159（+1.48dB），去模糊PSNR从27.491提升至29.351（+1.86dB）。

**消融实验：**
- 微调SAM全参/解码器导致零样本能力大幅下降（IoU从0.3056降至0.1871）。
- 各模块逐步叠加效果：w/AMFG（0.3455）→ w/AMFG-F（0.3535）→ +AOTG（0.3651）→ +ROT全模型（0.3717）。

## 相关工作脉络
- **SAM [29]**：本文基础模型，具备强零样本泛化能力，但对退化图像鲁棒性差。
- **HQ-SAM [27]**：提升分割质量的SAM改进版，但未针对退化场景优化。
- **AirNet [37] / URIE [64]**：通用图像恢复方法，分别面向视觉质量增强和识别任务优化，但非端到端与分割联合设计，且恢复目标为人眼感知而非分割性能。
- **QualNet [28] / FIFO [35]**：CNN-based退化鲁棒分割方法，分别针对质量无关特征和雾天场景，但单一退化类型泛化有限。
- **MPRNet [87] / HINet [6] / All-in-One [39] / IPT [3]**：多退化恢复模型，以图像恢复质量为目标，不直接优化分割任务。
- **定位差异**：本文不依赖预恢复步骤，而是直接在SAM架构内注入退化鲁棒性，同时冻结原参数以保留零样本能力，且配套构建大规模退化分割数据集。

## 局限性与未来方向
- 训练数据主要依赖15种合成退化，对真实世界退化（如传感器噪声、压缩伪影）的泛化能力需进一步验证。
- 仅使用点在和边界框提示，文本提示（text prompt）下的退化鲁棒性未充分测试。
- AMFG模块引入的频率域操作可能增加推理开销，未给出详细延迟分析。
- 未来可将该思想扩展至视频分割、多模态提示等场景，并探索与扩散模型等生成式恢复方法的结合。

## 研究启发与可借鉴点
- **冻结主干+轻量适配器**的设计范式可有效平衡性能提升与灾难性遗忘，适用于各类foundation model的领域适配。
- **频域退化抑制**思想可迁移至其他视觉任务（如检测、分类），作为通用的退化解耦模块。
- **一致性蒸馏损失**（退化特征对齐清晰特征）是一种无需额外标注的自监督正则化手段，可用于模型鲁棒性训练。
- **配套数据集构建**策略（多源数据+系统退化合成）可为后续退化鲁棒性研究提供基准。
- **下游任务验证**（去雾/去模糊）展示了分割模型作为结构先验的价值，可拓展至更多联合任务。

## 关键术语表
- **Segment Anything Model (SAM)**：由Meta提出的图像分割基础模型，基于SA-1B大规模数据集训练，支持点/框/文提示的零样本分割。
- **Anti-Degradation Mask Feature Generation (AMFG)**：抗退化掩码特征生成模块，通过双归一化、注意力融合与傅里叶抑制提取退化不变特征。
- **Anti-Degradation Output Token Generation (AOTG)**：抗退化输出Token生成模块，用轻量IN+MLP精炼输出Token以保持分类边界清晰。
- **Robust Output Token (ROT)**：微调后的SAM输出Token，用于增强退化图像下的分割鲁棒性。
- **Mask Feature Consistency Loss ($\mathcal{L}_{MFC}$)**：约束退化场景下提炼特征与清晰场景特征一致性的损失函数。
- **Token Consistency Loss ($\mathcal{L}_{TC}$)**：约束精炼Token与原清晰Token一致性的损失函数。
- **Robust-Seg Dataset**：本文构建的688K图像-掩码对数据集，包含7个源数据集及15种合成退化。
- **Catastrophic Forgetting**：神经网络在持续学习新数据时遗忘原有知识的现象，直接微调SAM会引发此问题。

## 可复现要素
- **数据集**：Robust-Seg数据集由作者构建，论文未明确声明是否公开；测试使用的基准数据集（LVIS、MSRA10K、NDD20、STREETS、FSS-1000、COCO、BDD-100k、LIS）多为公开数据集。
- **代码/权重**：论文未提及代码与权重开源情况。
- **关键超参**：学习率0.0005，训练40 epoch（130K迭代），batch size 8，8×A100 GPU；$\lambda_1$-$\lambda_3$未在正文中给出具体数值。
