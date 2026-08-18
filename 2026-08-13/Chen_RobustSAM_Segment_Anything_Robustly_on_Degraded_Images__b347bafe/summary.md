---
title: "RobustSAM: Segment Anything Robustly on Degraded Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:39"
field: "图像分割与鲁棒性"
keywords: ["Segment Anything Model", "鲁棒分割", "图像退化", "零样本分割", "频域处理"]
innovations: ["提出AOTG与AMFG模块实现退化不变特征提取", "傅里叶退化抑制策略在分割中的应用"]
benchmarks: ["MSRA10K", "LVIS", "NDD20", "STREETS", "FSS-1000", "COCO", "BDD-100k", "LIS"]
---

# 论文速读：RobustSAM: Segment Anything Robustly on Degraded Images

## 一句话总结
本文提出 RobustSAM，通过在原始 SAM 模型基础上添加轻量级抗退化模块（AOTG 和 AMFG），使其在低质量/退化图像上保持零样本分割能力；同时构建了包含 688K 图像-掩码对的 Robust-Seg 数据集用于训练与评估。

## 研究问题与动机
- SAM 在光照不足、噪声、模糊、恶劣天气、压缩伪影等退化图像上的分割性能显著下降，直接影响下游任务。
- 先用图像复原方法预处理再输入 SAM 的策略缺乏保证，因多数复原算法针对人类视觉感知优化而非分割模型需求。
- 直接微调 SAM 解码器或全部参数会导致灾难性遗忘，严重损害零样本泛化能力。
- SAM 作为结构先验被用于去雾、去模糊等下游任务，但其在退化条件下的脆弱性限制了这些应用的实际效果。

## 核心贡献（创新点）
- 提出 AOTG（Anti-Degradation Output Token Generation）模块：通过 Instance Normalization + MLP 过滤对退化敏感的 token 信息，与已有 SAM 微调方法本质不同，仅调整输出 token 而非编码器。
- 提出 AMFG（Anti-Degradation Mask Feature Generation）模块：结合 Instance/Batch Normalization、注意力机制与傅里叶退化抑制（Fourier Degradation Suppression），生成与清晰图像特征一致的退化不变特征。
- 引入 Robust Output Token（ROT）替代原始 SAM 输出 token，保留零样本能力的同时适应退化场景。
- 构建 Robust-Seg 数据集：整合 7 个现有数据集的 43K 精细标注图像，施加 15 种合成退化，形成 688K 图像-掩码对。
- 证明 RobustSAM 可显著提升基于 SAM 先验的下游任务（去雾、去模糊）性能。

## 方法详解
- **整体框架**：冻结原始 SAM 编码器与解码器参数，仅训练新增模块；清晰图像走原始 SAM，退化图像走 RobustSAM。
- **AMFG 模块**：输入特征先经 Instance Normalization（IN）去除风格/退化信息，另一支经 Batch Normalization（BN）保留细节，两路特征经注意力机制融合后与原始特征拼接，再通过 Squeeze-and-Excitation 通道注意力；之后送入傅里叶退化抑制模块——在频域通过 1×1 卷积去除幅度分量中的退化信息、保留相位分量维持结构完整性，最后逆变换回空间域。
- **AOTG 模块**：对 Robust Output Token（ROT）使用多层 Instance Normalization 加单层 MLP 进行轻量级过滤。
- **一致性损失**：Mask Feature Consistency Loss $\mathcal{L}_{MFC} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2$，约束退化特征与清晰特征对齐。
- **Token 一致性损失**：$\mathcal{L}_{TC} = \|\hat{T}_{RO} - T_{OC}\|_2$。
- **总体损失**：$\mathcal{L}_{Overall} = \mathcal{L}_{MFC} + \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{Seg}$，其中 $\mathcal{L}_{Seg} = \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{Focal}$。
- **训练效率**：可学习参数仅 403 MB（vs SAM 1250 MB），8 张 A100 GPU 上 30 小时完成训练。

## 实验与结果
- **数据集**：训练使用 MSRA10K、ThinObject-5k、LVIS；测试涉及 NDD20、STREETS、FSS-1000、COCO（合成退化）、BDD-100k、LIS（真实退化）。
- **MSRA10K（点提示，见表 2）**：RobustSAM Degrade IoU 0.8609 vs SAM 0.8194（+4.15%），Average IoU 0.8616 vs SAM 0.8207。
- **LVIS（框提示，见表 3）**：RobustSAM Degrade IoU 0.7506 vs SAM 0.7341（+1.65%），Average IoU 0.7511。
- **零样本 NDD20+STREETS+FSS-1000（点提示，见表 4）**：RobustSAM Degrade IoU 0.8195 vs SAM 0.7981（+2.14%），Average IoU 0.8216。
- **零样本 COCO（框提示，见表 5）**：RobustSAM AP 0.5130 vs SAM 0.5002。
- **真实退化 BDD-100k+LIS（点提示，见表 6）**：RobustSAM IoU 0.3717 vs SAM 0.3056（+21.6%），Dice 0.8926 vs SAM 0.3837。
- **下游任务（表 8）**：以 RobustSAM 为先验的去雾 PSNR 23.159 vs SAM 21.677（+1.48 dB），去模糊 PSNR 29.351 vs SAM 27.491（+1.86 dB）。
- **消融（表 7）**：逐模块添加均带来提升，AMFG 贡献最大；微调整个 SAM 或解码器导致零样本能力大幅下降。

## 相关工作脉络
- **SAM [29]**：本文基础模型，但其在退化图像上表现不佳；本文保持其零样本能力同时增强鲁棒性。
- **HQ-SAM [27]**：提升分割质量的高清版本，但未专门针对退化鲁棒性设计。
- **URIE [64]**：面向视觉识别的通用图像增强方法，本文作为基线对比，指出其面向分割的适配性优于通用复原方法。
- **AirNet [37]**：面向未知退化的通用视觉质量增强方法，与 SAM 串联作为两阶段基线。
- **FIFO [35]、QualNet [28]**：针对单一退化类型的分割鲁棒性方法，缺乏多退化通用性。
- **All-in-One [39]、IPT [3]**：多退化图像复原方法，优化目标为视觉质量而非下游分割性能。

## 局限性与未来方向
- 训练依赖 15 种合成退化，对未见过的真实退化类型泛化能力虽有实验支撑但仍有局限。
- 仅针对点提示和框提示进行了评估，未涉及文本提示或其他 Prompt 形式在退化条件下的表现。
- AMFG 模块中的傅里叶变换增加了计算复杂度，在极端低资源场景下可能受限。
- 未探索 RobustSAM 在视频分割或 3D 场景中的鲁棒性。
- 未来可将 RobustSAM 扩展到更多 SAM 下游任务（如图像编辑、视频跟踪）及更多真实退化场景。

## 研究启发与可借鉴点
- **频域退化抑制思路**：将退化视为"风格"在频域去除幅度信息、保留相位结构的策略可迁移至其他视觉任务的鲁棒性增强。
- **双归一化融合策略**：IN 去除退化风格 + BN 保留细节的并行分支设计，可作为通用特征净化模块复用。
- **一致性蒸馏范式**：用原始大模型的清晰特征监督增强模型的特征对齐，避免了微调导致的灾难性遗忘，适用于任何需要保持零样本能力的模型适配场景。
- **ROTI（Robust Output Token）设计**：仅微调输出 token 而非编码器/解码器的思路，为保持基础模型泛化能力的同时适应特定分布提供了轻量方案。
- **与下游任务的协同验证**：将鲁棒性模型应用于去雾/去模糊等 SAM-prior 任务，为评估方法价值提供了更丰富的视角。

## 关键术语表
**RobustSAM**：在 SAM 基础上增强退化图像鲁棒性的零样本分割模型。
**AOTG (Anti-Degradation Output Token Generation)**：抗退化输出 token 生成模块，通过 IN+MLP 过滤 token 中的退化敏感信息。
**AMFG (Anti-Degradation Mask Feature Generation)**：抗退化掩码特征生成模块，结合归一化、注意力与傅里叶抑制生成退化不变特征。
**ROT (Robust Output Token)**：替换原始 SAM 输出 token 的可学习 token，适应退化场景。
**Fourier Degradation Suppression**：在频域去除退化幅度信息、保留相位结构信息以实现特征净化的模块。
**Robust-Seg**：本文构建的 688K 退化图像-掩码对数据集，用于训练和评估鲁棒分割模型。
**LFC (Mask Feature Consistency Loss)**：约束退化特征与清晰特征的 L2 一致性损失。
**Catastrophic Forgetting**：微调过程中模型遗忘原始预训练知识的问题。

## 可复现要素
- **数据集**：Robust-Seg 由 LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO 七个数据集合成；BDD-100k 和 LIS 为公开数据集。论文未明确说明 Robust-Seg 是否单独开源。
- **代码/权重**：论文未提及代码开源声明。
- **关键超参**：学习率 0.0005，训练 40 epoch / 130,000 iterations，8 张 A100 GPU，Batch Size 8；退化类型 15 种加恒等映射；损失权重 $\lambda_1, \lambda_2, \lambda_3$ 论文未给出具体数值。
