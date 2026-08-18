---
title: "RobustSAM: Segment Anything Robustly on Degraded Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:24:43"
field: "鲁棒视觉基础模型"
keywords: ["零样本分割", "鲁棒分割", "Segment Anything Model", "图像退化", "频域去风格"]
innovations: ["以冻结 SAM 为骨干、AMFG+AOTG 双 adapter 保持零样本能力的轻量鲁棒分割框架", "在频域对 mask 与互补特征进行退化风格分离的傅里叶退化抑制模块"]
benchmarks: ["Robust-Seg (MSRA10K/LVIS/NDD20/STREETS/FSS-1000/COCO/BDD-100k/LIS)"]
---

# 论文速读：RobustSAM: Segment Anything Robustly on Degraded Images

## 一句话总结
RobustSAM 在冻结原始 SAM 参数的前提下，仅用约 403 MB 新增参数（可在 8 卡 A100 上 30 小时内训练完成）为 SAM 引入抗退化模块，使其在各类图像退化下仍保持高精度零样本分割能力；同时贡献了 68.8 万对 15 种合成退化图像掩码对的大规模数据集 Robust-Seg。

## 研究问题与动机
- SAM 的零样本分割能力依赖高质量图像，低光照、噪声、模糊、恶劣天气、压缩伪影等退化会显著降低其 mask 质量。
- 现有图像恢复方法（AirNet、URIE、HQ-SAM 等）以人眼视觉质量为目标优化，而非面向 SAM 这类下游分割模型的需求。
- 直接对 SAM 解码器或整模型做退化微调会引发灾难性遗忘，破坏其在干净图像上的零样本泛化能力。
- 下游任务（单图去雾、去模糊等）依赖 SAM 提供可靠结构先验，SAM 在退化图像上失效会连锁拖累这些应用。

## 核心贡献（创新点）
- 提出 Anti-Degradation Mask Feature Generation (AMFG) 模块，通过 Instance/Batch Norm 融合 + 注意力 + 通道注意（SEC）+ 傅里叶退化抑制，使互补特征与 mask 特征在空间/频率域同时去除退化风格、保留语义结构；与纯空间域增强方法本质区别在于引入了频域退化分离。
- 提出 Anti-Degradation Output Token Generation (AOTG) 模块，用轻量 IN+MLP 剔除输出 token 中的退化敏感信息；与直接微调 SAM 输出 token 的区别是引入来自干净图像的 token 一致性约束，避免灾难性遗忘。
- 引入 Robust Output Token (ROT)，冻结原 SAM 所有模块，仅替换输出 token 并联合训练上述模块；训练参数量仅 403 MB（对比 SAM 1250 MB），实现极低开销的稳健升级。
- 构建 Robust-Seg 数据集：从 7 个现有数据集（LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO）中精选 4.3 万张标注图像，施加 15 类合成退化，形成 68.8 万图像–掩码对，填补退化分割基准空白。
- 证明 RobustSAM 可作为 SAM-prior 下游任务（去雾、去模糊）的更强结构先验，在 SOTS/GoPro 上分别将 PSNR 提升 +1.48 dB / +1.86 dB，SSIM 提升 +0.023 / +0.016。

## 方法详解
**整体框架（图 2）**：训练时，清晰图像走原始 SAM（灰色冻结模块）得到干净特征；退化图像走 RobustSAM（紫色新增模块），经 AMFG/AOTG/ROT 后与干净特征对齐；推理时仅用 RobustSAM。

**AMFG 模块（图 3 左侧）**：
1. 输入特征并行过 Instance Normalization（去除退化风格）与 Batch Normalization（保留细节），二者经注意力融合。
2. 融合结果沿通道维拼接原始输入，并通过 Squeeze-and-Excitation 通道注意力（SEC）自适应加权。
3. 对互补特征 $F_{CFD}$ 与 mask 特征 $F_{MFD}$ 分别引入 **傅里叶退化抑制**：经 FFT 取振幅表征风格/退化成分，$1\times1$ 卷积滤除退化频带，相位（结构）保留，再逆 FFT 回空间域。
4. 与干净 SAM 对应特征对齐：
$$
\mathcal{L}_{\text{MFC}} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2
$$

**AOTG 模块（图 3 右侧）**：多级 IN 后接单层 MLP，对 ROT 做轻量退化过滤：
$$
\mathcal{L}_{\text{TC}} = \|\hat{T}_{RO} - T_{OC}\|_2
$$

**整体损失**：
$$
\mathcal{L}_{\text{Overall}} = \mathcal{L}_{\text{MFC}} + \lambda_1 \mathcal{L}_{\text{TC}} + \lambda_2 \mathcal{L}_{\text{Seg}}
$$
其中 $\mathcal{L}_{\text{Seg}} = \mathcal{L}_{\text{Dice}} + \lambda_3 \mathcal{L}_{\text{Focal}}$。训练 40 轮，lr=5e-4，8×A100，13 万步/30 小时。

## 实验与结果
**数据集与评估**：训练集 MSRA10K、ThinObject-5k、LVIS；测试集含合成退化 NDD20/STREETS/FSS-1000/COCO/MSRA10K/LVIS，以及含真实世界退化 BDD-100k/LIS；指标 IoU、PA、Dice、AP。

**关键结果（点提示，MSRA10K 合成退化）**：

| 方法 | 退化 IoU | 退化 PA | 清晰 IoU | 清晰 PA | 平均 IoU | 平均 PA |
|---|---|---|---|---|---|---|
| SAM | 0.8194 | 0.9108 | 0.8402 | 0.9235 | 0.8207 | 0.9116 |
| HQ-SAM | 0.8358 | 0.9202 | 0.8604 | 0.9328 | 0.8373 | 0.9210 |
| AirNet+SAM | 0.8157 | 0.9193 | 0.8236 | 0.9294 | 0.8162 | 0.9199 |
| URIE+SAM | 0.8217 | 0.9125 | 0.8450 | 0.9245 | 0.8231 | 0.9132 |
| **RobustSAM** | **0.8609** | **0.9640** | **0.8726** | **0.9649** | **0.8616** | **0.9641** |

- 在 LVIS 框提示上，RobustSAM 退化 IoU 达 0.7506（SAM 0.7341），平均 PA 0.9328。
- 在 NDD20/STREETS/FSS-1000 零样本上，平均 IoU 0.8216（SAM 0.8000）、PA 0.9780（SAM 0.9565）。
- 在 COCO 框提示零样本上，AP 达 0.5130（SAM 0.5002），APL 达 0.5518（SAM 0.5243）。
- 在 BDD-100k/LIS 真实退化点提示上，IoU 0.3717（SAM 0.3056）、Dice 0.8926（SAM 0.3837），提升极为显著。
- 下游去雾/去模糊：PSNR +1.48/+1.86 dB，SSIM +0.023/+0.016。

**消融**：冻结 SAM + 仅微调输出 token 仍弱于 RobustSAM；AMFG 贡献最大；ROT + AMFG + AOTG 三者叠加效果最佳。

## 相关工作脉络
- **SAM [29]**：本文基线；SAM 在退化图像上性能下降已被 Huang et al.、Qiao et al.、Wang et al. 等实证指出，本文聚焦“保零样本 + 抗退化”的协同优化。
- **HQ-SAM [27]**：提升 SAM 输出质量，但训练量大且未在退化场景系统评估；本文与之对比，强调在退化上 RobustSAM 的平均 IoU/PA 均领先。
- **AirNet [37]**、**URIE [64]**：通用/分割友好的图像恢复；两者的目标是提升视觉质量而非直接对齐 SAM 内部特征，本文以"feature consistency"思路绕过二次增强阶段。
- **QualNet [28]**、**FIFO [35]**、**SEANet/SE-Net [20]**：分别面向质量无关识别、雾不变特征、通道注意力；本文 AMFG 综合集成这些设计并在频域进一步扩展。
- **All-in-One / MPRNet / HINet 等退化修复**：以人眼感知为目标；本文指出其与 SAM 分割需求的 mismatch，提出面向 SAM 内部表示的对齐训练。
- **Segment Anything 鲁棒性调研 [24, 58, 61, 71]**：多为现象描述；本文首次给出可训练的轻量级修复方案并开源数据集。

## 局限性与未来方向
- 训练退化仅 15 种合成类型，对未知真实退化域的泛化依赖一致性损失假设，极限场景（极端运动模糊、严重压缩）下性能未见深入分析。
- 未讨论多提示（多点/文本）条件下的退化鲁棒性；当前仅报告点提示与框提示。
- 冻结全部 SAM 原始模块的设计虽避免灾难性遗忘，但也可能限制对特定退化域的特征自适应上限。
- 未来可将 AMFG/AOTG 推广至视频分割、多模态提示、3D 分割等下游，并探索无需配对干净图像的无监督一致性版本。

## 研究启发与可借鉴点
- **特征空间一致性蒸馏**：以干净图像作为教师、退化图像作为学生，在 feature/token 层面对齐，可有效在避免灾难性遗忘的同时注入退化鲁棒性，此范式可迁移至其他 foundation model 的微调场景。
- **频域退化抑制**：将傅里叶振幅视为退化风格、相位视为结构内容的解耦处理思路，可复用于去噪/去模糊中的 style removal。
- **极低开销 adapter**：403 MB 参数在 8 卡 30 小时完成训练，证明面向大型基础模型的局部 adapter 是性价比极高的稳健化路径。
- **统一退化数据集**：Robust-Seg 的“清晰 + 多类合成退化 + 保留原 mask"构造方式可作为领域 benchmark 的模板，推动后续工作的公平对比。
- **SAM-prior 下游增益**：本文证明了 segmentation robustness 可直接传导至去雾/去模糊等任务，为“基础模型鲁棒性×应用性能联动评估”提供实证范式。

## 关键术语表
- **RobustSAM**：在冻结原始 SAM 基础上引入抗退化 adapter 的零样本分割模型。
- **AMFG**：Anti-Degradation Mask Feature Generation，通过 IN/BN 融合 + SEC + 傅里叶抑制去除 mask 与互补特征中的退化风格。
- **AOTG**：Anti-Degradation Output Token Generation，轻量 IN+MLP 清理输出 token 中的退化敏感信息。
- **ROT**：Robust Output Token，替换 SAM 原输出 token、仅在该模块上训练以保持零样本能力。
- **Robust-Seg**：68.8 万对 15 种合成退化图像–掩码对构成的鲁棒分割基准数据集。
- **傅里叶退化抑制**：在频域以振幅表征退化风格并用卷积滤除、相位保留以维持结构的去风格化手段。
- **MFC/TC 一致性损失**：分别约束退化与干净图像在 mask 特征层和 token 层的输出对齐。

## 可复现要素
- 数据集：Robust-Seg，来源 7 个公开数据集（LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO、BDD-100k、LIS），论文给出了合成退化配置（15 类）。
- 代码/权重：论文未明确声明开源仓库，需另行查证；权重与代码以作者发布为准。
- 关键超参：训练 40 epoch、lr=5e-4、batch=8、8×A100、13 万步、30 小时；新增可学习参数 403 MB；SAM 冻结；点提示与框提示分别评测。
