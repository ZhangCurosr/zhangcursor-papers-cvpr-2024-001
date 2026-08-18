---
title: "RobustSAM: Segment Anything Robustly on Degraded Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:16:26"
field: "鲁棒计算机视觉"
keywords: ["图像分割", "退化鲁棒性", "Segment Anything Model", "零样本分割", "频域退化抑制", "基础模型适配"]
innovations: ["冻结原始SAM参数并仅添加AOTG/AMFG轻量模块实现退化鲁棒零样本分割", "在频域去除退化风格（振幅）同时保留结构信息（相位）的傅里叶退化抑制策略"]
benchmarks: ["MSRA10K", "LVIS", "NDD20", "STREETS", "FSS-1000", "COCO", "BDD-100k", "LIS"]
---

# 论文速读：RobustSAM: Segment Anything Robustly on Degraded Images

## 一句话总结
本文提出RobustSAM，在冻结原始SAM参数的基础上，通过新增抗退化Token生成模块(AOTG)和抗退化Mask特征生成模块(AMFG)，使SAM在低质量图像上仍能保持高精度的零样本分割能力，同时不损害其在清晰图像上的性能。

## 研究问题与动机
- SAM等基础分割模型在低光照、噪声、模糊、恶劣天气、压缩伪影等退化条件下性能显著下降，生成的掩码质量差，影响下游任务。
- 现有图像复原方法面向人类视觉感知优化，并不保证提升分割模型（如SAM）的性能。
- 直接微调SAM解码器或集成新解码器会严重损害零样本泛化能力；盲目用退化图像微调SAM还可能导致灾难性遗忘。
- SAM已被用于单图像去雾、去模糊等下游任务作为结构先验，但其退化鲁棒性不足限制了这些任务的实际应用价值。

## 核心贡献（创新点）
1. **提出RobustSAM框架**：在冻结原始SAM全部参数的同时，仅引入少量可学习模块（AOTG、AMFG、ROT），显著提升对15种合成退化的鲁棒性，且保持零样本能力。与微调SAM全参数/解码器的本质区别在于不破坏预训练特征空间。
2. **设计抗退化Mask特征生成模块(AMFG)**：结合Instance Normalization、Batch Normalization并行分支、注意力加权融合以及傅里叶退化抑制（将退化视为风格信息并在频域去除），使mask特征和互补特征与清晰图像特征保持一致。与单纯在空间域做增强方法的本质区别是引入了频域退化抑制。
3. **设计抗退化Output Token生成模块(AOTG)**：使用轻量级多实例归一化+MLP过滤token中的退化敏感信息，配合Token一致性损失约束。与直接微调原始output token的本质区别是不改变SAM原始token参数，仅通过后置模块对齐。
4. **构建Robust-Seg数据集**：整合7个数据集（LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO）共688K图像-掩码对，覆盖15种退化类型，为退化鲁棒分割建立新基准。
5. **验证下游任务增益**：证明RobustSAM作为先验可显著提升SAM-based去雾（PSNR +1.48dB）和去模糊（PSNR +1.86dB）任务性能。

## 方法详解
**整体架构**：冻结原始SAM（Image Encoder、Prompt Encoder、Mask Decoder全部参数不变），额外引入5个可学习组件（图中紫色部分）：AOTG模块、AMFG模块、ROT（微调后的output token）、Feature Fusion模块、以及相应的损失计算路径。

**训练流程**：
1. 对清晰图像施加15种合成退化之一（含恒等变换）得到退化图像。
2. 退化图像经Image Encoder提取特征后，通过原始SAM层（冻结）得到mask特征$F_{MFD}$、互补特征$F_{CFD}$和output token $T_{RO}$。
3. AMFG模块处理$F_{MFD}$和$F_{CFD}$：
   - **归一化并行分支**：Instance Normalization（去除退化风格）与Batch Normalization（保留细节）并行，再经attention机制融合。
   - **特征增强**：融合结果与原始输入沿通道拼接，加入SE channel attention。
   - **傅里叶退化抑制**：对幅值分量做1×1卷积去除退化风格，保留相位保持结构，再逆变换回空间域，得到$\hat{F}_{MFD}$和$\hat{F}_{CFD}$。
4. AOTG模块处理$T_{RO}$：多实例归一化层+单层MLP，得到$\hat{T}_{RO}$。
5. Feature Fusion模块融合$\hat{F}_{MFD}$、$\hat{F}_{CFD}$生成最终robust mask feature。
6. 平行地，原始清晰图像经冻结SAM得到$F_{MFC}$、$F_{CFC}$、$T_{OC}$。

**损失函数**：
- **Mask Feature Consistency Loss**：$\mathcal{L}_{MFC} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2$，确保退化特征与清晰特征对齐。
- **Token Consistency Loss**：$\mathcal{L}_{TC} = \|\hat{T}_{RO} - T_{OC}\|_2$，确保token一致。
- **Segmentation Loss**：$\mathcal{L}_{Seg} = \mathcal{L}_{Dice}(P,G) + \lambda_3 \mathcal{L}_{Focal}(P,G)$。
- **总损失**：$\mathcal{L}_{Overall} = \mathcal{L}_{MFC} + \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{Seg}$。

**推理**：仅使用RobustSAM（上半部分），输入退化图像直接输出分割掩码，无额外开销。

## 实验与结果
**数据集**：
- 训练：MSRA10K、ThinObject-5k、LVIS完整训练集 + 15种退化增强。
- 验证：MSRA10K、LVIS测试集。
- 零样本测试：NDD20、STREETS、FSS-1000、COCO（合成退化）；BDD-100k、LIS（真实世界退化：低光照、模糊、雨、雪等）。

**主要结果（点提示，见Table 2-6）**：
- **MSRA10K（合成退化）**：RobustSAM平均IoU = 0.8616，较SAM（0.8207）提升+4.09个百分点；退化场景IoU 0.8609 vs SAM 0.8194。
- **LVIS（框提示，合成退化）**：平均IoU = 0.7511，较SAM（0.7346）提升+1.65个百分点。
- **NDD20+STREETS+FSS-1000（零样本，合成退化）**：平均IoU = 0.8216，较SAM（0.8000）提升+2.16个百分点；退化场景IoU 0.8195 vs SAM 0.7981。
- **COCO（零样本，框提示，合成退化）**：AP = 0.5130，较SAM（0.5002）提升+1.28个百分点。
- **BDD-100k + LIS（零样本，真实退化，点提示）**：IoU = 0.3717，较SAM（0.3056）提升+6.61个百分点；Dice = 0.8926，较SAM（0.3837）大幅提升。
- **最强提升**：在真实退化数据集（BDD-100k+LIS）上IoU相对SAM提升约**21.6%**（从0.3056到0.3717）。

**消融实验（Table 7）**：
- 微调SAM全参数：IoU暴跌至0.1871，验证了冻结原始参数的重要性。
- 各模块贡献：AMFG > AMFG-F > +AOTG > +ROT，全部模块叠加（ALL）达到最佳IoU=0.3717。

**下游任务增益（Table 8）**：
- 去雾（SOTS）：PSNR从21.677提升至23.159（+1.482），SSIM从0.8451提升至0.8685。
- 去模糊（GoPro）：PSNR从27.491提升至29.351（+1.860），SSIM从0.9066提升至0.9229。

**训练效率（Table 1）**：
- 可学习参数仅403 MB（vs SAM 1250 MB），8×A100训练30小时，推理FPS 2.80（vs SAM 2.90），几乎无额外延迟。

## 相关工作脉络
1. **SAM [29]**：本文基础模型，提供零样本分割能力；本文不修改其任何参数，仅在其上方增加抗退化模块。
2. **HQ-SAM [27]**：高质量分割改进版，通过解码器增强提升掩码质量；本文与之对比，证明在退化场景下RobustSAM更优。
3. **URIE [64]**：面向识别任务的通用图像增强方法；本文将其与SAM组合为基线（URIE+SAM），证明端到端特征对齐方法优于先复原再分割的两阶段策略。
4. **AirNet [37]**：通用图像恢复SOTA方法；同样与SAM组合为基线（AirNet+SAM），体现"为视觉感知复原≠为分割复原"的核心观点。
5. **QualNet [28]、FIFO [35]**：针对特定退化（质量不可知/雾）的分割方法；本文指出这类方法仅处理单一退化类型，缺乏多退化通用性。
6. **Segment Anything under Corruptions [58]、On the Robustness of SAM [24]**：与本文同主题的评测工作，指出SAM在各类corruption下性能下降，本文提出实质性解决方案而非仅评测。

## 局限性与未来方向
- 退化增强仅使用15种合成退化，虽然能泛化到真实场景（BDD-100k、LIS结果已验证），但未见对未知退化类型的系统性讨论。
- 点提示和框提示均有实验，但未涉及文本提示（text prompt）下的退化鲁棒性验证。
- AOTG模块相对轻量（多实例归一化+单层MLP），对极端严重退化的处理能力未做深入分析。
- 未讨论在视频分割、3D分割等时序/三维应用场景下的鲁棒性。
- 训练依赖合成退化，实际应用中若存在训练分布外的退化类型，泛化能力仍有待进一步验证。

## 研究启发与可借鉴点
1. **"冻结主干+轻量适配器"范式**：对预训练基础模型（如SAM）做专项适配时，冻结原始参数、仅添加少量可学习模块是保持零样本能力的有效策略，可迁移到其他基础模型（如LLM、CLIP）的域适应任务中。
2. **频域退化抑制思想**：将图像退化视为"风格"信息并在频域（傅里叶振幅）去除、保留相位（结构）的策略，可借鉴用于其他感知任务（检测、分类）的退化鲁棒性增强。
3. **跨模态一致性蒸馏**：利用清晰图像的特征作为目标，约束退化图像处理后特征的接近程度（$\mathcal{L}_{MFC}$、$\mathcal{L}_{TC}$），是一种无需退化标签的自监督正则化手段，可推广至其他任务。
4. **面向下游任务的先验增强验证**：不仅评测本身任务指标，还验证对去雾/去模糊等下游任务的增益，为证明方法实用性提供了更有说服力的维度。
5. **大规模合成退化数据集构建**：Robust-Seg整合7个数据集+15种退化=688K样本的做法，为后续退化鲁棒性研究提供了可复用的数据构建模板。

## 关键术语表
- **SAM (Segment Anything Model)**：Meta提出的图像分割基础模型，基于SA-1B数据集预训练，支持点/框/文本提示的零样本分割。
- **AOTG (Anti-Degradation Output Token Generation)**：抗退化Output Token生成模块，通过归一化+MLP过滤token中的退化敏感信息。
- **AMFG (Anti-Degradation Mask Feature Generation)**：抗退化Mask特征生成模块，结合IN/BN、注意力融合与傅里叶退化抑制生成退化不变特征。
- **ROT (Robust Output Token)**：经微调的output token，替代原始SAM的output token以适配退化场景。
- **傅里叶退化抑制**：在频域利用1×1卷积处理振幅（风格/退化）而保留相位（结构），实现退化信息的剥离。
- **LFC (Loss of Catastrophic Forgetting)**：灾难性遗忘，指模型在新数据上训练后丢失原有知识；本文通过冻结SAM参数避免此问题。
- **Robust-Seg**：本文构建的688K图像-掩码对数据集，含15种合成退化，用于训练和评测鲁棒分割模型。

## 可复现要素
- **数据集**：Robust-Seg（688K）由7个公开数据集合成生成（LVIS、ThinObject-5k、MSRA10K、NDD20、STREETS、FSS-1000、COCO）；BDD-100k、LIS、SOTS、GoPro均为公开数据集。
- **代码/权重**：论文未明确声明开源链接（CVPR 2024，需另行核查项目页面）。
- **关键超参**：学习率0.0005，训练40 epoch/130000次迭代，8×A100 GPU，batch size=8；$\lambda_1$-$\lambda_3$为损失权重（论文未给出具体数值，需查阅附录或代码）。
