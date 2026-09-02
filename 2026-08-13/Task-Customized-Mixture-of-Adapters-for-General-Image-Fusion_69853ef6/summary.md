---
title: "Task-Customized-Mixture-of-Adapters-for-General-Image-Fusion"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhu_Task-Customized_Mixture_of_Adapters_for_General_Image_Fusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:48:28"
field: "计算机视觉-底层视觉"
keywords: ["图像融合", "MoE", "参数高效微调", "通用模型", "适配器", "提示学习"]
innovations: ["将MoE的稀疏专家机制与PEFT的Adapter结合，提出TC-MoA实现通用图像融合", "设计互信息正则化约束多源特征互补性，精准控制主导强度偏差", "构建可线性变换的Prompt控制接口，实现融合强度的显式调控"]
benchmarks: ["LLVIP", "MEFB", "MFIF"]
---

# 论文速读：Task-Customized Mixture-of-Adapters for General Image Fusion

## 一句话总结
本文提出一种**任务定制混合适配器（TC-MoA）**框架，基于MoE思想将预训练ViT作为冻结骨干，通过任务特定的路由网络与共享适配器银行动态生成融合提示，实现多模态（VIF）、多曝光（MEF）和多焦点（MFF）图像融合的**统一建模**，仅增加2.8%可学习参数即达到SOTA性能，并展现出对未见任务的卓越可控性与泛化能力。

## 研究问题与动机
1. **跨任务融合机制差异大**：MEF注重平衡融合亮度与结构；VIF侧重红外强度与可见纹理的互补；MFF则是极端不均衡的极化选择（每个像素仅依赖单一源），现有单一任务方法难以统一。
2. **统一模型面临"通病"困境**：已有通用融合方法[20, 41, 51]要么陷入主导任务偏差（如IFCNN偏向MFF），要么为追求跨任务共性而牺牲任务个体性，导致次优性能。
3. **效率与灵活性的矛盾**：预训练大模型参数量庞大，全量微调成本高；而传统PEFT方法（Adapter/Prefix-tuning）多面向高层语义任务，未在底层像素级融合任务中验证。
4. **缺乏可控的融合强度调节机制**：不同应用场景下用户对融合结果的主导倾向需求不同（如医疗图像中可能需突出病灶或背景），现有方法难以提供显式控制接口。

## 核心贡献（创新点）
1. **首次将MoE范式引入通用图像融合**：将每个Expert视为高效微调Adapter，通过任务特定路由网络动态组合，实现"一模型多任务自适应"，区别于U2Fusion/DeFusion等直接学习共性特征的方法。
2. **提出互信息正则化（MIR）约束Adapter提示**：通过最小化|prompt_x + prompt_y - 1|确保多源特征的互补性，使模型能精准识别不同源图像的主导强度比例，而非简单平均或选择。
3. **设计任务定制的损失函数体系**：VIF采用MaxPixel/MaxGrad保留高低频；MEF采用AvgPixel+MaxGrad维持平均亮度；MFF采用MaskPixel/MaskGrad强制单源决策，避免散焦边缘伪影。
4. **构建可线性变换的Prompt控制接口**：通过缩放因子α与偏移因子β对Prompt进行仿射变换，实现融合强度的显式调控（如"TC-MoA Light"更偏欠曝光源），为下游应用提供灵活性。
5. **参数量高效**：基于MAE-Large骨干（339M参数），仅新增9.58M可学习参数（约2.8%），显著低于全微调成本，且在Base版本（115M总参）上同样达到有竞争力结果。

## 方法详解

### 整体架构
- **骨干网络**：冻结的预训练Vision Transformer（ViT-Base/Large，源自MAE），包含Encoder（24层）与Decoder（8层），每τ=4层插入一个TC-MoA模块。
- **TC-MoA组成**：任务特定路由银行{G^V, G^E, G^F} + 共享适配器银行{A_1,...,A_N}（N=4）+ 提示驱动融合层F。

### 提示生成（Prompt Generation）
1. **特征拼接与降维**：
   $$f_x = E_j(X), \quad f_y = E_j(Y)$$
   $$\Phi = L(\text{Cat}(f_x, f_y))$$
   其中L由线性层+归一化组成，避免高维直接计算带来大量参数。

2. **Top-K路由选择**（K=2）：
   $$G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g + \mathcal{N}(0,1) \cdot \text{Softplus}(x \cdot W_{noise})))$$
   噪声项增强路由探索能力；不同任务的路由分布显著不同（如MFF路由更极化）。

3. **加权聚合生成Prompt**：
   $$\text{prompt} = \text{GAP}(\text{Sigmoid}(\sum_{i=1}^{N} G(\Phi)_i \cdot A_i(\Phi)))$$
   Prompt维度为$pH \times pH \times 2$，值域(0,1)，分别对应两源的重要性权重。

### 提示驱动融合（Prompt-Driven Fusion）
$$h_x = \text{prompt}_x \cdot f_x + S_x$$
$$h_y = \text{prompt}_y \cdot f_y + S_y$$
$$f_{TC-MoA} = \mathcal{F}(h_x + h_y)$$
其中$S_x, S_y$为源无关的可学习嵌入（引入源相关偏置）；$\mathcal{F}$包含卷积层以减少棋盘artifact并对齐后续Transformer解空间。

残差连接：
$$f'_x = \lambda_f f_x + (1-\lambda_f) f_{TC-MoA}, \quad \lambda_f \text{初始化为0.5}$$

### 互信息正则化（MIR）
$$\min |\text{prompt}_x + \text{prompt}_y - 1|$$
强制两源Prompt之和趋近于1，确保"此消彼长"的互补关系，避免冗余信息同时保留。

### 任务定制损失
- **VIF**：$\mathcal{L}_{MaxPixel}$ + $\mathcal{L}_{MaxGrad}$（保留最强高低频）+ $\mathcal{L}_{ssim}$ + $\mathcal{L}_{Pixel}$
- **MEF**：$\mathcal{L}_{AvgPixel}$（平均亮度）+ $\mathcal{L}_{MaxGrad}$ + $\mathcal{L}_{mefssim}$（专为MEF设计）
- **MFF**：$\mathcal{L}_{MaskPixel}$ + $\mathcal{L}_{MaskGrad}$（逐patch仅选单源计算损失，防止散焦边缘残留）
- **辅助损失**：借鉴MoE的$\mathcal{L}_{aux}$防止Adapter学习失衡

## 实验与结果

### 数据集
| 任务 | 训练集 | 测试集 |
|------|--------|--------|
| VIF | LLVIP (12025对) | LLVIP (70对随机采样) |
| MEF | SCIE (589对) | MEFB (100对) |
| MFF | RealMFF + MFI-WHU | MFIF协议 |

### 主要定量结果

**VIF (LLVIP)**：
- TC-MoA在8项指标中4项最优（红色加粗）：Q_cv=423.773↓、EN=7.428↑、MS-SSIM=0.949↑、Q_w=0.858↑
- 超越所有通用方法（U2Fusion/DeFusion/IFCNN）及多数任务专用方法（CDDFuse DD FM等）
- Q_cv较CDDFuse提升**14.4%**（人类感知质量）

**MEF (MEFB)**：
- MEF-SSIM=0.964↑（提升最显著），PSNR=57.213↑，Q_p=0.598↑
- 超越所有通用方法；与任务专用MEFNet相当但MEF-SSIM高出**5.4%**

**MFF (MFIF)**：
- NMI=0.875↑、MI=6.695↑、Q_cb=0.718↑、Q_cv=36.512↓（全部最优）
- 显著优于监督方法IFCNN（NMI +3.3%、Q_cv降低17.7%即感知质量更好）

### 消融实验关键结论
- **多Adapter vs 单Adapter**：MA配置在所有任务上全面优于SA（VIF的Q_abf从0.5997→0.6007，MEF从0.6362→0.6449）
- **TC-MoA vs 其他PEFT**：超越AdaptFormer（VIF的Q_abf从0.531→0.576，MEF从0.574→0.604）
- **骨干尺寸**：Base版本已具竞争力，Large版本进一步提升

### 效率对比
| 模型 | 总参数(M) | 可训练参数(M) | FPS(VIF 640×512) |
|------|-----------|---------------|-------------------|
| U2Fusion | 1.32 | 1.32 | 4.72 |
| DDFM | — | 1.78 | 0.01 |
| TC-MoA-Base | 115.40 | 3.87 | 3.33 |
| TC-MoA-Large | 348.70 | 9.58 | 1.60 |

经Shifted Windows优化后，Base/Large分别加速178%/167%，FPS可接受。

## 相关工作脉络
1. **通用图像融合方法**：U2Fusion[41]采用自监督学习跨任务共性但忽视个体性；DeFusion[20]同样忽略任务差异；IFCNN[51]虽通用但存在主导任务偏差（MFF偏向）。本文通过MoE动态路由实现"共性+个性"的统一。
2. **参数高效微调（PEFT）**：LoRA[10]、Prefix-tuning[18]、AdaptFormer[4]聚焦高层视觉/语言任务；本文首次将其拓展至底层像素级融合，且Adapter作为"提示"而非直接特征调制。
3. **MoE架构**：Shazeer等[33]提出Sparse-gated MoE；Switch Transformers[7]、GS-Shard[15]用于大模型缩放；Multi-gate MoE[24]处理多任务关系。本文借鉴其思想但将Expert替换为Adapter，适配冻结骨干的微调场景。
4. **任务专用融合**：Densefuse[31]、CDDFuse[54]（VIF）；MEFNet[28]、MEF-GAN[43]（MEF）；IFCNN[51]、MFF-GAN[46]（MFF）。本文证明统一模型可匹敌甚至超越专用方法。
5. **融合评估指标**：信息论类（EN、MI、FMI）、结构类（SSIM、MS-SSIM）、感知类（Q_cv、Q_cb）、梯度类（Q_abf、Q_g）。本文按任务定制评估体系。

## 局限性与未来方向
1. **训练数据量依赖**：VIF训练仅12K对，MEF仅589对，大规模泛化能力待进一步验证；MFF使用真实数据集但合成数据可能带来域偏移。
2. **任务边界假设**：路由网络需预先指定任务类型（VIF/MEF/MFF），对未知任务的零样本路由仍依赖人工调参（α, β），自动化任务识别尚缺。
3. **解码器深度限制**：仅Decoder后8层插入TC-MoA，浅层特征融合的潜力未充分挖掘；Encoder-Decoder对称设计可能带来收益。
4. **计算开销仍较高**：虽经优化，TC-MoA-Large的1.6 FPS仍难满足实时应用；Base版3.33 FPS对边缘设备仍吃力。
5. **仅实验三类融合**：未探索红外-紫外、SAR-可见光等其他模态组合，通用性边界有待扩展。

## 研究启发与可借鉴点
1. **MoE + PEFT的结合范式**：将MoE的"动态路由+稀疏激活"与Adapter的"参数高效微调"结合，可作为通用视觉基础模型适配的新思路，值得迁移至其他底层视觉任务（去噪、超分、去模糊）。
2. **互信息正则化的简洁性**：MIR通过一行公式约束互补性，避免了复杂的对抗训练或分解模块，计算开销极低且效果显著，可推广至多源特征融合的任何场景。
3. **Prompt的可控变换接口**：α-β仿射变换提供的显式控制能力为下游应用（如医疗影像中医生希望突出病灶还是背景）提供了实用价值，可设计更多维度的控制因子。
4. **任务定制损失的设计逻辑**：针对MFF的"单源决策"、MEF的"平均亮度"、VIF的"最大强度"分别设计不同损失，体现了"统一框架+任务适配"的平衡策略，可作为通用融合任务设计的参考模板。
5. **Top-K路由的噪声注入技巧**：在路由计算中加入$\mathcal{N}(0,1)$噪声增强探索，避免路由崩溃到单一Expert，对多任务学习的稳定性有参考价值。

## 关键术语表
- **General Image Fusion（通用图像融合）**：在同一模型中统一处理多种图像融合任务（VIF/MEF/MFF等），避免为每个任务单独设计网络。
- **Mixture of Experts (MoE)**：将多个子网络（Expert）与门控路由结合，根据输入动态选择激活Expert，提升容量同时控制计算量。
- **Task-Customized Mixture of Adapters (TC-MoA)**：本文提出的核心模块，将MoE的Expert替换为Adapter，通过任务特定路由动态组合生成融合提示。
- **Mutual Information Regularization (MIR)**：约束两源Prompt之和趋近于1的正则化项，确保融合时互补信息保留而非冗余叠加。
- **Dominant Intensity Bias（主导强度偏差）**：不同融合任务对源图像重要性的偏好差异，如MFF极化、MEF均衡、VIF中等偏置。
- **Parameter-Efficient Fine-Tuning (PEFT)**：仅微调少量参数即可适配预训练模型到下游任务的技术，如Adapter、LoRA、Prefix-tuning。
- **Prompt-Driven Fusion（提示驱动融合）**：利用生成的Prompt作为权重对多源特征进行加权调制，实现自适应信息选择。
- **Source Embedding（源嵌入）**：与输入无关的可学习参数，为不同源图像引入特征偏置，增强源相关性表达。

## 可复现要素
- **数据集**：LLVIP（公开）、SCIE（公开）、MEFB（公开）、RealMFF（公开）、MFI-WHU（公开）
- **代码**：已开源，URL https://github.com/YangSun22/TC-MoA
- **权重**：预训练MAE模型为公开权重；TC-MoA训练权重随代码提供
- **关键超参**：N=4（适配器数量）、K=2（Top-K路由激活数）、τ=4（TC-MoA插入间隔）、λ_f=0.5（初始残差权重）、学习率=1.5×10⁻⁴、batch size=3、训练epoch=20、输入尺寸=448×448
- **骨干模型**：MAE-Large（或Base）冻结，GAN loss用于重建
