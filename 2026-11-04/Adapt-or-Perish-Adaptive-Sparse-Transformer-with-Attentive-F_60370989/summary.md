---
title: "Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:56:49"
---

# 论文速读：Adapt-or-Perish-Adaptive-Sparse-Transformer-with-Attentive-F

## 一句话总结
提出自适应稀疏Transformer (AST)，通过双分支自适应稀疏自注意力 (ASSA) 与“增强-舒缓”特征精炼前馈网络 (FRFN)，在过滤空间无关交互与通道冗余的同时保障信息完整流动，在去雨纹、去雾、去雨滴等图像恢复任务上刷新SOTA。

## 研究问题与动机
1. **密集注意力引入噪声交互**：标准Transformer对所有token一视同仁地聚合特征，会无意引入无关区域的噪声干扰，降低恢复质量。
2. **密集特征图存在通道冗余**：高维特征图中并非所有通道都携带有效信息，直接套用常规FFN会导致计算浪费且阻碍关键特征表达。
3. **现有稀疏/去冗余方案存在缺陷**：Top-K硬选择（如DRSformer）对超参K敏感；超像素投影（如CODE）仍在投影空间内计算全部token关系；纯ReLU稀疏注意力熵值过低，易造成信息丢失。
4. **缺乏兼顾稀疏性与信息完整性的自适应机制**：如何在去除噪声交互的同时避免过度稀疏导致性能骤降，是高效视觉Transformer设计的核心痛点。

## 核心贡献（创新点）
1. **提出AST模型框架**：将自适应稀疏注意力与通道特征精炼结合，实现空间与通道双域的去冗余，区别于以往仅优化单一注意力形态的工作。
2. **设计ASSA双分支自适应模块**：将平方ReLU稀疏自注意力(SSA)与Softmax密集自注意力(DSA)并行计算，并通过可学习权重自适应融合；与仅依赖固定稀疏激活（如RePU²）或Hard-Top-K的方法本质不同，该设计以软选择缓解过稀疏带来的信息损失。
3. **开发FRFN“增强-舒缓”方案**：利用PConv强化高信息量通道，并通过DWConv+门控矩阵乘法弱化冗余通道；区别于Restormer的GDFN或标准FFN，FRFN在通道维度实现了更有选择性的特征变换。
4. **系统验证与可解释分析**：在雨纹、真实雾、雨滴去除三大任务上取得领先性能，并通过注意力熵分析量化不同机制的信息集中程度，论证双分支设计的必要性。

## 方法详解
- **整体架构**：U型编码器-解码器结构，编码器阶段仅保留FRFN（避免早期阶段注意力的低通滤波特性破坏局部细节），解码器阶段由ASSA+FRFN堆叠，中间插入瓶颈层捕获长程依赖。末端通过残差学习生成修复图像，采用Charbonnier损失训练：$\ell = \sqrt{\|I'-\hat{I}\|^2 + \epsilon^2}$，$\epsilon=10^{-3}$。
- **ASSA（自适应稀疏自注意力）**：将特征图切分为$M \times M$不重叠窗口，共享投影权重生成Q/K/V。密集分支：$DSA = Softmax(QK^T/\sqrt{d}+B)V$；稀疏分支：$SSA = ReLU^2(QK^T/\sqrt{d}+B)V$。自适应融合公式：$A = (w_1 \cdot SSA + w_2 \cdot DSA)V$，其中$w_n = e^{a_n}/\sum e^{a_i}$，可学习参数$a_1,a_2$初始化为1。随着网络加深，模型逐渐赋予稀疏分支更高权重，实现从保信息到强过滤的平滑过渡。
- **FRFN（特征精炼前馈网络）**：公式为$\hat{X}' = GELU(W_1 \text{PConv}(\hat{X}))$，拆分$[\hat{X}_1', \hat{X}_2'] = \hat{X}'$，经Reshape/Flatten转换维度后计算$\hat{X}_r' = \hat{X}_1' \otimes F(\text{DWConv}(R(\hat{X}_2'))) $，最终输出$\hat{X}_{out}' = GELU(W_2 \hat{X}_r')$。PConv仅对部分通道卷积实现“增强”，DWConv门控实现“舒缓”，两者协同完成通道级特征再平衡。

## 实验与结果
- **数据集与基线**：SPAD（去雨纹，对比DDN/RESCAN/MPRNet/Restormer/DRSformer等15项）、AGAN-Data（去雨滴，对比Eigen/Pix2pix/Uformer/WeatherDiff/AWRCP等15项）、Dense-Haze（真实去雾，对比RIDCP/FFA-Net/Restormer/DeHamer/MB-TaylorFormer等15项）。评估指标PSNR/SSIM/NIQE。
- **去雨纹 (SPAD)**：AST-B达到**49.51 dB / 0.9942 SSIM**，较最佳CNN基线(Fu et al., 45.03 dB)提升**+4.48 dB**，较最佳Transformer基线(DRSformer, 48.53 dB)提升**+0.98 dB**；AST-B+达49.72 dB。
- **去雨滴 (AGAN-Data)**：AST-B达到**32.32 dB / 0.935 SSIM**，较最佳方法(AWRCP)提升**+0.39 dB**，较扩散模型(WeatherDiff128)提升**+2.66 dB**；AST-B+达32.45 dB。
- **真实去雾 (Dense-Haze)**：AST-B达到**17.12 dB / 0.55 SSIM**，较最佳CNN(AECR-Net)提升**+1.37 dB**，较近期Transformer方法提升至少**+0.46 dB**；AST-B+达17.27 dB / 0.57。
- **感知质量**：NIQE降至**5.493**，显著优于Uformer(5.749)、Restormer(6.162)等，定性可视化显示对重度雨纹的结构保持能力更强。
- **消融结论**：ASSA较Swin SA/Top-k SA/Condensed SA分别提升0.96/0.76/0.49 dB；单用SSA导致性能下降0.51dB，验证双分支必要性；FRFN较GDFN提升0.77 dB，“增强”与“舒缓”组件分别贡献0.28 dB与0.14 dB。

## 相关工作脉络
1. **SwinIR / Uformer**：采用窗口注意力降低计算复杂度，但未解决窗口内token冗余与噪声交互问题，本文在其基础上引入自适应稀疏机制。
2. **Restormer**：提出通道注意力与GDFN，本文FRFN受其门控思想启发，但通过PConv+DWConv实现了更精细的通道级增强-舒缓变换。
3. **DRSformer**：利用Top-K通道选择过滤低相关token，参数K敏感且仍处理全部token关系；本文放弃硬阈值选择，改为双分支软融合。
4. **CODE**：将特征投影至超像素空间计算注意力，投影空间内仍涉及全量token交互；本文直接在原始窗口内完成稀疏化聚合。
5. **Primer / NLP稀疏Transformer**：早期将平方ReLU用于语言模型稀疏化，直接移植至视觉会导致注意力熵过低(1.543 vs 3.733)；本文通过DSA分支补偿熵值，实现视觉适配。
6. **SPANet / MPRNet**：CNN时代去雨代表性工作，本文在相同benchmark上大幅超越，验证Transformer+稀疏精炼架构的有效性。

## 局限性与未来方向
- **任务特异性**：当前模型针对单一退化类型训练，尚未实现多退化统一恢复框架。
- **极端退化鲁棒性不足**：如图6所示，面对极重度真实退化时仍可能出现伪影或结构失真。
- **未来方向**：构建跨退化类型的统一低质量图像恢复模型；注入领域先验（如暗通道先验用于去雾、Retinex先验用于低光照增强）以提升物理可解释性与泛化能力。

## 研究启发与可借鉴点
1. **“稀疏+密集”双分支自适应权重设计**可作为通用模板，迁移至目标检测、语义分割等下游视觉任务中平衡计算效率与信息完整性。
2. **FRFN的PConv增强+门控舒缓结构**参数开销小、即插即用，可直接替换标准Transformer的FFN模块，值得在图像编辑、超分等任务中验证。
3. **注意力熵分析**提供了一种可量化的解释工具，未来可用于指导稀疏注意力强度调度或动态门控阈值设计。
4. **编码器早期阶段剔除注意力机制**以避免低通滤波破坏局部高频细节，该经验对深层U型恢复网络的层间设计具有普适参考价值。

## 关键术语表
- **AST (Adaptive Sparse Transformer)**：本文提出的端到端图像恢复Transformer模型。
- **ASSA (Adaptive Sparse Self-Attention)**：融合稀疏分支(SSA)与密集分支(DSA)的双分支自适应自注意力模块。
- **SSA (Sparse Self-Attention)**：基于平方ReLU的稀疏注意力，过滤低query-key匹配分数的负向交互。
- **DSA (Dense Self-Attention)**：基于Softmax的标准密集注意力，保障特征流动的充分性。
- **FRFN (Feature Refinement Feed-forward Network)**：采用“增强-舒缓”范式的通道维度特征精炼前馈网络。
- **PConv (Partial Convolution)**：仅对部分输入通道执行卷积操作，用于选择性增强高信息量通道。
- **NIQE (Natural Image Quality Evaluator)**：无参考自然图像质量评价指标，数值越低代表感知质量越佳。
- **Charbonnier Loss**：论文采用的像素级损失函数，对离群值具鲁棒性，$\epsilon$设为$10^{-3}$。

## 可复现要素
- **数据集**：SPAD、AGAN-Data、Dense-Haze（均公开）。
- **代码/权重**：https://github.com/joshyZhou/AST（已开源）。
- **关键超参**：AdamW优化器；初始学习率2e-4，余弦衰减至1e-6；窗口大小8；AST-B配置：通道数C=32，阶段数$N_1$=4，各阶段基础块数$N_2$=[1,2,8,8]，瓶颈块数$N_3$=2；输入patch尺寸128×128；增强翻转数据增强。

<!--META
{"keywords": ["图像恢复", "自适应稀疏注意力", "Transformer", "去雨纹", "去雾", "特征精炼"], "field": "图像恢复/低级别视觉", "innovations": ["双分支自适应稀疏自注意力(ASSA)平衡稀疏性与信息完整性", "增强-舒缓特征精炼前馈网络(FRFN)在通道维度消除冗余", "端到端验证于多类图像退化去除任务并超越SOTA"], "benchmarks": ["SPAD",
