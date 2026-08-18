---
title: "Blur-aware-Spatio-temporal-Sparse-Transformer-for-Video-Debl"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Blur-aware_Spatio-temporal_Sparse_Transformer_for_Video_Deblurring_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:20:44"
field: "视频复原与增强"
keywords: ["video deblurring", "spatio-temporal transformer", "sparse attention", "blur map", "bidirectional feature propagation"]
innovations: ["非参数化模糊图估计驱动时空稀疏注意力，扩展Transformer时序窗口", "模糊图引导的双向特征传播抑制模糊像素误差累积", "奇偶交替采样的Key/Value时序稀疏策略减少50%计算量"]
benchmarks: ["GoPro", "DVD"]
---

# 论文速读：Blur-aware-Spatio-temporal-Sparse-Transformer-for-Video-Deblurring

## 一句话总结
本文提出BSSTNet，通过光流估计非参数化模糊图，将稠密时空注意力转化为稀疏注意力，并结合模糊图引导的双向特征传播（BBFP），有效扩展了Transformer的时序窗口长度，同时抑制了模糊区域的误差累积，在GoPro和DVD数据集上达到SOTA。

## 研究问题与动机
1. **时空Transformer的时序窗口受限**：现有方法（如VRT、RVRT）因自注意力计算开销大，通常仅使用2帧的短时序窗口，无法充分利用视频序列中远距离帧的信息。
2. **双向特征传播的误差累积问题**：基于光流的特征对齐在模糊帧中因光流不准确会引入模糊像素，并在传播过程中不断累积误差。
3. **模糊区域具有时空稀疏性**：运动模糊通常只出现在视频的局部时空区域，且模糊程度与像素位移量正相关，这为稀疏计算提供了先验依据。

## 核心贡献（创新点）
1. **提出非参数化模糊图估计方法**：基于前后向光流的模长平方和直接计算模糊图，无需额外网络训练，为后续稀疏化提供空间先验。
2. **设计Blur-aware Spatio-temporal Sparse Transformer (BSST)**：通过模糊图筛选Query中的模糊token和Key/Value中的清晰token进行稀疏注意力计算，使Transformer能够使用更长的时序窗口（如48帧）而不显著增加计算量。
3. **提出Blur-aware Feature Alignment (BFA)**：在双向特征传播中引入清晰图作为deformable convolution的基础掩码，阻止相邻帧模糊区域的特征传播，减少误差累积。
4. **在GoPro（PSNR 35.98 / SSIM 0.9792）和DVD（PSNR 34.95 / SSIM 0.9703）上超越SOTA Shift-Net+**，同时FLOPs降低13G，推理速度提升1.6倍。

## 方法详解
**整体架构**（Figure 2）：BSSTNet由三个组件构成：Blur Map Estimation → BBFP → BSST → Decoder。

**1. 模糊图估计**：
- 输入：降采样视频序列的前向后向光流 $\mathbf{O}_{t\to t+1}$ 和 $\mathbf{O}_{t\to t-1}$
- 未归一化模糊图：$\hat{\mathbf{B}}_t = \sum_{i=1}^{2}((\mathbf{O}_{t\to t+1})_i^2 + (\mathbf{O}_{t\to t-1})_i^2)$
- 归一化：$\mathbf{B}_t = \frac{\hat{\mathbf{B}}_t - \min(\hat{B})}{\max(\hat{B}) - \min(\hat{B})}$，清晰图 $\mathbf{A}_t = 1 - \mathbf{B}_t$

**2. BBFP（模糊感知双向特征传播）**：
- 聚合特征：$\hat{\mathbf{F}}_t^j = \text{BFA}(\hat{\mathbf{F}}_t^{j-1}, \hat{\mathbf{F}}_t^{j-1}, \hat{\mathbf{F}}_{t-2}^j, \text{Warp}(\cdot), \mathbf{O}, \mathbf{A})$
- BFA核心：将清晰图 $\mathbf{A}$ 作为额外条件生成deformable convolution的offset和mask，并作为基础mask叠加到DCN输出，确保只传播清晰区域特征。

**3. BSST（模糊感知时空稀疏Transformer）**：
- 将聚合特征soft split为patch embedding，通过模糊图下采样得到窗口级模糊图 $\mathbf{U}$
- **空间稀疏**：设置阈值 $\theta=0.3$，生成空间稀疏掩码 $\mathbf{S} = \text{Clip}(\sum_t \mathbf{Q}_t, 1)$，仅保留模糊窗口参与注意力计算
- **Query时序稀疏**：对每个空间位置选择模糊程度最高的 $K_q$ 个帧（Top-$K_q$策略）
- **Key/Value时序稀疏**：选择清晰区域帧，奇偶BSST交替选择奇偶帧，减少50% K/V计算量
- 最终自注意力：$\text{Attention}(\hat{\mathbf{Y}}_q, \hat{\mathbf{Y}}_k, \hat{\mathbf{Y}}_v) = \text{Softmax}(\frac{\hat{\mathbf{Y}}_q\hat{\mathbf{Y}}_k^T}{\sqrt{C_z}})\hat{\mathbf{Y}}_v$
- 未选中窗口仍应用标准window attention以保持分辨率完整性，最后soft composition合并

## 实验与结果
**数据集**：GoPro（2103训练/1111测试）和DVD（61训练/10测试）

**主要结果**：
- GoPro：PSNR **35.98** / SSIM **0.9792**，超越Shift-Net+（35.88/0.9790）
- DVD：PSNR **34.95** / SSIM **0.9703**，超越Shift-Net+（34.69/0.9690），提升0.26dB/0.0013
- 相比Shift-Net+：FLOPs降低13G（133 vs 146），推理快1.6倍（28ms vs 45ms）

**消融实验**：
- BBFP移除导致PSNR下降0.21dB；BFA替代为SFFA导致PSNR下降0.13dB
- BSST移除导致PSNR下降0.85dB；Top-25%稀疏策略仅需43% FLOPs即可接近全token性能
- 时序长度扩展：SST在TL=60时OOM，BSST可稳定运行至TL=60并获得小幅增益

## 相关工作脉络
1. **BasicVSR++ / RNN-MBP / STDAN / FGST**：基于光流引导的双向特征传播方法，依赖密集对齐易引入模糊像素且存在误差累积，本文BBFP通过模糊图引导选择性对齐加以改进。
2. **VRT / RVRT**：基于时空Transformer的视频恢复方法，受限于自注意力计算复杂度仅使用短时序窗口（2帧），本文BSST通过稀疏化实现更长时序窗口。
3. **Shift-Net+**：当前最强基线之一，采用grouped spatio-temporal shift机制，本文在精度超越的同时降低了计算量。
4. **Fuseformer / Propainter**：采用类似window partition和soft composition的Transformer设计，本文在此基础上引入模糊感知的时空稀疏策略。
5. **RAFT**：光流估计器，本文冻结预训练权重用于模糊图生成，不引入额外可学习参数。

## 局限性与未来方向
1. **模糊图估计依赖光流精度**：非参数化方法在光流估计极不准的场景（如极端模糊或纹理缺失区域）可能失效。
2. **仅针对运动模糊**：模糊图基于光流模长估计，对离焦模糊等非运动类模糊可能不适用。
3. **稀疏策略超参依赖调优**：阈值$\theta$、$K_q$、$K_{kv}$需根据数据集调整，泛化到不同场景可能存在适应性挑战。
4. **未来可探索**：可学习的模糊图估计、对多种模糊类型的统一建模、向其他视频恢复任务（如去噪、插帧）的迁移。

## 研究启发与可借鉴点
1. **稀疏先验驱动的高效注意力设计**：模糊图引导的时空稀疏策略可将密集Transformer的计算复杂度从$O(T\cdot HW)$降至$O(m_s n_s \cdot K_q \cdot hw)$，该方法可迁移至视频超分、视频补全等长时序任务。
2. **无参数先验的轻量集成**：利用光流模长直接构造模糊图无需训练额外网络，参数开销为零，这种"physics-informed prior"思路可推广至其他需要运动先验的视频理解任务。
3. **双向传播中的误差隔离机制**：BFA通过清晰图掩码阻止模糊像素传播，本质上是一种"误差阻断"策略，对于所有基于对齐的多帧融合任务均有借鉴价值。
4. **奇偶交替采样策略**：Key/Value中奇偶BSST交替选择帧，以50%计算代价获得等效长时序感受野，该策略可推广至其他需扩展时序范围的Transformer架构。

## 关键术语表
**Blur Map（模糊图）**：基于光流模长估计的逐像素模糊程度图，数值越高表示该区域运动模糊越严重，本文核心先验信息。
**BBFP（Blur-aware Bidirectional Feature Propagation）**：模糊感知双向特征传播，利用模糊图引导的双向递归特征聚合，减少模糊像素的跨帧传播。
**BFA（Blur-aware Feature Alignment）**：模糊感知特征对齐，在deformable convolution中引入清晰图作为基础掩码，选择性对齐清晰区域特征。
**BSST（Blur-aware Spatio-temporal Sparse Transformer）**：模糊感知时空稀疏Transformer，通过模糊图筛选Query/KV中的关键token进行稀疏自注意力计算。
**Soft Split / Soft Composition**：可微分的重叠patch切分与合并操作，用于将特征划分为patch embedding并聚合回原分辨率。
**Top-K 稀疏策略**：仅保留模糊程度最高（或最低）的K个窗口参与注意力计算，其余窗口被丢弃以降低计算量。
**Deformable Convolution（DCN）**：可变形卷积，根据输入特征自适应调整采样位置，本文利用其生成偏移量和掩码实现选择性对齐。

## 可复现要素
- **数据集**：GoPro和DVD均为公开数据集（GoPro: Nah et al., CVPR 2017；DVD: Su et al., CVPR 2017）
- **代码开源**：论文未明确提及代码开源声明（URL: https://vilab.hit.edu.cn/projects/bsstnet）
- **关键超参**：$\theta=0.3$，patch size $p=4$，stride $s=2$，训练时序长度24帧（推理48帧），$K_q=12$（训练）/24（推理），$K_{kv}=12$（训练）/24（推理）
- **训练配置**：batch size 8，8×A100 GPU，初始学习率$4\times10^{-4}$，Adam优化器，L1 loss，随机crop至256×256
- **光流估计器**：冻结预训练RAFT权重，不随训练更新
