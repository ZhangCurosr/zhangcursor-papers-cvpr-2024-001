---
title: "HIR-Diff-Unsupervised-Hyperspectral-Image-Restoration-Via-Im"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:44:59"
field: "高光谱图像复原与生成模型"
keywords: ["Hyperspectral Image Restoration", "Diffusion Model", "Unsupervised Learning", "Low-rank Decomposition", "Total Variation Guidance", "Noise Schedule"]
innovations: ["基于预训练扩散模型的无监督HSI复原框架HIR-Diff，将HSI分解为约简图像与系数矩阵分别恢复", "提出TV引导函数将传统正则化嵌入扩散采样过程", "设计指数噪声调度实现约5倍加速且保持性能"]
benchmarks: ["Washington DC Mall", "Houston", "Salinas"]
---

# 论文速读：HIR-Diff-Unsupervised-Hyperspectral-Image-Restoration-Via-Im

## 一句话总结
论文提出 HIR-Diff，一种基于预训练扩散模型的无监督高光谱图像（HSI）复原框架，将 HSI 分解为约简图像（reduced image）与系数矩阵的乘积，分别通过改进的条件扩散采样与 SVD+RRQR 估计，并结合 TV 引导函数与指数噪声调度，在去噪、超分辨率、掩码填补三项任务上均优于现有方法，且推理速度显著提升。

## 研究问题与动机
1. **模型基方法先验不足**：传统方法依赖手工先验（低秩、TV 等），难以充分刻画 HSI 复杂结构，且优化过程耗时长、易陷入次优解。
2. **深度学习泛化与数据瓶颈**：现有 DL 方法需要大量配对数据，而 HSI 标注数据稀缺，导致在未见数据集上泛化能力差。
3. **已有无监督扩散方法效率低下**：如 DDS2M 虽能无监督恢复 HSI，但需为每个数据集重新训练一个未训练的 DNN，耗时极长（数百至数千秒）。
4. **HSI 多波段特性使直搬自然图像扩散模型不可行**：DDRM 等方法无法直接处理 HSI 波段数多且各数据集不一致的问题。

## 核心贡献（创新点）
1. **无监督 HSI 复原框架 HIR-Diff**：将 HSI 分解为约简图像与系数矩阵的张量乘积（$\mathcal{X} = \mathcal{A} \times_3 \mathbf{E}$），分别估计后再乘回，本质上是"降维扩散+代数重建"的分步范式，与端到端扩散或纯自监督训练方法不同。
2. **TV 引导函数设计**：在反向采样中引入由数据保真项与全变分正则化构成的引导函数 $\mathcal{L} = \lambda \| \mathbf{H}(\hat{\mathcal{A}}_0 \times_3 \mathbf{E}) - \mathcal{Y} \|_F^2 + \beta \| \hat{\mathcal{A}}_0 \times_3 \mathbf{E} \|_{TV}$，将先验知识嵌入扩散采样，区别于无条件生成或仅靠退化图约束的方法。
3. **SVD + RRQR 系数矩阵估计**：利用 SVD 分解退化图估计系数矩阵 $\mathbf{E} = \mathbf{V}\mathbf{V}_s^{-1}$，并用 Rank-Revealing QR 优化波段索引选择，最大化 $|\det(\mathbf{V}_s)|$ 以提升数值稳定性与信息多样性，比直接选等间隔波段或最小二乘估计更鲁棒。
4. **指数噪声调度加速**：提出 $\bar{\alpha}_t = e^{-kt/T}$ 的指数调度，使扩散初期快速去噪、后期精细恢复，实现约 5 倍加速（仅 20 步即可达到与 100 步线性/余弦调度相当甚至更优的性能）。

## 方法详解
1. **整体流程**（Algorithm 1）：输入退化 HSI $\mathcal{Y}$ → 用 SVD+RRQR 估计系数矩阵 $\mathbf{E}$ → 以 $\mathcal{Y}$ 和 $\mathbf{E}$ 为条件，经 20 步 DDIM 反向采样恢复约简图像 $\mathcal{A}$ → 最终输出 $\mathcal{X}_0 = \mathcal{A}_0 \times_3 \mathbf{E}$。
2. **系数矩阵估计**：对 $\mathbf{Y}_{(3)}^\top$ 做 rank-K SVD 得 $\mathbf{Y}_{(3)}^\top = (\mathbf{U}\mathbf{S})\mathbf{V}^\top$，取 $\mathbf{V}$ 中对应所选波段索引的行组成 $\mathbf{V}_s$，则 $\mathbf{E} = \mathbf{V}\mathbf{V}_s^{-1}$。该推导基于退化算子 $\mathbf{H}$ 在线性且仅作用于空间维的假设下成立。
3. **RRQR 波段选择**：对 $\mathbf{V}^\top$ 做 RRQR 分解 $\mathbf{V}^\top \boldsymbol{\Pi} = [\mathbf{Q}\mathbf{R}_1 \;\; \mathbf{Q}\mathbf{R}_2]$，取前 K 个置换列对应的原始波段索引作为 $\mathbf{V}_s$ 的行索引，使得 $|\det(\mathbf{V}_s)| = |\det(\mathbf{R}_1)|$ 被最大化，避免数值不稳定。
4. **条件扩散采样**：每步先由 DDIM 公式 $\hat{\mathcal{A}}_0 = (\mathcal{A}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta)/\sqrt{\bar{\alpha}_t}$ 估计去噪结果，再计算梯度 $\nabla_{\mathcal{A}_t} \mathcal{L}(\hat{\mathcal{A}}_0, \mathbf{E}, \mathcal{Y})$ 调整噪声预测 $\hat{\epsilon}_\theta = \epsilon_\theta + s(t) \nabla_{\mathcal{A}_t} \mathcal{L}$，最后按 DDIM 更新 $\mathcal{A}_{t-1}$。
5. **指数噪声调度**：$\bar{\alpha}_t = e^{-kt/T}$ 经归一化后映射到 $[\epsilon, 1-\epsilon]$ 范围，相比线性/余弦调度，前期 $\bar{\alpha}_t$ 上升更快（帮助从随机噪声快速收敛到观测数据附近），后期变化平缓（利于细节精修）。

## 实验与结果
- **数据集**：Washington DC Mall ($256 \times 256 \times 191$)、Houston ($256 \times 256 \times 124$)、Salinas ($128 \times 128 \times 190$)，均为公开数据集。
- **评估指标**：PSNR、SSIM；推理时间（秒）。
- **基准方法**：
  - 去噪：BM4D、NGMeet、ETPTV、T3SC、MACNet、SST、SERT、DDS2M
  - 超分辨：DIP2d、DIP3d、SFCSR、SSPSR、MCNet、RFSR、PDENet
  - 掩码填补：TRPCA、TRLRF、S2NTNN、HLRTF、DIP2d、DIP3d、DDS2M
- **主要结果**：
  - **去噪**（σ=30）：WDC Mall 42.85 dB / 0.97（最佳），Houston 38.13 dB / 0.94（最佳），Salinas 43.79 dB / 0.96（最佳）；推理时间 13–25 秒，远低于 DDS2M 的 846–3132 秒。
  - **超分辨**（×4）：WDC Mall 34.68 dB / 0.74（最佳），Houston 30.68 dB / 0.72（最佳），Salinas 37.53 dB / 0.88（最佳）；推理时间 14–28 秒。
  - **掩码填补**（rate=0.7）：WDC Mall 37.90 dB / 0.87（次优），Houston 32.42 dB / 0.82（最佳），Salinas 40.70 dB / 0.94（最佳）；推理时间 16–20 秒。
- **提升幅度**：相比强基线 DDS2M，去噪任务在 WDC Mall（σ=30）上提升 1.27 dB，超分辨 ×4 Salinas 提升约 2.99 dB，同时速度提升约 50–200 倍。
- **消融结论**：
  - SVD 分解策略优于"SVD Only（伪图像）"和"Least Square"两种替代方案（去噪 PSNR 40.73 vs 37.53 vs 28.88）。
  - RRQR 选带（$|\det(\mathbf{V}_s)| = 0.0115$，E 最大值 1.01）远优于等间隔选带（如 (1,48,96) 对应 PSNR 27.85）。
  - 移除 TV 项后去噪 PSNR 从 36.14 降至 34.99；移除所有引导后骤降至 10.12。
  - 指数调度在 20 步下去噪 PSNR 36.01，优于线性（34.34）和余弦（34.61），且在 50/100 步仍保持领先。

## 相关工作脉络
1. **BM4D / NGMeet / ETPTV**：基于手工先验的模型基方法，依赖低秩或 TV 正则化，泛化好但先验主观、计算慢；HIR-Diff 用扩散先验替代手工先验，同时保持无监督特性。
2. **T3SC / SST / SERT / MACNet**：监督式 DL 方法，需大量配对数据，在跨数据集泛化上受限；HIR-Diff 无需微调即可迁移。
3. **DDRM**：将扩散模型引入线性逆问题，通过 SVD 分解退化矩阵后在谱域采样；但无法直接处理 HSI 波段多变且非线性的退化场景，HIR-Diff 通过约简图像+系数矩阵分解绕过此限制。
4. **DDS2M**：首个针对 HSI 的无监督自监督扩散方法，用未训练网络逐数据集学习；HIR-Diff 改用预训练扩散模型，无需额外训练，速度大幅提升。
5. **DIP2d / DIP3d**：基于 Deep Image Prior 的单幅图像复原方法，无需训练但收敛慢；HIR-Diff 结合预训练扩散先验，在速度和质量上均更优。
6. **Rui et al. (Unsupervised Pansharpening)**：本文灵感来源之一，同样利用预训练扩散模型结合低秩分解进行多光谱图像复原，HIR-Diff 将思路扩展到更广义的 HSI 退化任务并新增 TV 引导与指数调度。

## 局限性与未来方向
1. 系数矩阵估计依赖退化图像的 SVD，当退化严重（如极低信噪比、极高缺失率）时，SVD 分解可能不够稳健，影响后续扩散质量。
2. 预训练扩散模型仅在 RGB 遥感图像上训练，对非遥感领域（如生物医学 HSI）的跨域泛化能力未经验证。
3. 波段选择索引 $(i_1, \dots, i_K)$ 的 K 值（约简谱维数）由用户预设，缺乏自适应确定方法，可能在不同 dataset 上需调参。
4. 仅验证了三种标准退化（去噪、超分辨、掩码填补），未覆盖混合退化（如同时存在模糊+噪声+缺失）等更复杂场景。
5. 指数噪声调度的超参数 $k$ 和 $\epsilon$ 需手动设定，不同任务/数据集的最优值可能不同。

## 研究启发与可借鉴点
1. **"分解-独立恢复-重组"范式**：将高维多光谱数据分解为低谱维图像+代数系数矩阵，分别用不同工具处理，可迁移至多光谱/彩色图像修复、视频复原等任务。
2. **RRQR 优化选带策略**：利用矩阵分解的数值秩揭示特性来选择代表性子空间索引，这一思想可用于其他需要选取"代表样本/通道"的问题（如多任务学习中的任务选择、传感器布设优化）。
3. **TV 引导条件扩散**：将传统正则化项转化为扩散采样中的梯度引导，兼具生成模型的质量优势与传统先验的可解释性，可推广至任何需要物理/结构约束的扩散复原任务。
4. **指数噪声调度用于条件扩散**：论文揭示了条件引导下线性/余弦调度的缺陷，提出的指数调度思路可作为一般性加速技巧，适用于其他存在类似收敛波动的条件扩散应用。
5. **预训练扩散+无训练适配**：利用大规模预训练扩散模型作为通用图像先验，结合退化图和代数约束实现零样本/少样本复原，是连接生成模型与传统反问题的高效路线。

## 关键术语表
**Hyperspectral Image (HSI)**：在高光谱范围内采集的多波段图像，每个像素含连续光谱信息，空间-谱维三元张量结构使其具有强低秩特性。
**Reduced Image (约简图像)**：从 HSI 中选取 K 个独立波段构成的低谱维图像（$K \ll B$），其分布与预训练扩散模型的数据分布一致，可直接作为扩散模型输入。
**Coefficient Matrix (系数矩阵)**：表征约简图像各波段与原始 HSI 各波段之间线性关系的矩阵 $\mathbf{E} \in \mathbb{R}^{B \times K}$，通过 SVD 从退化图中估计。
**Rank-Revealing QR (RRQR)**：一种带列置换的 QR 分解，能揭示矩阵的数值秩并使右上三角块的行列式绝对值最大化，用于确定最优波段索引。
**Total Variation (TV) Regularization**：对图像各像素梯度绝对值求和的正则化项，在保持边缘锐利的同时抑制噪声，此处作为扩散引导函数的组成部分。
**Diffusion Posterior Sampling / Guided Diffusion**：在扩散反向采样过程中，通过对预测噪声施加由数据保真和正则化项计算的梯度，使生成结果满足观测约束。
**Exponential Noise Schedule**：以指数形式安排 $\bar{\alpha}_t$ 的噪声调度策略，前期快速增大以减少噪声、后期缓慢变化以精修细节，适配条件引导扩散的收敛特性。
**Mode-3 Tensor-Matrix Multiplication ($\times_3$)**：沿张量第三维（谱维）与矩阵相乘的运算，$\mathcal{X} \times_3 \mathbf{E}$ 将 K 维谱信号映射为 B 维谱信号。

## 可复现要素
- **数据集**：Washington DC Mall、Houston、Salinas，均为公开可用。
- **代码**：已开源，地址 https://github.com/LiPang/HIRDiff。
- **预训练模型**：论文使用在 RGB 遥感图像上预训练的扩散模型（引用 [10]），论文未提供模型权重下载链接，说明需自行获取或下载该预训练权重。
- **关键超参**：扩散采样步数 $T = 20$；约简谱维 $K$（文中 Salinas 使用 $K=3$，其他数据集类似）；引导强度 $s(t)$、$\lambda$、$\beta$（论文未给出具体数值，需从原文或代码补充）；指数调度参数 $k$、$\epsilon$（论文未给出具体数值）。
