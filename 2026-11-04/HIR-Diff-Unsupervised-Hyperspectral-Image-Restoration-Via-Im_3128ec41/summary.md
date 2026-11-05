---
title: "HIR-Diff-Unsupervised-Hyperspectral-Image-Restoration-Via-Im"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:01:39"
field: "高光谱图像恢复与遥感影像处理"
keywords: ["Hyperspectral Image Restoration", "Diffusion Model", "Unsupervised Learning", "Low-Rank Decomposition", "Total Variation Prior", "Noise Schedule"]
innovations: ["提出基于预训练扩散模型的无监督 HSI 恢复框架，将干净 HSI 分解为降采样图像与系数矩阵并分步恢复", "利用 SVD 与 RRQR 从退化观测中鲁棒估计系数矩阵并优选波段索引", "设计含 TV 正则的条件引导与指数噪声调度，在 20 步内实现去噪加速并提升条件扩散稳定性"]
benchmarks: ["Washington DC Mall", "Houston", "Salinas", "HSI Denoising", "Noisy HSI Super-Resolution", "Noisy HSI Inpainting"]
---

# 论文速读：HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models

## 一句话总结
论文提出一种基于预训练扩散模型的无监督高光谱图像（HSI）恢复框架 HIR-Diff，将干净的 HSI 分解为低秩的低光谱维度的"降采样图像"与系数矩阵的乘积，利用 SVD+RRQR 从退化观测中先验估计系数矩阵，再通过带 TV 正则引导函数与指数噪声调度改进的扩散模型恢复降采样图像，从而实现免训练的高光谱去噪、超分辨率与修复任务。

## 研究问题与动机
- **HSI 恢复数据稀缺与泛化难题**：HSI 获取成本高，高质量配对数据有限；既有深度学习方法在未知传感器/场景上泛化性较差。  
- **模型类方法先验主观**：BM4D/ETPTV 等传统方法依赖手工正则，难以充分刻画 HSI 复杂结构，且优化过程耗时，解未必最优。  
- **已有扩散无监督方案仍受限**：如 DDS2M 仅用退化图像自监督训练扩散模型，虽通用性强但不利用已有数据集先验知识，且每个新数据集都需重新训练，耗时巨大（如表中 DDS2M 推理/训练时间显著高于本文）。  
- **直接复用 RGB 扩散模型的适配性不足**：HSI 波段数多且因传感器而异，线性/余弦噪声调度在引入退化条件后的扩散采样中容易出现早期 PSNR 剧烈波动、后期收敛困难等问题，需改进调度与引导策略。

## 核心贡献（创新点）
- **提出无监督扩散恢复框架 HIR-Diff**：利用 HSI 低秩性将恢复拆分为"系数矩阵 + 降采样图像"两步，避免端到端训练，与依赖配对数据的监督方法形成本质区别。  
- **设计带 TV 正则的扩散引导函数**：在 DDIM 反向采样中构造包含数据保真项与 TV 正则项的梯度引导，使扩散过程更贴合 HSI 退化约束并保留边缘，区别于 DDRM/DDS2M 中仅依赖扩散先验或单一保真项的引导方式。  
- **基于 SVD + RRQR 的鲁棒系数矩阵估计**：用 SVD 估计系数矩阵，并用 Rank-Revealing QR 选择波段索引以最大化 |det(Vs)|，降低系数矩阵数值不稳定风险，相较直接最小二乘选带或对 SVD 伪图像的扩散更稳健。  
- **提出指数噪声调度加速扩散采样**：在仅 20 步采样下获得与 50/100 步相近性能，实现约 5× 去噪加速，缓解线性/余弦调度在条件引导下的收敛不稳问题。

## 方法详解
- **整体思路与低秩分解**：设干净 HSI $\mathcal{X}\in\mathbb{R}^{H\times W\times B}$，利用谱相关性假设 $\mathcal{X}=\mathcal{A}\times_3 \mathbf{E}$，其中 $\mathcal{A}$ 为低光谱维"降采样图像"，$\mathbf{E}\in\mathbb{R}^{B\times K}$ 为系数矩阵，$K\ll B$。恢复时分别估计 $\mathbf{E}$ 与 $\mathcal{A}$，再相乘得 $\mathcal{X}_0$。  
- **退化建模**：退化过程 $\mathcal{Y}=\mathbf{H}(\mathcal{X})+\mathcal{Z}$，其中 $\mathbf{H}$ 对而去噪为单位阵、超分含模糊+下采样、修复含随机掩膜；评估使用 PSNR 与 SSIM。  
- **系数矩阵估计（SVD + RRQR）**：对退化 HSI 作 rank-$K$ SVD，$\mathbf{Y}_{(3)}^T=(\mathbf{U}\mathbf{S})\mathbf{V}^T$，按选定波段索引构成 $\mathbf{V}_s$，系数矩阵近似为 $\mathbf{E}=\mathbf{V}\mathbf{V}_s^{-1}$。为避免选取高度相关波段导致 $|\det(\mathbf{V}_s)|$ 过小和 $\mathbf{E}$ 数值不稳定，采用 RRQR 对 $\mathbf{V}^T$ 分解并选取使 $|\det(\mathbf{R}_1)|$ 最大化的前 $K$ 列对应波段作为索引。  
- **扩散引导恢复降采样图像**：采用 DDIM 采样，预测初始值 $\hat{\mathcal{A}}_0=\frac{\mathcal{A}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathcal{A}_t,t)}{\sqrt{\bar{\alpha}_t}}$；噪声预测引入梯度引导：
  $\hat{\epsilon}_\theta(\mathcal{A}_t,t)=\epsilon_\theta(\mathcal{A}_t,t)+s(t)\nabla_{\mathcal{A}_t}\mathcal{L}(\hat{\mathcal{A}}_0,\mathbf{E},\mathcal{Y})$，
  随后按 DDIM 更新 $\mathcal{A}_{t-1}$。  
- **引导损失函数**：$\mathcal{L}(\hat{\mathcal{A}}_0,\mathbf{E},\mathcal{Y})=\lambda\|\mathbf{H}(\hat{\mathcal{A}}_0\times_3\mathbf{E})-\mathcal{Y}\|_F^2+\beta\|\hat{\mathcal{A}}_0\times_3\mathbf{E}\|_{\mathrm{TV}}$，数据保真项保证与观测一致，TV 项发挥去噪与边缘保持作用。  
- **指数噪声调度**：将 $\bar{\alpha}_t$ 设计为先快速上升、后缓慢变化的指数形式并做归一化微小偏移，从而在扩散初期使噪声快速衰减收敛到含结构信息的解，后期更利于细节精细化；采样步数设为 $T=20$。  
- **实现与预训练模型**：使用在大规模 RGB 遥感图像上预训练的扩散模型，未针对目标 HSI 微调；代码开源至 GitHub。

## 实验与结果
- **数据集**：Washington DC Mall（191 波段）、Houston（124 波段）、Salinas（190 波段），各裁剪至 256×256 或 128×128 并去除部分噪声波段。  
- **基线覆盖**：模型类 BM4D、NGMeet、ETPTV、TRPCA 等；监督 DL 类 T3SC、SST、SERT、SFCSR 等；无监督扩散/DIP 类 DDS2M、DIP2d、DIP3d。  
- **去噪结果（代表性最高值）**：WDC Mall σ=30 时，Ours PSNR=42.85 / SSIM=0.97，优于 DDS2M（41.58 / 0.95）；Salinas σ=30 时 Ours 43.79 / 0.96，优于 NGMeet（44.76）但与强基线接近；Houston σ=30 时 Ours 38.13 / 0.94，优于 DDS2M（35.64）。  
- **超分辨率结果（最佳提升显著）**：WDC Mall ×2 时 Ours 36.67 / 0.81，较次优 MCNet（34.55 / 0.74）提升明显；Salinas ×2 时 Ours 39.44 / 0.91，超越 MCNet（36.01 / 0.89）。  
- **修复结果**：WDC 掩码率 0.7 时 Ours 37.90 / 0.87，弱于 DDS2M 38.32 / 0.91，但推理时间显著更低（17s vs 4920s）；Salinas 0.7 时 Ours 40.70 / 0.94 与 DDS2M 40.49 / 0.94 持平，速度仍具优势。  
- **速度对比**：本文在 20 步采样下整体耗时约 13–28s；DDS2M 等无监督训练方法动辄数百至数千秒，体现扩散加速与免训练的实用价值。  
- **消融结论要点**：① 低秩分解策略中选择"SVD 估 E + 扩散恢复 A"显著优于直接使用 SVD 伪图像或最小二乘估计系数；② RRQR 选带相比等间距选带可大幅提升 $|\det(\mathbf{V}_s)|$、压低 $\mathbf{E}$ 最大元素并显著提高 PSNR；③ 引导损失中同时加入数据保真与 TV 项效果最佳，缺任何一项性能下降；④ 指数调度在 20/50/100 步均优于线性与余弦调度，且在步数减少时性能下降更慢。

## 相关工作脉络
- **DDRМ (Denoising Diffusion Restoration Models)**：通过线性退化矩阵 SVD 在谱空间进行扩散恢复，适用于线性逆问题；本文面向多波段 HSI，不能直接套用，改由低秩分解 + 条件引导的扩散恢复降采样图像。  
- **DDS2M (Self-supervised Diffusion Spatio-Spectral Model)**：仅用退化 HSI 自监督训练扩散模型，泛化性好但每个数据集重复训练耗时；本文不重新训练扩散模型，借助预训练先验与低秩分解显著提速。  
- **Deep Hyperspectral Prior (DIP2d/DIP3d)**：基于单幅图像无监督深度先验做去噪/超分/修复；属于"退化图像 + 网络迭代优化"范式，训练慢且易受初始化和架构影响，本文以预训练扩散先验替代网络训练路径。  
- **传统模型类方法 (BM4D/NGMeet/ETPTV/TRPCA/TRLRF/S2NTNN/HLRTF)**：依赖手工先验与张量低秩/稀疏正则，泛化与细节恢复受限；本文用数据驱动扩散先验并结合低秩约束，取得更高 PSNR/SSIM。  
- **监督 HSI 超分/去噪方法 (SFCSR/SSPSR/MCNet/RFSR/PDENet/T3SC/SST/SERT/MACNet)**：在特定数据集上表现优异但跨域泛化有限；本文走无监督路线，强调对 unseen 传感器/任务的适应能力。  
- **先验遥感扩散工作 (如 pansharpening 无监督低秩扩散、RemoDiff 等)**：启发本文利用预训练扩散与低秩分解结合；差异在于面向通用 HSI 恢复三类任务，并引入 TV 引导与指数调度改进条件扩散稳定性与速度。

## 局限性与未来方向
- **预训练模型域适配**：当前使用 RGB 遥感预训练扩散模型，跨模态/跨分布差异可能在极端退化或特殊纹理场景下影响细节生成。  
- **低秩假设边界**：HSI 在强噪声或高压缩掩码下局部低秩性可能被破坏，RRQR 选带与固定 $K$ 的选择对退化程度敏感，未见自适应 $K$ 或鲁棒秩估计讨论。  
- **引导强度与调度超参依赖**：指数调度中的 $k,\epsilon$ 及引导权重 $\lambda,\beta,s(t)$ 需人工设定；不同传感器与任务可能需要重新调参。  
- **修复任务相对 DDS2M 仍有小幅差距**：在部分高掩码率场景下，PSNR/SSIM 略逊于 DDS2M，显示纯扩散先验在极度缺失信息下的补全能力仍可加强。  
- **泛化与复杂度权衡**：虽显著快于无监督训练方法，但 20 步采样相对于单前向网络仍然偏慢，未来可进一步压缩步数或联合蒸馏加速。

## 研究启发与可借鉴点
- **"低秩分解 + 预训练扩散"的无监督范式**：将多通道/多模态数据拆分为低频成分与系数矩阵，可在免训练前提下复用已有扩散先验，适用于 SAR、医学影像等多维数据恢复。  
- **条件引导损失的设计思路**：将数据保真项与 TV 正则结合并在扩散梯度中应用，对高光谱/视频序列这类需要兼顾频域一致性与空间边缘的任务具有迁移价值。  
- **RRQR 波段/通道选择策略**：通过最大化子矩阵行列式来选取信息独立度更高的通道子集，可推广到多光谱信道选择、传感器校准与压缩感知编码设计。  
- **指数噪声调度用于条件扩散**：在引导强度较大、起始噪声与条件图像差异显著的场景中，先快收敛后慢精修的调度更稳，可作为通用扩散加速经验复用。  
- **与团队的结合机会**：可将本框架引入低资源多光谱/高光谱退化场景（如卫星遥感、显微光谱），或把 RRQR 选带与自适应秩估计结合，探索免训练多任务统一恢复器。

## 关键术语表
- **Hyperspectral Image (HSI)**：在宽谱范围内采集的连续多波段图像，兼具空间与光谱信息，常见于遥感与矿检。  
- **Diffusion Model / DDIM**：通过学习逐步加噪与去噪过程进行生成的概率模型；DDIM 是非马尔可夫加速采样变体。  
- **Low-rank Decomposition / Tensor Product**：将高维数据表示为低秩因子乘积，HSI 常用 $\mathcal{X}=\mathcal{A}\times_3\mathbf{E}$ 表示。  
- **SVD (Singular Value Decomposition)**：将矩阵分解为正交矩阵与奇异值矩阵的乘积，用于提取主要能量方向与估计系数矩阵。  
- **RRQR (Rank-Revealing QR)**：通过列置换揭示矩阵数值秩的 QR 分解，这里用于选取信息独立性更强的波段索引。  
- **Total Variation (TV) Prior**：利用图像梯度稀疏性进行正则化，常用于去噪与边缘保持。  
- **Noise Schedule**：控制每一步噪声强度的函数，本文提出指数型调度以适配条件扩散的收敛节奏。  
- **PSNR / SSIM**：图像恢复常用的像素级质量与结构相似度评价指标。

## 可复现要素
- **数据集**：Washington DC Mall、Houston、Salinas 等公开遥感 HSI 数据集；论文提及 publicly available，代码链接已给出。  
- **代码/权重**：代码开源在 https://github.com/LiPang/HIRDiff；使用在 RGB 遥感图像上预训练的扩散模型，论文未提供该预训练权重的直接链接，需查阅引用 [10] 或其官方仓库。  
- **关键超参**：采样步数 $T=20$；rank-$K$ SVD 与 RRQR 选带；指数噪声调度参数 $k,\epsilon$；引导损失权重 $\lambda,\beta$ 与引导强度 $s(t)$；论文实验细节中对后几项仅作说明性描述，具体数值建议参考开源代码与附录（如有）。  
- **硬件环境**：所有 DL 基线使用 NVIDIA GeForce RTX 3090 训练；本文方法主要为推理阶段扩散采样。
