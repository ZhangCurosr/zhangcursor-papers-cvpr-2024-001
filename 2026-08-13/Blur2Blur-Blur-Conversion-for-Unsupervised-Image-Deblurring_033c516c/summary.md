---
title: "Blur2Blur-Blur-Conversion-for-Unsupervised-Image-Deblurring"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Pham_Blur2Blur_Blur_Conversion_for_Unsupervised_Image_Deblurring_on_Unknown_Domains_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:21:30"
---

# 论文速读：Blur2Blur-Blur-Conversion-for-Unsupervised-Image-Deblurring

## 一句话总结
本文提出 Blur2Blur，一种插件式无监督图像去模糊框架，通过将目标摄像机捕获的未知模糊图像转换为已知模糊分布（如 GoPro）的模糊图像，再结合预训练监督去模糊网络，实现对特定摄像机的高效去模糊。

## 研究问题与动机
- **场景驱动需求**：实际应用中（手机摄影、工业监控、执法摄像头）往往需要针对特定设备定制去模糊算法，而非泛用型解法。
- **监督模型跨域失效**：在配对数据上训练的 SOTA 去模糊网络（如 NAFNet、Restormer）在面对真实世界 unseen blur 时泛化能力差，容易出现过拟合或输出近似原图的退化现象。
- **配对数据获取困难**：获取严格时空对齐的模糊-清晰配对数据需要分束器、双机同步、色彩标定等高成本硬件，多数商用摄像机并不支持。
- **无监督去模糊瓶颈**：现有无配对学习方法直接学习模糊→清晰映射时，因缺乏高频细节监督，且难以跨越巨大的域偏移，重建质量往往不理想。

## 核心贡献（创新点）
1. **模糊到模糊的任务分解策略**：将高难度的未知域去模糊（$C \rightarrow$ 清晰）拆解为“未知模糊→已知模糊”的域迁移与“已知模糊→清晰”的标准恢复两步，显著降低优化难度。
2. **多尺度模糊翻译网络与联合损失设计**：提出基于对抗与多尺度感知重建的联合训练框架，确保翻译过程仅改变模糊核分布而严格保留原始语义内容。
3. **可控的已知模糊靶域构建方法**：利用模糊核提取器将已知配对数据集的模糊核迁移到目标摄像机捕获的清晰图像上，使靶域 K 与源域 B 在噪声、色彩、分辨率上保持一致，迫使判别器仅关注模糊分布差异。
4. **Plug-and-play 兼容性与真实场景验证**：Blur2Blur 可作为独立模块无缝接入任意预训练去模糊骨干（NAFNet/Restormer），并在手机摄影与 webcam 康复监控等真实场景中验证了实用价值。

## 方法详解
- **问题形式化**：模糊图像建模为 $y = \mathcal{F}_C(x, k) + \eta$，目标求解 $\mathcal{G}_C^*(y) = x$。本文将其分解为 $\mathcal{G}_C^* = \mathcal{G}_{C'}^* \circ G$，其中 $G: y \rightarrow y'$ 将图像从未知域 $C$ 映射到已知域 $C'$（$y' = \mathcal{F}_{C'}(x, k') + \eta'$）。
- **翻译网络 G**：采用 MIMO-UNet 多尺度架构（基于 Pix2Pix 实现），支持粗到细的模糊模式转换。
- **对抗损失**：$\mathcal{L}_{adv}(G,D) = \mathbb{E}_{y \sim \mathcal{K}}[\log D(y)] + \mathbb{E}_{y \sim \mathcal{B}}[\log(1 - D(G(y)))]$，G 与 D 进行 minimax 博弈。
- **梯度惩罚**：$\mathcal{L}_{grad}^D(D) = \mathbb{E}_{\hat{y} \sim \hat{B}}[(\|\nabla_{\hat{y}} D(\hat{y})\|_2 - 1)^2]$，对判别器施加 Lipschitz 约束（WGAN-GP 风格）。
- **感知重建损失**：$\mathcal{L}_{rec}^G(G) = \frac{1}{M}\sum_{i=1}^{M}\frac{1}{t_i}\mathbb{E}_{y_i \sim \mathcal{B}}[||\phi(y_i) - \phi(G(y_i))||_1]$，其中 $\phi(\cdot)$ 为预训练 VGG19 特征提取器，$t_i$ 为归一化元素数。防止 G 篡改图像语义，仅聚焦模糊核替换。
- **总损失**：$\mathcal{L}_{total}^G = \mathcal{L}_{adv} + \lambda_{rec}\mathcal{L}_{rec}$，$\mathcal{L}_{total}^D = -\mathcal{L}_{adv} + \lambda_{grad}\mathcal{L}_{grad}$。
- **靶域 K 构建**：从目标摄像机收集未配对模糊集 $\mathcal{B}$ 与清晰序列 $\mathcal{S}$；利用 Blur Kernel Extractor [37] 从已知配对数据集（如 GoPro/REDS/RSBlur/RB2V）提取模糊核并应用到 $\mathcal{S}$ 上生成 $\mathcal{K}$，保证非模糊属性一致。
- **训练细节**：按拉普拉斯方差对图像排序，前 50% 数据预热，200K 步后逐步扩至全量；Adam 优化器，1M 步（约 3 天，2×A100），前 500K 步学习率固定，之后线性衰减；随机裁剪至 256×256，配合旋转、翻转、颜色抖动。

## 实验与结果
- **数据集与设置**：使用 RB2V_Street、REDS、RSBlur、GoPro 作为未知源域，统一以 GoPro 作为已知目标域（K）。对比基线包括监督类（NAFNet、Restormer）、无配对类（CycleGAN、DualGAN）、泛化类（BSRGAN+NAFNet、RSBlur+NAFNet）。
- **定量结果（Table 2）**：
  - NAFNet + Blur2Blur(GoPro)：RB2V_Street 26.98/0.812，REDS 28.11/0.893，RSBlur 29.00/0.857，较直接应用源域监督模型提升约 **2.20~2.67 dB**。
  - Restormer + Blur2Blur(GoPro)：RB2V_Street 25.97/0.750，REDS 27.55/0.885，RSBlur 28.89/0.850，提升约 **2.12~2.91 dB**。
  - 无监督/泛化基线（CycleGAN、DualGAN、BSRGAN、RSBlur）在真实模糊上均表现较差，PSNR/SSIM 明显落后。
- **消融实验**：B:S 比例实验中，6:4 最优（GoPro-RB2V 26.98，GoPro-REDS 28.11）；比例过高（9:1）会导致清晰内容过拟合，比例过低则模糊特征学习不充分。
- **真实场景验证**：在自建 PhoneCraft（Samsung Galaxy Note 10 Plus 手机视频）和 WritingHands（webcam 手写康复监控）数据集上，结合 RSBlur 预训练模型与 Blur2Blur 后，NIQE 分数降至 8.8~9.8，视觉恢复效果显著优于直接使用预训练模型。

## 相关工作脉络
- **经典盲去卷积**（TV、Hyper-laplacian priors）：依赖线性均匀模糊假设，对真实非线性非均匀运动模糊泛化极差。
- **监督去模糊**（NAFNet、Restormer、SRN、MPRNet）：依赖大规模配对数据，在跨设备/ unseen blur 时性能断崖式下降。
- **合成退化增强**（BSRGAN、RSBlur）：通过数学退化模拟模糊，但无法覆盖真实摄像机的复杂噪声与光学特性，易过拟合合成分布。
- **无配对图像翻译**（CycleGAN、DualGAN）：直接尝试模糊→清晰映射，受限于大跨度域偏移与语义失真，低频细节重建能力有限。
- **本文定位**：避开端到端域鸿沟，采用“模糊核迁移+已知域恢复”的两阶段范式，既复用强监督去模糊模型的能力，又规避配对数据与高难度零样本恢复，属于无监督/域适配去模糊的新分支。

## 局限性与未来方向
- **仍需物理接触目标设备**：虽只需未配对数据，但仍需前往目标摄像机现场采集 B 与 S，无法完全远程自动化。
- **靶域构建依赖核提取质量**：K 数据集的生成依赖 Blur Kernel Extractor [37] 的准确性，若源域模糊核与目标域分布差异过大，可能导致翻译漂移。
- **B:S 比例敏感**：实验显示比例偏离 6:4 会明显损害性能，实际应用需根据数据可得性进行调参。
- **未来方向**：可探索全自动靶域自适应策略以替代人工选择 K；将框架扩展至视频时序去模糊；结合神经辐射场或视频生成先验进一步抑制内容伪影。

## 研究启发与可借鉴点
- **任务分解降维范式**：将“跨域低层恢复”拆分为“分布特征迁移”+“标准下游恢复”，显著降低 GAN 训练难度，可迁移至去噪、超分、去雨等域外适配任务。
-
