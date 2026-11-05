---
title: "Blur2Blur-Blur-Conversion-for-Unsupervised-Image-Deblurring"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pham_Blur2Blur_Blur_Conversion_for_Unsupervised_Image_Deblurring_on_Unknown_Domains_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:57:34"
field: "低层视觉-图像去模糊"
keywords: ["无监督去模糊", "模糊转换", "域自适应", "GAN", "图像复原", "相机定制"]
innovations: ["将去模糊分解为模糊到模糊转换+已知域去模糊两步，降低学习难度", "基于模糊核迁移构建设备特征对齐的已知模糊集，使判别器专注模糊核判别"]
benchmarks: ["RB2V Street", "REDS", "RSBlur", "GoPro", "PhoneCraft"]
---

# 论文速读：Blur2Blur — Blur Conversion for Unsupervised Image Deblurring on Unknown Domains

## 一句话总结
Blur2Blur 提出了一种即插即用的无监督图像去模糊框架，通过训练一个模糊转换器将目标相机捕获的未知模糊图像转化为已知模糊分布的图像，再借助预训练去模糊网络完成最终复原，从而在无需配对数据的前提下实现面向特定相机的定制化去模糊。

## 研究问题与动机
- **实际场景需要相机定制化**：工业质检、执法记录仪、手机摄影等应用要求去模糊算法针对特定相机优化，而非泛化于任意模糊。
- **预训练模型存在域偏移**：在 GoPro/REDS 等数据集上训练的 supervised 去模糊网络（如 NAFNet、Restormer）面对真实相机产生的未见模糊时严重过拟合，性能大幅下降。
- **配对数据采集成本高昂**：获取 blur-sharp 配对数据需要分光镜、高速/低速双相机、时间同步与几何对齐等复杂硬件，许多目标相机不具备此条件。
- **既有无监督方法效果有限**：Cycle-GAN、DualGAN 等直接学习模糊→清晰映射的方法受限于模糊程度差异大、真实模糊模式复杂，难以有效重建细节；广义去模糊方法（BSRGAN、RSBlur）同样无法处理未见模糊。

## 核心贡献（创新点）
1. **提出 Blur2Blur 框架**：将去模糊任务分解为"未知模糊→已知模糊转换" + "已知模糊去模糊"两步，避免直接学习高难度的模糊→清晰映射。
2. **设计模糊-模糊转换的损失体系**：结合对抗损失（WGAN-GP）、多尺度感知重建损失（VGG19）与梯度惩罚，确保转换只修改模糊核而保留图像内容。
3. **基于模糊核迁移的已知模糊集构建**：利用 Sharp 图像与模糊核提取器 [37]，将已知模糊核移植到目标相机拍摄的 Sharp 图像上，生成与未知模糊图像共享设备特征的已知模糊数据集 K。
4. **即插即用兼容任意预训练去模糊骨干**：可与 NAFNet、Restormer 等最强监督去模糊模型无缝结合，实验显示结合后 PSNR 提升最高达 2.91 dB。
5. **真实场景验证**：在 PhoneCraft（三星手机）和 WritingHands（Webcam 手写康复）两个实际应用中证明框架的工程可行性。

## 方法详解
**问题形式化**：设模糊图像 $y = \mathcal{F}_C(x, k) + \eta$，目标为学习去模糊函数 $\mathcal{G}_C^*$。Blur2Blur 将其分解为 $\mathcal{G}_C^* = \mathcal{G}_{C'}^* \circ G$，其中 $G: y \to y'$ 将未知模糊域 $C$ 映射到已知模糊域 $C'$，$\mathcal{G}_{C'}^*$ 为预训练去模糊网络。

**模糊转换网络 G 的训练**：
- **对抗损失**（WGAN-GP）：引入判别器 D 区分真实已知模糊图像与 G 生成的图像，损失函数 $\mathcal{L}_{adv}(G,D) = \mathbb{E}[\log D(y)] + \mathbb{E}[\log(1-D(G(y)))]$，并对 D 施加梯度惩罚 $\mathcal{L}_{grad}^D = \mathbb{E}[(\|\nabla_{\hat{y}} D(\hat{y})\|_2 - 1)^2]$ 保证 Lipschitz 连续性。
- **多尺度感知重建损失**：使用 VGG19 特征提取器在 M 个尺度上计算 $\mathcal{L}_{rec}^G = \frac{1}{M}\sum_i \frac{1}{t_i}\mathbb{E}[\|\phi(y_i) - \phi(G(y_i))\|_1]$，防止 G 篡改图像语义内容。
- **总损失**：$\mathcal{L}_{total}^G = \mathcal{L}_{adv} + \lambda_{rec}\mathcal{L}_{rec}$，$\mathcal{L}_{total}^D = -\mathcal{L}_{adv} + \lambda_{grad}\mathcal{L}_{grad}$，其中 $\lambda_{rec}=0.8$，$\lambda_{grad}=0.005$。

**已知模糊集 K 的构建**：从目标相机收集未配对的模糊图像集 B 和清晰图像集 S，利用模糊核提取器 [37] 从已知 blur-sharp 数据集（如 GoPro）提取模糊核，移植到 S 上生成 K，使 K 与 B 共享设备颜色/分辨率/噪声特征，仅模糊核分布不同，确保判别器专注于模糊核判别。

**训练策略**：G 基于 MIMO-UNet 在 Pix2Pix 框架实现；按 Laplacian 方差对图像排序，前期仅用约 50% 数据训练，200K 迭代后逐步扩展至全量；学习率前 500K 步恒定，之后线性衰减。

## 实验与结果
**数据集**：RB2V Street（11K 真实配对）、REDS（300 段高速视频合成模糊）、RSBlur（13K 真实模糊）、GoPro（3K 配对）作为已知模糊源。

**定量结果（NAFNet + Blur2Blur 最优）**：

| 配置 | RB2V Street | REDS | RSBlur |
|------|------------|------|--------|
| NAFNet w/ GoPro（直接迁移）| 24.78 / 0.714 | 25.80 / 0.880 | 26.33 / 0.790 |
| NAFNet w/ Blur2Blur (GoPro) | **26.98 / 0.812** | **28.11 / 0.893** | **29.00 / 0.857** |
| NAFNet w/ 源域配对数据（上界）| 28.72 / 0.883 | 29.09 / 0.927 | 33.06 / 0.888 |

- 相对直接迁移，Blur2Blur + NAFNet 分别提升 **+2.20 / +2.31 / +2.67 dB** PSNR。
- Blur2Blur + Restormer 在 RSBlur 上提升 **+2.91 dB**，为最大增益。
- 所有无监督/广义基线（CycleGAN、DualGAN、BSRGAN、RSBlur）在所有数据集上均大幅落后。

**消融**：B:S 比例实验中 6:4 最优（Table 3），过多模糊图像导致过拟合。

**实际应用**：PhoneCraft 数据集上 NIQE 得分 8.8–9.8（越低越好）；WritingHands Webcam 场景经过两天训练即显著恢复清晰度。

## 相关工作脉络
1. **经典盲去卷积**（Chan & Wong 1998; Krishnan & Fergus 2009）：假设均匀线性模糊核，在真实非均匀模糊上泛化差。
2. **监督配对去模糊**（NAFNet 2022, Restormer 2022, MIMO-UNet 2021）：依赖 GoPro/REDS 等数据集训练，面对未见相机模糊严重退化。
3. **无监督域适应去模糊**（CycleGAN 2017, DualGAN 2017）：直接学习模糊→清晰映射，因模糊语义变化大且真实模糊模式复杂而效果不佳。
4. **广义去模糊**（BSRGAN+NAFNet 2021, RSBlur 2022）：通过退化增强提升泛化，但 augmentation 仍不足以覆盖真实相机模糊。
5. **模糊核迁移**（Tran et al. 2021 [37]）：从 blur-sharp 对中提取并转移模糊核，本文借用此技术构建已知模糊集 K。
6. **本文定位**：与无监督方法相比不尝试重建丢失细节，而是将任务简化为模糊核转换；与监督方法相比无需配对数据；与广义方法相比针对特定相机定制。

## 局限性与未来方向
- 仍需访问目标相机采集未配对模糊/清晰图像集（B 和 S），对完全黑盒部署场景受限。
- 转换过程可能引入 artifacts，尤其当未知模糊与已知模糊差异过大时。
- 依赖预训练去模糊模型的质量，若 $C'$ 域无良好骨干则整体上限受限。
- 目前主要针对运动模糊，对散焦模糊、压缩模糊等其它退化类型的泛化未验证。
- 未来可将 Blur2Blur 思想迁移至去噪、去雨、超分辨等其他低层视觉任务的域自适应。

## 研究启发与可借鉴点
1. **任务分解策略**：将高难度的模糊→清晰映射分解为"模糊→模糊转换 + 已知域去模糊"两步，降低学习难度，此思路可迁移至其他复原任务的域自适应。
2. **感知重建而非像素重建**：用 VGG19 多尺度感知损失替代 L1/L2 损失保护语义内容，是图像转换任务中平衡 fidelity 与 style 的有效实践。
3. **渐进式训练调度**：按模糊程度排序图像、初期仅用难样本训练，逐步扩展至全量，对 GAN 训练稳定性有显著帮助。
4. **设备特征对齐的已知域构建**：通过 blur kernel transfer 确保 K 与 B 共享除模糊核外的所有设备特征，使判别器专注模糊核判别，这一设计避免了域偏移干扰。
5. **即插即用范式**：框架与任意预训练去模糊骨干兼容，为后续研究者提供了模块化扩展的基础设施。

## 关键术语表
- **Blur2Blur**：本文提出的无监督去模糊框架，通过模糊到模糊的转换实现未知域去模糊。
- **未知模糊域（C）**：目标相机产生的、未被预训练模型见过的模糊分布。
- **已知模糊域（C'）**：已有强预训练去模糊模型的模糊分布（如 GoPro、REDS）。
- **模糊核迁移（Blur Kernel Transfer）**：从已知 blur-sharp 对中提取模糊核并应用到目标清晰图像的技术 [37]。
- **WGAN-GP**：带梯度惩罚的 Wasserstein GAN，提供更稳定的对抗训练。
- **感知重建损失**：基于 VGG19 多层特征的 L1 损失，用于约束内容一致性。
- **NIQE**：无参考自然图像质量评估指标，越低表示图像越自然。
- **MIMO-UNet**：多输入多输出 U-Net，本文作为模糊转换器 G 的网络架构。

## 可复现要素
- **代码开源**：https://github.com/VinAIResearch/Blur2Blur
- **数据集**：GoPro（训练已知模糊）、REDS、RSBlur、RB2V Street（测试未知域）；PhoneCraft 和 WritingHands 为自建数据集，论文未公开。
- **关键超参**：$\lambda_{rec}=0.8$，$\lambda_{grad}=0.005$，batch size=16，图像裁剪 256×256，训练 1M 迭代约 3 天（2×A100）。
- **网络架构**：G 使用 MIMO-UNet（Pix2Pix 默认配置），去模糊骨干可选 NAFNet 或 Restormer。
- **优化器**：Adam，学习率前 500K 步恒定后线性衰减。
