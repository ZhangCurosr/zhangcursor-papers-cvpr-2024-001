---
title: "Towards-Robust-Event-guided-Low-Light-Image-Enhancement-A-La"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liang_Towards_Robust_Event-guided_Low-Light_Image_Enhancement_A_Large-Scale_Real-World_Event-Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:14:27"
field: "低光照视觉增强"
keywords: ["低光照图像增强", "事件相机", "多模态融合", "SNR引导特征选择", "时空对齐数据集"]
innovations: ["提出SDE大规模真实世界事件-图像数据集（30K+对，0.03mm空间精度）", "设计SNR引导的区域特征选择机制自适应融合异构传感器特征", "匹配对齐策略消除事件-图像时序偏差（90%误差<0.01s）"]
benchmarks: ["SDE", "SDSD"]
---

# 论文速读：Towards-Robust-Event-guided-Low-Light-Image-Enhancement-A-La

## 一句话总结
本文提出了大规模真实世界事件-图像数据集SDE（超30K对时空对齐样本），并在此基础上设计了EvLight框架，通过SNR引导的区域特征选择与多尺度整体融合，实现了鲁棒的低光照图像增强。

## 研究问题与动机
1. **低光照图像增强瓶颈**：帧相机在细节（如边缘）缺失时易产生曝光不均和色彩失真，现有Retinex等方法在极低光照下性能受限。
2. **数据集缺失制约事件相机应用**：现有事件-图像数据集（如LIE、DVS-Dark）存在规模小（仅2K+）、静态场景、无配对GT或时空未对齐等问题，难以支撑真实场景训练。
3. **直接融合易加剧噪声**：事件和图像在不同区域具有不同噪声分布，简单特征融合可能放大局部噪声，需自适应选择策略。
4. **时空对齐技术挑战**：非线性运动轨迹下，事件流与图像帧的时间戳偏差难以精确消除，需新策略保障对齐质量。

## 核心贡献（创新点）
1. **提出SDE大规模真实世界数据集**：采用UR5机器人臂+DAVIS346事件相机，实现非线性轨迹采集，空间对齐精度<0.03mm（优于SDSD的1mm），包含31,477对时空对齐的Low-light/Normal-light图像-事件对。
2. **设计匹配对齐策略**：通过多次采集序列并匹配最小时间间隔偏差，使90%数据时间对齐误差<0.01s，显著降低随机时间偏差。
3. **提出SNR引导的区域特征选择机制**：利用信噪比先验，IRFS模块选择高SNR图像特征保留色彩，ERFS模块选择低SNR区域的事件特征增强结构细节，避免直接融合噪声。
4. **设计多尺度整体-区域融合框架EvLight**：结合Attention-based整体特征提取与区域选择性融合，在SDE和SDSD数据集上分别超越Retinexformer 1.14dB和2.62dB PSNR提升。

## 方法详解
**EvLight框架包含三个核心模块：**

**1. Preprocessing（预处理）**
- **初始增强**：采用Retinexformer估计光照图L，通过逐像素乘法增强低光照图像：$I_{lu} = I \odot L$
- **SNR图估计**：将增强图转灰度$I_g$，经均值滤波去噪得$\tilde{I}_g$，计算$M_{snr} = \tilde{I}_g / |I_g - \tilde{I}_g|$作为区域噪声先验
- **特征提取**：使用conv3×3提取图像特征$F_{img}$和事件体素网格特征$F_{ev}$

**2. SNR-guided Regional Feature Selection（SNR引导区域特征选择）**
- **多尺度下采样**：在三个尺度$i=0,1,2$上对特征和SNR图进行下采样
- **IRFS模块**：对图像特征经残差块+ECA注意力处理后，与归一化SNR图$\hat{M}_{snr}^i$逐元素相乘：$F_{sel-img}^i = \hat{M}_{snr}^i \odot \hat{F}_{img}^i$，保留高SNR区域特征
- **ERFS模块**：对事件特征采用逆SNR图$\bar{M}_{snr}^i = 1 - \hat{M}_{snr}^i$加权：$F_{sel-ev}^i = \bar{M}_{snr}^i \odot \hat{F}_{ev}^i$，增强低可见度/高噪声区域的边缘细节

**3. Holistic-Regional Fusion Branch（整体-区域融合分支）**
- **UNet-like架构**：编码器-解码器结构，含跳跃连接
- **HFE模块**：使用通道级self-attention（类比Restormer）捕获长程依赖：$\hat{F}_{mid} = \text{Attention}(F_{ho}) + F_{ho}$，$\hat{F}_{ho} = \text{FFN}(\text{LN}(\hat{F}_{mid})) + \hat{F}_{mid}$
- **HRF模块**：将上采样整体特征与选择后的区域特征拼接，通过卷积生成空间注意力图，再与输入特征融合：$F_{ho}^i = \mathcal{F}_3(\sigma(\mathcal{F}_1(F_{cat}^i)) \odot \mathcal{F}_2(F_{cat}^i) + F_{cat}^i)$
- **损失函数**：$L = \sqrt{||I_{en} - I_{gt}||^2 + \epsilon^2} + \lambda||\Phi(I_{en}) - \Phi(I_{gt})||_1$，含像素级_smooth L1_loss_和AlexNet特征级_L1_loss_

## 实验与结果
**数据集与评估设置**
- **SDE数据集**：91对序列（43室内+48户外），分辨率346×260，76对训练/15对测试
- **SDSD数据集**：25对动态视频序列，经v2e模拟器生成事件流，分辨率降至346×260
- **评估指标**：PSNR、PSNR*（亮度归一化后评估）、SSIM

**主要结果（Table 2）**
| 数据集 | 方法 | PSNR↑ | PSNR*↑ | SSIM↑ |
|--------|------|-------|--------|-------|
| SDE-in | Retinexformer | 21.30 | 23.78 | 0.6920 |
| **SDE-in** | **Ours** | **22.44** | **24.81** | **0.7697** |
| SDE-out | Retinexformer | 22.92 | 23.71 | 0.6834 |
| **SDE-out** | **Ours** | **23.21** | **25.60** | **0.7505** |
| SDSD-in | Retinexformer | 25.90 | 25.97 | 0.8515 |
| **SDSD-in** | **Ours** | **28.52** | **29.73** | **0.9125** |
| SDSD-out | Retinexformer | 26.08 | 28.48 | 0.8150 |
| **SDSD-out** | **Ours** | **26.67** | **30.30** | **0.8356** |

- **最强结果**：SDSD-in上PSNR达28.52，较Retinexformer提升**2.62dB**
- SDE-in上较Retinexformer提升**1.14dB**，PSNR*提升0.93-1.21dB
- 优于事件引导基线ELIE（SDE-in: 19.98）、Liu et al.（SDE-in: 21.79）

**消融实验**
- 移除事件：PSNR下降0.23dB
- 移除SNR引导选择：PSNR下降0.86dB（Table 3）
- IRFS+ERFS联合贡献：PSNR分别提升0.34dB和0.60dB（Table 4）

## 相关工作脉络
1. **Retinexformer [4]**：单阶段RetinexTransformer方法，本文帧基线；优势在于全局光照估计，但低光照下边缘细节恢复不足。
2. **ELIE [18]**：首个事件引导LIE方法，使用残差融合模块；局限在于静态场景采集、直接融合未考虑噪声差异。
3. **Liu et al. [25]**：AAAI'23事件引导视频增强；采用合成事件+融合变换模块；在SDSD-out上过拟合严重（PSNR与PSNR*差距大）。
4. **SNR-Net [48]**：CVPR'22利用SNR感知Transformer；本文借鉴其SNR估计思想但扩展至事件-图像跨模态选择融合。
5. **eSL-Net [36]**：ECCV'20事件增强图像恢复；侧重高分辨率重建，非专门针对低光照场景。
6. **DVS-Dark [52]/EvLowLight [24]**：无配对GT或仅灰度输出；本文提供完整彩色GT和大样本监督训练。

## 局限性与未来方向
1. **硬件限制**：DAVIS346相机导致部分图像存在色差和摩尔纹，影响训练数据质量。
2. **同步触发需求**：当前需手动匹配序列，未来需改进硬件实现机器人-事件相机同步触发以降低采集成本。
3. **泛化性验证有限**：仅在CED、MVSEC等小规模事件数据集验证泛化，未测试极端动态场景。
4. **计算复杂度**：多尺度特征选择+Attention机制推理开销较大，未讨论实时部署可行性。
5. **未来方向**：探索端到端时空对齐采集系统、扩展至低光照视频增强/去模糊任务、引入更轻量级注意力设计。

## 研究启发与可借鉴点
1. **SNR先验指导跨模态融合**：将噪声统计先验作为特征选择门控信号，可迁移至其他多模态融合任务（如红外-可见光、深度-图像融合）。
2. **机器人轨迹多样性设计**：非线性轨迹+高精度重复定位（0.03mm）为多模态数据集构建提供可复用的硬件方案。
3. **匹配对齐策略**：多次采集序列匹配最小时间偏差的方法适用于任何存在时序偏差的多传感器同步问题。
4. **PSNR*评估指标**：亮度归一化后的PSNR评估能更准确反映结构恢复能力，值得在图像恢复任务中推广使用。
5. **事件-图像互补性建模**：区分"高SNR保留图像细节"与"低SNR借用事件边缘"的选择策略，为异构传感器融合提供新范式。

## 关键术语表
- **Event Camera**：事件相机，生物启发式传感器，仅记录像素亮度变化事件，具有HDR和高时间分辨率特性。
- **Event Voxel Grid**：事件体素网格，将事件流按极性分配到三维体素（x, y, t）中形成的张量表示。
- **SNR (Signal-to-Noise Ratio)**：信噪比，本文用于量化图像各区域的噪声强度，作为特征选择的先验指导。
- **IRFS (Image-Regional Feature Selection)**：图像区域特征选择模块，利用SNR图选择高可信度图像特征。
- **ERFS (Event-Regional Feature Selection)**：事件区域特征选择模块，利用逆SNR图选择低可见度区域的事件边缘特征。
- **HFE (Holistic Feature Extraction)**：整体特征提取模块，基于channel-wise自注意力捕获长程依赖。
- **HRF (Holistic-Regional Fusion)**：整体-区域特征融合模块，通过空间注意力机制融合多尺度特征。
- **PSNR***：亮度归一化后的PSNR，通过均值比率校正预测图像亮度后计算，更聚焦结构恢复性能。

## 可复现要素
- **SDE数据集**：公开（Project Page: https://vlislab22.github.io/eg-lowlight/），含31,477对配对数据
- **代码**：论文未提及开源状态
- **权重**：论文未提及开源状态
- **关键超参**：
  - 事件体素网格bin size: 32
  - 学习率: SDE数据集1e-4，SDSD数据集2e-4
  - 训练epoch: 80
  - Batch size: 8
  - 裁剪尺寸: 256×256
  - 旋转角度: 90°, 180°, 270°
  - 损失权重λ: 论文未明确给出数值
  - ε: 10^-4
