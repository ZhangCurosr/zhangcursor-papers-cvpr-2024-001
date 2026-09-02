---
title: "Towards Progressive Multi-Frequency Representation for Image Warping"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:50:37"
---

# 论文速读：Towards Progressive Multi-Frequency Representation for Image Warping

## 一句话总结
本文提出MFR（Progressive Multi-Frequency Representation），通过可学习的Gabor小波滤波器与串联式渐进滤波网络，从输入隐特征的多频子带中逐层提取频率表示，以粗到细的方式合成形变图像，显著提升了同构图变换、ERP转透视投影及非对称超分任务的重建质量与分布外（OOD）泛化能力。

## 研究问题与动机
1. 传统插值形变（如双三次）在不规则网格中易产生锯齿与模糊伪影，导致局部细节丢失。
2. 现有神经形变方法（SRWarp、LTEW）虽将任务建模为广义超分或连续空间表征，但仍难以捕捉形变区域的局部高频变化，生成图像易出现失真且缺乏锐利边缘。
3. 现有方法在OOD场景（训练集未覆盖的缩放因子或几何变换）下性能显著衰退，限制实际部署。
4. 亟需一种能够同时兼顾空域局部定位与频域多尺度表征的建模机制，以实现高质量、低畸变的渐进式形变合成。

## 核心贡献（创新点）
1. **渐进式多频滤波网络**：设计$L$个串联的频率学习模块，逐层从隐向量中提取不同频带的频率表示。区别于SRWarp/LTEW的单阶段特征映射，本文显式构建了多频成分的递进合成流水线。
2. **可学习Gabor小波滤波器**：在滤波层引入可微分的1-D Gabor小波，通过反向传播联合优化衰减率$\alpha$与调制频率$\omega$。区别于SIREN或高斯滤波器，该设计在空域与频域均具紧支撑，能更精准地捕获形变局部的强度突变。
3. **门控短连接融合架构**：采用1-D注意力门控自适应调控频率信息的注入比例，并结合短连接将原始空间特征与频域修正量相加。区别于纯频域生成或无残差结构，该设计有效避免了深层变换导致的细节退化。
4. **系统的OOD泛化评估**：在同构图变换、ERP→透视投影、非对称超分三个任务中统一区分In-scale与Out-of-scale设置。区别于仅报告In-scale基准的工作，本文方法在未见尺度与变换上均取得显著增益。

## 方法详解
- **特征编码阶段**：使用预训练SR模型将输入图像$I_{in}$投影至隐特征空间，再结合局部纹理估计器（LTE）融合相对坐标与曲率几何信息，得到隐表示$X \in \mathbb{R}^{W \times H \times d}$。
- **渐进式频率学习**：将$X$向量化为$\{x_i\}$，输入$L$个串联模块。第$\ell$模块的频表示更新为：$z_i^{(0)} = \mathcal{G}_{\theta_0}(x_i) + x_i$，$z_i^{(\ell+1)} = \mathcal{G}_{\theta_\ell}(x_i) \otimes (W^{(\ell)}z_i^{(\ell)} + b^{(\ell)})$。随着模块堆叠，等效可合成的正弦基函数数量呈指数增长，从而覆盖宽频带。
- **门控输出与粗到细合成**：每层输出$y_i^{(\ell)} = f_{\phi_\ell}(g
