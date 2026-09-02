---
title: "Towards-Progressive-Multi-Frequency-Representation-for-Image"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:14:40"
---

# 论文速读：Towards-Progressive-Multi-Frequency-Representation-for-Image

## 一句话总结
本文提出 MFR（Multi-Frequency Representation），通过渐进式多频率表征学习与可学习 Gabor 小波滤波器，从输入潜特征中逐频带提取高频信息，以自粗到细的方式生成形变图像；在齐次变换、ERP 投影及非对称超分等任务上，显著提升细节恢复能力与分布外泛化性能。

## 研究问题与动机
- 传统形变依赖双三次等插值方法，易引入锯齿与模糊伪影，图像质量退化明显。
- 现有 NN-based 方法（如 SRWarp、LTEW）将形变视为广义超分或连续坐标 MLP 表征，难以捕捉形变区域的局部强度变化，生成结果常含畸变且缺失高频细节。
- 上述方法在面对分布外（out-of-distribution, OOD）缩放因子或几何变换时泛化能力显著下降，限制实际部署。
- 频率域表征已在分类、域适应、生成等任务中验证有效，但如何针对形变任务设计兼顾空间局部性与频率选择性的表征机制仍缺乏系统探索。

## 核心贡献（创新点）
- 提出渐进式多频率表征学习框架：通过串联频率学习模块从潜特征的不同频带子空间提取表示，并以残差累加方式自粗到细合成图像，区别于依赖单一频带或纯坐标 MLP 的现有形变方法。
- 引入可学习 1-D Gabor 小波滤波器层：利用其在空间与频率域均具紧支撑的特性，通过反向传播自适应学习衰减率与调制率，显著增强局部高频细节的捕获能力，优于传统 SIREN 或高斯滤波。
- 设计短连接与 1-D 门控机制联合融合结构：在每模块输出端通过 gate 自适应控制频率表征的信息流，并与原始空间特征短接，避免纯频域建模导致的细节丢失。
- 系统验证跨任务泛化能力：在齐次变换、ERP→Perspective 投影、非对称超分三项任务上均取得 SOTA，尤其 OOD 设置下 PSNR/mPSNR 提升显著，视觉质量最优。

## 方法详解
- **特征编码阶段**：使用预训练 SR 模型（RRDB 或 RCAN）将输入 RGB 图像 $I_{in} \in \mathbb{R}^{W \times H \times 3}$ 投影至潜空间，再结合局部纹理估计器 LTE 融合坐标信息（相对位置）与几何信息（曲率），得到潜特征 $X \in \mathbb{R}^{W \times H \times d}$。
- **渐进频率学习**：将 $X$ 向量化为 $\{x_i\}_{i=1}^N$（$N=H \times W$），堆叠 $L$ 个频率学习模块。第 $\ell$ 模块的频率表示更新为：$z_i^{(0)} = \mathcal{G}_{\theta_0}(x_i) + x_i$，$z_i^{(\ell+1)} = \mathcal{G}_{\theta_\ell}(x_i) \otimes (W^{(\ell)} z_i^{(\ell)} + b^{(\ell)})$，其中 $\otimes$ 为逐元素乘法。理论推导表明多层串联可近似为大量正弦函数的线性组合，用较少模块即可覆盖丰富频带。
- **模块输出与重建**：$y_i^{(\ell)} = f_{\phi_\ell}(g_\ell(z_i^{(\ell)}) + x_i)$，$g_\ell$ 为 1-D 注意力门控，$f_{\phi_\ell}$ 为解码函数。最终图像 $y = y_{bic} + \sum_{\ell=1}^L y^{(\ell)}$，$y_{bic}$ 为双三次插值粗基准，实现粗到细的渐进生成。
- **Gabor 小波滤波器层**：滤波层采用 $\mathcal{F}_{Gabor}(x) = e^{-\frac{(x-x_0)^2}{\alpha^2}} e^{-i\omega(x-x_0)}$，$\alpha$ 控制空间衰减，$\omega$ 控制频率调制，参数可学习；相比 SIREN/Gaussian 能同时感知局部空间变化与对应频带响应。
- **损失函数**：训练阶段仅使用 $\ell_2$ loss（重建损失），无额外正则或感知损失。

## 实验与结果
- **数据集与设置**：训练使用 DIV2K（800 张 HR 图像，随机裁切 48×48 patch，batch=16，Adam β1=0.9/β2=0.999，600 epochs，lr=2e-4 余弦退火）。测试涵盖齐次变换（mPSNR）、ERP→Perspective（视觉）、非对称超分（PSNR），区分 in-scale（训练含该因子）与 out-of-scale/OOD（训练不含）设置。基线包括 Bicubic、RRDB/RCAN、SRWarp、LTEW、MetaSR、ArbSR。
- **齐次变换**：in-scale 下 MFR-RRDB 平均 mPSNR 略超 LTEW；out-of-scale 下全面领先，DIV2KW 提升 0.20dB，Urban100W 提升 0.26dB，视觉边缘与纹理更清晰、畸变更少。
- **ERP→Perspective**：跨域泛化验证，MFR 能有效校正全景图极区严重变形并保留局部内容，视觉效果显著优于 Bicubic 与 LTEW。
- **非对称超分**：以 RCAN 为骨干，MFR-RCAN 在 in-scale 与 out-of-scale 下均取得最佳或次佳 PSNR；大缩放因子（如 ×8/×7.6）下优势更明显，高频细节恢复能力最强。
- **消融实验**：可学习 Gabor 滤波器（GW-L）在 OOD 下优于 SIREN、普通 Gabor 与静态 Gabor（GW-S）；短连接（SC）与门控（Gate）二者结合时性能最佳（DIV2KW OS: 26.90dB，Urban100W OS: 24.51dB）。

## 相关工作脉络
- **SRWarp [41]**：将形变建模为广义超分并引入自适应重采样核。本文定位：SRWarp 对局部高频变化建模不足，大尺度/形变复杂时易畸变；MFR 显式多频带渐进学习弥补该缺陷。
- **LTEW [22]**：基于连续空间与 Fourier 特征 MLP 合成不规则网格内容。本文定位：LTEW 的 MLP 表征容量有限，难捕捉局部突变；MFR 在潜特征空间逐频带滤波并结合 Gabor
