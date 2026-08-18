---
title: "Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_Learning_Spatial_Adaptation_and_Temporal_Coherence_in_Diffusion_Models_for_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:25"
---

# 论文速读：Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution

## 一句话总结
本文提出SATeCo方法，通过在冻结的预训练扩散模型（UNet/VAE）解码器中插入空间特征自适应（SFA）与时间特征对齐（TFA）模块，利用低分辨率视频提供像素级空间引导与管状窗时序校准，使扩散过程同时兼顾高分辨率细节生成与跨帧视觉一致性。

## 研究问题与动机
1. **扩散模型直接用于视频超分存在时空双重失配**：现有图像超分扩散模型（如StableSR）逐帧独立推理，缺乏跨帧时序建模，导致相邻帧出现对象形状/纹理突变；同时扩散过程的随机性易破坏原始外观信息。
2. **传统回归类VSR模型细节生成上限受限**：如VRT、EDVR等虽能保证时序一致，但在严重退化场景下难以恢复高频纹理，感知指标落后于扩散先验方法。
3. **现有扩散超分的空间引导粒度粗糙**：多数工作依赖全局条件插拔或零初始化卷积加权求和，无法为每个像素提供精确的仿射调节，难以在“保真”与“生成”间取得平衡。
4. **潜空间去噪与像素空间重建需统一协调**：仅校准潜变量易忽略色彩/结构细节，仅依赖像素重建则浪费扩散先验，需设计能在两个空间协同注入视频先验的机制。

## 核心贡献（创新点）
1. **提出SFA模块实现像素级仿射自适应**：通过LR潜特征逐像素预测Scale与Bias，对HR中间特征做归一化仿射调制，提供细粒度空间引导；区别于以往仅依赖全局注意力或全局向量条件的控制方法。
2. **提出TFA模块实现Tubelet级时序对齐**：在HR特征局部管状窗内执行自注意力增强跨帧交互，并与LR对应管状窗执行交叉注意力完成时序校准，本质是将3D局部时空关联显式引入扩散解码流程。
3. **构建“冻结骨干+轻量适配器”的高效范式**：完全冻结预训练UNet与VAE权重，仅在各自解码器块插入SFA/TFA，在保留强大图像生成先验的同时大幅降低训练成本与显存占用。
4. **端到端集成视频上采样器与视频精炼器**：前期用Transformer时序互注意力完成高质量上采样，后期用可学习残差融合网络调和扩散生成内容与原始LR视频，兼顾感知质量与保真度。

## 方法详解
- **整体流程**：输入LR视频$X_L$ → Transformer视频上采样器生成$X_u$ → VAE编码器得到潜码$Z$ → 加噪后送入冻结UNet解码器（每块插入SFA+TFA）得到$Z_0$ → VAE解码器（同样插入SFA+TFA）得到$X_d$ → 视频精炼器融合$X_u$与$X_d$输出最终HR视频$X_H$。
- **SFA（Spatial Feature Adaptation）**：对帧$i$的LR潜特征$g^i$经两层2D卷积预测$M^i$与$S^i$，对HR中间特征$f^i$做空间归一化后仿射变换：$\tilde{f}^i = S^i \odot \frac{f^i - \mu^i}{\sigma^i} + M^i$。该模块同时作用于UNet（潜空间去噪）与VAE（像素空间重建）解码器。
- **TFA（Temporal Feature Alignment）**：将$\tilde{f}^i$在时空维度切分为$N$个非重叠Tubelet（空间$h\times w$，时间$L$帧），先用3D卷积提取Q/K/V做自注意力：$\hat{F}_{tub} = \text{Attention}(Q,K,V)$；再以LR Tubelet特征$G_{tub}$为Key/Value，HR特征为Query做交叉注意力：$\bar{F}_{tub} = \text{Attention}(Q', K', V')$，将输出reshape回原尺寸送入下一解码块。实验设$h=w=8$。
- **Video Refiner**：拼接$X_d$与$X_u$后过残差块，按$X_H = w X_u + (1-w) X_d + \text{ResBlock}([X_u, X_d])$融合，$w=0.5$平衡原始保真与扩散生成。
- **训练策略（四阶段）**：①训练视频上采样器（Charbonnier Loss）；②冻结UNet，仅训其中SFA/TFA（标准扩散损失）；③冻结VAE，仅训VAE解码器SFA/TFA（MSE重建Loss）；④冻结前三部分，训练视频精炼器。全程基于Stable Diffusion与Diffusers库。

## 实验与结果
- **数据集与设置**：训练集Vimeo-90K（64,612 clips，448×256，每clip 7帧）；测试集REDS4与Vid4（均为4倍下采样，推理clip长度$L=6$）。
- **定量结果**：REDS4上PSNR 31.62 dB、SSIM 0.8932，与SOTA回归模型IconVSR（31.67/0.8948）几乎持平；全面领先感知指标（LPIPS 0.1735、DISTS 0.0607、NIQE 4.104、CLIP-IQA 0.6622）。Vid4上PSNR 27.44，DISTS 0.1015较次优方法VRT（0.1372）相对提升26.0%。
- **定性与人评**：Visual案例显示SATeCo能恢复锐利边缘（如屋檐、车轮辐条），且相邻帧纹理/物体形态高度一致；Amazon MTurk百人盲测中用户偏好胜率最高。
- **消融结论**：SFA+TFA同时插入UNet与VAE解码器效果最佳（PSNR 28.56→31.62）；时序互注意力上采样器显著优于PixelShuffle；Refiner参数$w=0.5$为生成质量与保真度的最优折衷。

## 相关工作脉络
1. **StableSR [46]**：图像超分扩散代表，仅嵌入轻量编码器且冻结主干；本文在其扩散先验上进一步引入显式时空适配器，补齐视频时序一致性短板。
2. **VRT [23]**：基于Transformer滑动窗口的VSR SOTA，其时序互注意力思想启
