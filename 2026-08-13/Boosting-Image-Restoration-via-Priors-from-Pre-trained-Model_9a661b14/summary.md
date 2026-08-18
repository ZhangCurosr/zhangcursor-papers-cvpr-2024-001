---
title: "Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xu_Boosting_Image_Restoration_via_Priors_from_Pre-trained_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:23"
---

# 论文速读：Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model

## 一句话总结
本文提出轻量级即插即用模块 **PTG-RM**，首次系统性地挖掘 CLIP、Stable Diffusion 等大规模多模态预训练模型的现成特征（OSF）作为退化先验，通过自适应空间可变增强与通道-空间注意力机制，显著提升低光照增强、去雨、去模糊、去噪等多种低层视觉恢复任务的性能，且推理阶段完全卸载预训练模型。

## 研究问题与动机
- 图像恢复属病态逆问题，仅优化目标网络结构或堆叠参数量易过拟合，且各任务突破现有上界日益困难。
- 传统方法依赖物理先验（语义图、深度图、预计算 SNR 等），这些先验跨场景泛化能力弱，且需针对特定任务设计复杂对齐机制，难以统一迁移。
- 大规模预训练模型在海量数据中学习到的表征隐含多样退化模式，但其特征异构性（如 CLIP 的 1D 全局向量 vs 恢复网络的 2D 空间特征）阻碍了直接复用。
- 现有工作仅利用预训练特征进行高层理解或图像质量评估（如 CLIP-IQA），尚未探索将其转化为低层恢复的隐性先验。

## 核心贡献（创新点）
- **提出通用轻量插件 PTG-RM**：仅增加 <1M 参数，即可作为即插即用模块适配不同架构的恢复网络，打破预训练大模型仅服务于高层任务的局限。
- **设计 PTG-SVE（空间可变增强）**：摒弃固定 SNR 参考，通过可学习空间嵌入与 OSF 引导生成动态融合掩码，自适应权衡短程 CNN 局部细节与长程 Transformer 全局上下文。
- **设计 PTG-CSA（通道-空间注意力）**：利用 OSF 生成空间可变卷积核合成注意力权重，实现细粒度的通道校正与空间自适应关注，避免异构特征的直接维度对齐。
- **统一训练范式**：细化模块与目标网络共享同一损失函数联合优化，支持有监督与无监督（如 ZeroDCE、RUAS）训练的无缝兼容，推理时仅需恢复网络。

## 方法详解
- **整体流程**：退化图像 $I_d$ 经目标恢复网络 $\mathcal{F}$ 得到初始结果 $\hat{I}_c$，预训练模型 $\mathcal{G}$ 提取 OSF $g=\mathcal{G}(I_d)$。轻量编码器 $\mathcal{R}_e$ 对 $[\hat{I}_c, I_d]$ 计算潜特征 $f$，依次经 PTG-SVE 与 PTG-CSA 得到 $\bar{f}$，解码器 $\mathcal{R}_d$ 输出校正掩码 $I_m$ 与残差项 $I_r$，最终结果 $\bar{I}_c = I_d + (\hat{I}_c - I_d) \times I_m + I_r$。
- **PTG-SVE**：引入空间位置嵌入 $S_m$，将 $g$ 映射后与 $f$、$S_m$ 拼接送入 $\mathcal{R}_m$ 生成融合权重 $M$。短程特征 $f_s=\mathcal{R}_s(f)$（ResNet）与长程特征 $f_l=\mathcal{R}_l(f)$（Restormer）按 $M$ 动态加权：$\hat{f} = M \times f_s + (1-M) \times f_l$，实现“差质量区域用长程、好质量区域用短程”的自适应策略。
- **PTG-CSA**：通道注意力 $M_c=\mathcal{R}_c(\text{Pool}(\hat{f}) \oplus \mathcal{T}_c(g))$，得 $\hat{f}_c=\hat{f}\times M_c$。空间注意力通过位置嵌入 $S_c$ 与 $\hat{f}$ 生成 location-wise 卷积核 $\mathcal{C}_p$，经卷积与映射得到 $M_s$，得 $\hat{f}_s=\hat{f}\times M_s$。二者融合为 $\bar{f}=\mathcal{R}_f(\hat{f}_c \oplus \hat{f}_s)$。
- **损失函数**：$\mathcal{L} = \mathcal{L}_g(\hat{I}_c, I_c) + \lambda_1 \mathcal{L}_g(\bar{I}_c, I_c)$，其中 $\lambda_1=1$，$\mathcal{L}_g$ 与目标网络保持一致（像素损失、感知损失或无监督等价损失均可）。

## 实验与结果
- **数据集与基线**：低光照（LOL-real, SID）、去雨（Rain100H/L, Test100/1200/2800）、运动去模糊（GoPro, HIDE, RealBlur-R/J）、散景去模糊（DPDD）、高斯去噪（Set12, BSD68, CBSD68, Kodak, McMaster, Urban100, SIDD）。对比基线包括 UHD、URetinex、SNR、SKF、SMG、SPAIR、Restormer、MPRNet、DRUNet、GRL 及无监督方法 ZeroDCE/RUAS/SCI。
- **关键数值**：LOL-real +UHD 提升至 PSNR 22.91（+3.04）、SSIM 0.767（+0.061）；SID +SNR 提升至 25.50（+4.02）、SSIM 0.892（+0.043）。无监督任务中 +ZeroDCE 在 LOL-real 提升 0.73 dB。去噪方面，Restormer 在 SIDD 上 PSNR 突破 40.22 dB（+0.20）。
- **消融结论**：移除 SP、CA 或 SA 任一组件均导致性能下降；使用固定 SNR 掩码、直接拼接 1D/2D 先验的特征对齐方案均劣于本文；将通道数扩大 4 倍的“Large R”版本仍不及 Full Setting，验证了机制设计优于单纯堆参。用户研究（80 人 A/B 测试）亦证实主观质量显著提升。

## 相关工作脉络
- **多模态先验恢复**（SKF、SMG）：依赖预计算的语义/深度/边缘图，需配对标注且先验误差易级联放大；本文直接用预训练 OSF，无需额外配对数据，泛化更鲁棒。
- **预训练模型下游应用**（CLIP-IQA 等）：此前仅探索特征在高层分类、编辑、质量评估中的潜力；本文首次将其转化为低层恢复的隐性先验，填补多模态大模型在低层任务的应用空白。
- **自适应操作范围机制**（SNR-aware 低光照方法）：传统策略依赖预计算信噪比或固定参考确定融合权重；本文通过可学习映射与 OSF 引导生成空间可变权重，更好适应未知退化分布。
- **无监督低层恢复**（ZeroDCE、Enlightengan 等）：现有无监督方法依赖特定自监督损失设计；本文证明 PTG-RM 可与任意无监督损失兼容，统一提升现有基线表现。
- **轻量即插即用模块**（PnP Denoiser 等）：以往插件多聚焦单一任务；本文验证跨任务通用性，证明 OSF 先验的可迁移价值高于传统任务定制模块。

## 局限性与未来方向
- 性能
