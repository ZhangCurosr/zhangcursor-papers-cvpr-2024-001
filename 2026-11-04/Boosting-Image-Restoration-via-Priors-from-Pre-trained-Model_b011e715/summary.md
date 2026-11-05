---
title: "Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Restoration_via_Priors_from_Pre-trained_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:57:50"
field: "低层视觉图像恢复"
keywords: ["图像恢复", "预训练模型", "CLIP", "轻量级插件", "空间可变增强", "通道-空间注意力"]
innovations: ["首次系统探索大规模预训练模型（CLIP/Stable Diffusion）在低层图像恢复任务中的应用", "提出<1M参数的PTG-RM轻量插件，通过空间可变增强和通道-空间注意力自适应融合多尺度先验", "利用预训练先验学习空间自适应融合mask，替代传统手工设计的SNR值融合策略"]
benchmarks: ["LOL-real", "SID", "Rain100H", "Test100", "GoPro", "HIDE", "RealBlur-R", "DPDD", "BSD68", "SIDD"]
---

# 论文速读：Boosting-Image-Restoration-via-Priors-from-Pre-trained-Model

## 一句话总结
论文首次系统性探索将CLIP、Stable Diffusion等大规模预训练模型的off-the-shelf features (OSF)作为退化相关先验，用于增强图像恢复任务，并提出轻量级PTG-RM模块（<1M参数）在低光增强、去雨、去模糊、去噪等多个任务上实现显著性能提升。

## 研究问题与动机
1. **图像恢复的瓶颈**：恢复任务是病态问题，单纯修改网络结构或增加参数难以突破性能上限，且容易过拟合。
2. **现有物理先验的局限**：深度图、语义图等预计算多模态先验在不同场景中泛化能力有限，且依赖复杂配对标注。
3. **预训练模型潜力未被挖掘**：CLIP-IQA发现CLIP特征含退化相关信息可用于图像质量评估，但无人探索其在恢复任务中的应用。
4. **特征异构性挑战**：预训练模型特征（如CLIP的1D特征）与恢复任务特征（2D）存在表征和维度不匹配，直接对齐困难。

## 核心贡献（创新点）
1. **首次将大规模多模态预训练模型的隐藏先验引入低层视觉恢复**：不同于CLIP-IQA仅做质量评估，本文证明预训练模型蕴含可用于恢复的退化信息，无需显式标注。
2. **提出PTG-RM轻量级插件**：仅增加<1M参数，作为通用增强模块可适配不同恢复网络（Restormer、MPRNet等）和任务；训练时引入预训练先验，推理时移除。
3. **设计PTG-SVE自适应空间融合机制**：摒弃传统固定SNR值作为融合mask的做法，通过可学习空间映射自适应决定每个区域的短程（CNN）/长程（Transformer）操作权重。
4. **设计PTG-CSA双重视图注意力**：利用预训练先验同时生成通道注意力和空间自适应卷积核，实现不同区域的差异化特征精炼。

## 方法详解
**整体流程**：给定退化图像$I_d$，初始恢复结果$\hat{I}_c = \mathcal{F}(I_d)$，提取预训练模型OSF $g = \mathcal{G}(I_d)$，通过PTG-RM $\mathcal{R}$精炼得到$\bar{I}_c = \mathcal{R}(\hat{I}_c, I_d, g)$。

**编码阶段**：潜空间蒸馏避免直接与恢复特征对齐，$f = \mathcal{R}_e(\hat{I}_c \oplus I_d)$，$\mathcal{R}_e$为简单编码器。

**PTG-SVE（空间可变增强）**：
- 引入位置嵌入$S_m = \mathcal{P}(S)$处理二维坐标信息
- 可学习映射$\mathcal{T}_m$将OSF转换到适配空间
- 范围学习模块生成融合mask：$M = \mathcal{R}_m(f \oplus \mathcal{T}_m(g) \oplus S_m)$
- 短程操作$\mathcal{R}_s$（ResNet）和长程操作$\mathcal{R}_l$（Restormer backbone）分别处理：$f_s = \mathcal{R}_s(f), f_l = \mathcal{R}_l(f)$
- 自适应融合：$\hat{f} = M \times f_s + (1-M) \times f_l$

**PTG-CSA（通道-空间注意力）**：
- 通道注意力：$M_c = \mathcal{R}_c(\mathcal{O}(\hat{f}) \oplus \mathcal{T}_c(g))$，$\mathcal{O}$为池化，得$\hat{f}_c = \hat{f} \times M_c$
- 空间注意力：学习局部卷积核$\mathcal{C}_p = \mathcal{R}_p(\hat{f}, \mathcal{T}_c(g), S_c)$，生成$\hat{M}_s = \hat{f} * \mathcal{C}_p$，经$\mathcal{R}_o$得$M_s$，$\hat{f}_s = \hat{f} \times M_s$
- 融合：$\bar{f} = \mathcal{R}_f(\hat{f}_c \oplus \hat{f}_s)$

**解码与输出**：解码器$\mathcal{R}_d$输出修正mask $I_m$和残差细化$I_r$，最终：$\bar{I}_c = I_d + (\hat{I}_c - I_d) \times I_m + I_r$

**损失函数**：与基础网络使用相同损失，$\mathcal{L} = \mathcal{L}_g(\hat{I}_c, I_c) + \lambda_1 \mathcal{L}_g(\bar{I}_c, I_c)$，实验中$\lambda_1=1$。

## 实验与结果
**数据集**：LOL-real、SID（低光）；Rain13K训练，Rain100H/L、Test100/1200/2800测试（去雨）；GoPro训练，HIDE、RealBlur-R/J测试（运动去模糊）；DPDD训练，EBDB、JNB测试（散焦去模糊）；Set12、BSD68、CBSD68、Kodak、McMaster、Urban100、SIDD（去噪）。

**主要结果**：
- **低光增强**：LOL-real上URetinex+Ours-c提升+3.54dB PSNR/+3.8 SSIM×100；SNR基线+3.04dB/+4.02dB。相比SKF（+2.15M参数）和SMG（+16.76M参数），Ours仅增0.67M参数且效果更优。
- **去雨**：Restormer+Ours-c在Test100达PSNR 32.30（↑0.30），SSIM 0.934（↑0.011）。
- **运动去模糊**：Restormer+Ours-c在GoPro达PSNR 33.18（↑0.26），SSIM 0.966（↑0.005）。
- **散焦去模糊**：GRLD-B+Ours-c在DPDD综合PSNR 27.27（↑0.23），SSIM 0.855（↑0.008）。
- **高斯去噪**：BSD68 σ=25下Restormer+Ours-c达29.78（↑0.27）。
- **真实去噪**：SIDD上Restormer+Ours-c达PSNR 40.22 dB，突破40dB门槛。

**消融实验**：SP、CA、SA各组件均有效；位置嵌入至关重要；基于预训练先验的mask优于直接SNR值；将PTG-RM插入中间层（+Ours-f）同样有效。

## 相关工作脉络
1. **图像恢复先验方法**：SKF用语义图优化低光特征空间；SMG整合边缘/深度/语义信息。本文不使用额外多模态标注，直接利用预训练模型泛化知识。
2. **预训练模型下游应用**：CLIP、BLIP、BLIP2在零样本分类、图像编辑、分割等高层任务成功；CLIP-IQA仅探索质量评估。本文首次系统探索其在低层恢复中的应用。
3. **多模态先验辅助恢复**：NIIR使用近红外信息；部分方法依赖预计算物理图。本文利用预训练模型的隐式先验，无需显式标注。
4. **无监督恢复方法**：ZeroDCE、RUAS、SCI无需配对数据。本文方法与无监督框架天然兼容，进一步提升性能。
5. **SOTA恢复架构**：Restormer、MPRNet、SPAIR等。本文作为通用插件可无缝集成。

## 局限性与未来方向
1. **提升幅度不均**：不同任务和模型的性能增益存在差异，与目标网络容量和任务复杂度相关。
2. **预训练模型选择影响**：CLIP、BLIP2、Stable Diffusion的增益效果不同，最优选择策略未深入研究。
3. **训练/推理分离**：预训练模型仅在训练阶段参与，推理时移除，可能限制了进一步优化空间。
4. **未来方向**：探索更有效的蒸馏框架以获得精细恢复特征先验；开发针对性技术产品。

## 研究启发与可借鉴点
1. **"预训练先验蒸馏"范式**：利用大规模预训练模型的隐藏知识而非直接提取特征，可迁移到超分辨率、inpainting等其他低层视觉任务。
2. **轻量级插件设计参考**：<1M参数实现跨任务泛化，为高效架构设计提供范例。
3. **空间自适应操作范围学习**：用预训练先验学习最优融合权重，替代手工设计的先验（如SNR），可扩展到其他需自适应操作的场景。
4. **无监督/弱监督兼容性**：为缺乏配对数据的场景利用预训练先验提供新思路。
5. **跨层集成灵活性**：PTG-RM可部署在输出端或中间层，为网络结构设计提供参考。

## 关键术语表
- **OSF (Off-the-Shelf Features)**：直接从预训练模型提取的未微调特征，作为恢复任务的退化相关先验信息源。
- **PTG-RM (Pre-Train-Guided Refinement Module)**：提出的轻量级精炼模块，总参数<1M，包含PTG-SVE和PTG-CSA两个子组件。
- **PTG-SVE (Pre-Train-Guided Spatial-Varying Enhancement)**：基于预训练先验学习空间自适应映射，动态融合短程CNN与长程Transformer操作结果。
- **PTG-CSA (Pre-Train-Guided Channel-Spatial Attention)**：利用预训练先验同时生成通道注意力和空间自适应卷积核的双重视图注意力机制。
- **Restoration Prior**：图像恢复中用于约束解空间的已知信息，如去噪的噪声水平、去模糊的模糊核。
- **SNR Value as Mask**：信噪比值直接作为短/长程特征融合mask的基准方法，本文消融实验的对比对象。

## 可复现要素
- **数据集**：LOL-real、SID、Rain13K、Rain100H/L、Test100/1200/2800、GoPro、HIDE、RealBlur-R/J、DPDD、EBDB、JNB、Set12、BSD68、CBSD68、Kodak、McMaster、Urban100、SIDD（均为公开数据集）
- **代码**：论文未明确提及代码开源状态，论文未提及
- **关键超参**：λ₁=1（损失权重），模块参数量<1M；具体学习率、训练轮数等未在解析文本中详细列出
