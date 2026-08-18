---
title: "One-Shot Structure-Aware Stylized Image Synthesis"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cho_One-Shot_Structure-Aware_Stylized_Image_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:12:49"
field: "图像编辑与风格迁移"
keywords: ["One-Shot Style Transfer", "Structure Preservation", "Diffusion Models", "Image Stylization", "Semantic-Structure Disentanglement", "Text-Driven Manipulation"]
innovations: ["提出基于扩散模型的一站式结构感知风格化方法，有效解耦并控制图像语义与结构", "引入结构编码时间步和结构保持网络(SPN)以在风格迁移中鲁棒地保留输入图像结构", "直接优化解耦的语义潜在代码实现文本驱动的图像属性操控，同时保持结构与风格"]
benchmarks: ["ArtFID", "ID Similarity (ArcFace)", "Structure Distance", "FFHQ", "AAHQ", "MetFaces"]
---

# 论文速读：One-Shot Structure-Aware Stylized Image Synthesis

## 一句话总结
提出了OSASIS，一种基于扩散模型的单样本结构感知风格化方法，通过解耦图像语义与结构信息，在风格迁移过程中鲁棒地保持输入图像结构，尤其擅长处理训练中罕见属性的OOD参考图像及文本驱动操控。

## 研究问题与动机
*   GAN-based风格化方法在保持输入图像结构（尤其是手部、麦克风等复杂/罕见元素）方面存在显著困难，且容易将参考图像的结构性伪影混入生成结果。
*   现有扩散模型风格化方法（如DiffuseIT, InST）主要关注框架适配，未能有效解决结构保持问题，且存在领域鸿沟或过度依赖文本引导的局限。
*   如何设计一种扩散模型风格化方法，既能从单个参考图像中提取纯风格信息，又能精确控制并保留输入图像的高级别内容与低级结构细节，是核心挑战。

## 核心贡献（创新点）
*   **提出OSASIS框架**：构建了一个在扩散模型中有效解耦结构信息与可迁移语义信息的一站式风格化方法，通过条件调节控制内容与风格的注入程度。
*   **结构编码时间步控制机制**：利用结构潜在代码 $\mathbf{x}_{t_0}$ 并通过调整编码时间步 $t_0$ 来精细控制结构保留的强度，区别于直接使用完整特征的方法。
*   **语义编码器与方向性CLIP损失微调**：借鉴DiffAE获取语义有意义的潜在变量 $\mathbf{z}_{sem}$，并采用MTG的思路，通过组合方向性CLIP损失微调预训练DDIM以弥合领域鸿沟。
*   **结构保持网络(SPN)**：引入一个基于1x1卷积的网络来补偿扩散编码过程引入的噪声，专门用于增强空间信息和结构完整性，防止边缘物体畸变。
*   **直接优化语义潜在用于文本驱动操控**：无需微调模型，直接对输入图像的语义潜在代码 $\mathbf{z}_{sem}^{in}$ 进行CLIP方向性损失优化，实现保留结构与风格属性下的文本驱动属性修改。

## 方法详解
*   **训练流程**：
    1.  **图像准备**：使用预训练DDPM $\epsilon_\theta^A$ 生成与风格图像 $I_B^{style}$ 语义对齐的自然域图像 $I_A^{style}$。将 $I_B^{style}$ 通过前向DDPM编码到特定时间步得到结构潜在代码 $\mathbf{x}_{t_0}$。
    2.  **模型设置**：冻结预训练DDIM $\epsilon_\theta^A$ 和DiffAE语义编码器 $\text{Enc}_\phi$。创建副本 $\epsilon_\theta^B$ 进行微调。
    3.  **结构编码**：输入图像 $I_A^{in}$ 通过前向DiffAE编码，先得到语义潜在 $\mathbf{z}_{sem}^{in}$，再在冻结的 $\epsilon_\theta^A$ 条件下编码得到结构潜在 $\mathbf{x}_{t_0}^{in}$。
    4.  **结构保持网络(SPN)**：使用SPN（1x1卷积）提取 $I_A^{in}$ 的空间信息 $\mathbf{x}_t^{SPN}$，在反向DiffAE步骤中与当前噪声预测输出相加，并通过超参 $\lambda_{SPN}$ 调节影响程度，生成下一步输入 $\mathbf{x}_t'$。
    5.  **损失函数**：总损失包括：(a) **跨域损失**：在CLIP空间衡量 $I_A^{in} \to I_B^{in}$ 的变化方向与 $I_A^{style} \to I_B^{style}$ 的变化方向的一致性；(b) **域内损失**：衡量同一域内变化方向的相似性；(c) **重建损失**：比较重建构建出的风格图像 $\hat{I}_B^{style}$ 与真实 $I_B^{style}$，包含 $L_1$、感知损失和CLIP嵌入 $L_1$ 损失。
*   **采样/推理流程**：
    1.  **内容与风格混合**：在UNet的特征图上对称地条件化两个语义潜在代码：将风格图像的 $\mathbf{z}_{sem}^{style}$ 注入到**低层特征图**以传递风格，将输入图像的 $\mathbf{z}_{sem}^{in}$ 注入到**高层特征图**以保留内容。使用结构潜在代码 $\mathbf{x}_{t_0}^{in}$ 作为结构基础。分界点设为 $f_{ch}$。
    2.  **文本驱动操控**：对输入图像的语义潜在 $\mathbf{z}_{sem}^{in}$ 直接应用CLIP方向性损失进行优化，然后将优化后的潜在与微调后的 $\epsilon_\theta^B$ 及风格条件结合进行采样生成。

## 实验与结果
*   **数据集**：训练主要利用预训练的FFHQ域模型 $\epsilon_\theta^A$。评估使用了 **AAHQ**, **MetFaces** 以及先前研究中使用的风格图像。为了突出结构保持能力，特别从FFHQ中筛选出 **低密度区域图像**（基于随机重建的高LPIPS得分）进行重点评测。扩展验证了 **AFHQ-dog**, **LSUN-church**, **DeepFashion** 数据集。
*   **评估基线**：**MTG+HFGI**, **JoJoGAN+HFGI**, **DiffuseIT**, **InST**。
*   **评估指标**：风格化质量用 **ArtFID↓**，内容保持用 **ID Similarity↑** (ArcFace)，结构保持用 **Structure Distance↓**。
*   **主要结果**：
    *   在低密度图像上，OSASIS在 **ID Similarity** 和 **Structure Distance** 上显著优于所有基线（见表1）。例如在MetFaces上，ID Similarity为0.7323，Structure Distance为0.0295，均优于最佳基线。
    *   虽然在部分ArtFID指标上不是最优（如Prev数据集33.20 vs DiffuseIT的35.86），但综合考虑内容与结构保持，OSASIS表现最佳。
    *   OOD参考图像实验中，OSASIS能有效转移风格而无结构性伪影，而其他方法出现严重失真。
    *   消融实验证实了 **SPN的有效性**（$\lambda_{SPN}=0.5$时ID Sim提升至0.7177，SD降至0.0348）以及**语义潜在代码分层注入策略的正确性**。

## 相关工作脉络
*   **MTG/JoJoGAN (GAN-based)**：同样是一站式风格化，但依赖GAN逆技术，在处理OOV结构时保留能力弱，易受风格图像结构干扰。
*   **DiffuseIT (Diffusion-based)**：无训练风格迁移，利用CLIP和ViT引导，但存在输入与风格域间的鸿沟，风格化效果受限。
*   **InST (Diffusion-based)**：使用文本反转从风格图像提取概念并条件化生成，但引导策略易导致面部表情/身份变化，且难以忠实传递颜色。
*   **DiffusionCLIP/Asyrp (Diffusion-based Manipulation)**：聚焦于文本引导的图像编辑，通过CLIP损失微调或训练h-space模块，与本文的图片引导风格化目标不同。
*   **StyleGAN-NADA**：可用于一站式风格化，但主要面向文本驱动风格转移，能力有限。
*   **OSASIS定位**：区别于上述方法，专注于**基于扩散模型的结构感知图片引导风格化**，核心创新在于显式解耦并分别控制结构与语义/内容。

## 局限性与未来方向
*   **训练时间较长**：相较于比较方法，OSASIS的训练耗时更长。
*   **每风格需单独训练**：目前需要针对每个风格图像进行微调，限制了多风格快速部署的实用性。
*   **未来方向**：优化训练效率，探索减少或避免为每个新风格单独训练的需求，提升实际应用可行性。

## 研究启发与可借鉴点
*   **解耦结构/语义用于条件控制**：将图像信息分离为结构潜在（控制时间步）和语义潜在（通过DiffAE），并在UNet不同层次分别注入，是一种精细控制生成内容的有效范式。
*   **结构保持网络(SPN)的设计**：利用轻量级1x1卷积网络直接增强反扩散过程中的空间信息，以对抗编码噪声导致的结构损失，设计巧妙且有效。
*   **CLIP方向性损失用于域适应**：借鉴MTG思路，通过方向性CLIP损失微调预训练扩散模型来桥接自然域与风格域，避免了大量配对数据需求。
*   **直接优化潜在进行文本操控**：跳过模型微调，直接对已解耦的语义潜在代码进行优化以实现文本驱动的属性修改，保留了原有结构和风格，思路清晰。
*   **低密度图像筛选评估**：利用预训练扩散模型的重建残差（LPIPS）来识别和筛选训练数据分布中罕见的“低密度”样本，用于针对性评估模型的结构保持鲁棒性，实验设计具有参考价值。

## 关键术语表
*   **OSASIS**：本文提出的One-Shot Structure-Aware Stylized Image Synthesis方法的全称。
*   **结构潜在代码 ($\mathbf{x}_{t_0}$)**：将图像通过扩散过程编码到特定时间步$t_0$得到的潜在表示，主要承载图像的整体轮廓和结构信息。
*   **语义潜在代码 ($\mathbf{z}_{sem}$)**：由DiffAE的语义编码器生成的潜在变量，包含丰富、线性、可解码的对象语义和内容信息。
*   **结构保持网络 (SPN)**：一个基于1x1卷积的网络，用于在扩散反演过程中增强空间信息，帮助保留输入图像的结构完整性。
*   **方向性CLIP损失**：用于微调扩散模型或优化潜在代码的损失函数，旨在使CLIP嵌入空间中的变化沿着特定期望方向进行。
*   **低密度图像**：在预训练数据分布中较少出现的、包含罕见结构元素的图像，通过重建难度较高来识别，用于测试模型结构保持的鲁棒性。
*   **内容与风格混合**：在采样阶段，通过将风格语义潜在注入低层特征、输入语义潜在注入高层特征，并在UNet中对齐结构潜在，从而实现内容与风格的解耦融合。

## 可复现要素
*   **代码**：开源，地址 https://github.com/hansam95/OSASIS
*   **权重**：论文未明确提及开源权重，但使用公开预训练模型（FFHQ上的DDIM $\epsilon_\theta^A$）。
*   **数据集**：FFHQ（公开），AAHQ, MetFaces（需申请或自行获取），AFHQ-dog, LSUN-church, DeepFashion（公开）。
*   **关键超参**：编码时间步 $t_0$，SPN系数 $\lambda_{SPN}$（实验表明0.5左右较优），特征注入分界点 $f_{ch}$，文本优化步数/学习率（详见补充材料）。
