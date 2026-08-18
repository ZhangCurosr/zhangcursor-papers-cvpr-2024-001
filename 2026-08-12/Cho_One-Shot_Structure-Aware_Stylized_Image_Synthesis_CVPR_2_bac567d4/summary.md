---
title: "One-Shot Structure-Aware Stylized Image Synthesis"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cho_One-Shot_Structure-Aware_Stylized_Image_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:18:01"
field: "图像风格迁移与生成"
keywords: ["Image Stylization", "Diffusion Models", "One-shot Learning", "Structure Preservation", "Style Transfer", "Latent Code Disentanglement"]
innovations: ["提出结构-语义显式解耦的扩散模型单样本风格迁移框架", "设计轻量级结构保持网络（SPN）增强结构完整性", "实现无需数据集的零样本域适应单样本风格迁移"]
benchmarks: ["FFHQ", "AAHQ", "MetFaces", "ArtFID", "ID Similarity", "Structure Distance"]
---

# 论文速读：One-Shot Structure-Aware Stylized Image Synthesis

## 一句话总结
本文提出 OSASIS，一种基于扩散模型（Diffusion Model）的单样本风格迁移方法，通过显式解耦图像的结构与语义信息，在保持输入图像结构完整性的同时，将单张参考图像的风格有效迁移，并在 OOD 参考图像处理和文本驱动编辑任务中展现出优于现有 GAN 和 Diffusion 基线的性能。

## 研究问题与动机
1. **GAN 风格迁移的结构保持难题**：现有基于 GAN 的单样本风格迁移方法在处理罕见结构或复杂物体（如手、麦克风）时，难以保持输入图像的原始结构，且容易产生结构性伪影。
2. **扩散模型在风格迁移中的局限**：虽然扩散模型在图像生成上表现优异，但直接用于风格迁移时，难以兼顾高保真度与结构保持，现有方法（如 DiffuseIT、InST）在处理域间隙（Domain Gap）和精细结构控制上存在不足。
3. **风格与结构的纠缠问题**：GAN 等方法在推理时无法有效分离参考图像的结构与风格，导致参考图像中的结构信息泄露到生成结果中。
4. **低密度区域图像的处理能力**：对于训练集中罕见出现的属性（低密度区域图像），现有方法的结构保持能力显著下降，需要更强的结构感知机制。

## 核心贡献（创新点）
1. **结构-语义解耦框架**：提出通过结构潜在码（Structural Latent Code）和语义潜在码（Semantic Latent Code）分别控制图像的结构和内容，实现风格迁移中的结构感知。
   - 与已有工作的区别：不同于 DiffuseIT 等直接利用扩散模型隐式保持结构的方法，OSASIS 显式分离并控制结构和语义信息。
2. **结构保持网络（SPN）**：设计一个轻量级 SPN，通过 1x1 卷积有效保留输入图像的空间信息和结构完整性，缓解编码过程中噪声添加导致的结构丢失。
   - 与已有工作的区别：这是首次针对扩散模型风格迁移任务引入专门的结构保持模块，弥补了 DiffAE 编码过程固有的结构损失问题。
3. **无需数据集的单样本微调策略**：利用预训练扩散模型生成域内和域外参考图像，结合 CLIP 方向损失实现无数据集的单样本风格迁移模型微调。
   - 与已有工作的区别：相比 MTG 等方法需要特定训练数据，OSASIS 仅需单张参考图像即可完成风格迁移，且无需额外数据集。
4. **文本驱动的属性编辑集成**：通过直接优化语义潜在码并结合 CLIP 方向损失，实现文本驱动的图像属性编辑与风格迁移的统一框架。
   - 与已有工作的区别：不同于 DiffusionCLIP 仅关注文本引导编辑，OSASIS 将文本编辑与图像风格迁移无缝整合。

## 方法详解
**整体框架**：OSASIS 基于预训练的 DDIM，通过引入语义编码器（来自 DiffAE）和结构保持网络（SPN）实现风格迁移。

**关键设计**：
1. **结构潜在码提取**：使用 Forward DiffAE 将输入图像编码到特定时间步 $t_0$，得到结构潜在码 $\mathbf{x}_{t_0}^{in}$，该时间步可调节结构保持强度。
2. **语义潜在码控制**：利用 DiffAE 的语义编码器 $\mathrm{Enc}_{\phi}$ 提取语义潜在码 $\mathbf{z}_{sem}$，在推理时分别对 UNet 的低层和高层特征图进行条件控制，实现风格和内容的混合。
3. **结构保持网络（SPN）**：采用 1x1 卷积网络 $SPN(\cdot)$ 提取空间信息，通过公式 $\mathbf{x}_t' = \mathbf{x}_t + \lambda_{SPN} * \mathbf{x}_t^{SPN}$ 将结构信息融合到去噪过程中，$\lambda_{SPN}$ 控制融合强度。
4. **训练策略**：冻结预训练 DDIM $\epsilon_{\theta}^A$ 和语义编码器，复制得到 $\epsilon_{\theta}^B$ 进行微调。使用三种损失函数：
   - **Cross-domain loss**：对齐域间变化方向
   - **In-domain loss**：测量域内变化相似性
   - **Reconstruction loss**：包含 $L_1$ 损失、感知相似性损失和 CLIP embedding 损失，确保风格重建质量
5. **采样过程**：推理时将风格图像的 $\mathbf{z}_{sem}^{style}$  conditioning 到 UNet 低层特征图（传递风格），将输入图像的 $\mathbf{z}_{sem}^{in}$  conditioning 到高层特征图（传递内容），并结合结构潜在码 $\mathbf{x}_{t_0}^{in}$ 保持结构。

## 实验与结果
**数据集**：主要在 FFHQ 数据集上进行评估，并扩展到 AFHQ-dog、LSUN-church、DeepFashion 等数据集验证泛化能力。

**评估指标**：ArtFID（风格迁移相关性）、ID Similarity（内容保持，使用 ArcFace）、Structure Distance（结构保持）。

**主要结果**（Table 1，针对低密度图像）：
- **ArtFID**：OSASIS 在 AAHQ (34.89)、MetFaces (43.20)、Prev (33.20) 上均优于所有基线方法
- **ID Similarity**：OSASIS 达到 0.6825/0.7323/0.7029，显著高于 GAN 方法（MTG: 0.3730/0.4656/0.4063，JoJoGAN: 0.5145/0.5207/0.4743）
- **Structure Distance**：OSASIS 在 MetFaces (0.0295) 和 Prev (0.0391) 上表现最佳，略优于 DiffuseIT
- **最强提升**：相比 JoJoGAN+HFGI，OSASIS 的 ID Similarity 平均提升约 20-30%，Structure Distance 降低约 15-25%

**实验结论**：OSASIS 在低密度区域图像上展现出更强的结构保持能力，特别是在处理训练集中罕见属性时表现优异。在 OOD 参考图像处理上，能够成功提取纯风格信息而避免结构伪影。

## 相关工作脉络
1. **MTG [39] 与 JoJoGAN [2]**：作为 GAN 基线方法，使用 CLIP 方向损失进行单样本域适应，但依赖传统逆方法难以保持多样结构；OSASIS 针对相同任务提出基于扩散模型的替代方案。
2. **DiffuseIT [15]**：早期扩散模型风格迁移方法，缺乏针对域间隙的专门处理；OSASIS 通过跨域/域内损失组合更有效地桥接域间差异。
3. **InST [37]**：使用文本逆技术提取风格概念并条件生成，但存在颜色传递不准确和过度风格集中的问题；OSASIS 通过语义码的条件控制实现更精细的风格-内容平衡。
4. **DiffusionCLIP [13] 与 Asyrp [17]**：聚焦文本引导的图像编辑；OSASIS 将其思想扩展到图像引导的风格迁移任务，实现了更统一的处理框架。
5. **DiffAE [23]**：提供了语义编码器和可解码潜在空间的基础；OSASIS 在此基础上引入结构编码和 SPN，解决了原有方法在风格迁移中的结构保持不足。

## 局限性与未来方向
**自述局限**：
1. **训练时间较长**：相比对比方法，OSASIS 的训练时间显著增加，这是结构保持增强的 trade-off。
2. **每风格需单独训练**：当前方法需要为每个目标风格图像进行单独微调，限制了多风格快速部署的场景适用性。

**可推断的局限**：
1. 对于极端结构差异的参考图像，可能仍存在风格丢失或结构扭曲的风险。
2. 高 $λ_{SPN}$ 值可能导致过度强调结构而损害风格质量，需要精细调参。

**未来方向**：
1. 优化训练效率，探索更高效的微调策略。
2. 实现无需每风格单独训练的通用风格迁移模型。
3. 扩展到其他图像编辑和合成任务。

## 研究启发与可借鉴点
1. **结构-语义显式解耦的设计思路**：将图像分解为结构潜在码和语义潜在码分别控制，为其他图像转换任务（如超分辨率、去噪）提供了可复用的架构模式。
2. **轻量级结构保持模块的设计**：SPN 使用简单的 1x1 卷积即可有效保留空间信息，证明在复杂生成任务中，轻量级结构增强模块可能比复杂的注意力机制更高效。
3. **无数据集单样本微调策略**：通过预训练模型生成辅助数据并结合 CLIP 损失实现零样本域适应，为资源受限的场景提供了实用方案。
4. **低密度图像评估协议**：利用 DiffAE 的随机重建误差来识别训练集低密度图像，为模型鲁棒性评估提供了可量化的测试协议。
5. **文本驱动编辑与风格迁移的统一框架**：通过优化语义潜在码实现文本编辑，与风格迁移共享同一套参数，展示了多任务学习的潜力。

## 关键术语表
**OSASIS**：One-Shot Structure-Aware Stylized Image Synthesis 的缩写，本文提出的基于扩散模型的单样本风格迁移方法。
**Structural Latent Code** ($\mathbf{x}_{t_0}$)：通过 Forward DiffAE 编码到特定时间步的潜在表示，主要携带图像的结构和轮廓信息。
**Semantic Latent Code** ($\mathbf{z}_{sem}$)：由 DiffAE 语义编码器提取的潜在变量，具有线性、可解码和语义有意义的特性，控制图像的内容和风格。
**Structure-Preserving Network (SPN)**：轻量级 1x1 卷积网络，用于在扩散去噪过程中增强结构信息的保留。
**Cross-domain Loss**：衡量域 A 到域 B 变化方向一致性的损失函数，确保风格迁移的方向正确性。
**In-domain Loss**：测量同一域内变化相似性的损失函数，补充跨域损失的不足。
**Low-density Image**：训练集中罕见属性或结构的图像，通过高 LPIPS 重建误差识别，用于评估模型的结构保持鲁棒性。
**CLIP Directional Loss**：基于 CLIP 嵌入空间的方向损失，用于引导图像风格或属性的定向变化。

## 可复现要素
- **数据集**：FFHQ（公开）、AAHQ（公开）、MetFaces（公开）、AFHQ-dog、LSUN-church、DeepFashion（均公开）
- **代码开源**：是，GitHub: https://github.com/hansam95/OSASIS
- **预训练权重**：使用公开的 FFHQ 预训练 DDIM 和 DiffAE
- **关键超参数**：
  - 编码时间步 $t_0$：论文未明确提及具体数值
  - SPN 权重系数 $\lambda_{SPN}$：最优值为 0.1（Table 2）
  - 特征图条件切换点 $f_{ch}$：论文未明确提及
  - 损失函数权重：详细设置在补充材料中
