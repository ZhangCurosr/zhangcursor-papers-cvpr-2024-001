---
title: "Z-sup-sup-Zero-shot-Style-Transfer-via-Attention-Reweighting"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Deng_Z_Zero-shot_Style_Transfer_via_Attention_Reweighting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:52:37"
field: "图像风格迁移"
keywords: ["zero-shot style transfer", "diffusion model", "attention reweighting", "latent diffusion", "cross-attention"]
innovations: ["提出基于预训练 latent diffusion 的零样本风格迁移框架，无需训练/微调", "设计 cross-attention reweighting 机制将加权系数嵌入 Softmax 内部实现自适应内容-风格平衡", "双路径 DDIM 反转确保内容和风格特征在时间维度对齐"]
benchmarks: ["User Study (55 participants, 1760 votes)", "Qualitative comparison with ArtFlow, AdaAttN, StyTr², CAST, InST, VCT"]
---

# 论文速读：Z*-STAR: Zero-shot Style Transfer via Attention Reweighting

## 一句话总结
论文提出了 **Z-STAR（Z\*）**，一种基于预训练 latent diffusion 模型的零样本（训练无关）图像风格迁移方法，通过双路径 DDIM 反转变为内容和风格图像构建去噪轨迹，并设计 cross-attention reweighting 机制在不重新训练的情况下实现风格与内容的自适应平衡融合。

## 研究问题与动机
1. **传统方法对精细风格表征能力有限**：基于 Gram 矩阵的方法（如 AdaIN、StyTr²、ArtFlow）仅捕捉全局二阶统计量，无法迁移局部精细风格特征（如笔触、头发和眼睛细节）。
2. **现有扩散模型方法需额外训练**：InST、VCT 等方法需要将每个风格图像蒸馏为 style embedding 并训练约 20 分钟，易出现风格偏差或内容失真。
3. **朴素 cross-attention 破坏内容结构**：直接用风格特征的 Key/Value 查询内容 Query 时，Softmax 会过度放大弱相关像素的注意力权重，导致内容结构丢失（如图 3 所示低相似度区域）。
4. **文本提示过于粗糙**：纯文本条件难以表达精细的风格细节，缺乏图像级引导的精度。

## 核心贡献（创新点）
1. **零样本风格迁移框架**：直接利用预训练 latent diffusion（Stable Diffusion v1.5）的生成先验，无需任何训练/微调即可完成风格迁移，与 InST/VCT 等需要训练 style embedding 的方法本质不同。
2. **双路径 DDIM 反转机制**：同时对内容图像和风格图像进行独立 DDIM 反转，获取时间对齐的去噪轨迹，解决了风格特征需随去噪过程自适应演化的问题。
3. **Cross-attention Reweighting 策略**：将加权系数 λ 嵌入 Softmax 内部而非外部线性加权，使注意力权重能自适应根据内容-风格相关性动态调整，解决了简单加法引入噪声项 C 导致的次优问题。
4. **支持条件控制与多风格扩展**：通过掩码函数 φ(·) 实现区域化风格控制，并可将多风格图像的 Key/Value 并行拼接，为下游应用提供灵活接口。

## 方法详解
**整体流程**：基于 Stable Diffusion v1.5，文本 prompt 设为空字符串，去噪步数 T=30 步。

**双路径网络（Dual-path Networks）**：
- 分别对内容图像 I_c 和风格图像 I_s 进行 DDIM 反转，得到去噪轨迹 {x^[0:T]^c} 和 {x^[0:T]^s}。
- 在去噪过程中，提取 U-Net 各层的空间特征 f_c 和 f_s，作为后续 attention 计算的输入。
- 最终生成：f̂_c = G_θ(ε_Ic, {f_s, f_c}, T)，即融合两种特征后经 T 步去噪得到风格化结果。

**Cross-attention Reweighting**：
- 朴素设置（Eq.7）：f̂_c = Softmax(Q_c K_s^T / √d) V_s，倾向过度强调风格而丢失内容。
- 简单加法（Eq.8）：f̂_c = λ·Attn(Q_c, K_s, V_s) + (1-λ)·Attn(Q_c, K_c, V_c)，但 Softmax 会放大负相关像素的权重，需手动调 λ。
- **提出的重加权方案（Eq.11-12）**：
  - A' = σ([λ · Q_c K_s^T / √d, Q_c K_c^T / √d])
  - f̂_c' = A' * [V_s; V_c]^T
  - 将 λ 内置于 Softmax，使相关性强的像素获得更高权重，弱相关像素自动被抑制，无需手动补偿 (1-λ)。
- **条件控制（Eq.16-17）**：通过掩码区域 Ω 和函数 φ(x) 将非目标区域设为 −∞，实现局部风格迁移。
- **多风格扩展（Eq.18）**：将 N 个风格图像的 Key/Value 并行拼接，但受限于显存留待未来研究。

**关键超参**：λ=1.2（固定），注意力模块注入 U-Net 第 10-15 层，去噪步数 5-30 步。

## 实验与结果
**数据集与基线**：与 ArtFlow、AdaAttN、IEST、StyTr²、CAST、QuanArt、InST、VCT 等 SOTA 方法对比。

**定性结果**：如图 6 所示，Z-STAR 能保留内容结构的同时准确迁移笔触等精细风格，优于 InST/VCT 的内容/风格偏差问题，以及 CNN/Transformer 方法（如 CAST、StyTr²）的风格不忠实问题。

**用户研究（Table 1）**：
- 55 名参与者，100 组测试，共收集 1760 票。
- **内容保留**：Z-STAR 平均得票率最高，仅 CAST（23.6%）和 StyTr²（45.0%）在用户偏好上与其接近。
- **风格表征**：CAST（56.4%）和 StyTr²（51.4%）略优，但与 Z-STAR 差异不显著。
- **整体效果**：Z-STAR 显著优于所有对比方法，证明其在内容与风格间取得了最佳平衡。

**消融实验**：
- 注意力注入步数：第 5 步开始、第 30 步结束效果最佳（图 7）。
- 注意力注入层数：U-Net 解码器高层（第 10-15 层）效果最优（图 8）。
- Reweighting 有效性：移除内容分量或仅用简单加法均导致内容丢失或风格不足（图 9）。
- λ 敏感性：λ ≥ 1.2 时效果稳定（图 10）。

## 相关工作脉络
1. **Gatys et al. [15] / AdaIN [18]**：开创 CNN 风格迁移范式，使用 Gram 矩阵衡量风格，但仅捕捉全局二阶统计，无法处理精细本地特征。
2. **StyTr² [13] / ArtFlow [2]**：分别利用 Transformer 长程建模和可逆流网络改进风格表达，但仍依赖固定损失约束，难以保证风格忠实度。
3. **CAST [50]**：引入对比学习替代 Gram 矩阵损失，在精细风格上有所提升，但受限于风格损失约束，仍出现"不像艺术作品"的问题。
4. **InST [51] / VCT [9]**：基于扩散模型的图像翻译/风格迁移方法，需为每个风格图像训练约 20 分钟的 style embedding，存在训练开销和风格/内容偏差问题。
5. **Prompt-to-Prompt [16] / Plug-and-Play [40]**：在文本到图像扩散过程中操控 cross-attention 以实现图像编辑，本文借鉴其 attention 分析思路但将其应用于无文本条件的纯图像引导场景。
6. **StyleDiffusion [24]**：通过 mapping network 将图像逆转为 context embedding，仍需额外网络学习；本文证明无需任何额外训练即可直接从图像中提取风格特征。

## 局限性与未来方向
1. **λ 参数固定**：虽实验表明 λ≥1.2 时鲁棒，但针对不同内容-风格图像对可能仍需自适应调整。
2. **多风格扩展受限于显存**：多风格并行处理的理论框架已提出，但受显存约束尚未深入探索（论文明确留待未来研究）。
3. **未公开代码与权重**：模型训练细节和实验代码均未开源，影响可复现性。
4. **仅使用 Stable Diffusion v1.5**：未验证在更新架构（如 SDXL）上的泛化能力。

## 研究启发与可借鉴点
1. **双路径同步反转设计**：同时反转内容和风格图像并确保时间对齐，为其他图像到图像生成任务（如图像翻译、风格化编辑）提供了可复用的特征对齐范式。
2. **Attention 内部重加权思想**：将加权系数嵌入 Softmax 而非外部线性组合，这一设计可有效避免简单加法引入的噪声项，可迁移至其他需要融合多源特征的注意力模块中。
3. **零样本利用预训练扩散先验**：证明了未经微调的 Stable Diffusion 本身已蕴含丰富的风格-内容分离表征能力，为减少训练开销、实现即插即用的图像编辑任务提供了新思路。
4. **区域化条件控制接口**：通过掩码函数 φ(·) 修改 attention 权重，为局部风格迁移、选择性编辑等应用提供了灵活的控制机制。

## 关键术语表
**Zero-shot Style Transfer**：无需针对特定风格图像进行训练即可实现风格迁移的方法。
**DDIM Inversion**：确定性采样逆过程，将真实图像转换为带噪声的潜在表示并记录去噪轨迹。
**Cross-attention Reweighting**：在 Softmax 内部嵌入加权系数，自适应调节内容特征查询风格特征的注意力权重。
**Dual-path Network**：同时对内容图像和风格图像执行独立 DDIM 反转，获取时间对齐的双轨迹特征。
**Latent Diffusion Model**：在压缩潜在空间而非像素空间执行去噪的扩散模型，典型代表为 Stable Diffusion。
**U-Net**：扩散模型中用于预测噪声的编码器-解码器结构，包含多层 self-attention 和 cross-attention 层。

## 可复现要素
- **数据集**：论文未明确说明使用的外部数据集，实验在自定义内容/风格图像对上进行评估。
- **代码/权重开源状态**：论文未提及代码或权重开源，模型基于开源的 Stable Diffusion v1.5 checkpoint。
- **关键超参**：Stable Diffusion v1.5；去噪步数 30 步；注意力模块注入 U-Net 第 10-15 层；去噪步数范围第 5-30 步；λ=1.2；文本 prompt 为空字符串。
