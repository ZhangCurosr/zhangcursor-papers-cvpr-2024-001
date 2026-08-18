---
title: "Z<sup>∗</sup>: Zero-shot Style Transfer via Attention Reweighting"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Deng_Z_Zero-shot_Style_Transfer_via_Attention_Reweighting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:16:58"
field: "图像风格迁移"
keywords: ["Zero-shot Style Transfer", "Diffusion Model", "Attention Reweighting", "Latent Diffusion", "Image Editing"]
innovations: ["提出零样本扩散风格迁移框架Z-STAR，无需训练即可利用预训练扩散先验", "设计交叉注意力重加权机制，在Softmax内部自适应平衡内容与风格特征", "通过双去噪路径和DDIM反转向实现内容-风格在时间维度的自然对齐"]
benchmarks: ["用户研究（55参与者，1760票）", "定性对比（ArtFlow/AdaAttN/IEST/StyTr2/CAST/QuanArt/InST/VCT）"]
---

# 论文速读：Z<sup>∗</sup>: Zero-shot Style Transfer via Attention Reweighting

## 一句话总结
本文提出了一种零样本（训练无关）图像风格迁移方法 Z-STAR，利用预训练扩散模型的生成先验知识，通过双去噪路径和交叉注意力重加权策略，在不重新训练/微调的情况下实现内容与风格的自然融合。

## 研究问题与动机
- 现有风格迁移方法（如 CNN-based AdaIN、Transformer-based StyTr²）依赖 Gram 矩阵或对比损失衡量风格相似性，难以捕捉精细风格细节（如笔触、纹理），且无法将风格特征正确映射到内容图像的对应区域。
- 近期基于扩散的方法（InST、VCT）需要对每张风格图像训练 style embedding（约 20 分钟/张），且存在内容失真或风格偏离的问题。
- 文本提示（text prompt）作为风格引导过于粗糙，无法精确描述细粒度艺术风格（如特定画作的笔触、色彩分布）。
- 直觉上，预训练的扩散模型已隐含学习了风格迁移的先验知识，如何在不训练的前提下直接利用这一先验是关键挑战。

## 核心贡献（创新点）
1. **零样本风格迁移框架**：直接利用预训练 Stable Diffusion（v1.5）的生成先验，无需对任何输入图像进行 fine-tuning 或训练，区别于 InST/VCT 需要 per-style 训练 embedding 的方案。
2. **双去噪路径设计**：同时维护内容和风格图像在扩散潜空间中的逆向轨迹，确保特征在时间维度上自然对齐，而非依赖固定文本 embedding。
3. **交叉注意力重加权机制**：在 Softmax 内部引入可学习/可调参数 λ，使注意力权重自适应地增强显著的相关值、抑制弱相关噪声，从根本上解决了朴素交叉注意力偏向风格而丢失内容结构的问题。
4. **条件控制扩展**：支持基于二元掩码的区域风格控制（将不需要迁移的风格区域设为 −∞），以及一对多风格迁移的扩展形式。

## 方法详解
**整体流程**（图 2）：给定内容图像 $I_c$ 和风格图像 $I_s$，分别通过 DDIM 反转向（inversion）获得扩散轨迹 $\boldsymbol{x}_{[0:T]}^c$ 和 $\boldsymbol{x}_{[0:T]}^s$；在去噪过程中提取 U-Net 各层的 spatial features $\{f_c\}$ 和 $\{f_s\}$；通过重加权交叉注意力融合后，经 T 步去噪得到风格化输出 $\hat{I}_c$。

**双路径生成**：
$$
I_s = \mathcal{G}_\theta(\epsilon_{I_s}, \{f_s\}, T), \quad I_c = \mathcal{G}_\theta(\epsilon_{I_c}, \{f_c\}, T)
$$
风格化结果：
$$
\hat{I}_c = \mathcal{G}_\theta(\epsilon_{I_c}, \{f_s, f_c\}, T)
$$

**注意力重加权核心公式**（Eq. 11-12）：
$$
A' = \sigma\left(\left[\lambda \cdot \frac{Q_c K_s^T}{\sqrt{d}}, \quad \frac{Q_c K_c^T}{\sqrt{d}}\right]\right)
$$
$$
\hat{f}_c' = A' * V'^T = \sigma\left(\left[\lambda \cdot \frac{Q_c K_s^T}{\sqrt{d}}, \quad \frac{Q_c K_c^T}{\sqrt{d}}\right]\right) * \begin{bmatrix} V_s \\ V_c \end{bmatrix}
$$
其中 $Q_c = K_c = V_c = f_c$（内容自注意力），$Q_c = f_c, K_s = V_s = f_s$（风格交叉注意力），$\lambda \geq 1.2$ 为风格缩放因子。关键在于 λ 被置于 Softmax 内部，使归一化自动平衡两路特征，避免朴素相加中 Softmax 放大小值噪声的问题（图 4 对比）。

**条件控制**（Eq. 16-17）：通过映射 $\phi(\cdot)$ 对指定区域 $\Omega$ 施加风格控制，$\phi(x) = -\infty$ 表示禁止该区域受风格影响，线性渐变实现自然边界过渡。

## 实验与结果
- **实现细节**：基于 Stable Diffusion v1.5 checkpoint，null 文本提示，30 步去噪；注意力模块注入第 5–30 步、U-Net 层 10–15。
- **对比基线**：ArtFlow、AdaAttN、IEST、StyTr²、CAST、QuanArt、InST、VCT（均包括 CNN/Transformer/Flow/Diffusion 类方法）。
- **定性结果**：Z-STAR 能保留内容结构的同时忠实传递风格笔触（图 6）；InST/VCT 存在内容偏差或风格失真；CNN/Transformer 类方法细节表现不足。
- **用户研究**（55 参与者，1760 票）：Z-STAR 在"整体效果"上显著优于所有对比方法；内容保持优于所有方法；风格表现略低于 CAST 和 StyTr² 但差异不显著（表 1）。
- **消融**：
  - 注入步骤过早（<第 5 步）丢失内容结构，过晚则风格弱化；最优为第 5–30 步。
  - 注入层过低（0–5）破坏内容，过高（5–10）风格不足；最优为 decoder 高分辨率层（10–15）。
  - 去掉内容项导致结构丢失；仅用简单相加（λ=0.5）丢失细节；重加权取得最佳平衡。
  - λ ≥ 1.2 时效果稳定，默认取 λ = 1.2。

## 相关工作脉络
1. **Gatys et al. [15]**：开创性提出 CNN 特征 + Gram 矩阵衡量风格，Z-STAR 与之本质区别在于无需训练、不依赖手工设计的风格损失。
2. **AdaIN [18] / StyTr² [13]**：基于特征统计匹配或 Transformer 长距离建模的风格迁移，Z-STAR 指出其 Gram 矩阵局限（二阶统计无法捕获细粒度局部风格）。
3. **CAST [50]**：用对比损失替代 Gram 矩阵风格损失，Z-STAR 认为其仍依赖固定 encoder，无法充分利用扩散模型的生成先验。
4. **InST [51] / VCT [9]**：基于扩散的风格迁移，需 per-style 训练 embedding（约 20 分钟/张），Z-STAR 在不训练前提下直接利用原始模型实现同等甚至更优效果。
5. **Prompt-to-Prompt [16] / Plug-and-Play [40] / MasaCtrl [6]**：通过修改 cross-attention/self-attention 实现文本引导编辑，Z-STAR 将注意力重加权从文本域扩展到图像域，解决零样本风格对齐问题。
6. **StyleDiffusion [24]**：用 mapping network 将风格图像反转为 context embedding，Z-STAR 避免额外网络，直接在 latent space 操作 attention。

## 局限性与未来方向
- 一对多风格迁移（N 张风格图融合至单内容图）因显存限制仅给出公式扩展，未做充分实验（Sec. 4.2 末段）。
- 超参数 λ 虽在 ≥1.2 时稳定，但仍需手动设定，缺乏自动自适应机制。
- 方法依赖预训练 Stable Diffusion v1.5，对高质量艺术风格图像迁移效果好，但面对极端风格偏移或低相似度的内容-风格对时效果未详细评估。
- 论文未提及大规模定量评估指标（如 FID、KID、LPIPS 等），主要依靠用户投票和定性比较。

## 研究启发与可见借鉴点
1. **注意力重加权的设计思路**（将权重系数嵌入 Softmax 内部而非外部加权求和）可迁移至其他需要融合多源特征的扩散生成任务（如图像编辑、条件生成），避免 Softmax 放大噪声的问题。
2. **双逆向轨迹 + 同步去噪对齐**的策略：同时维护多个输入的扩散轨迹并在相同时间步进行操作，可推广至多图融合、多模态对齐等场景。
3. **区域控制接口**（通过 −∞ 掩码实现空间选择性风格注入）设计简洁通用，可直接复用于扩散模型的各种 conditional generation 任务。
4. **零样本利用预训练扩散先验**的思路为减少 style transfer 的训练成本提供了新范式，可探索在其他生成任务（如颜色迁移、材质替换）中的迁移价值。
5. **消融中对注入步数和层数的系统分析**提供了扩散模型 attention injection 调参的实用经验，可直接指导同类工作的实验设计。

## 关键术语表
- **Zero-shot Style Transfer**：无需针对目标风格进行训练或微调，直接利用预训练模型实现风格迁移。
- **DDIM Inversion**：确定性采样逆过程，将真实图像转化为扩散轨迹，用于后续编辑或条件生成。
- **Cross-Attention Reweighting**：在 Softmax 内部引入缩放系数 λ，自适应平衡交叉注意力与自注意力的输出权重。
- **Latent Diffusion Model**：在压缩的潜空间（而非像素空间）执行去噪过程的扩散模型，典型代表为 Stable Diffusion。
- **Gram Matrix Style Loss**：通过特征图的二阶统计量（Gram 矩阵）衡量图像风格差异的经典损失函数。
- **Content Self-Attention**：用内容特征作为 Query/Key/Value 的自注意力操作，用于保持生成结果的结构一致性。
- **Style Cross-Attention**：用内容特征作为 Query、风格特征作为 Key/Value 的交叉注意力操作，用于注入风格信息。
- **Null Text Prompt**：空字符串作为文本条件，避免文本引导对风格迁移产生干扰。

## 可复现要素
- **模型**：Stable Diffusion v1.5 checkpoint（Hugging Face 公开，无需额外训练）。
- **代码**：论文未明确声明开源，但方法基于公开 Stable Diffusion，核心公式清晰可复现。
- **关键超参**：去噪步数 T=30，注意力注入步数 5–30，注入层 10–15，λ=1.2，文本 prompt 为空字符串。
- **数据集**：论文未使用特定 benchmark 数据集，以自定义内容-风格对进行展示和用户研究。
