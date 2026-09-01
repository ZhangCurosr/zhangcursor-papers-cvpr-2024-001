---
title: "TIGER: Time-Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ren_TIGER_Time-Varying_Denoising_Model_for_3D_Point_Cloud_Generation_with_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:42:39"
field: "3D点云生成"
keywords: ["3D point cloud generation", "diffusion model", "time-varying mask", "position encoding", "Transformer", "CNN"]
innovations: ["时间自适应双流去噪架构，通过可学习时间掩码动态融合全局Transformer特征与局部CNN特征", "提出PSPE和BλPE两种新型3D连续位置编码，保留相对位置线性表达能力"]
benchmarks: ["ShapeNetv2"]
---

# 论文速读：TIGER: Time-Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process

## 一句话总结
本文提出 TIGER，一种时间自适应双流去噪扩散模型，通过可学习时间掩码动态调节 Transformer 全局特征与 CNN 局部特征的权重，实现高质量 3D 点云生成。

## 研究问题与动机
- 现有 3D 点云扩散模型（如 DPM、PVD、LION）均直接套用专为 2D 图像设计的 UNet/CNN 架构，缺乏针对点云特性定制的去噪网络设计。
- CNN 受限于感受野，难以在扩散早期快速捕捉全局形状分布；而 Transformer 注意力机制擅长建模长程依赖，适合早期粗形状构建。
- 卷积在局部细节建模上表现更优，应在扩散后期占据更高权重，但现有方法无法实现这种时间自适应的权重分配。
- 点云的排列不变性要求引入有效的 3D 位置编码，传统图像位置编码方式不适用于 3D 空间中的无序点集。

## 核心贡献（创新点）
- **时间自适应双流架构**：设计结合浅层 CNN 与 Transformer 的双流去噪网络，通过可学习的时间掩码在每个 timestep 动态重加权全局与局部特征，区别于以往固定权重或单流架构。
- **两种新型 3D 连续位置编码**：提出相位移位置编码（PSPE）和 Base-λ 位置编码（BλPE），在连续空间内保留相对位置的线性表达能力，优于不可学习的标准编码和可学习编码。
- **位置感知自注意力机制**：将位置关系矩阵显式乘入注意力图，使 Transformer 在聚合特征时充分利用 3D 空间位置信息，区别于标准自注意力。

## 方法详解
- **扩散过程设定**：前向过程按预设噪声调度 $\beta_t$ 逐步添加高斯噪声：$q(\mathbf{X}_t|\mathbf{X}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\mathbf{X}_{t-1}, \beta_t\mathbf{I})$；反向过程学习噪声预测 $\epsilon_\theta(\mathbf{X}_t, t)$，优化 MSE 损失 $\mathcal{L} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(\mathbf{X}_t, t)\|_2^2$。
- **点云编码器**：将噪声点云体素化到 $L\times L\times L$ 网格，经 3D 卷积 + Swish + GroupNorm 得到隐式体素特征 $\hat{\mathbf{V}}$；通过最远点采样降采样至 $M$ 个点，再用三线性插值查询体素特征，得到潜点云 $\hat{\mathbf{X}}_t \in \mathbb{R}^{M\times d}$。
- **Tokenization**：使用 Dual PatchNorm（两层 LayerNorm 间夹 MLP）将潜特征投影到 Token 空间：$\mathbf{T} = \mathrm{LN}(\mathrm{MLP}(\mathrm{LN}(\hat{\mathbf{X}}_t)))$。
- **PSPE 位置编码**：对 x、y、z 各轴施加不同相位的正弦/余弦函数，最大化轴向相位差（$4\pi/3$）以区分坐标轴，同时保证相对位置的线性可表示性。
- **BλPE 位置编码**：将三维坐标压缩为单标量 $pos = \lambda^2 z + \lambda y + x$（$\lambda=1000$），再用标准正弦编码，提升通道利用率。
- **位置感知自注意力**：通过位置编码计算位置关系矩阵 $\mathbf{H}=\mathrm{Softmax}((\mathbf{P}_{emb}\mathbf{W}_p)(\mathbf{P}_{emb}\mathbf{W}_p)^T)$，将其与标准 QK^T 逐元素相乘后输入 Softmax，使注意力图具备位置感知能力。
- **时间掩码生成器**：将 timestep 编码为正弦位置嵌入，经两层 MLP + LeakyReLU + Sigmoid 生成 CNN 分支掩码 $\mathbf{M}_c$，Transformer 分支掩码为 $\mathbf{M}_{tr}=\mathbf{1}-\mathbf{M}_c$，两者之和恒为 1；对两分支特征做 MLP 对齐维度后，按掩码加权融合：$\hat{\mathbf{X}}_F = \mathrm{MLP}(\mathbf{X}_c^*\odot\mathbf{M}_c + \mathbf{X}_{tr}^*\odot\mathbf{M}_{tr})$。
- **解码器**：将融合特征体素化为 $\tilde{\mathbf{V}}_F$，以原始点云坐标 $\mathbf{X}_t$ 为查询点进行三线性插值，再经投影矩阵得到噪声预测 $\epsilon_\theta \in \mathbb{R}^{N\times 3}$。

## 实验与结果
- **数据集**：ShapeNetv2，预处理后取飞机、椅子、汽车三类，每形状采样 2048 个点。
- **评估指标**：1-NN 准确率（CD 和 EMD 两种距离度量），分数越接近 50% 表示质量与多样性越好。
- **主要结果（Car 类，1-NN 准确率）**：TIGER 在 CD 度量下达到 54.12%，EMD 度量下达到 50.24%，均优于 PVD（54.55%/53.83%）和 LION（53.41%/51.14%）。
- **飞机类提升**：TIGER EMD 结果为 56.26%，较 PVD 提升 13.45%（相对），较 LION 提升 8.12%，归因于 Transformer 对全局分布的高效聚合。
- **训练效率**：TIGER 训练耗时 164 GPU 小时，仅为 LION（550h）的约 1/4，推理时间 9.73s，接近 PVD（8.46s）且显著快于 LION（27.12s）。
- **消融结论**：channel-wise 时间掩码（CD=59.43，EMD=53.51）优于 scalar 掩码（CD=60.79，EMD=54.12）；BλPE + 位置感知注意力组合取得最优结果（CD=59.43，EMD=53.51）。
- **权重趋势**：CNN 分支权重随 timestep 单调递增，验证了早期依赖全局形状、后期依赖局部细节的发现；飞机和椅子类别权重变化曲线比汽车更剧烈，可能与形状复杂度相关。

## 相关工作脉络
- **PVD [55] / LION [53]**：采用 PVCNN 作为扩散去噪骨干，仅依赖卷积建模局部邻域，缺乏全局感受野；TIGER 通过引入 Transformer 分支弥补此缺陷，并实现时间自适应融合。
- **DPM [34]**：首个将扩散模型应用于 3D 点云生成工作的代表，同样使用 CNN 架构；TIGER 在相同范式基础上探索了双流结构与时变权重设计。
- **PointFlow [51]**：基于归一化流的 3D 点云生成方法，使用 1D 卷积+max-pooling 编码；TIGER 在扩散框架下取得了更优的生成质量和多样性。
- **Point Transformer [54] / PCT [16]**：将 Transformer 应用于点云处理的前作；TIGER 并非简单套用，而是将 Transformer 限定于全局分支并与 CNN 分工协作，且引入专用位置编码。
- **SetVAE [24]**：基于 VAE 的集合结构化数据生成方法；TIGER 在扩散框架下显著超越其在 3D 点云生成任务上的表现。

## 局限性与未来方向
- **无条件生成限制**：当前模型无法控制生成对象的类别，可通过将类别编码融入潜空间并训练条件扩散模型来解决。
- **双网络结构效率不足**：全局与局部特征分别通过独立子网络提取，计算效率有损失；未来可探索单网络内实现时间自适应属性的方案。
- **泛化验证范围有限**：实验仅在 ShapeNetv2 三类上进行定量对比，通用模型（55 类）仅在定性展示，未进行系统性定量评估。

## 研究启发与可借鉴点
- **时间自适应特征融合**：将 timestep 信息转化为可学习掩码以动态调节不同模块权重，该设计可迁移至其他扩散生成任务（如图像、点云补全、上采样）。
- **位置编码设计思路**：PSPE 通过相位偏移区分坐标轴、BλPE 通过多项式压缩保持相对位置线性性，两种策略均可直接复用于其他 3D 点云感知任务。
- **位置感知注意力机制**：将位置关系显式注入注意力图的做法，增强了 Transformer 在无序点云上的空间建模能力，可与各类点云 Transformer 骨干结合使用。
- **实验设计借鉴**：用 1-NN 准确率同时衡量质量与多样性、可视化 timestep 权重变化曲线验证假设，均为可复用的实验分析方法。

## 关键术语表
**Diffusion Process**：通过在数据上逐步添加高斯噪声构建前向过程，再学习反向去噪过程以从噪声中生成新数据的生成范式。
**Time-varying Mask**：根据扩散 timestep 动态生成的可学习权重掩码，用于在不同去噪阶段调节全局与局部特征的融合比例。
**PSPE（Phase-Shifted Position Encoding）**：利用不同频率和相位的正弦/余弦函数为 3D 点坐标生成位置编码，通过最大化轴向相位差区分 x/y/z 轴。
**BλPE（Base-λ Position Encoding）**：将三维坐标通过多项式 $pos=\lambda^2 z+\lambda y+x$ 压缩为单一标量后再施加正弦编码，以提高通道利用率的位置编码方法。
**Position-Aware Self-Attention**：在标准自注意力计算中显式乘入由位置编码生成的位置关系矩阵，使注意力分布具备 3D 空间感知能力的机制。
**1-NN Accuracy**：用最近邻分类器在生成样本与真实样本混合集上的准确率评估生成质量与多样性的指标，越接近 50% 表现越好。
**Chamfer Distance (CD)**：衡量两个点云之间最近点对距离的平方和，用于评估点云生成的几何保真度。
**Earth Movers' Distance (EMD)**：衡量将一个点云分布匹配到另一个点云分布所需的最小代价，对全局形状分布更为敏感。

## 可复现要素
- **数据集**：ShapeNetv2，经 PointFlow 预处理，代码仓库：https://github.com/Zhiyuan-R/Tiger-Diffusion。
- **代码/权重**：代码已开源（见论文提供的 GitHub 链接），论文未明确声明预训练权重是否开源。
- **关键超参**：点云采样数 N=2048，降采样后点数 M（由 furthest point sampling 确定），体素分辨率 L（论文未明确给出具体数值），位置编码参数 λ=1000，训练平台为 NVIDIA V100 GPU。
