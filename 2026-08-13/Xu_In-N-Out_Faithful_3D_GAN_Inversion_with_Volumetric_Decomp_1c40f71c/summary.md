---
title: "In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xu_In-N-Out_Faithful_3D_GAN_Inversion_with_Volumetric_Decomposition_for_Face_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:51:00"
---

# 论文速读：In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing

## 一句话总结
本文提出一种基于体素分解的3D GAN反演方法（In-N-Out），通过将输入图像中的分布内（InD）人脸内容与分布外（OOD）物体/妆容显式分离为两个独立的神经辐射场，并结合复合体渲染技术，在实现高保真重建的同时有效保留了预训练3D生成模型（EG3D）的语义编辑能力。

## 研究问题与动机
- **核心问题**：现有3D-aware GAN反演方法（如基于EG3D的方法）在处理包含重妆容、面具、大眼镜等OOD成分的输入时，难以在重建保真度与编辑可塑性之间取得平衡。
- **动机1**：预训练的3D GAN通常仅在自然人脸数据集（如FFHQ）上训练，缺乏对复杂纹理或遮挡物的建模先验。强制用单一潜在码重建OOD内容会导致重建失真，或被迫微调生成器而破坏潜在空间的编辑属性。
- **动机2**：现有GAN反演方法多假设输入可由单个潜在码完整表示，无法区分“可编辑的人脸主体”与“不应被编辑的OOD背景/附件”，导致重构-editability trade-off 恶化。
- **动机3**：视频场景下OOD物体可能存在轻微运动或形变，静态单一辐射场难以稳定表征，需要引入时间维度的适应性设计与解耦机制。

## 核心贡献（创新点）
- **提出InD/OOD双辐射场分解架构**：将3D人脸反演显式拆分为自然人脸（InD）与异常成分（OOD）两个独立辐射场，本质区别在于首次将复合体渲染（composite volume rendering）引入3D GAN反演，而非像ChunkyGAN那样仅做2D图像级segmentation拼接。
- **设计学习型混合权重与熵正则化**：为OOD辐射场引入可学习的融合权重 $b$ 并辅以二进制熵损失 $\mathcal{L}_b$，强制空间位置在InD/OOD间二选一，避免成分混叠干扰后续语义编辑。
- **保留预训练模型可编辑性的正则策略**：通过 $w$ 均值正则 $\mathcal{L}_w$ 与风格变差正则 $\mathcal{L}_\Delta$ 约束InD潜在码，确保即便引入OOD组件后，原始StyleGAN的编辑轴（如微笑、年龄、表情）仍可有效作用且身份不漂移。
- **仅微调SR模块的廉价高分辨率适配**：针对OOD tri-plane破坏原SR模块输入分布的问题，提出冻结主干、仅微调EG3D的SR模块即可输出512×512高质量图像，避免重新训练整个生成器。

## 方法详解
- **骨干网络**：采用 EG3D，其利用 tri-plane 表示3D几何，沿视线 $\mathbf{r}$ 采样点投影至三个正交2D平面经双线性插值聚合，再由解码器 MLP 预测颜色与密度，生成128×128低分辨率特征图，最后经SR模块上采样至512×512。
- **InD 人脸反演**：冻结EG3D的预训练tri-plane与解码器，优化每帧潜在码 $w_t \in \mathbb{R}^{14 \times 512}$，并施加 $\mathcal{L}_w = ||w_t - \bar{w}||_2^2$（$\bar{w}$ 为FFHQ采样均值）与 $\mathcal{L}_\Delta$ 正则，保持潜在码位于预训练分布内。
- **OOD 内容建模**：引入额外随机初始化的 tri-plane $\mathbf{T}^O$ 与逐帧潜在码 $\phi_t \in \mathbb{R}^{32}$，新解码器 $D^O$ 输出颜色 $\mathbf{c}^O$、密度 $\sigma^O$ 及融合权重 $b \in [0,1]$。
- **复合体渲染**：沿光线对InD与OOD两路辐射场加权积分合成：$\mathbf{C}^C(\mathbf{r}) = \sum_k T^C(t_k)\big(b \alpha^O \sigma^O \delta_k \mathbf{c}^O + (1-b) \alpha^I \sigma^I \delta_k \mathbf{c}^I\big)$，其中 $T^C$ 为累积透射率。
- **损失函数**：低分辨率阶段总损失 $\mathcal{L}^{LR} = \sum_t \mathcal{L}_t^C + \lambda_\Delta \mathcal{L}_\Delta + \lambda_w \mathcal{L}_w + \lambda_\mathcal{D} \mathcal{L}_\mathcal{D}$，其中 $\mathcal{L}_t^C$ 包含LPIPS与像素MSE，$\mathcal{L}_b$ 以二元熵惩罚 $b$ 偏离0/1，单图输入时额外引入MiDaS深度正则 $\mathcal{L}_\mathcal{D}$。
- **SR微调**：固定所有tri-plane与解码器，仅以 $\mathcal{L}^{SR} = ||\mathbf{x}-\hat{\mathbf{x}}||_2^2 + \mathcal{L}_{LPIPS}(\mathbf{x},\hat{\mathbf{x}})$ 微调SR模块100轮，$\mathbf{x}$ 为原图，$\hat{\mathbf{x}}$ 为SR输出。
- **编辑流程**：重建完成后，仅对 $w_t$ 施加 InterfaceGAN 或 StyleCLIP 定义的编辑方向，OOD组件完全冻结，从而
