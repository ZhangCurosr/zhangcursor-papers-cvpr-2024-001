---
title: "DiffusionAvatars-Deferred-Diffusion-for-High-fidelity-3D-Hea"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Kirschstein_DiffusionAvatars_Deferred_Diffusion_for_High-fidelity_3D_Head_Avatars_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:58:19"
field: "三维头 avatar 生成与动画"
keywords: ["3D Head Avatar", "Diffusion Model", "Neural Rendering", "NPHM", "ControlNet", "Deferred Diffusion"]
innovations: ["基于 ControlNet 的延迟扩散神经渲染器，结合 NPHM 几何与 2D 扩散先验", "通过 TriPlane 将可学习特征绑定至隐式 NPHM 表面", "通过跨注意力层直接注入 NPHM 表情编码以增强细粒度表情控制"]
benchmarks: ["NeRSemble (self-reenactment)", "用户研究 (Visual Quality, Driving Fidelity)"]
---

# 论文速读：DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars

## 一句话总结
本文提出了 DiffusionAvatars，一种基于扩散模型的神经渲染器，通过结合隐式 3D 头模型 (NPHM) 的几何先验与预训练 2D 潜在扩散模型 (LDM) 的图像生成能力，从多视角视频中合成具有高逼真度和精确 pose/expression 控制能力的可动画 3D 头部头像。

## 研究问题与动机
1.  **核心问题**：如何从个人多视角视频中创建高保真、可动画、且支持任意视角和表情控制的 3D 头部头像。
2.  **现有方法不足**：纯 2D 网络（如 GAN、Diffusion）生成的图像视觉效果虽好，但缺乏跨视角和时间的一致性，且对 3D 姿态/表情的精确控制有限；而基于 3D 表征（如 NeRF、3DMM）的方法能提供更好的几何一致性和控制性，但渲染结果的 photographic realism（照片级真实感）通常不及 2D 生成模型。
3.  **技术挑战**：该任务本质上是一个欠约束的 4D 光度重建问题。需要在保持精细几何控制的同时，利用强大的 2D 视觉先验来提升渲染细节和质感。

## 核心贡献（创新点）
1.  **将 ControlNet 范式应用于 3D 头像生成**：创新性地使用预训练的 Latent Diffusion Model (LDM) 作为骨干，并通过 ControlNet 架构条件化于 NPHM 网格的光栅化渲染图，从而继承了强大的 2D 图像合成先验，同时利用 3D 几何提供可控的 guidance。与仅使用 2D 条件（如 landmarks）的方法相比，提供了更准确的 3D 视图一致性。
2.  **通过 TriPlane 将可学习空间特征绑定至隐式几何表面**：针对 NPHM 无统一 UV 空间的问题，提出在 NPHM 的规范空间 (canonical space) 中，使用 TriPlane 和 Ambient Map 根据顶点规范坐标查询和绑定个人专属的可学习特征图。这与传统在显式网格（如 FLAME）上绑定神经纹理的方法不同，能更好地适应隐式 SDF 表示的几何细微变化。
3.  **提出通过跨注意力机制的直接表情条件注入**：除了光栅化渲染提供的粗略表情形状，还通过新增的跨注意力层，将 NPHM 的表情编码 ($z_{exp}$) 直接注入到 LDM 的 U-Net 中。这使得模型能够区分和利用表情编码中的细微细节，从而生成更富表现力、更准确的复杂表情（如口型、舌头），这是仅靠光栅化渲染难以实现的。

## 方法详解
1.  **基础模型**：以 Stable Diffusion v2.1 为骨干，采用 v-prediction 目标函数和重缩放的噪声调度进行训练，以提升收敛速度和泛化能力。
2.  **3D 几何代理**：使用 NPHM (Neural Parametric Head Model) 从多视角视频序列中拟合每个人的身份码 ($z_{id}$) 和每帧的表情码 ($z_{exp}^t$)。NPHM 输出 Signed Distance Field (SDF)，可通过 Marching Cubes 提取网格 $M_t$。
3.  **光栅化与渲染**：使用 nvdiffrast 将 NPHM 网格 $M_t$ 在目标视角 $\pi_t$ 下光栅化，生成包含法线、深度和**规范坐标渲染图** ($R_{can}^t \in \mathbb{R}^{H \times W \times (3+2)}$) 的多通道输入 buffer $R^t$。规范坐标用于后续的特征映射。
4.  **TriPlane 特征映射**：利用光栅化得到的规范坐标 $R_{can}^t$，分别在三个 2D 平面上查询空间特征 (TriPlane)，并对额外 2 维环境坐标查询环境特征图。这些特征图被拼接到光栅化 buffer $R^t$ 后，共同构成 ControlNet 的输入。
5.  **直接表情条件注入**：NPHM 的表情码 $z_{exp}^t$ 经过一个线性层 (EXP) 映射为 4 个表情 tokens ($f_{exp}^t$)，作为 Key 和 Value，插入到 LDM U-Net 的跨注意力层中，为模型提供细粒度的表情指导。
6.  **延迟扩散渲染**：训练时，将加噪后的潜变量 $x_\tau$、光栅化及特征输入 $R^t$、表情 tokens $f_{exp}^t$ 一同送入模型。模型由两部分组成：冻结的预训练 LDM (S) 和其下的 ControlNet (C)。损失函数为预测噪声与真实 v 值之间的 L2 损失：$\mathcal{L} = \mathbb{E}[||\mathcal{D}(x_\tau^t, R^t, f_{exp}^t) - v||_2]$。训练参数包括 ControlNet、跨注意力参数、EXP 层及 TriPlane/环境特征图。

## 实验与结果
1.  **数据集**：NeRSemble 数据集，包含 8 个人的多视角视频序列。共使用约 3300 个时序帧/人进行训练。
2.  **评估任务与基线**：
    *   **自重演 (Self-Reenactment)**：在未见视角和表情上测试。基线包括 NeRFace, DiffusionRig, DNR, DNR+GAN, MVP。
    *   **头像动画 (Avatar Animation)**：将另一人的表情驱动到生成的头像上。通过用户研究评估。
    *   **评估指标**：PSNR, LPIPS, JOD, AKD, CSIM, AED, APD。
3.  **主要结果**：
    *   **自重演**：在 8 个人上的平均结果表明，本文方法在所有客观指标上均取得最优。特别是 **LPIPS (0.081)** 和 **CSIM (0.882)** 提升显著，表明生成图像更清晰、感知质量更好，且身份一致性更高。PSNR (24.9) 也达到最高。
    *   **头像动画**：用户研究中，本文方法在视觉质量 (VQ: 4.02) 和表情驱动忠实度 (DF: 4.14) 上均大幅领先于所有基线（最高 VQ 4.02 vs 次高 DNR+GAN 3.06）。
4.  **消融实验**：证明了扩散模块、NPHM 几何代理（优于 FLAME）、直接表情条件注入、预训练 LDM 先验以及 TriPlane 特征映射对最终性能均有积极贡献。

## 相关工作脉络
1.  **Deferred Neural Rendering (DNR)**：早期工作，在像素空间训练神经网络渲染器，将神经纹理绑定到 BSFM 网格。本文借鉴了“延迟渲染”思想，但将渲染器替换为基于扩散的神经渲染器，并使用更准确的 NPHM 几何和 TriPlane 特征绑定，显著提升了逼真度和细节。
2.  **DiffusionRig**：近期工作，使用扩散模型并条件化于 FLAME 网格的光照渲染图。本文与其定位类似，但核心区别在于：(1) 使用隐式 NPHM 而非显式 FLAME 网格，几何表达更丰富；(2) 提出了 TriPlane 空间特征绑定和直接表情条件注入，增强了细节和表情控制能力；(3) 使用 ControlNet 范式。
3.  **NeRFace / MVP**：基于 NeRF 或体积原语的 3D 头像方法。本文相比这些方法，放弃了 3D 辐射场的显式表示，转而利用更成熟的 2D 扩散先验来追求更高的单帧图像质量，同时通过 3D 几何代理来维持视角一致性。
4.  **ControlNet / IPAdapter**：本文直接采用了 ControlNet 框架将预训练 LDM 适配为图像到图像的翻译器，并借鉴了 IPAdapter 的跨注意力注入机制来实现条件控制，是对现有可控扩散模型技术的创造性组合与应用。
5.  **风格化/视频转视频方法**：如 Tune-a-Video, MagicProp 等。本文方法专注于创建静态人物的 3D 头像，而非视频风格转换，且强调 3D 几何驱动的一致性，与这些纯 2D 视频处理方法不同。

## 局限性与未来方向
1.  **光照控制**：当前方法将光照烘焙到生成图像中，缺乏对独立光照属性的控制能力。未来可探索结合环境光照建模或分离渲染路径。
2.  **实时性**：依赖扩散模型的迭代去噪过程计算成本高，无法实时应用。未来可结合扩散模型蒸馏技术 (如 Consistency Models, Progressive Distillation) 加速推理。
3.  **全身/身体部分**：当前方法主要聚焦头部，颈部以下身体部分的处理未在文中详述，可考虑与身体表征方法结合。
4.  **极端表情与遮挡**：虽然通过 NPHM 和直接条件注入改进了复杂表情，但在极度夸张表情或严重自遮挡情况下，生成质量可能下降，需要更鲁棒的几何-外观联合建模。

## 研究启发与可借鉴点
1.  **"Deferred Diffusion" 架构范式**：将预训练 2D 扩散模型与 3D 几何代理（尤其是隐式表示）结合的思路，通过 ControlNet 提供几何 conditioning、通过额外注意力机制注入细粒度参数，这一范式可有效迁移到其他需要高保真度和可控性的 3D 生成任务（如全身数字人、物体化身）。
2.  **TriPlane 在隐式模型上的应用**：将 TriPlane 等显式空间结构特征绑定到隐式 SDF 模型的规范空间顶点的做法，为在各类隐式 3D 表示（如 SDF, NeRF）上高效引入可学习的空间细节提供了一种简洁有效的解决方案。
3.  **细粒度参数直接注入扩散模型**：通过跨注意力层将关键驱动参数（如表情码、姿态码）直接作为条件 tokens 注入预训练 LDM 内部，比仅依赖光栅化图像的隐式编码更能利用参数信息，此设计可用于需要精准控制生成内容的扩散模型应用。
4.  **训练策略细节**：采用 v-prediction 和重缩放噪声调度（允许纯噪声输入训练）可以提升训练稳定性和收敛速度，这些技巧值得在相关扩散模型微调任务中尝试。
5.  **评估体系**：结合了丰富的客观指标（感知、身份、关键点距离）和人工用户研究（视觉质量、驱动忠实度）来全面评估头像的生成质量与可用性，评估体系较为完备。

## 关键术语表
*   **Deferred Neural Rendering**: 一种神经渲染范式，先渲染出几何属性图（如法线、深度），再由一个神经网络将这些属性图翻译为最终图像，将几何与外观分离处理。
*   **NPHM (Neural Parametric Head Model)**: 一种神经参数化头部模型，给定身份和表情编码，输出连续的有向距离场 (SDF) 来表示头部几何形状，能捕捉比传统 3DMM 更丰富的细节。
*   **Latent Diffusion Model (LDM)**: 在压缩的潜空间中进行去噪扩散过程的扩散模型，如 Stable Diffusion，能高效生成高分辨率图像。
*   **ControlNet**: 一种为预训练扩散模型添加额外条件控制（如边缘、深度、姿态）的架构，通过复制并冻结原骨干网络，增加一个可训练的卷积编码器网络来注入条件信息。
*   **TriPlane**: 一种高效的 3D 空间特征表示，使用三个二维特征平面（分别对齐 XY, XZ, YZ 平面）来存储和查询三维空间中的特征。
*   **Canonical Space**: 在 NPHM 等变形场模型中，指代个体处于中性、未变形状态时的参考三维空间。通过逆向变形场可以将任意表情状态下的顶点映射回此空间。
*   **Cross-Attention**: 注意力机制的一种，其中 Query (Q) 和 Key/Value (K/V) 来自不同的输入序列，常用于将外部条件信息（如文本、编码向量）注入到模型主路径中。

## 可复现要素
*   **数据集**：NeRSemble 数据集（部分公开，需申请获取），包含 8 人的多视角视频用于实验。
*   **代码开源情况**：论文未明确声明代码完全开源，但项目页面 (https://tobias-kirschstein.github.io/diffusion-avatars/) 可能包含相关链接。作者单位 TUM 的相关开源项目中有部分代码。
*   **权重开源情况**：预训练的 Stable Diffusion v2.1 权重公开可用。本文训练的特定人物头像权重未在论文中声明公开。
*   **关键超参**：优化器 Adam；ControlNet 及表达式条件层学习率 1e-4；TriPlane 特征查找学习率 1e-2；batch size 8；分辨率 512x512；TriPlane 分辨率 512x512，通道数 16；总训练步数 100k；训练时长约 2 天/人 (单张 RTX A6000)。
