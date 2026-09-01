---
title: "DiffusionAvatars-Deferred-Diffusion-for-High-fidelity-3D-Hea"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Kirschstein_DiffusionAvatars_Deferred_Diffusion_for_High-fidelity_3D_Head_Avatars_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:50:49"
field: "3D 头像生成与动画"
keywords: ["3D Head Avatar", "Diffusion Model", "Neural Rendering", "Deferred Rendering", "NPHM", "ControlNet", "Cross-attention"]
innovations: ["结合 NPHM 隐式几何与预训练 LDM 的 Deferred Diffusion 渲染器", "基于 TriPlanes 的规范空间表面特征绑定机制", "直接表达式代码的 Cross-attention 条件注入"]
benchmarks: ["NeRSemble", "PSNR", "LPIPS", "JOD", "AKD", "CSIM"]
---

# 论文速读：DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars

## 一句话总结
本文提出 DiffusionAvatars，一种将神经参数头部模型 (NPHM) 的精确几何控制与预训练潜扩散模型 (LDM) 的强大图像合成能力相结合的 Diffusion-based Neural Renderer，能够从多视图视频中学习并合成具有高度视图一致性、细节丰富且表情可控的高保真 3D 头部头像。

## 研究问题与动机
*   **核心问题**：如何从单个人物的多视图视频中，创建一个既能保持视图和时间一致性，又能提供精细表情与姿态控制的高保真可动画 3D 头部头像。
*   **现有方法不足**：
    1.  纯 2D 生成方法（如 GANs、直接微调的 Diffusion Models）在生成逼真面部图像方面表现优异，但通常缺乏对三维视图的一致性和精细的 3D 几何控制。
    2.  纯 3D 重建/表征方法（如基于 NeRF 或 3DMM 的方法）能提供良好的视图一致性，但其渲染结果在照片级真实感和细节丰富度上往往不及顶级 2D 生成模型。
    3.  早期的 2D+3D 结合方法（如 DNR）受限于所用几何代理（如 Basel Face Model）的精度，难以捕捉复杂的细微表情和拓扑变化区域（如口腔内部）。

## 核心贡献（创新点）
*   **提出了 DiffusionAvatars 框架**：一种将预训练 LDM 作为后端的扩散神经网络渲染器，通过 ControlNet 范式条件化于 NPHM 网格的栅格化结果，实现了高质量图像合成与 3D 可控性的结合。 *区别于先前仅使用简单 3DMM 作为条件输入的方法，本文采用了表达能力更强的隐式 NPHM 几何代理。*
*   **设计了基于 TriPlanes 的表面特征绑定机制**：通过将可学习的空间特征（TriPlanes）与 NPHM 的规范空间坐标关联，为神经渲染器提供了弥补几何细节不足、增强跨视图一致性的“神经纹理”。 *与基于显式 UV 展开或球面映射的特征绑定相比，该方法天然适配 NPHM 的隐式几何表示。*
*   **引入了直接的表达式交叉注意力条件注入**：将 NPHM 的表达式代码通过线性层转换为 token，并直接作为额外条件注入到预训练 LDM U-Net 的交叉注意力层中，以增强对细微表情的建模和控制。 *与仅依赖栅格化图像间接传递表达式信息的方式不同，这种直接条件注入为扩散模型提供了更明确、更细粒度的表达式指导。*

## 方法详解
1.  **几何代理与渲染准备**：使用 MonoNPHM 对每帧多视图视频进行拟合，获得身份代码 $z_{id}$ 和表达式代码 $z_{exp}^t$，进而生成 Signed Distance Field (SDF) 并提取网格 $M_t$。利用 nvdiffrast 将 $M_t$ 从目标视点进行栅格化，得到包含法线、深度和**规范坐标** ($R_{can}^t$) 的渲染图。
2.  **表面特征映射 (TriPlanes)**：利用栅格化得到的规范坐标 $R_{can}^t$，查询一个可学习的 TriPlane 结构（用于 3D 空间）和一个 Ambient Map（用于 2D 环境维度），生成特征图 $R_{feat}^t$ 和 $R_{feat\_amb}^t$。这些特征图被拼接到栅格化缓冲区中，为渲染器提供表面先验。
3.  **直接表达式条件化**：将表达式代码 $z_{exp}^t$ 通过一个线性层 EXP 映射为 4 个表达式 token ($f_{exp}^t$)。在预训练 LDM 的 U-Net 中，插入额外的交叉注意力层，以这些 token 作为 Key 和 Value，与 U-Net 中间的自注意力特征进行交互，实现直接的条件注入。
4.  **基于 ControlNet 的扩散渲染**：主干采用冻结权重的 Stable Diffusion v2.1 LDM。将步骤 1 和 2 中得到的所有渲染缓冲区（法线、深度、规范坐标、特征图）以及步骤 3 中的表达式 token 共同输入到可训练的 ControlNet 分支中。通过迭代去噪过程（采用 **v-parameterization** 训练策略以加速收敛并更好处理纯噪声输入），最终生成目标图像。
5.  **训练损失**：最小化预测的 v 值与目标 v 值之间的 L2 损失，同步优化 ControlNet 分支、表达式条件化模块以及 TriPlane 特征图。

## 实验与结果
*   **数据集**：NeRSemble 数据集的多视图序列，选取 8 名人物进行训练和评估。
*   **评估任务与基线**：在**自我重演**和**头像动画**两个任务上进行评估。基线包括 NeRFace, MVP, DNR, DNR+GAN, DiffusionRig。
*   **主要定量结果**：
    *   在自我重演任务（Table 1）中，本方法在 **LPIPS** (0.081，更低越好)、**CSIM** (0.882，越高越好) 和 **AKD/AED/APD** 等多项感知和面部细节指标上取得**最优**或接近最优的结果。PSNR (24.9) 略低于 NeRFace (23.0) 和 DNR (24.5)，但 LPIPS 显著更低，表明主观视觉质量更优。
    *   在头像动画的用户研究中（Table 2），本方法在**视觉质量 (VQ)** (4.02/5.0) 和**驱动保真度 (DF)** (4.14/5.0) 上均获得最高平均分，且视觉质量的优势尤为明显。
*   **消融实验** (Table 3)：验证了各组件的有效性。**移除扩散模块**导致 LPIPS 显著下降；**使用 FLAME 替代 NPHM** 在 AKD 等指标上表现更差；**移除表达式条件化**会降低整体性能；**移除空间特征**也对多项指标有负面影响。

## 相关工作脉络
*   **Deferred Neural Rendering (DNR) [70]**：本文直接继承并扩展了 DNR 的思想（deferred rendering + neural texture），但用更高保真度的 NPHM 网格替换了 BFM，并用预训练扩散模型替换了 Pix2Pix 架构，同时增加了直接的条件注入机制。
*   **DiffusionRig [12]**：同样基于扩散模型和 FLAME 网格进行头像动画。本文与之关键区别在于使用了**隐式的 NPHM 几何**（能表达更丰富的细节和拓扑变化）和**显式的跨视图 TriPlane 特征绑定**以增强一致性，而 DiffusionRig 主要依赖单张照片的微调。
*   **NeRFace [16], MVP [43]**：这些是基于 NeRF 或 3D 体素原语的方法，具有良好的视图一致性，但受限于神经辐射场的渲染速度和照片级纹理生成能力。本文方法在保持视图一致性的同时，追求更接近 2D 扩散模型的渲染真实感。
*   **基于 FLAME 的 3DMM 方法**：大量前期工作依赖 FLAME 等显式参数化网格进行表情控制。本文采用 **NPHM** 这一隐式表示，其 SDF 定义和变形场能提供更平滑、细节更丰富的头部几何，从而为渲染器提供更好的几何指导。
*   **ControlNet / IP-Adapter 范式**：本文借鉴了 ControlNet [86] 利用额外条件控制预训练扩散模型的思想，以及 IP-Adapter [82] 通过 cross-attention 注入外部条件的技术，将其应用于 3D 头像生成的特定领域。

## 局限性与未来方向
*   **当前局限性**：
    1.  **光照控制缺失**：目前模型将光照烘焙在生成的图像中，缺乏对场景光照的独立控制能力。
    2.  **计算效率低**：依赖多步扩散去噪过程，无法达到实时应用的要求。
    3.  **几何代理依赖**：性能受限于 NPHM 拟合的质量，对于极度夸张的表情或遮挡情况可能仍存在偏差。
*   **未来方向**：
    1.  集成显式的光照估计或可控光照模块。
    2.  应用扩散模型蒸馏技术（如 Consistency Models, Distillation）来加速推理，迈向实时应用。
    3.  探索更通用的几何表示或结合其他细节增强技术。

## 研究启发与可借鉴点
*   **隐式精细几何与 2D 扩散先验的结合**：将高阶隐式几何代理（NPHM SDF）的输出作为条件，指导预训练扩散模型进行图像合成，是一条兼顾可控性与高保真度的有效路径。可迁移至身体、动物或其他复杂形状的高保真生成任务。
*   **规范空间特征绑定 (TriPlanes)**：将可学习特征绑定到对象的规范空间（canonical space），而非视图空间或显式 UV，可以有效解耦几何与外观，提升跨视角的一致性。此技巧可用于其他基于隐式几何的神经渲染任务。
*   **多层级条件注入策略**：同时利用**栅格化渲染图**（提供全局姿势、形状和粗略表情）和**直接嵌入的条件 token**（通过 cross-attention 提供细粒度控制），这种分层条件注入策略可以更稳定、更精确地引导扩散过程。适用于需要多源条件控制的生成任务。
*   **v-parameterization 在条件生成中的应用**：在需要强条件输入（包括纯噪声输入）的场景下，采用 v-parameterization 训练扩散模型有助于获得更快的收敛速度和更稳定的生成结果。

## 关键术语表
*   **NPHM (Neural Parametric Head Model)**：一种神经参数化头部模型，通过身份和表达式编码定义一个隐式的 Signed Distance Field (SDF) 来表征头部几何及其动态变形。
*   **ControlNet**：一种在预训练扩散模型旁添加可训练副本网络（通常是 U-Net）的技术，通过约束该网络来精确控制预训练模型的生成过程，而不显著改变其原始能力。
*   **TriPlanes**：一种高效的 3D 特征表示方法，使用三个相互垂直的 2D 特征平面来存储和查询 3D 空间中的特征，常用于神经渲染和生成。
*   **v-parameterization**：扩散模型的一种训练目标参数化方式，预测值 v 是噪声和原始潜变量的线性组合，相较于预测噪声 $\epsilon$，在信噪比极低时仍能提供更稳定的梯度信号。
*   **Deferred Diffusion**：本文提出的范式，即先通过传统的可微分渲染管线（基于 3D 几何）生成一组中间渲染缓冲区，再将这些缓冲区作为条件输入到一个扩散模型中进行最终图像合成，类比于图形学中的“延迟渲染”。
*   **Cross-attention Conditioning**：将外部条件（如文本、特征向量）通过注意力机制引入到模型主流程中，使其能够根据条件动态调整内部特征的生成，是实现精准条件控制的关键技术。

## 可复现要素
*   **数据集**：NeRSemble 数据集，论文已公开多视图视频及对应的 NPHM 拟合网格数据用于实验。
*   **代码**：论文项目页面 (https://tobias-kirschstein.github.io/diffusion-avatars/) 可能提供代码和模型权重，但正文未明确说明。
*   **关键超参**：训练使用 Adam 优化器，ControlNet 和表达式条件层学习率为 $1e-4$，TriPlane 特征图学习率为 $1e-2$。批大小为 8，图像分辨率为 512x512。TriPlane 尺寸为 $512 \times 512 \times 16$。使用 Stable Diffusion v2.1 作为预训练 LDM。训练约 100k 步，耗时约两天（单张 RTX A6000）。
