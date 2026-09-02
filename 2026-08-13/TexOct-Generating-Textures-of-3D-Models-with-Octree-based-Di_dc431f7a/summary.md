---
title: "TexOct-Generating-Textures-of-3D-Models-with-Octree-based-Di"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_TexOct_Generating_Textures_of_3D_Models_with_Octree-based_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:48:38"
field: "3D 视觉生成"
keywords: ["3D Texture Generation", "Diffusion Model", "Octree", "Point Cloud", "Text-to-3D", "ShapeNet"]
innovations: ["首次将DDPM扩散去噪过程直接作用于八叉树节点以生成3D纹理", "端到端在3D空间生成纹理，无需UV展开或多视图聚合", "在八叉树U-Net中引入octree-based multi-head cross-attention实现文本/图像条件生成"]
benchmarks: ["ShapeNet", "FID", "KID"]
---

# 论文速读：TexOct-Generating-Textures-of-3D-Models-with-Octree-based-Diffusion

## 一句话总结
TexOct 提出了一种基于八叉树的扩散模型，直接在三维空间中利用稠密点云为给定 3D 网格生成高质量、完整的纹理贴图；该方法避免了多视图生成中的自遮挡问题，并以单阶段方式实现了比 Point-UV 更细粒度的纹理细节。

## 研究问题与动机
- **UV 映射不统一**：2D Diffusion 方法依赖 UV 展开，但任意 3D 网格都存在无限多种 UV 映射方式，难以直接生成一致的 3D 纹理。
- **多视图自遮挡**：Text2Tex 等基于多视图投影的方法受 NP-hard 集合覆盖问题限制，最优视角集存在遮挡，导致纹理错误（如椅子凹陷处的纹理错误）。
- **稀疏点云导致纹理粗糙**：Point-UV (Stage-1) 仅使用约 4096 个点，分辨率不足，无法准确表达纹理细节；GPU 显存限制了直接使用稠密点云的可行性。
- **如何高效利用稠密点云**：在有限算力下将稠密点云结构化地输入扩散模型并实现端到端 3D 纹理生成，是一个尚未被充分解决的问题。

## 核心贡献（创新点）
1. **端到端三维空间直接纹理生成框架**：避免了 UV 映射依赖和多视图一致性难题，与 Text2Tex 等多视图方法相比从根本上消除了自遮挡引起的纹理错误。
2. **基于八叉树的 3D 扩散模型（TexOct）**：首次将 DDPM 的去噪过程直接作用于八叉树节点，相比 Point-UV 的点云扩散，能更高效地处理稠密点云，生成更精细纹理。
3. **支持文本/图像条件的多尺度八叉树交叉注意力模块**：在每个 ResNet Block 后引入 octree-based multi-head cross-attention（除第一阶段外），使用 CLIP 提取条件特征，使生成可被文本或图像引导，与无条件 Point-UV 形成条件生成能力的补充。
4. **系统的超参数分析**：定量分析了八叉树深度（10–13）与采样点数（10K–200K）对重建误差、训练时间和 FID/KID 的影响，给出深度=12、点数=100K 的最优配置。

## 方法详解
- **八叉树构建**：用 meshsampling 在网格表面采样 $M$ 个带 RGB 颜色值的点 $\mathcal{P}$，进行偏移和平移量化：$\mathcal{P}_Q = \text{round}((\mathcal{P} - \text{offset}) / qs)$，其中量化步长 $qs \geq (max(P) - min(P))/(2^L - 1)$，$L$ 为八叉树最大深度；每个叶子节点对应边长为 $qs$ 的立方体，点被合并到最近立方体，重建误差 $\leq qs/2$。
- **扩散过程**：对每个八叉树节点的颜色值随机采样高斯噪声 $\epsilon_t$，按 DDPM 前向过程加噪得到 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t$；训练目标为简化损失 $\mathcal{L} = ||\epsilon_\theta(x_t) - \epsilon_t||_2^2$，实际训练中预测干净信号而非噪声（参考 RenderDiffusion/Point-UV 的训练策略）以获得更稳定训练。
- **网络架构**：U-Net 主干，含四个阶段，对应的树深度分别为 12、11、10、9，保证不同感受野下的局部-全局一致性；每个 ResNet Block 包含两个 Oct-conv、两个 Oct-norm 及 MLP；条件生成时在除第一阶段外每个 Block 后插入 octree-based multi-head cross-attention 模块（将节点特征划分为 patch，通过线性层生成 Q/K/V 并与 CLIP 提取的条件特征做注意力计算）。
- **推理流程**：从 $\mathcal{N}(0,1)$ 采样纯噪声赋给已知点云节点，以 $t=1000$ 开始，用标准 DDPM 采样逐步去噪；最终通过 Reverse Octree 将去噪后八叉树还原为彩色点云，再映射回原始网格生成纹理化网格（使用 [34] 提供的工具）。
- **条件生成**：文本条件下使用预训练 CLIP 提取 text embedding，图像条件下随机渲染一个视角图并通过 CLIP 提取 image embedding，均通过 cross-attention 注入 U-Net 特征。

## 实验与结果
- **数据集**：ShapeNet [6]，类别包括 Chair、Table、Car、Bench，每模型采样 100K 个点用于训练。
- **评估指标**：FID 和 KID（渲染 512×512 图像，从 4 个随机视角渲染；公平比较时 Ours* 和 Text2Tex* 均用 20 视角渲染 20 个样本）。
- **基线方法**：Texture Fields [24]、Texturify [32]、Point-UV (1-Stage/2-Stage) [43]、Text2Tex [8]。
- **主要结果（Table 1，全部类别平均）**：
  - TexOct：**FID=14.75，KID=0.13**，在全部四个类别上均取得最优或次优。
  - 相比最强基线 Point-UV (2-Stage) 提升：FID 降低 2.56（17.31→14.75），KID 降低 0.17（0.30→0.13）。
  - 相比 Text2Tex*：FID 降低 15.11（53.22→38.11），KID 降低 0.65。
  - 对比 Point-UV (1-Stage)（稀疏 4096 点）：FID 从 57.69 降至 14.75，提升幅度巨大。
- **用户研究（Table 2）**：20 名用户，2000 份回复，TexOct 以 **80.7%** 偏好率显著优于 Point-UV [43]（19.3%）。
- **定性结果**：TexOct 在 Chair、Table、Bench 等类别上生成的纹理在颜色多样性、高频细节和 3D 一致性上均优于基线。
- **超参分析（Table 3）**：Depth=12 时 FID=23.67、KID=0.13 为最佳折衷；Depth=13 虽重建误差更低（0.002）但过拟合严重（FID 升至 28.19）。点数在 100K 后收益递减但显存成本急剧上升（Figure 8）。

## 相关工作脉络
1. **AUV-Net [10]**：学习对齐 UV 映射进行纹理合成，与 TexOct 的根本差异在于前者仍依赖 UV 展开，后者直接在 3D 空间操作避免 UV 不统一问题。
2. **Text2Tex [8]**：基于多视图 2D Diffusion 的文本驱动纹理生成，存在自遮挡导致的纹理错误（Figure 1-(a)）；TexOct 从根本上绕过了视角选择问题。
3. **Point-UV [43]**：两阶段点云+UV 扩散方法，Stage-1 使用稀疏点云（4096 点）导致纹理粗糙；TexOct 在单阶段内通过八叉树表示 100K 稠密点云，直接超越其 Stage-1，并在 FID 上优于其 Stage-2。
4. **Texturify [32]**：GAN-based 直接在 3D 表面操作的纹理生成，缺乏高分辨率细节控制；TexOct 利用扩散模型的渐进去噪能力获得更高频纹理细节。
5. **TexFusion [5]**：使用 2D Diffusion 聚合多视角 latent texture map 再融合，仍需 UV 一致性；TexOct 完全规避 UV 流程。
6. **Texture Fields [24]**：隐式函数空间学习纹理表示，分辨率受限于 SDF 网格粒度；TexOct 的八叉树结构可实现更灵活的分辨率自适应。

## 局限性与未来方向
- **未利用额外几何先验**：当前方法仅使用点坐标和颜色，未引入法线、曲率、Laplace–Beltrami 算子等几何信息，作者指出这是未来可改进的方向。
- **八叉树深度与泛化的权衡**：过深（如 Depth=13）导致过拟合，过浅（如 Depth=10）分辨率不足，存在一个较窄的最优区间，对不同形状类别可能需要自适应深度策略。
- **推理时间较长**：DDPM 多步采样本身耗时，且单次对象生成时间未与 Text2Tex（~15 分钟）直接对比，推断效率有优化空间。
- **生成结果为单阶段**：相比 Point-UV 的两阶段（粗→细）范式，本文未引入 refinement 阶段，可能在极端高分辨率场景下不如两阶段方法精细。

## 研究启发与可借鉴点
1. **八叉树作为 3D 扩散的骨架结构**：将 Diffusion 模型作用于八叉树节点而非原始点云，兼顾了计算效率与空间层次感知，该范式可迁移到其他 3D 生成任务（如点云补全、3D 形状生成）。
2. **端到端消除 UV 依赖的设计思路**：对于任何需要纹理/颜色的 3D 生成任务，可直接在 3D 空间操作以避免 UV 展开带来的拓扑敏感性问题，值得在多类别 3D 内容生成中验证。
3. **条件注入的通用接口设计**：在 U-Net 各阶段后插入 octree-based cross-attention，将 2D 视觉语言模型（CLIP）的嵌入高效接入 3D 扩散生成，这一接口设计可直接复用到其他条件 3D 生成任务。
4. **预测干净信号而非噪声的训练技巧**：沿用 RenderDiffusion / Point-UV 的做法，直接预测 $x_0$ 而非 $\epsilon_t$，可提升扩散模型训练的稳定性，适合在后续工作中作为 baseline trick 使用。
5. **超参系统性分析的方法**：对八叉树深度和采样点数分别做消融，绘制 FID/KID vs 计算成本的 trade-off 曲线，为后续工作的实验设计提供了可复用的分析框架。

## 关键术语表
- **Octree（八叉树）**：一种三维空间层次数据结构，递归地将空间划分为 8 个子立方体，用于高效表示和压缩稠密点云。
- **DDPM（Denoising Diffusion Probabilistic Model）**：基于随机微分过程的生成模型，通过逐步去噪从纯高斯噪声中生成样本。
- **UV Mapping（UV 映射）**：将 3D 网格表面参数化到 2D 平面的坐标变换，是传统纹理贴图的核心步骤。
- **Cross-Attention（交叉注意力）**：在 Transformer 架构中，Query 来自目标序列、Key/Value 来自条件序列的注意力机制，用于将外部信息注入生成过程。
- **FID / KID**：Fréchet Inception Distance 和 Kernel Inception Distance，衡量生成图像与真实图像分布差异的常用指标，值越低表示质量越高。
- **Meshsampling**：在三角网格表面进行均匀采样的方法，用于生成带几何和纹理信息的点云。
- **OCNN（Octree-based CNN）**：基于八叉树结构的卷积神经网络，支持在八叉树节点上进行卷积和归一化操作。
- **Reverse Octree**：将八叉树结构还原为原始点云（含颜色值）的过程，与构建过程互为逆操作。

## 可复现要素
- **数据集**：ShapeNet [6]，训练集拆分遵循 Point-UV [43] 的方法；meshsampling 采样 100K 点/模型。
- **代码/权重**：论文未提及代码开源声明。
- **关键超参**：八叉树深度 L=12，采样点数=100K，训练 2000 epoch，AdamW 优化器，学习率 1e-4，batch size=128；推理时初始噪声 timestep t=1000。
- **条件模型**：CLIP [26] 预训练模型提取文本/图像 embedding。
- **评估设置**：渲染分辨率 512×512；公平比较时 Ours* 和 Text2Tex* 均采用 20 个随机样本×20 视角渲染。
