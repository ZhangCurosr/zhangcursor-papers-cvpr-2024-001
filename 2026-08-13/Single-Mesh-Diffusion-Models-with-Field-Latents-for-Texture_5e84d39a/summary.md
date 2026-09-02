---
title: "Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Mitchel_Single_Mesh_Diffusion_Models_with_Field_Latents_for_Texture_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:17:01"
field: "3D生成式建模"
keywords: ["3D纹理生成", "扩散模型", "流形深度学习", "等距等变性", "单样本生成", "场潜变量"]
innovations: ["提出Field Latents将纹理编码为切向量场，利用对数坐标函数实现丰富特征插值", "构建等距等变的场卷积扩散模型(FLDM)，支持单网格纹理生成与跨几何纹理转移", "将卷积核扩展至C×R^e→C实现条件注入，在保持等变性的同时稳定训练"]
benchmarks: ["Google Scanned Objects", "Objaverse"]
---

# 论文速读：Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture

## 一句话总结
本文提出了一种直接在3D三角网格表面上操作的**内生潜在扩散模型（FLDM）**，核心创新是引入**Field Latents（场潜变量）**——将纹理编码为网格顶点处的切向量场，配合等距等变的场卷积去噪网络，实现了从单个带纹理网格生成高质量纹理变体，并支持标签引导生成、修复填充及跨几何体的生成式纹理转移。

## 研究问题与动机
1. **2D到3D的纹理映射存在视图不一致性**：主流方法（如DreamFusion系列）依赖不同视角渲染2D图像并使用预训练2D LDM作为先验，但将2D细节映射回3D表面时会产生不自然的色调/光照残留，且SDS方法仅适用于低分辨率LDM与小NeRF，难以合成细粒度细节。
2. **栅格化表示丢失高频细节**：基于体素/三平面（triplane）的方法（如Sin3DM）需先将几何与纹理栅格化到3D网格，受显存限制导致高频纹理细节混叠；UV atlas方法（如Point-UV Diffusion）在网格UV展开不连通时表现不佳。
3. **高质量3D纹理数据稀缺**：大规模3D数据集（如Objaverse）中仅少数样本具有复杂非均匀纹理，许多资产为手工制作或扫描所得，几何/风格独特，难以直接训练大规模多类别生成模型。
4. **表面缺乏标准方向基准**：与图像不同，曲面点无规范方向，局部坐标系存在任意旋转模糊，直接移植2D VAE/Diffusion面临等变性挑战。

## 核心贡献（创新点）
1. **Field Latents（场潜变量）**：将高维纹理压缩为网格顶点处的切向量特征（复数表示），利用向量表征局部方向信息，相比标量特征可实现更优重建质量；与INFs相比，本文使用对数映射坐标函数而非重心插值，在面片内提供更丰富的特征连续扩展。
2. **Field Latent Diffusion Model（FLDM）**：在切向量场的流形空间上定义前向加噪过程与反向去噪网络，去噪器基于场卷积（Field Convolutions, FC）构建，并通过扩展卷积核至$\mathbb{C} \times \mathbb{R}^e \to \mathbb{C}$实现时间步与条件嵌入的注入，同时保持等距等变性——与Sin3DM将纹理与3D嵌入绑定的做法本质不同。
3. **单网格纹理生成范式**：借鉴SinDiffusion/SinGAN的单样本生成思路，在单个带纹理网格上训练FLDM，规避大规模3D纹理数据需求，同时通过浅层U-Net控制感受野防止过拟合。
4. **等距等变性与生成式纹理转移**：FL-VAE与FLDM均满足等距等变性（commute with isometries），使得预训练模型可直接在新的相似几何上采样，无需重新训练即可实现纹理风格迁移，这与基于点映射或token条件化的已有纹理转移方法形成本质区别。

## 方法详解
**Field Latents 编码**：在曲面$M$上每点$p$关联切空间$T_pM \cong \mathbb{C}_p$，纹理$\psi$在$p$的one-ring邻域$B_p$内采样标量值，经VN-Transformer编码为$d$维复数潜变量$z_p^\psi = \mu_p^\psi + \sigma_p^\psi \odot \epsilon(p)$，其中$\epsilon(p) \sim \mathcal{TN}_M(0, I_d)$为切丛上的多维高斯噪声。

**对数映射解码器**：解码器$\mathcal{D}_p$以$\log_p q \in \mathbb{C}_p$（点$q$在$p$处切空间的对数坐标）和潜码$z_p^\psi$为输入，构造两类特征：(1) 不变标量特征：$z_p^\psi[z_p^\psi]^*$的半上三角部分向量化；(2) 位置感知特征$c_{pq}^\psi = \log_p q \cdot \overline{z_p^\psi}$（含内积与行列式信息）。二者拼接后输入5层real-MLP预测该点纹理值。与INFs不同，仅推理时进行重心插值，训练时每个面独立预测。

**FLDM前向/反向过程**：前向加噪$Z_t = \sqrt{\alpha_t}Z + \sqrt{1-\alpha_t}\epsilon$，$\epsilon \sim \mathcal{TN}_M(0, I_d)$；反向去噪由场卷积浅层U-Net实现，损失为$\mathbb{E}\|\epsilon - \varepsilon(Z_t, t, \rho)\|_2^2$。时间步$t$与条件$\rho$（如用户标签）通过扩展卷积核$\mathbf{f}_{c'c} \in L^2(\mathbb{C} \times \mathbb{R}^e, \mathbb{C})$注入，避免加法嵌入破坏等变性。

**预训练策略**：FL-VAE在Open-Images V4上预训练——将$512\times512$图像叠加随机10K顶点的平面网格作为"纹理"，强制模型在低分辨率网格上重建高频细节，提升对任意3D网格的鲁棒性。FLDM训练时生成500个不同三角剖分的30K顶点副本，防止去噪网络学习网格连通性。

## 实验与结果
**数据集**：Google Scanned Objects [13]（16个复杂纹理物体，高/低分辨率网格30K/5K顶点）、Objaverse [8]与Scanned Objects联合（10个资产用于生成评估）。

**重建质量**（Table 1）：FL-VAE（30K顶点）PSNR **22.38**、DSSIM **0.51**、LPIPS **1.02**，全面优于INFs（PSNR 21.33/0.66/1.31）及FL-VAE-Barycentric（20.59/0.83/1.81）；低分辨率下优势更大（20.59 vs 19.61/16.45），验证对数坐标函数的表征能力提升。

**生成质量**（Table 2）：FLDM SIFID **3.27** vs Sin3DM **6.58**；FLDM样本LPIPS **0.94** vs Sin3DM **2.91**（后者因生成新几何导致多样性虚高）。定性分析显示Sin3DM的细节呈沿主轴重复/外推模式，而FLDM可在局部相似的 mesh 区域无缝复现纹理细节。

**下游任务**：标签引导生成（Figure 4）成功区分鞋底/鞋身/内里纹理；修复填充（Figure 5）在掩码边界处自然融合；生成式纹理转移（Figure 6）在零亏格与三亏格拓扑间成功迁移纹理风格。

## 相关工作脉络
1. **DreamFusion/SDS系列** [41]：通过优化NeRF使渲染图匹配预训练2D LDM分布，计算成本受限于低分辨率LDM，无法合成细粒度纹理——本文从网格表面直接建模，完全规避2D渲染瓶颈。
2. **Sin3DM** [58]：唯一在单网格范式下的LDM基线，但采用三平面体素栅格化表示，纹理与3D嵌入强耦合导致细节异常（重复/外推）——本文的内生切向量场表示解耦了纹理与几何嵌入。
3. **Intrinsic Neural Fields (INFs)** [27]：使用Laplacian特征函数作为顶点特征并通过重心插值扩展至面片，在低压缩比下重建"破碎"——本文的对数坐标函数提供更丰富的面内插值，显著改善重建质量。
4. **Texturify [48]/Mesh2Tex [2]**：GAN-based方法，将纹理压缩至facet属性并通过可微渲染监督——本文采用扩散模型，生成多样性与图像质量更高。
5. **Manifold Diffusion Fields** [14,64]：首个定义在2D Riemann流形上的全内生扩散模型，但依赖全局注意力，仅适用于低分辨率粗网格——本文基于局部支撑的场卷积，可扩展至高分辨率网格。
6. **SinDiffusion/SinGAN系列** [28,38,46,54]：单图像扩散生成范式，通过浅层U-Net控制感受野防过拟合——本文将其推广至流形表面，引入等变卷积替代标准卷积。

## 局限性与未来方向
1. **训练时间长**：FLDM训练耗时较长，可能限制迭代实验效率。
2. **无法表达合成纹理的方向性**：当前FLDM无法生成反映用户指定方向信息的纹理（如毛发生长方向），作者建议可通过与网络特征做点积的向量场嵌入解决。
3. **纹理转移依赖近似等距**：跨几何体纹理转移要求两网格近似等距，对高度不规则几何可能失效。
4. **UV展开质量依赖**：预处理阶段需共享UV atlas映射，劣质展开可能影响结果。

## 研究启发与可借鉴点
1. **等距等变性的隐式解法**：通过切向量场+对数坐标实现内生的等变表示，无需显式求解全局标架（global frame），可迁移至其他流形上学习的等变网络设计。
2. **单网格预训练策略**：在平面网格上预训练FL-VAE后可泛化至任意3D曲面，这种"平坦化预训练+流形部署"思路可推广至其他表面任务（如法场生成、材质分解）。
3. **条件注入的等变技巧**：将时间步/条件嵌入扩展至卷积核（$\mathbb{C} \times \mathbb{R}^e \to \mathbb{C}$）而非加法/乘法扰动，既保留等变性又避免训练不稳定，为等变扩散模型的条件注入提供通用范式。
4. **生成式纹理转移的应用扩展**：等距等变性天然支持跨几何纹理迁移，可与神经辐射场（NeRF）/3D Gaussian Splatting的纹理细化结合，为单样本人物/物体定制纹理提供新思路。

## 关键术语表
**Field Latents (FLs)**：将纹理编码为网格顶点处切空间的离散复数向量场的潜在表示，捕获局部纹理方向信息，实现感知压缩。

**Field Convolution (FC)**：定义在流形切向量场上的卷积算子 [35]，支持旋转等变的局部特征聚合，是FLDM去噪网络的核心组件。

**Isometry-Equivariance**：模型输出随输入几何的等距变换（保距变形）同步旋转/变换的性质，本文FL-VAE与FLDM均满足此性质，使纹理细节可在局部相似区域无缝复制。

**Logarithmic Coordinate Function**：解码器中利用$\log_p q$将面上点$q$映射至$p$处切空间的对数坐标，相比重心插值提供丰富的位置感知特征（内积+行列式）。

**Single-Textured-Mesh Paradigm**：从单个带纹理3D网格训练生成模型、合成同类纹理变体的设定，规避大规模3D纹理数据集需求。

**VN-Transformer**：Vector Neuron Transformer [1]，处理切向量特征的旋转等变注意力模块，用于FL-VAE编码器。

**Score Distillation Sampling (SDS)**：DreamFusion提出的优化策略，通过预训练2D LDM的梯度蒸馏指导3D神经场优化，本文指其无法合成高频细节的局限性。

**Triplane Latent**：Sin3DM等方法使用的3D平面特征表示，需先将网格栅格化到体素再提取，易产生混叠。

## 可复现要素
- **数据集**：Google Scanned Objects [13]、Objaverse [8]、Open-Images V4 [29]（预训练用）；论文未明确数据集公开状态，但链接均指向公开数据集。
- **代码**：论文声明"Code and visualizations are available at https://single-mesh-diffusion.github.io/"，代码开源。
- **权重**：未明确声明是否公开预训练FL-VAE权重。
- **关键超参**：FL-VAE latent dim $d=8$；FLDM max timestep $T=1000$；网格重采样500个副本、30K顶点；Open-Images预训练使用1K个平面网格（各10K顶点、512×512图像）；优化器Adam；线性噪声调度。
