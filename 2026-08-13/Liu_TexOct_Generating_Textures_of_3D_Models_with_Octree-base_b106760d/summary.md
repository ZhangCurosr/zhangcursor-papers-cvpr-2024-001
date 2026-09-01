---
title: "TexOct: Generating Textures of 3D Models with Octree-based Diffusion"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_TexOct_Generating_Textures_of_3D_Models_with_Octree-based_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:01:09"
field: "3D内容生成"
keywords: ["3D纹理生成", "八叉树扩散模型", "点云深度学习", "无条件/条件生成", "ShapeNet", "OCNN", "DDPM"]
innovations: ["提出基于八叉树的DDPM直接在3D空间生成密集点云纹理，避免自遮挡与UV映射问题", "设计八叉树多头交叉注意力模块支持文本/图像条件引导的高质量纹理生成"]
benchmarks: ["ShapeNet"]
---

# 论文速读：TexOct: Generating Textures of 3D Models with Octree-based Diffusion

## 一句话总结
本文提出 TexOct，一种基于八叉树（Octree）的扩散模型，直接在 3D 空间中利用密集采样点云生成高质量、完整的 3D 纹理，有效避免了多视角方法中的自遮挡问题以及稀疏采样导致的纹理细节丢失。

## 研究问题与动机
- **UV 映射困境**：给定 3D 网格存在无限种 UV 展开方式，无法统一构建 UV 映射以直接生成 3D 纹理。
- **多视角自遮挡**：多视角纹理生成需选择最优视角集，但这是 NP-hard 集合覆盖问题；自遮挡会导致纹理错误（如 Text2Tex 中椅子被遮挡区域纹理错误）。
- **点云分辨率瓶颈**：点云表示可规避拓扑和 UV 差异，但 GPU 显存限制使此前方法仅能使用稀疏点云（如 Point-UV 仅 4096 点），导致纹理粗糙。
- **目标**：如何在 3D 空间内高效利用密集点云，实现高质量纹理生成。

## 核心贡献（创新点）
1. **端到端 3D 空间纹理生成**：直接在 3D 空间优化纹理，避免多视角方法的自遮挡问题，与 Point-UV 等多阶段 UV 方法形成本质区别。
2. **八叉树扩散模型（Octree-based Diffusion）**：将 DDPM 的降噪过程直接作用于八叉树节点，利用 OCNN 的八叉树卷积算子处理层级结构，区别于传统点云扩散（如 Point-E、Lion）对均匀点云的建模。
3. **多条件生成支持**：提出八叉树多头交叉注意力模块（after each ResNet block），通过 CLIP 编码器接入文本/图像条件，实现 text/image-conditional 纹理生成。
4. **SOTA 性能**：在 ShapeNet 上优于 Point-UV (Stage-2) 等基线，平均 FID 为 14.75，用户研究偏好率达 80.7%。

## 方法详解
### 八叉树构建
- 从 3D 网格表面采样 $M$ 个点 $\mathcal{P} = \{\mathbf{p}_i\}_{i=1}^{M}$，每个点含坐标 $(x,y,z)$ 和纹理 RGB。
- 平移并量化：$\mathcal{P}_Q = \text{round}(\frac{\mathcal{P} - offset}{qs})$，其中 $qs \geq \frac{max(P) - min(P)}{2^L - 1}$。
- 递归空间细分：根节点按最大包围盒边长分为 8 个子立方体，用 8-bit 占据码编码；递归至最大深度 $L$，叶节点聚合最近点。
- 重建误差上界：$\|\tilde{P}_i - P_i\|_\infty \leq \frac{qs}{2}$。

### 扩散模型设计
- **U-Net 架构**：4 个下采样/上采样阶段，树深度分别为 12→11→10→9，对应不同感受野。
- **ResNet Block**：含 2 个 Oct-Conv + 2 个 Oct-Norm + MLP；第一阶段省略注意力以提升显存效率。
- **条件模块**：每个 ResNet Block 后接八叉树多头交叉注意力（源自 Octformer），CLIP 特征经线性层生成 K/V，与 Q 做注意力计算。
- **训练策略**：遵循 RenderDiffusion/Point-UV 建议，预测干净纹理值（而非噪声），采用 MSE 损失：$\mathcal{L} = \|\hat{x}_0 - x_0\|_2^2$。

### 推理流程
1. 对已知点云各点赋予 $\mathcal{N}(0,1)$ 噪声，取 $t=1000$。
2. 构建八叉树并输入扩散模型，经 DDPM 有限步去噪。
3. "Reverse Octree" 还原为着色点云，再映射回网格生成纹理化 3D 模型。

## 实验与结果
- **数据集**：ShapeNet，包含 Chair、Table、Car、Bench 四类；训练集采样 100K 点/模型。
- **基线**：Texture Fields、Texturify、Point-UV (1-Stage / 2-Stage)、Text2Tex。
- **评价指标**：FID、KID（渲染 512×512 图像，随机 20 视角）。
- **主要结果**：
  - 无条件生成：Ours 平均 FID = 14.75，KID = 0.13，优于 Point-UV (2-Stage) 的 17.31/0.30，优于 Text2Tex 的 53.22/0.79。
  - 各品类均取得最低 FID/KID（Chair: 9.46/0.18，Car: 21.53/0.10，Table: 7.92/0.09，Bench: 20.08/0.13）。
- **用户研究**：2000 份回复，Ours 偏好率 80.7% vs Point-UV 19.3%。
- **条件生成**：文本/图像条件实验展示语义一致性，纹理细节更清晰。
- **超参分析**：八叉树深度 $L=12$ 为最优（FID 23.67），深度过大（13）导致过拟合；采样点数 100K 为平衡点。

## 相关工作脉络
1. **UV 空间方法**（AUV-Net [10]）：学习无监督 UV 对齐，但依赖单一展开，不适配多形状；本文直接操作 3D 空间，绕过 UV。
2. **多视角扩散方法**（Text2Tex [8]、Texfusion [5]）：利用 2D 扩散生成多视角贴图；存在自遮挡与视角覆盖 NP-hard 问题；本文在 3D 点云上直接去噪，避免视角选择。
3. **点云扩散方法**（Point-UV [43]、Lion [44]）：Point-UV 分两阶段先粗后精，受限于稀疏点云；本文用八叉树编码密集点云，单阶段生成高频细节。
4. **八叉树 CNN**（OCNN [37]、Octformer [36]）：本文继承 OCNN 的八叉树卷积算子，首次将其与扩散模型结合用于纹理生成。
5. **3D 纹理 GAN/Diffusion**（Texture Fields [24]、Texturify [32]）：Texture Fields 用函数空间建模；Texturify 用 GAN 在密集点上训练；本文采用扩散框架，生成质量更优且可控。

## 局限性与未来方向
- 未引入额外几何先验（法向量、曲率、Laplace–Beltrami 算子），可能限制纹理贴合度。
- 八叉树深度与泛化能力存在权衡：深度增加提升分辨率但易过拟合，训练耗时呈指数增长。
- 密集点云采样（100K）消耗较多 GPU 显存，大规模场景下可扩展性待验证。
- 未来可探索几何-纹理联合生成、自适应深度八叉树、条件引导机制改进等方向。

## 研究启发与可借鉴点
1. **八叉树 + 扩散的组合范式**：将 OCNN 的高效空间划分与 DDPM 生成能力结合，为其他 3D 生成任务（如点云补全、网格变形）提供新思路。
2. **预测干净值而非噪声**：采用 $\mathcal{L} = \|\hat{x}_0 - x_0\|^2$ 替代标准噪声 MSE，训练更稳定，可直接迁移至其他扩散应用。
3. **空间层级注意力机制**：八叉树多头交叉注意力在跳过首阶段的情况下实现条件注入，兼顾显存与语义控制，适合资源受限的 3D 生成任务。
4. **深度-精度权衡分析**：系统实验展示了八叉树深度对重建误差、训练时间和 FID 的非单调影响，其评估方法可作为同类工作的参考模板。
5. **密集点云高效表示**：100K 点通过八叉树压缩，相比原始点云扩散降低计算负担，提示可探索其他层级数据结构（如 k-d 树、FAISS 索引）加速 3D 生成。

## 关键术语表
**Octree（八叉树）**：将 3D 空间递归划分为 8 个子节点的层级树结构，用于高效编码密集点云的空间位置关系。
**DDPM（Denoising Diffusion Probabilistic Model）**：通过前向加噪与反向去噪学习数据分布的概率生成模型，本文将其适配至八叉树节点。
**FID / KID**：Fréchet Inception Distance / Kernel Inception Distance，衡量生成图像与真实图像分布的距离，值越小越好。
**OCNN（Octree-based CNN）**：基于八叉树的卷积神经网络，支持非均匀采样点云的卷积、归一化和下/上采样操作。
**Reverse Octree**：将八叉树的层级表示还原为原始点云坐标与属性值的逆过程。
**CLIP**：Contrastive Language–Image Pre-training 模型，用于提取文本/图像条件嵌入，驱动条件纹理生成。
**Cross-Attention（交叉注意力）**：允许扩散模型特征与外部条件（文本/图像）进行特征交互的注意力机制。
**UV Mapping**：将 3D 网格表面展开为 2D 坐标空间的映射，是传统纹理生成的核心步骤。

## 可复现要素
- **数据集**：ShapeNet [6]，训练集拆分遵循 Point-UV [43]；meshsampling [20] 提供 100K 点 + 纹理配对。
- **代码开源**：论文未明确声明代码开源状态（需进一步核查项目主页）。
- **关键超参**：八叉树深度 $L=12$；采样点数 100K；训练 2000 epochs；AdamW，lr=1e-4，batch size=128；预测干净信号；$t_{max}=1000$。
- **环境依赖**：PyTorch，OCNN 库，CLIP 预训练模型。
