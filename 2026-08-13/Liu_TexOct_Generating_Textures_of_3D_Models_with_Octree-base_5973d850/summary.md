---
title: "TexOct: Generating Textures of 3D Models with Octree-based Diffusion"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_TexOct_Generating_Textures_of_3D_Models_with_Octree-based_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:01:11"
field: "3D图形生成"
keywords: ["3D纹理生成", "八叉树扩散", "DDPM", "点云生成", "条件生成"]
innovations: ["基于八叉树的扩散模型，首次在OCTREE结构上执行DDPM去噪实现密集点云纹理生成", "端到端3D空间纹理生成，规避UV映射复杂性和多视角自遮挡问题"]
benchmarks: ["ShapeNet"]
---

# 论文速读：TexOct: Generating Textures of 3D Models with Octree-based Diffusion

## 一句话总结
论文提出 TexOct，一种直接在3D空间中生成高质量、完整纹理的方法，通过八叉树结构高效表示密集点云并结合DDPM扩散模型逐步去噪，避免了多视角方法的自遮挡问题和稀疏采样的纹理退化问题。

## 研究问题与动机
- **UV映射的复杂性**：2D扩散方法需将3D纹理映射到2D UV空间，但给定3D网格存在无限种UV展开方式，难以构建统一映射；
- **多视角方法的自遮挡缺陷**：基于多视角投影的方法受限于NP-hard集合覆盖问题，自遮挡导致视角无法完整覆盖模型表面，产生纹理错误（如Text2Tex在椅子腿部的纹理错误）；
- **点云采样密度受限**：点云可规避拓扑与UV变化，但GPU显存限制使现有方法只能使用稀疏点云（如Point-UV仅4096点），导致纹理粗糙、细节缺失；
- **3D一致性难以保证**：2D图像生成的纹理在投影到3D表面时容易出现局部不一致和模糊。

## 核心贡献（创新点）
1. **端到端3D空间纹理生成**：直接在3D空间优化纹理，避免多视角方法的自遮挡错误，与基于UV/多视角投影的方法本质不同；
2. **八叉树扩散模型**：首次将DDPM的逐层去噪过程应用于八叉树节点，利用OCNN的八叉树算子（Oct-conv、Oct-norm）处理层级点云数据，突破显存限制支持密集采样；
3. **条件生成扩展**：设计八叉树多头交叉注意力模块（octree-based cross-attention），将CLIP提取的文本/图像特征注入扩散过程，实现文本引导和图像引导纹理生成；
4. **定量与定性优势**：在ShapeNet上FID均值为14.75，显著优于Point-UV (2-Stage) 的17.31，用户研究偏好率达80.7%。

## 方法详解
**整体流程**：采样表面点云 → 构建八叉树 → 八叉树扩散去噪 → 逆八叉树还原彩色点云 → 映射回网格。

1. **八叉树构建**：
   - 使用meshsampling从3D网格表面采样 $M$ 个点（坐标+RGB），本文采样100K点；
   - 对点云平移并对齐至原点，量化处理：$\mathcal{P}_Q = round((\mathcal{P} - offset) / qs)$，量化步长 $qs$ 需满足 $qs \geq (max(P)-min(P))/(2^L-1)$，$L$ 为最大深度；
   - 根节点开始递归8分，用8位二进制编码记录子节点占用状态，直到深度 $L$；叶节点对应边长为 $qs$ 的立方体，内部点合并到最近立方体；
   - 重建误差控制为 $error \leq qs/2$。

2. **八叉树扩散模型**：
   - 采用U-Net架构，含ResNet块，四级不同树深度（12/11/10/9）处理多尺度信息；
   - 每个ResNet块含两个Oct-conv、两个Oct-norm和一个MLP；
   - 训练时预测干净信号（非噪声），采用 $\mathcal{L}_{simple} = ||\epsilon_\theta(x_t) - \epsilon_t||_2^2$；
   - 推理时从 $N(0,1)$ 初始化噪声，经DDPM采样逐步去噪。

3. **条件生成**：
   - 在ResNet块后插入八叉树多头交叉注意力模块（首层省略以节省显存）；
   - 输入 $X \in \mathbb{R}^{N \times C}$ 被切分为patch，通过线性层生成Query；Key/Value由CLIP特征的线性层生成；
   - 文本条件使用TextShape数据集的CLIP embedding，图像条件使用单视角渲染图的CLIP embedding。

4. **关键超参**：树深度 $L=12$，采样点数100K，训练2000 epoch，AdamW优化器，学习率 $1e^{-4}$，batch size 128。

## 实验与结果
- **数据集**：ShapeNet（chair/table/car/bench），训练集划分同Point-UV [43]，每模型采样100K点及对应地面真实纹理。
- **基线**：Unconditional：Texture Fields、Texturify、Point-UV (1-Stage/2-Stage)、Text2Tex；Conditional：Point-UV、Text2Tex。
- **评价指标**：FID、KID（渲染512×512，4个随机视角）。
- **主要结果**：

| 方法 | Avg FID↓ | Avg KID↓ |
|------|----------|----------|
| Point-UV (2-Stage) | 17.31 | 0.30 |
| Ours | **14.75** | **0.13** |
| Text2Tex* | 53.22 | 0.79 |
| Ours* | 38.11 | 0.14 |

- 最优类别：Table类FID=7.92，Car类KID=0.10；
- **用户研究**：2000条响应，TexOct偏好率80.7% vs Point-UV的19.3%；
- **超参分析**：深度12为最佳（depth=13过拟合，FID回升至28.19）；采样点100K在FID/KID与显存间取得平衡（10K不足，200K收益有限）。

## 相关工作脉络
1. **UV/2D空间方法**（AUV-Net、TexFusion）：依赖UV对齐，受限于展开唯一性，本文直接在3D空间操作，规避此问题；
2. **多视角投影方法**（Text2Tex、Texture）：需解决视图优化与自遮挡，属NP-hard集合覆盖，本文无此约束；
3. **点扩散方法**（Point-UV、Lion）：Point-UV用稀疏点云（4096点）生成粗糙纹理，本文通过八叉树实现密集点云高效处理；
4. **3D表面卷积**（Texturify、O-CNN）：Texturify用GAN直接作用于3D面片，本文引入扩散模型实现更稳定训练；
5. **Tri-plane方法**（3DGen、Sin3DM）：在隐式三角平面中生成纹理，依赖预训练VAE，本文直接在点云/八叉树上操作，无需额外编解码器。

## 局限性与未来方向
- **仅使用颜色信息**：当前模型只输入/输出RGB，未融合法向量、曲率、Laplace-Beltrami算子等几何先验，可能限制几何感知纹理质量；
- **自监督训练局限**：无条件生成依赖ShapeNet中提供的真实纹理作为监督，未涉及无标注数据或自监督预训练；
- **超参敏感**：八叉树深度影响训练时间和重建误差，depth过大易过拟合，需经验调优；
- **条件生成依赖CLIP**：文本/图像条件通过CLIP嵌入注入，对CLIP语义理解能力依赖较强，极端描述下效果未验证。

## 研究启发与可借鉴点
1. **八叉树+扩散的组合范式**：将Oct-conv/Oct-norm等八叉树算子与DDPM结合，可用于其他3D点云任务（形状生成、分割、补全），是可复用的架构模板；
2. **密集采样+层级压缩策略**：100K点通过八叉树高效编码，避免显存爆炸，为点云生成任务提供了显存友好型设计参考；
3. **条件注入方式**：八叉树多头交叉注意力模块（patch划分+线性投影）可与现有3D扩散模型（如Point-E、LION）结合，提升文本/图像条件控制能力；
4. **预测干净信号替代噪声**：训练中预测 $x_0$ 而非 $\epsilon_t$（同RenderDiffusion、Point-UV），提升训练稳定性，值得在3D扩散任务中尝试；
5. **超参消融设计**：对八叉树深度和采样点数的联合分析（图8、表3）提供了清晰的显存-质量权衡范式，可直接迁移至类似点云生成工作。

## 关键术语表
**DDPM（Denoising Diffusion Probabilistic Models）**：通过前向加噪和反向去噪的马尔可夫链学习数据分布的生成模型，本文以 $\mathcal{L}_{simple}$ 作为训练目标；
**Octree（八叉树）**：将3D空间递归8分树的层级数据结构，本文用于高效编码密集点云并支持多尺度特征学习；
**Oct-conv / Oct-norm**：基于OCNN的八叉树卷积/归一化算子，在八叉树节点间传递特征，替代传统3D卷积；
**CLIP**：OpenAI的多模态预训练模型，提取文本和图像的联合语义嵌入，本文用于条件生成；
**FID（Fréchet Inception Distance）**：衡量生成图像与真实图像分布差异的指标，越低越好；
**KID（Kernel Inception Distance）**：类似FID的度量，基于MMD核距离，对样本量更鲁棒；
**Reverse Octree**：将八叉树层级结构逆变换回点云的过程，保留每个点的预测纹理值；
**Meshsampling**：从3D网格表面均匀采样点集的算法，本文用于获取100K表面点及对应纹理。

## 可复现要素
- **数据集**：ShapeNet [6]，训练集划分同Point-UV [43]，**公开可用**；
- **代码/权重**：论文未提供开源声明，**未开源**；
- **关键超参**：树深度 $L=12$，采样点数100K，训练2000 epoch，batch size 128，学习率 $1e^{-4}$，AdamW优化器；
- **评估协议**：渲染分辨率512×512，4个随机视角（无条件）或20个视角（有条件对比），FID/KID计算。
