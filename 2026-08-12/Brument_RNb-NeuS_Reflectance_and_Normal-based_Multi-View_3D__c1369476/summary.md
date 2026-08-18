---
title: "RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Brument_RNb-NeuS_Reflectance_and_Normal-based_Multi-View_3D_Reconstruction_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:15:55"
field: "多视图3D重建"
keywords: ["多视图光度立体", "神经体积渲染", "SDF", "反射率", "法线重建", "重参数化"]
innovations: ["像素级联合重参数化将反射率与法线映射为模拟辐射向量，实现单目标优化", "首次将反射率作为先验系统性地融入MVPS神经体积渲染框架"]
benchmarks: ["DiLiGenT-MV"]
---

# 论文速读：RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction

## 一句话总结
本文提出了 RNb-NeuS，通过像素级联合重参数化将反射率与法线映射为模拟辐射向量，实现了基于神经体积渲染的单目标多视图光度立体重建；在 DiLiGenT-MV 基准上显著超越现有方法，尤其在高曲率和低可见度区域取得了突破性细节恢复效果。

## 研究问题与动机
- **非朗伯场景下 MVS 失效**：传统多视图立体（MVS）依赖亮度一致性假设，在非朗伯表面或弱纹理区域难以精确重建。
- **现有 MVPS 方法的多目标冲突**：最新多视图光度立体（MVPS）方法采用多目标优化，不同目标之间可能存在冲突，导致细尺度细节丢失。
- **高曲率与低可见度区域重建困难**：固定光照条件下，薄结构、高曲率区域及遮挡区域的几何细节恢复仍是未解难题。
- **反射率未被充分利用**：现有方法多依赖法线信息，较少系统性地利用光度立体估计的反射率（albedo）作为几何重建先验。

## 核心贡献（创新点）
1. **像素级联合重参数化**：将反射率与法线联合重参数化为模拟辐射向量，使异构输入转化为齐次优化量，避免多目标冲突。
2. **单目标优化框架**：基于重参数化实现了神经体积渲染（NVR）框架下的单一损失函数，不同于现有方法的多目标权衡。
3. **首次将反射率作为先验用于 MVPS**：利用高质量 PS 方法估计的反射率信息辅助几何重建，显著提升细节恢复能力。
4. **通用性与灵活性**：该方法与任意现有或未来的光度立体方法兼容，无论是否校准、是否基于深度学习。

## 方法详解
- **表面参数化**：使用符号距离函数（SDF）$f: \mathbb{R}^3 \to \mathbb{R}$ 表示几何，其零水平集 $S = \{\mathbf{x} | f(\mathbf{x}) = 0\}$ 为 reconstructed surface；使用标量函数 $\rho: \mathbb{R}^3 \to \mathbb{R}$ 编码反照率（albedo）。
- **重参数化核心设计**：对每个像素，将法线 $\mathbf{n}_k$ 和反射率 $r_k$ 通过 Lambertian 模型重参数化为辐射向量：
  $$\mathbf{v}(\mathbf{n}_k, r_k) = r_k \mathbf{L}_k \mathbf{n}_k \in \mathbb{R}^3$$
  其中 $\mathbf{L}_k$ 为逐像素任意选取的非奇异光照矩阵（3 个线性无关光照方向）。
- **优化目标**：
  $$\min_{f,\rho} \sum_{k=1}^{m} \| \mathbf{v}(\mathbf{n}_k, r_k) - \tilde{\mathbf{v}}_k(f, \rho) \|_1 + \lambda \mathcal{L}_{\text{reg}}(f)$$
- **体渲染函数**：扩展 NeuS 渲染方程，颜色函数替换为：
  $$c_l(\mathbf{x}_k(t)) = \rho(\mathbf{x}_k(t)) \nabla f(\mathbf{x}_k(t))^\top \mathbf{l}_{k,l}$$
  去除了视角依赖性以匹配 Lambertian 假设。
- **正则化**：采用 eikonal 正则项 $\mathcal{L}_{\text{reg}}(f)$ 确保 $\|\nabla f\|=1$。
- **完整流水线**：① 使用 SDM-UniPS 估计每视角反射率与法线图；② 基于不确定性阈值（$\tau=15°$）过滤不可靠像素；③ 跨视角缩放反射率值消除尺度模糊；④ 选取逐像素最优光照三元组；⑤ 在 NeuS 架构上优化 300k 次迭代。

## 实验与结果
- **数据集**：DiLiGenT-MV，包含 5 个真实物体（Bear、Buddha、Cow、Pot2、Reading），每物体 20 个视角、96 个光照图像。
- **评估指标**：F-score、Chamfer Distance（CD）、法线平均角度误差（MAE）；另统计高曲率（8.27%顶点）与低可见度（8.70%顶点）区域性能。
- **最强结果**：平均 CD 为 **0.23**，较次优方法 MVPSNet（0.27）提升 **17.4%**；平均 MAE 为 **4.95°**，优于所有全自动基线。
- **高曲率区域**：CD 仅上升 **4%**（对比 PS-NeRF 上升 36%、MVPSNet 上升 96%）；低可见度区域 CD 仅上升 **13%**（对比基线上升 46%-81%）。
- **消融实验**：去除反射率（W/o reflectance）使平均 CD 从 0.23 升至 0.26；去除最优光照（W/o opt. l.）升至 0.24；去除不确定性过滤（W/o uncert.）影响较小。

## 相关工作脉络
- **经典 MVPS**：Hernandez et al. [6]、Park et al. [20] 基于网格变形/参数化，依赖 Lambertian 假设且无法处理复杂非朗伯表面。
- **SDF-based MVPS**：Logothetis et al. [14] 首次用 SDF 解决 MVPS，但依赖已知校准光照与相机位姿。
- **DiLiGenT-MV 基准**：Li et al. [12] 提出该benchmark，验证了基于网格传播与 BRDF 估计的方法优势。
- **NeRF-based MVPS**：Kaya et al. [10, 9, 11] 系列工作将 NeRF/SDF 引入 MVPS，但需多目标优化与不确定性调参；PS-NeRF [26] 显式建模 BRDF 与光照，计算成本高。
- **快速 MVPS**：MVPSNet [27] 采用聚合阴影匹配预测深度与法线，速度较快但精度受限。
- **本文定位**：通过单目标重parameterization统一反射率与法线输入，避免了多目标冲突，同时保持与标准 NVR 管道的兼容性。

## 局限性与未来方向
- **依赖 PS 输入质量**：SDM-UniPS 偶尔产生错误法线，导致跨视角不一致；需更鲁棒的 PS 方法。
- **计算成本较高**：单物体重建需 8-16 小时（GPU），未来可适配 NeuS2 [23] 将时间降至约 10 分钟。
- **仅使用漫反射分量**：当前假设反射率为标量（albedo），未利用镜面反射、粗糙度等更丰富的 PBR 参数。
- **扩展方向**：使用 $n>3$ 光照向量加 pseudo-inverse 提升鲁棒性；结合反射率不确定性评估；支持更多 PBR 模型。

## 研究启发与可借鉴点
- **异构输入的统一重参数化范式**：将法线（几何）与反射率（光度）映射为齐次辐射向量，避免了多目标优化的超参调优问题，该思路可迁移到其他多源数据融合场景。
- **逐像素最优光照配置**：基于 [4] 的最优光照三元组选择策略，可用于最小化法线估计不确定性，提升重建鲁棒性。
- **不确定性过滤机制**：采用角度偏差阈值（$\tau=15°$）剔除不可靠 PS 输入，简单有效，可推广至其他依赖预训练模块的 pipeline。
- **与先进 PS 方法的即插即用**：本文 MVPS pipeline 与任何 PS 方法兼容，可快速集成最新 SOTA 方法提升性能。
- **高曲率/低可见度区域量化评估**：将评估分解至难区域，比全局均值更能反映方法在细节恢复上的真实能力。

## 关键术语表
- **Multi-View Photometric Stereo (MVPS)**：结合多视角立体与光度立体的 3D 重建任务，同时利用多视角几何约束与多变光照下的光度信息。
- **Signed Distance Function (SDF)**：隐式表面表示，空间中任一点到表面signed距离，零水平集即 reconstructed surface。
- **Neural Volume Rendering (NVR)**：基于 NeuS/NeRF 的体渲染技术，沿视线积分累积颜色/辐射值以合成图像。
- **Re-parameterization**：将原始异构输入（法线+反射率）通过数学变换映射为同质量（辐射向量），便于统一优化。
- **Eikonal Regularization**：约束 SDF 梯度范数为 1 的正则项，确保 SDF 为符号距离函数。
- **Albedo**：表面反射率标量，表示材质的固有颜色/反射能力，与光照无关。
- **DiLiGenT-MV**：多视图光度立体基准数据集，包含 5 个具有复杂反射特性的真实物体，每物体 20 视角×96 光照。

## 可复现要素
- **数据集**：DiLiGenT-MV（公开基准）。
- **代码/权重**：论文未提及代码开源情况。
- **关键超参**：不确定性阈值 $\tau = 15°$；迭代次数 300k；batch size 512 pixels；PS 方法使用 SDM-UniPS [8]，$N_{\text{trials}} = 100$，每 trial 随机选 10 张图像；光照三元组按 [4] 最优配置选取。
- **硬件**：标准 GPU，单物体 8-16 小时。
