---
title: "RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Brument_RNb-NeuS_Reflectance_and_Normal-based_Multi-View_3D_Reconstruction_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:15:38"
---

# 论文速读：RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction

## 一句话总结
本文提出 RNb-NeuS，通过将多视角光度立体（PS）估计的法线与反射率图联合重参数化为模拟多变光照下的辐射度向量，构建了单一目标的神经体积渲染（NVR）优化框架；在 DiLiGenT-MV 基准上显著提升了重建精度，尤其在高曲率与低可见性区域的细节恢复上达到 SOTA。

## 研究问题与动机
- MVS 结合 NVR 能处理复杂结构与自遮挡，但在非朗伯表面、低纹理或窄基线配置下仍难以恢复极薄几何细节，且亮度一致性假设常失效。
- 传统光度立体（PS）擅长恢复高频法线与估计反射率，但低频大尺度几何重建能力不足。
- 现有 MVPS 方法多采用多目标联合优化，不同来源的法线、深度与颜色目标可能存在内在冲突，导致精细细节丢失或需繁琐的超参调优。
- 核心动机：如何在不引入多目标冲突的前提下，将 PS 提供的高精度法线与反射率先验无缝融入 NVR 流水线，实现单一目标的稳定优化。

## 核心贡献（创新点）
1. **像素级联合重参数化策略**：将异质的反射率标量与单位球面法线向量统一映射为任意多变光照下的辐射度向量，使输入数据与 NVR 渲染值处于同构度量空间。
2. **单目标优化框架**：摒弃现有 MVPS 的多目标权衡机制，仅保留数据保真项与 Eikonal 正则项，从优化结构上根除目标冲突导致的细节退化。
3. **首个显式利用反射率先验的 MVPS 范式**：不仅融合法线几何先验，还首次将反射率（反照率）作为直接约束融入体积渲染，显著增强了几何-光度一致性。
4. **高度通用的流水线架构**：方法独立于前端 PS 算法，可无缝兼容任意校准/非校准、传统优化或深度学习驱动的光度立体后端。

## 方法详解
- **表面参数化**：几何由 SDF $f:\mathbb{R}^3\to\mathbb{R}$ 的零水平集定义，光度属性由标量反照率场 $\rho:\mathbb{R}^3\to\mathbb{R}$ 编码。
- **辐射度重参数化**：对每个像素选取 3 个线性无关的任意光照方向构成矩阵 $\mathbf{L}_k$，将输入转换为 $\mathbf{v}(\mathbf{n}_k, r_k) = r_k \mathbf{L}_k \mathbf{n}_k \in \mathbb{R}^3$。该映射在 $r_k \neq 0$ 时保持双射，可反向恢复原始反射率与法线；负辐射度值用于自然表达自阴影。
- **单目标优化目标**：
  $$\min_{f,\rho} \sum_{k=1}^m \| \mathbf{v}(\mathbf{n}_k, r_k) - \tilde{\mathbf{v}}_k(f, \rho) \|_1 + \lambda \mathcal{L}_{\mathrm{reg}}(f)$$
  第一项衡量输入模拟辐射度与体积渲染辐射度的一致性，第二项为 Eikonal 正则 $\mathcal{L}_{\mathrm{reg}}(f) = \frac{\sum_k \int_{t_n}^{t_f} (\|\nabla f\|^2 - 1)^2 dt}{m(t_f - t_n)}$，确保 $\|\nabla f\|=1$ 并稳定法线。
- **体积渲染适配**：在 NeuS 基础上移除视角依赖以匹配朗伯模型；体素颜色替换为 $c_l(\mathbf{x}) = \rho(\mathbf{x}) \nabla f(\mathbf{x})^\top \mathbf{l}_{k,l}$，沿射线积分得 $\tilde{\mathbf{v}}_k$。
- **完整 MVPS 流水线**：① 使用 SDM-UniPS 估计各视角法线与反射率；② 基于 $N_{\mathrm{trials}}=100$ 次随机试验的中位角偏差过滤低置信度像素（阈值 $\tau=15^\circ$）；③ 跨视角中位数对齐反射率尺度；④ 采用最优光照三元组（倾斜均分120°、倾角54.74°）模拟辐射度；⑤ 300k 次迭代、batch size 512 联合优化 $f$ 与 $\rho$；⑥ Marching Cubes 提取网格。

## 实验与结果
- **数据集与基线**：DiLiGenT-MV（5个复杂材质物体，20视角×96光照）；对比基线包括 Park16、Li19、Kaya22、PS-NeRF、Kaya23、MVPSNet。
- **评估指标**：Chamfer Distance (CD)、Mean Angular Error (MAE)、F-score；额外统计高曲率区域（8.27% 顶点）与低可见性区域（<5视角，8.70% 顶点）的误差。
- **主要结果**：平均 CD 为 **0.23**，较第二名 MVPSNet（0.27）提升 **17.4%**；平均 MAE 为 **4.95°**，优于 Kaya23（5.14°）及所有其他全自动方法；F-score 曲线整体领先。
- **细节优势**：在高曲率区域，PS-NeRF 与 MVPSNet 的 CD 误差分别飙升 36% 与 96%，本方法仅上升 **4%**；在低可见性区域，其他方法误差上升 46%~81%，本方法仅上升 **13%**，证明单目标优化有效避免了冲突导致的细节丢失。
- **消融实验**：移除反射率
