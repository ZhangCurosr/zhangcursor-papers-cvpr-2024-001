---
title: "HashPoint-Accelerated-Point-Searching-and-Sampling-for-Neura"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_HashPoint_Accelerated_Point_Searching_and_Sampling_for_Neural_Rendering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:02:05"
field: "神经辐射场与点云渲染"
keywords: ["神经渲染", "点云渲染", "光线追踪", "哈希搜索", "自适应采样", "NeRF"]
innovations: ["哈希表加速的点云搜索（3D转2D哈希查找）", "自适应主表面采样（仅对首次相交表面采样）", "结合光栅化速度与光线追踪质量的新范式"]
benchmarks: ["Synthetic-NeRF", "Replica", "Waymo", "ShapeNet"]
---

# 论文速读：HashPoint: Accelerated Point Searching and Sampling for Neural Rendering

## 一句话总结
本文提出 HashPoint 方法，通过哈希表加速的点云搜索和自适应主表面采样技术，结合光栅化的高效搜索与光线追踪的高质量渲染优势，显著提升了神经体积渲染中的点搜索与采样效率，在保持甚至提升重建精度的同时实现最高 80 倍加速。

## 研究问题与动机
- **核心问题**：神经体积渲染中点云搜索和采样效率低，现有方法在速度和质量之间难以兼得。
- **光栅化方法不足**：NPBG、FreqPCR 等方法虽然可实现实时渲染，但存在可见空洞，且 U-Net 修复难以生成高保真细节。
- **光线追踪方法不足**：Point-NeRF、Point-SLAM 等方法质量高但搜索复杂、对多个表面过度采样，导致渲染时间过长。
- **现有加速结构局限**：Uniform grid、K-d tree、Octree 等方法在射线查询时仍有较高计算开销，且固定半径或 K 近邻策略无法区分表面重要性。

## 核心贡献（创新点）
- **哈希点搜索（Hashed-Point Searching）**：将 3D 点云搜索空间转换至 2D 图像平面，通过哈希表实现 O(1) 复杂度的高效邻居点检索，与 K-d tree（O(m log n + mq)）等传统方法本质不同。
- **自适应主表面采样（Adaptive Primary Surface Sampling）**：基于点云几何分布自适应选择主表面样本点，仅对光线首次相交的表面进行特征聚合，区别于多表面均匀采样（Point-NeRF）和 K 近邻提取（NPLF/Pointersect）策略。
- **混合渲染框架**：结合光栅化的快速搜索与光线追踪的高质量渲染，既避免光栅化的空洞问题，又克服光线追踪的计算瓶颈。
- **广泛兼容性**：方法可无缝集成到 Point-NeRF、Point-SLAM、NPLF、Pointersect 等多种现有光线追踪点云渲染方法中。

## 方法详解
- **加速结构构建**：将所有点云投影到图像平面，保留每个像素内的所有点，使用 Morton code（Z-order）排列点列表，通过 CUDA 原子操作在 O(1) 时间内完成重排，构建哈希表（键为像素索引，值为点列表中起始位置和点数）。
- **自适应搜索半径**：从相机光心沿射线方向发射圆锥进行搜索，搜索核半径 s 根据像素尺寸和焦距自适应计算，样本点搜索半径 r 随距离 t 线性增长（公式 3），避免固定半径引入的噪声。
- **主表面采样策略**：将邻近点投影到射线上形成样本候选点，计算"伪无符号距离函数（pseudo-UDF）"：$$d_j = \frac{1}{K}\sum_{i=1}^K \|x_j^{sp} - x_i^{pc}\|_2$$（公式 4）。
- **置信度权重计算**：将距离转换为置信度 $$\alpha_j = \gamma \exp(-d_j^2/\beta^2)$$，再结合体积渲染的遮挡感知权重 $$w_j = \alpha_j \prod_{k=1}^{j-1}(1-\alpha_k)$$（公式 5），仅保留 w_j > 0 的近主表面样本点。
- **与现有方法集成**：采样后可接 MLP 预测密度和颜色（如 Point-NeRF），或直接基于点特征估计颜色（如 NPLF、Pointersect）。

## 实验与结果
- **数据集**：Synthetic-NeRF（合成）、Replica（室内）、Waymo（户外）、ShapeNet（3D 模型）。
- **评估指标**：PSNR、SSIM、LPIPS、FPS。
- **Synthetic-NeRF**：集成 Point-NeRF 后 PSNR 达 33.22（仅次于 Point-NeRF 的 33.31），FPS 达 9.60，实现 **80 倍加速**（对比 Point-NeRF 的 0.12 FPS）。
- **Replica**：集成 Point-SLAM 后 PSNR 35.43、SSIM 0.983，分别比深度引导采样快 **1.8 倍**、比多表面采样快 **11.5 倍**。
- **Waymo**：集成 NPLF 后 PSNR 30.57、SSIM 0.912，实现 **6 倍加速**（1.98 FPS vs 0.33 FPS）。
- **ShapeNet**：集成 Pointersect 后 PSNR 29.5、SSIM 1.0、LPIPS 0.02，实现 **8 倍加速**（10.12 FPS vs 1.25 FPS）。
- **搜索效率**：100 万点云在 RTX 4090 上仅需 0.5ms 搜索 + 3.5ms 采样；CPU 测试显示优于 Uniform grid、K-d tree、Octree。

## 相关工作脉络
- **Point-NeRF [62]**：采用 Uniform grid 搜索 + 多表面采样，本文用哈希搜索和主表面采样替代，在保持质量的同时大幅提升速度。
- **NPLF [41]**：暴力搜索 + K 近邻点特征聚合，本文方法避免了对非主表面点的噪声采样，精度更高且速度快 6 倍。
- **Pointersect [10]**：Uniform grid 搜索 + K=40 近邻，本文减少至 K=6 且只取主表面点，加速 8 倍同时提升 LPIPS。
- **Point-SLAM [48]**：依赖密集深度图 + 单表面/多表面采样，本文无需深度输入且自适应采样数量，速度提升 1.8-11.5 倍。
- **3D Gaussian Splatting [24]**：虽也用光栅化加速搜索，但 Gaussian 形状小需更多点，本文用插值方式只需更少点（Lego 场景 35MB vs 200MB）。
- **Hash 编码方法（Instant NGP [38]、MixNeRF [28]）**：聚焦内存与优化效率，本文聚焦搜索与采样效率，两者可互补。

## 局限性与未来方向
- **几何噪声敏感**：ablation 显示初始点云质量影响主表面采样效果，需 10 分钟点优化（pruning & growing）才能发挥最佳性能。
- **超参数依赖**：β 依赖点云密度、γ 控制采样范围，需针对不同类型场景调整。
- **仅主表面采样**：对于透明或半透明场景可能丢失次表面信息。
- **未来方向**：探索无需几何优化的端到端训练、扩展到动态场景、与 3D Gaussian Splatting 等最新方法结合。

## 研究启发与可借鉴点
- **3D→2D 搜索转换思想**：将三维空间搜索降维至二维图像平面进行哈希查找，可迁移至其他 3D 视觉任务（如点云配准、邻居查询）。
- **伪 UDF 概念**：用点到邻居平均距离近似表面距离，比严格 UDF 计算更高效，可用于快速表面评估。
- **自适应采样数量**：根据点云分布动态决定样本点数（1 到 n），而非固定 K 值，可推广至其他采样敏感任务。
- **搜索与渲染解耦**：先加速搜索定位候选点，再按需采样，模块化设计便于集成到不同渲染框架。
- **混合架构思路**：结合光栅化（快速筛选）和光线追踪（高质量渲染）各自优势，为神经渲染提供新范式。

## 关键术语表
- **Hashed-Point Searching**：将点云投影到图像平面构建哈希表，实现 O(1) 复杂度的射线邻居点快速检索。
- **Adaptive Primary Surface Sampling**：基于点云几何分布自适应选择主表面样本点，仅对光线首次相交表面进行特征聚合。
- **Pseudo-UDF**：伪无符号距离函数，通过计算样本点到 K 个邻居点的平均距离近似表面距离。
- **Volume Rendering**：体积渲染，沿射线采样并累积透明度与颜色以生成像素值。
- **Z-order / Morton Code**：多维数据线性排序方法，用于将同像素点有序存储以优化缓存访问。
- **K-NP Extract**：K 近邻点提取，直接选取射线附近 K 个最近点特征进行颜色预测。
- **Multi-Surface Sampling**：多表面采样，在所有与射线相交表面上均匀采样，计算成本高。

## 可复现要素
- **数据集**：Synthetic-NeRF、Replica、Waymo、ShapeNet（论文使用标准数据集）。
- **代码开源**：是，https://jiahao-ma.github.io/hashpoint/。
- **关键超参**：β（依赖点云密度）、γ（控制采样范围）、K（邻居点数，默认 8 或 6）、搜索核大小 s。
- **训练策略**：Point-NeRF 集成需先用多表面采样优化 10K 迭代，再切换至主表面采样；其他方法可直接集成。
- **硬件**：NVIDIA RTX 4090 GPU（搜索 0.5ms + 采样 3.5ms / 百万点）。
