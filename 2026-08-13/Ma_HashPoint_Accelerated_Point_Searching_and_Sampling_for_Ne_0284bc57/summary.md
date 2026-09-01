---
title: "HashPoint: Accelerated Point Searching and Sampling for Neural Rendering"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ma_HashPoint_Accelerated_Point_Searching_and_Sampling_for_Neural_Rendering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:01:52"
field: "神经渲染与点云表示"
keywords: ["神经渲染", "点云渲染", "光线追踪", "哈希搜索", "主表面采样", "体积渲染", "实时渲染"]
innovations: ["哈希点搜索：将三维点云搜索降维至二维图像平面哈希表，实现O(1)邻点查询", "自适应主表面采样：基于伪UDF与体积渲染权重自动保留靠近主表面的采样点，避免多表面冗余", "通用可集成架构：与Point-NeRF/Point-SLAM/NPLF/Pointersect无缝结合，实现6-80倍加速"]
benchmarks: ["Synthetic-NeRF", "Replica", "Waymo", "ShapeNet"]
---

# 论文速读：HashPoint: Accelerated Point Searching and Sampling for Neural Rendering

## 一句话总结
论文提出 HashPoint 方法，通过将点云投影到 2D 图像平面构建哈希表实现 O(1) 加速搜索，并结合"伪 UDF"进行自适应主表面采样，在不损失渲染质量的前提下实现 6–80 倍加速。

## 研究问题与动机
- **体神经渲染的低效性**：NeRF 等全局 MLP 方法需对整条射线均匀采样，大量样本落入空空间，导致训练和推理时间过高。
- **点云渲染两类策略的权衡困境**：光栅化方法可实时渲染但存在空洞且缺乏高频细节；光线追踪方法保真度高但需遍历 3D 空间搜索邻点，效率低。
- **现有光线追踪搜索方法的冗余**：K-d 树、八叉树、均匀网格等加速结构的搜索复杂度为 O(m log n) 或更高，且实际上只有最近的主表面才对渲染贡献最大，多表面采样存在冗余计算。
- **稀疏点云下 K 近邻采样的误差**：NPLF、Pointersect 等取固定 K 个最近点的方法在点云稀疏时容易引入非主表面噪声。

## 核心贡献（创新点）
- **哈希点搜索（Hashed-Point Searching）**：将三维搜索转化为沿相机视锥的二维图像平面查找，通过哈希表实现 O(n + mq) 搜索复杂度，相比 K-d 树/八叉树的 O(m log n + mq) 更快。
- **自适应主表面采样（Adaptive Primary Surface Sampling）**：基于"伪 UDF"评估候选点到点云表面的距离，结合体积渲染的 alpha 混合权重自动保留靠近主表面的样本点，避免多表面冗余采样。
- **通用可集成架构**：提出的搜索与采样模块可无缝嵌入 Point-NeRF、Point-SLAM、NPLF、Pointersect 等主流光线追踪点云渲染方法，带来一致的速度提升。
- **与 3D Gaussian Splatting 的本质区分**：Gaussian 需存储形状参数（200MB vs 35MB），本文通过插值邻近点特征实现高质量渲染，存储更紧凑。

## 方法详解
- **哈希表构建**：将所有 3D 点投影到相机图像平面，同一像素内的点按 Morton Z-order 排序后存入点列表；哈希表 key 为像素索引，value 为起始位置和点数，构建复杂度 O(n)，查询复杂度 O(1)。
- **自适应搜索半径**：每个像素发射一个圆锥搜索邻点，搜索核大小 s = 2·⌈ṙ/ṙ̇⌉ + 1 随采样点距射线原点距离 t 线性放大，避免固定半径引入越界噪声。
- **候选点生成**：将搜索半径内的邻近点投影到相机射线上，得到一组候选采样点 x_j^sp。
- **伪 UDF 重要性评估**：候选点重要性由 d_j = (1/K) Σ ||x_j^sp − x_i^pc|| 计算，反映点到点云表面的平均距离（非严格 UDF，故称"pseudo-UDF"）。
- **自适应权重分配**：α_j = γ·exp(−d_j²/β²)，再结合体积渲染的 occlusion-aware 权重 w_j = α_j · ∏_{k<j}(1 − α_k)，保留 w_j > 0 的主表面候选点作为最终采样点；数量从 0 到 n 自适应变化。
- **与基线方法的集成方式**：Point-NeRF 中先用多表面采样（MS）优化 10K 迭代获取几何，再切换至主表面采样（PS）；NPLF 中减少 K 值（40→8）并结合主表面筛选；Pointersect 中 K 值从 40 降至 6。
- **几何精炼**：MVS 初始化点云含噪声时需配合点修剪与生长（Pruning & Growing, P&G）进行约 10 分钟几何优化，否则主表面采样会受偏移影响。

## 实验与结果
- **数据集**：Synthetic-NeRF（合成）、Replica（室内真实）、Waymo（室外真实）、ShapeNet（通用 3D 形状）。
- **评估指标**：PSNR、SSIM、LPIPS、FPS。
- **Point-NeRF + HashPoint（Synthetic-NeRF）**：PSNR 33.22、FPS 9.60，**80 倍加速**，精度仅次于最优光栅化方法 FreqPCR。
- **Point-SLAM + HashPoint（Replica）**：PSNR 35.43 / SSIM 0.983，比深度引导采样快 1.8 倍、比多表面均匀采样快 **11.5 倍**。
- **NPLF + HashPoint（Waymo）**：PSNR 30.57 / SSIM 0.912，**6 倍加速**，主表面采样比 K 近邻更准确（图 6 可视化）。
- **Pointersect + HashPoint（ShapeNet）**：PSNR 29.5±5.6、SSIM 1.0±0.0，**8 倍加速**，K 值从 40 降至 6 仍更优。
- **搜索效率基准（CPU 测试）**：百万级点云搜索仅需 **0.5ms**（GPU RTX 4090），采样 3.5ms，总耗时 4ms。
- **消融结论**：HS 比 Uniform Grid 快 5 倍；PS 比多表面采样快 60–80 倍；无几何精炼时精度下降（图 9），验证了 P&G 的必要性。

## 相关工作脉络
- **Point-NeRF [62]**：使用均匀网格搜索 + 多表面均匀采样；本文用哈希搜索替代均匀网格、用主表面自适应采样替代多表面均匀采样，定位效率与精度双提升。
- **NPLF [41]**：暴力搜索 + K 近邻（K=8）颜色预测；本文同样利用邻点特征但通过主表面筛选替代固定 K 近邻，避免稀疏点云的噪声干扰。
- **Pointersect [10]**：均匀网格搜索 + K=40 近邻；本文将其 K 值降至 6 并结合主表面筛选，实现更高 FPS（1.25→10.12）。
- **Point-SLAM [48]**：依赖稠密深度图进行单表面采样；本文无需深度传感器输入即可自适应主表面采样，适用范围更广。
- **3D Gaussian Splatting [24]**：基于光栅化的可微分化渲染，但需存储 3D 高斯形状参数（200MB vs 35MB）；本文用轻量插值方案实现同等质量，存储效率更高。
- **NeRF [36] / Mip-NeRF [4] / Zip-NeRF [6]**：体渲染基线；本文不直接替换 MLP 渲染器，而是加速其前置的点云搜索与采样环节，属于互补型优化。

## 局限性与未来方向
- **依赖几何质量**：点云初始化含噪声时（如无 P&G 精炼），主表面采样精度显著下降，限制了在低质量重建场景的直接应用。
- **超参数敏感性**：β 依赖点云密度、γ 控制采样范围，不同场景需手动调参，泛化性有待验证。
- **稀疏点云适应性未充分讨论**：论文仅在 Synthetic-NeRF、Replica 等中等密度数据集验证，极端稀疏场景下的表现存疑。
- **未对比 3D Gaussian Splatting**：虽在文中简要区分，但未在相同条件下进行定量比较，难以判断相对优势边界。
- **未来方向**：可探索无几何精炼的鲁棒主表面采样策略；将哈希搜索与动态点云更新结合用于 SLAM 场景；扩展至动态场景与视频级渲染。

## 研究启发与可借鉴点
- **三维→二维的降维搜索思想**：将 3D 点云空间搜索映射到 2D 图像平面并通过哈希表加速，这一范式可迁移至点云配准、3D 目标检测等需要快速邻域查询的任务。
- **伪 UDF + 体积渲染权重结合**：用简单距离度量替代严格 SDF/UDF 计算，再叠加 occlusion-aware 的 alpha 混合公式，可在保证主表面优先的同时避免复杂几何推理。
- **光栅化+光线追踪的混合架构**：光栅化负责快速粗筛邻点，光线追踪负责高质量渲染，两者分工协作的思路可推广至神经渲染加速、实时 SLAM 等场景。
- **消融中"先 MS 后 PS"的训练策略**：10K 迭代多表面采样进行几何精炼，再切换至主表面采样进行高效渲染，这种"先学习后加速"的两阶段训练策略具有通用参考价值。
- **K 值动态缩减的空间**：Pointersect 原 K=40，本文降至 K=6 后精度不降反升，提示在神经渲染中"质量优先于数量"的设计哲学值得深入探索。

## 关键术语表
- **Hashed-Point Searching**：将 3D 点云投影到 2D 图像平面并按像素构建哈希表，实现 O(1) 复杂度的邻点快速查询。
- **Adaptive Primary Surface Sampling**：基于伪 UDF 和体积渲染权重自适应保留靠近主表面的采样点，避免多表面冗余采样。
- **Pseudo-UDF（伪无符号距离函数）**：候选点到其 K 个最近邻点云点的平均距离，虽非严格 UDF 但可近似表征点到表面的远近程度。
- **Rasterization-based rendering**：将 3D 点投影到图像平面、通过 z-buffer 确定可见性并进行渲染的方法，速度快但易产生空洞。
- **Ray-tracing-based point cloud rendering**：沿相机射线搜索点云邻点并进行特征插值或体积渲染的方法，精度高但搜索开销大。
- **Volume rendering**：沿射线对离散采样点按透射率 α 和密度 σ 进行累积积分以计算像素颜色的物理渲染公式。
- **Morton code（Z-order）**：将多维坐标映射为一维线性顺序的编码方式，用于加速同像素内点云的有序存储与检索。
- **Pruning & Growing（P&G）**：点云几何精炼过程，通过删除噪声点并生长新点来改善初始 MVS 点云的几何质量。

## 可复现要素
- **代码**：已开源，地址 https://jiahao-ma.github.io/hashpoint/
- **数据集**：Synthetic-NeRF、Replica、Waymo、ShapeNet（均为公开数据集）
- **关键超参数**：β（依赖点云密度）、γ（控制采样范围）、K（近邻数量，不同方法取 6/8/40）、搜索半径 r_min（对应 t_near 处最小半径）——论文正文未给出具体数值，需在 Supplementary 中进一步查阅。
