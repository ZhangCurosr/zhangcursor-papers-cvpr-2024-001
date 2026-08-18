---
title: "ASH-Animatable-Gaussian-Splats-for-Efficient-and-Photoreal-H"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Pang_ASH_Animatable_Gaussian_Splats_for_Efficient_and_Photoreal_Human_Rendering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:20:18"
field: "可动画人体神经渲染"
keywords: ["3D Gaussian Splatting", "Human Rendering", "Real-time Rendering", "Animatable Avatar", "Neural Rendering"]
innovations: ["将3D高斯分布附着在可变形网格上并在2D纹理空间高效学习", "运动感知纹理编码实现骨骼姿态到高斯参数的2D到2D翻译"]
benchmarks: ["DynaCap", "Novel View Synthesis", "Novel Pose Synthesis"]
---

# 论文速读：ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering

## 一句话总结
本文提出ASH，一种将3D高斯分布附着在可变形模板网格上的实时人体渲染方法，通过在2D纹理空间高效学习高斯参数，实现了高质量、可可控骨骼运动的人体avatar实时渲染。

## 研究问题与动机
- **核心问题**：如何在保持photorealistic渲染质量的同时，实现人体avatar的实时可动画渲染
- **现有方法不足**：
  - NeRF类混合方法渲染质量高但速度慢（约5秒/帧），无法满足实时需求
  - 纯网格方法（如DDC）虽然实时但渲染质量较差，缺乏细节
  - 原始3D Gaussian splatting仅适用于静态场景，难以直接扩展到动态可动画人体
  - 在3D空间中直接学习骨骼姿态到高斯参数的映射对计算资源要求极高

## 核心贡献（创新点）
1. **提出ASH框架**：将3D高斯分布参数化在可变形模板网格表面，实现实时高质量人体渲染，与NeRF类方法相比速度提升数百倍
2. **2D纹理空间参数化**：通过将高斯参数表示为纹理空间的texel，将3D问题转化为2D图像翻译任务，避免了难以扩展的3D架构
3. **运动感知解码器设计**：设计几何网络$\mathcal{E}_{geo}$和外观网络$\mathcal{E}_{app}$，从运动感知纹理（法线贴图+位置贴图）高效预测高斯参数
4. **两阶段训练策略**：提出warmup阶段预训练伪地面真实高斯参数，再在完整序列上进行端到端训练，解决长序列训练收敛难题

## 方法详解
**整体框架**：
- 输入：骨骼运动$\bar{\theta}_f$和虚拟相机视图
- 输出：实时（~30fps） photorealistic渲染图像

**关键组件**：

1. **可动画模板网格**：
   - 使用Habuermann等人的character model
   - 在unposed-canonical空间中通过embedded deformation和per-vertex displacement进行非刚性变形
   - 通过Dual Quaternion skinning将canonical顶点定位为posed空间顶点

2. **高斯纹理参数化**：
   - 每个texel代表一个3D高斯分布，参数为$(\bar{\mu}_{uv,i}, \bar{d}_{uv,i}, q_{uv,i}, s_{uv,i}, \alpha_{uv,i}, \eta_{uv,i})$
   - Canonical空间位置通过重心坐标插值得到：$\bar{\mu}_{uv,i} = w_{a,i}\bar{V}_{f,j} + w_{b,i}\bar{V}_{f,k} + w_{c,i}\bar{V}_{f,l}$
   - 通过Dual Quaternion skinning变换到posed空间：$\mu_{uv,i} = T_{uv,i}(\bar{\mu}_{uv,i} + \bar{d}_{uv,i})$

3. **运动感知解码器**：
   - 从posed mesh计算normal textures $\mathbf{T}_{n,f}$和position textures $\mathbf{T}_{p,f}$
   - 几何网络：$\mathcal{E}_{geo}(\mathbf{T}_{n,f}, \mathbf{T}_{p,f}) = (\bar{d}_{uv,i}, s_{uv,i}, q_{uv,i}, \alpha_{uv,i})$
   - 外观网络：$\mathcal{E}_{app}(\mathbf{T}_{n,f}, \mathbf{T}_{p,f}, \Phi_f) = \eta_{uv,i}$，其中$\Phi_f$编码全局外观特征

4. **训练策略**：
   - Warmup阶段：采样t帧，固定高斯位置，优化其他参数作为伪ground truth，损失$\mathcal{L}_{pre} = \mathcal{L}_2(\{\mathcal{G}_i'\}, \{\mathcal{G}_i''\})$
   - 最终训练：像素级L1 + SSIM损失$\mathcal{L}_{main} = 0.1\mathcal{L}_1 + 0.9\mathcal{L}_{ssim}$

## 实验与结果
**数据集**：
- DynaCap数据集（tight和loose服装两个subject）
- 自建数据集：120相机系统，25fps，27000帧训练/7000帧测试

**评估指标**：PSNR、LPIPS（1K分辨率，每10帧采样）

**主要结果**：

| 方法 | Tight PSNR | Tight LPIPS | Loose PSNR | Loose LPIPS |
|------|-----------|-------------|------------|-------------|
| DDC (RT) | 31.21 | 22.56 | 28.10 | 31.68 |
| HDHumans | 30.98 | 15.09 | 29.24 | 15.79 |
| **ASH (RT)** | **35.84** | **11.92** | **35.47** | **8.30** |

- 实时方法中ASH显著超越DDC（Tight PSNR提升4.63dB，LPIPS降低10.64）
- 与离线方法HDHumans相比，ASH在novel-view synthesis上PSNR更高（35.84 vs 30.98），且实现实时渲染（29.64fps）
- Ablation显示motion condition和motion-aware offset均必要，256分辨率在质量和效率间取得最佳平衡

## 相关工作脉络
1. **DDC [13]**：实时mesh-based方法，使用learned dynamic textures，ASH在渲染质量上大幅超越
2. **HDHumans [15]**：混合方法，jointly优化neural implicit fields和template mesh，渲染质量接近ASH但需数秒/帧
3. **Neural Actor [33]**：使用parametric human body mesh的texture map作为local pose features，无法处理loose outfits
4. **TAVA [29]**：canonical space中implicit field表示shape/appearance/skinning weights，通过iterative root finding canonicalize
5. **3D Gaussian Splatting [24]**：静态场景表示，ASH将其扩展到动态可动画场景的关键基础

## 局限性与未来方向
- **自述局限**：当前不更新底层deformable template mesh的几何结构
- **未来方向**：探索是否可以直接用Gaussian splatting改进3D mesh几何
- **隐含局限**：需要多视角视频和3D骨骼标注进行训练，数据采集成本较高
- **隐含局限**：texture resolution固定为256，higher resolution会显著增加计算复杂度

## 研究启发与可借鉴点
1. **2D参数化范式**：将3D空间的高斯参数映射到2D纹理空间，用2D CNN高效学习，这一思路可迁移到其他3D生成任务
2. **运动感知纹理编码**：使用normal map + position map作为motion condition，比直接使用skeletal pose更利于2D卷积捕获空间上下文
3. **两阶段训练策略**：先用固定位置的高斯参数做warmup，再端到端训练，这一策略对长序列训练具有通用参考价值
4. **UV参数化优势**：利用mesh的UV parameterization保证Gaussian数量恒定，避免了原始3D Gaussian splatting中splitting/merging带来的不连续性

## 关键术语表
**3D Gaussian Splatting**：一种基于3D高斯分布的显式场景表示方法，通过高效splatting实现实时渲染
**Dual Quaternion Skinning**：一种刚体变形技术，相比传统linear blend skinning能更好保持体积
**Canonical Space**：未变形的参考空间，所有运动和变形都映射到此空间进行处理
**Embeded Deformation**：一种学习非刚性变形的方法，通过embedded graph捕获局部变形模式
**Spherical Harmonics**：用于表示view-dependent appearance的谐波函数，ASH中使用3阶（48维系数）

## 可复现要素
- 数据集：DynaCap公开可用；自建数据集120相机setup未公开
- 代码：论文未提及开源计划
- 关键超参：texture resolution=256，$\lambda_{pix}=0.1$，$\lambda_{ssim}=0.9$，使用U-Net架构
- 训练框架：基于多视角RGB视频和3D骨骼标注
