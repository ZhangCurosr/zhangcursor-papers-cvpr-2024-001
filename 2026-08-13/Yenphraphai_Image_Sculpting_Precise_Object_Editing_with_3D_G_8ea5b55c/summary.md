---
title: "Image Sculpting: Precise Object Editing with 3D Geometry Control"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Yenphraphai_Image_Sculpting_Precise_Object_Editing_with_3D_Geometry_Control_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:52:15"
---

# 论文速读：Image Sculpting: Precise Object Editing with 3D Geometry Control

## 一句话总结
本文提出 Image Sculpting 框架，通过将单张图像中的目标对象重建为带纹理的 3D 模型，在 3D 空间利用图形学变形工具进行精确、可量化的操控，再经由粗到细的扩散增强管线将其渲染回高分辨率 2D 图像，实现了 pose 编辑、旋转平移、3D 组合、雕刻与序列添加等高保真图像编辑任务。

## 研究问题与动机
- 现有基于文本提示或 2D 潜在空间操控的图像编辑方法（如 InstructPix2Pix、DragGAN 系列）在实现精确、可量化操控（如“旋转 42 度”、“上移 5 cm”）时存在严重歧义与物理不可控问题。
- 2D 交互式方法受限于平面特征空间，难以准确表达三维空间变换、处理遮挡关系，且缺乏骨骼/物理结构感知能力。
- 传统计算机图形学（CG）管线虽能提供高精度 3D 操控，但依赖昂贵硬件、专业软件与人工建模/绑定/渲染流程，普通用户难以直接使用。
- 单视角 3D 重建技术近年发展迅速，但其生成的几何与纹理仍较粗糙，直接渲染无法达到照片级真实感，需结合生成式模型进行后期增强与背景融合。

## 核心贡献（创新点）
- 提出首个将单图对象转化为 3D 模型并在 3D 空间进行可交互变形、最终返回高质量 2D 图像的端到端编辑框架，弥合了生成式 AI 的创意自由度与传统 CG 管线的精确可控性之间的差距。
- 设计粗到细（coarse-to-fine）生成增强管线，通过 One-shot DreamBooth、深度 ControlNet 与特征/注意力注入技术协同，在保留用户编辑后几何结构的同时恢复原始纹理细节。
- 引入背景融合（Background Blend-In）机制，在扩散去噪过程中对背景区域进行掩码混合，避免简单拷贝粘贴带来的光影断裂与反射失真。
- 构建新基准 **SculptingBench**（28 张图像，六类精确编辑任务），并提出 **D-RMSE** 指标定量评估编辑后几何信息的保留程度。
- 实验证明该方法在 pose 编辑、旋转平移、3D 组合、雕刻与序列添加等任务上显著优于 OBJect-3DIT、DragDiffusion、InstructPix2Pix 等现有基线。

## 方法详解
框架分为三个阶段：
1. **去渲染与 3D 重建（De-Rendering）**：使用 SAM 分割目标对象，基于 Score Distillation Sampling (SDS) 从零样本单图重建 NeRF；利用 threestudio 将 NeRF 密度转换为 SDF，提取等值面并生成带纹理三角网格（使用 Instant-NGP，网格尺寸 256）。
2. **3D 模型变形（Deformation）**：用户手动构建骨架并计算蒙皮权重，支持 ARAP（As-Rigid-As-Possible）、线框笼变形（Cage-based）或线性 blend 蒙皮（Linear Blend Skinning）等经典图形学算法进行精确交互编辑；变形仅改变顶点位置，UV 纹理随网格同步形变。
3. **粗到细生成增强（Coarse-to-Fine Enhancement）**：
   - **One-shot DreamBooth**：仅用输入图像对 SDXL-1.0 进行 LoRA 微调（800 步，学习率 1e-5），捕获对象细节纹理并填补粗渲染缺失。
   - **深度控制**：从变形后的 3D 模型直接渲染深度图，通过 ControlNet 注入为空间控制信号；背景区域不使用深度图，以减少不确定性。
   - **特征注入（Feature Injection）**：对粗渲染图进行 DDIM 反演，在每步去噪中同步对粗糙潜在与优化潜在去噪，提取两者的自注意力图（transformer 块）与特征图（residual 块）；用粗渲染的特征/注意力覆盖优化过程的对应层，强制保留编辑后的几何与布局结构。
   - **背景融合（Background Blend-In）**：在去噪步骤中对背景区域施加掩码，将未掩码区域的去噪结果与原始背景混合；使用 SDXL Refiner 在 $t = 0.1T$ 之后进一步减少伪影，背景修复使用 Adobe generative fill。

## 实验与结果
- **数据集与评测**：自建 **SculptingBench**（28 张图像，覆盖 pose 编辑、旋转、平移、组合、雕刻、序列添加六类任务）。使用 **DINO** 分数衡量纹理/语义相似性，提出 **D-RMSE** 指标：$\mathrm { D - R M S E } = \sqrt { \mathbb { E } \left[ ( \mathrm { d e p t h } _ { \mathrm { c o a r s e } } - \mathrm { d e p t h } _ { \mathrm { enhanced } } ) ^ { 2 } \right] }$（基于 MiDaS 深度图计算），用于量化几何保真度。
- **基线对比**：对比
