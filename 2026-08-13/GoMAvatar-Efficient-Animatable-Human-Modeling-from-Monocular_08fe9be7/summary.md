---
title: "GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Wen_GoMAvatar_Efficient_Animatable_Human_Modeling_from_Monocular_Video_Using_Gaussians-on-Mesh_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:44:33"
field: "神经辐射场与可动画人体建模"
keywords: ["3D Gaussian Splatting", "Human Avatar", "Monocular Reconstruction", "Real-time Rendering", "Animatable Human Modeling"]
innovations: ["Gaussians-on-Mesh混合表示：将3D高斯附着于可变形网格三角面，结合高斯溅射渲染质量与显式几何正则化", "前向运动学架构：基于显式网格实现正向LBS变形，避免NeRF类方法的隐式反向映射问题", "伪漫反射-伪着色分解：通过高斯溅射与法线图的神经着色模块分离外观与光照效应"]
benchmarks: ["ZJU-MoCap", "PeopleSnapshot"]
---

# 论文速读：GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular

## 一句话总结
本文提出GoMAvatar，一种基于Gaussians-on-Mesh（GoM）表示的新型单目视频人体建模方法，能够实时渲染高质量、可动画化的人体数字分身，兼具显式几何建模能力与3D高斯溅射的高效渲染优势。

## 研究问题与动机
1. **现有方法无法同时满足高质量渲染、实时性能、内存效率和图形引擎兼容性**：Neural Field类方法（如HumanNeRF、MonoHuman）渲染质量高但缺乏显式几何、难以与游戏引擎集成；Mesh类方法可动画化但难以建模拓扑变化和高质量外观。
2. **3D高斯溅射在可动画人体建模中的应用存在知识空白**：虽然3D Gaussian Splatting在静态场景渲染上成功，但自由形变的人体需要显式几何正则化，而原始高斯溅射缺乏表面建模。
3. **显式正向运动学优于隐式反向映射**：NeRF类方法需从观测空间到规范空间的隐式反向映射（ill-posed），而基于网格的前向运动学（LBS）更直接明确。

## 核心贡献（创新点）
1. **提出Gaussians-on-Mesh (GoM)表示**：将3D高斯附着于网格三角面上，结合高斯溅射的渲染质量与可变形网格的几何建模能力，本质区别在于高斯参数在局部坐标系中学习并随网格变形自适应调整。
2. **设计伪漫反射-伪着色分解的神经渲染器**：将RGB颜色分解为高斯溅射渲染的伪漫反射图与基于法线图预测的伪着色图，有效捕获视图相关光照效应。
3. **前向动画化架构**：基于显式网格实现正向运动学（LBS+非刚性变形），避免了NeRF类方法中从观测空间到规范空间的模糊反向映射问题。
4. **端到端可微分训练与高效的细分上采样**：通过GoM subdivision动态增加网格面数以提升细节，同时保持实时渲染速度。

## 方法详解
**GoM表示（Sec. 3.1）**：
- 规范空间中GoM由顶点集合$\{v_{\theta,i}^c\}$和三角面集合$\{f_{\theta,j}\}$组成
- 每个顶点包含坐标$p_{\theta,i}^c$和线性blend蒙皮权重$w_i$（J个关节）
- 每个面对应一个3D高斯，其局部参数$(r_{\theta,j}, s_{\theta,j}, c_{\theta,j})$定义旋转、缩放和颜色
- 高斯均值$\mu_j$为三角面质心，协方差$\Sigma_j$通过局部到世界的变换矩阵$A_j$适应三角形形状变化

**渲染流程（Sec. 3.2）**：
- 最终图像$I = I_{GS} \cdot S$，分解为伪漫反射图$I_{GS}$和伪着色图$S$
- $I_{GS}$通过3D高斯溅射渲染，高斯参数在三角面局部坐标系中学习
- $S$通过$1\times1$卷积网络从mesh光栅化的法线图$N_{mesh}$预测，输入包含位置编码$\gamma(N_{mesh})$
- 使用SoftRasterizer获取主体mask

**动画化（Sec. 3.3）**：
- 前向运动学：规范空间顶点经非刚性变形（NRDeformer MLP）后，再通过LBS变换到观测空间
- LBS公式：$p_i^o = \frac{\sum_{j=1}^{J} w_i^j(R_j^p p_i^{nr} + t_j^p)}{\sum_{k=1}^{J} w_i^k}$
- 非刚性变形：$p_i^{nr} = p_{\theta,i}^c + \text{NRDeformer}_{\theta}(\gamma(p_{\theta,i}^c), P)$

**姿势精炼（Sec. 3.4）**：
- 仅在新视角合成和训练阶段使用，学习对估计姿势的纠正$\xi_j \in SO(3)$
- 动画时无需此模块

**训练（Sec. 3.5）**：
- 总损失：$L = L_I + \alpha_{lpips}L_{lpips} + \alpha_M L_M + \alpha_{reg}L_{reg}$
- 正则化项：$L_{reg} = L_{mask} + \alpha_{lap}L_{lap} + \alpha_{normal}L_{normal} + \alpha_{color}L_{color}$
- 初始化：基于SMPL网格，高斯初始为沿法线方向薄的椭球
- GoM subdivision：在每个边上插入新顶点，将每个面细分为4个小面

## 实验与结果
**数据集**：
- ZJU-MoCap（6个主体，按MonoHuman分割）
- PeopleSnapshot（4个主体）
- YouTube舞蹈视频（定性验证）

**基线方法**：Neural Body、HumanNeRF、NeuMan、MonoHuman、Anim-NeRF、InstantAvatar

**主要结果（ZJU-MoCap，Tab.1）**：
- 新视角合成：PSNR 30.37 dB / SSIM 0.9689 / LPIPS* 32.53（最优或次优）
- 新姿势合成：PSNR 30.34 dB / SSIM 0.9688 / LPIPS* 32.39
- 推理时间：**23.2 ms/frame（43 FPS）**，比MonoHuman快257倍、比HumanNeRF快76倍
- 内存：**3.63 MB/subject**，仅次于NeuMan（2.27 MB）但远低于其他方法

**PeopleSnapshot结果（Tab.3）**：
- PSNR 30.68 / SSIM 0.9767 / LPIPS 0.0213，显著优于InstantAvatar（28.61/0.9698/0.0242）
- 推理时间25.82 ms，比Anim-NeRF（217 ms）快8.4倍

**几何质量（Tab.2）**：
- 法向一致性NC：0.6201（所有方法最优）
- Chamfer Distance：2.8364，略高于MonoHuman（2.6303）但优于Neural Body

**消融实验（Tab.4, Tab.5）**：
- GoM表示优于纯高斯或纯网格
- 局部可变高斯优于世界坐标系高斯和固定局部高斯
- 着色模块提升PSNR约0.23 dB
- Subdivision显著改善几何质量（CD从3.07降至2.84）

## 相关工作脉络
1. **NeRF类人体建模**（HumanNeRF [70]、MonoHuman [81]、NeuMan [27]）：使用隐式神经场表示人体，需反向映射到规范空间；本文用显式网格+前向运动学解决这一问题。
2. **Mesh类方法**（如[58] Real-time Volumetric Rendering）：先训练NeRF再烘焙到网格，二次烘焙损害质量；GoM端到端训练避免此问题。
3. **3D Gaussian Splatting**（Kerbl et al. [30]、4D Gaussian [72]、Dynamic 3D Gaussians [45]）：用于静态/动态场景渲染，但缺乏显式表面；GoM将高斯约束在网格上实现几何正则化。
4. **模板化人体模型**（SMPL [43]、SCAPE [2]）：提供参数化人体形状和蒙皮权重，本文以此初始化GoM。
5. **实时NeRF加速**（PlenOctrees [79]、BakedSDF [78]、MobileNeRF [13]）：通过烘焙或缓存隐式表示加速；本文用高斯溅射+网格光栅化实现原生实时渲染。
6. **人体动画化方法**（ARCH [19,76]、S3 [77]、ARAH [69]）：支持重动画但渲染质量不足；GoM结合高斯溅射提升渲染质量。

## 局限性与未来方向
1. **Chamfer Distance略逊于MonoHuman**：作者归因于3D高斯在法线方向有厚度，导致渲染mask比实际网格稍大，网格偏小。
2. **姿态估计依赖**：YouTube视频实验中使用PARE估计姿态、MediaPipe生成mask，精度有限；但方法对不完美输入仍有一定鲁棒性。
3. **仅支持场景特定建模**：与InstantAvatar等场景无关方法相比，本文方法针对单个人体训练。
4. **Mesh分辨率限制**：尽管使用subdivision，基础SMPL拓扑结构仍可能限制极端姿态下的表现。

## 研究启发与可借鉴点
1. **混合表示的设计思路**：将高斯溅射与显式几何结合的思路可迁移到其他领域（如衣物、动物、手等），用显式结构正则化高斯自由度。
2. **颜色分解策略**：伪漫反射+伪着色的分解方式简洁高效，可用于其他需要视图相关外观建模的场景（如透明材质、次表面散射近似）。
3. **前向运动学vs反向映射**：显式正向变形避免隐式方法的ill-posed问题，这一原则可应用于其他可形变对象建模。
4. **细分上采样策略**：GoM subdivision在训练时增加几何细节、推理时保持紧凑表示，是"训练-推理解耦"的典型应用。
5. **与图形引擎兼容**：OpenGL兼容的设计使得成果可直接用于游戏/AR应用，为科研到工业落地提供范式。

## 关键术语表
**Gaussians-on-Mesh (GoM)**：将3D高斯溅射附着于可变形网格三角面上的混合表示，高斯参数在局部坐标系中学习并随网格变形自适应。
**Linear Blend Skinning (LBS)**：标准蒙皮算法，通过关节旋转/平移的加权组合将顶点从规范空间变换到观测空间。
**Pseudo Albedo/Shading Decomposition**：将渲染图像分解为高斯溅射生成的伪漫反射图和基于法线图的伪着色图，以分离几何外观与光照效应。
**Non-rigid Deformer (NRDeformer)**：预测姿态相关的顶点偏移量的MLP网络，在LBS之前补充非刚性形变。
**Pose Refinement**：在学习过程中校正估计姿态误差的模块，仅在新视角合成时使用，不影响纯动画应用。
**LPIPS\***：LPIPS指标乘以1000后的值，用于更方便地比较不同方法的感知质量差异。
**Steiner Ellipse**：三角形内切椭圆的最大面积版本，本文初始化高斯使其投影恢复Steiner椭圆。
**SoftRasterizer**：可微分光栅化器，用于生成软mask和法线图，支持端到端训练。

## 可复现要素
- **数据集**：ZJU-MoCap（公开）、PeopleSnapshot（公开）、YouTube视频（公开）
- **代码**：论文未明确说明开源状态，但项目页面为https://wenj.github.io/GoMAvatar/
- **关键超参**：损失权重$\alpha_{lpips}, \alpha_M, \alpha_{reg}, \alpha_{lap}, \alpha_{normal}, \alpha_{color}$（论文未列出具体数值）；Subdivision仅在训练时启用
- **硬件**：NVIDIA A100 GPU测试推理速度
- **初始化**：SMPL模板，高斯初始旋转为零、缩放为1（薄椭球沿法线方向）
