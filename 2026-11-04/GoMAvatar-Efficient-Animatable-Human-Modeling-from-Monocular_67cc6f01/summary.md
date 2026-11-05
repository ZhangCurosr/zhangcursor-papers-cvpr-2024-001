---
title: "GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_GoMAvatar_Efficient_Animatable_Human_Modeling_from_Monocular_Video_Using_Gaussians-on-Mesh_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:01:28"
field: "可动画数字人重建与实时渲染"
keywords: ["Gaussians-on-Mesh", "单目人体建模", "实时渲染", "可形变网格", "Gaussian Splatting", "数字人", "Novel View Synthesis", "Articulatable Avatar"]
innovations: ["提出Gaussians-on-Mesh混合表示，将3D Gaussian绑定到可形变三角面片局部坐标系实现高效正向articulation", "设计伪漫反射/伪阴影分解渲染管线，以I=I_GS·S解耦视图相关着色效应", "端到端可微GoM表示兼容OpenGL图形管线，实现43 FPS实时渲染与3.63 MB极致压缩"]
benchmarks: ["ZJU-MoCap", "PeopleSnapshot", "YouTube in-the-wild videos"]
---

# 论文速读：GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular

## 一句话总结
GoMAvatar 提出了一种 **Gaussians-on-Mesh（GoM）混合表示**，将 3D Gaussian Splatting 的渲染质量与速度、与可形变网格的几何显式性相结合，仅用**单目视频**即可重建出高保真、可重articulation 的数字化人，在 NVIDIA A100 上实现 **43 FPS 实时渲染**，每主体仅占用 **3.63 MB** 参数。

## 研究问题与动机
1. **高质量动画化数字人的低门槛采集需求**：传统 MoCap 依赖昂贵穿戴设备和多视角相机阵列，而消费级应用亟需仅用单目 RGB 视频即可生成动画化数字人的方法。
2. **现有 3D 表示各有缺陷**：Neural field 类（如 NeRF）渲染质量高但缺乏显式几何、难以 forward 正向 articulation；Mesh 类（如 SMPL-based）可快速渲染且兼容图形管线，但难以建模拓扑变化和高质量外观；Point cloud 类（如 3D Gaussian splatting）渲染快，但缺乏显式表面建模，不利于 avatar 建模。
3. **Gaussian splatting 在可 articulation 人体上的知识空白**：虽然 4D Gaussian splatting 已用于动态场景，但将其与可变形网格结合、用于规则化新姿态下 Gaussians 形变的机制尚未被充分探索。
4. **视图依赖着色与几何质量的矛盾**：纯 Gaussian splatting 渲染难以捕捉视图相关的 shading 效应；纯网格光栅化又丢失细节，需要一种融合两者优势的分解渲染策略。

## 核心贡献（创新点）
1. **提出 Gaussians-on-Mesh（GoM）混合表示**：在每个三角面片上绑定一个 3D Gaussian，将 Gaussian splatting 的渲染速度与网格的显式几何/正向 articulation 能力融为一体；与纯 Gaussian splatting 的本质区别在于 Gaussian 参数定义在**局部三角坐标系**而非世界坐标，从而实现随姿态自适应形变。
2. **设计基于网格法线的伪光照分解渲染模块**：将 RGB 颜色分解为伪漫反射图（$I_{\mathrm{GS}}$，由 Gaussian splatting 渲染）与伪阴影图（$\mathcal{S}$，由网格法线图经 1×1 卷积网络预测），以 $I = I_{\mathrm{GS}} \cdot \mathcal{S}$ 相乘合成最终图像；与纯 end-to-end RGB 预测的区别在于显式解耦了视图相关着色效应。
3. **实现正向 kinematics 驱动的实时 articulation 管线**：通过非刚性形变 MLP + 线性混合蒙皮（LBS）完成从 canonical 空间到 observation 空间的正向映射，避免了 NeRF 类方法所需的病态反向映射（observation → canonical）；与 InstantAvatar 等间接方法的区别在于无需逆向体素 LBS，可直接处理自相交姿态。
4. **端到端可微渲染，兼容 OpenGL 图形管线**：整个 GoM 表示可直接衔接传统光栅化引擎，支持 novel view / novel pose 合成，且文件体积极小（3.63 MB/人），远超同类方法的工程可用性。

## 方法详解

### 3.1 Gaussians-on-Mesh（GoM）表示
在 canonical 空间中，GoM 由顶点集合 $\{v_{\theta,i}^c\}_{i=1}^V$ 和三角面片集合 $\{f_{\theta,j}\}_{j=1}^F$ 构成：
- **顶点** $v_{\theta,i}^c = (p_{\theta,i}^c, w_i)$，其中 $p_{\theta,i}^c \in \mathbb{R}^3$ 为 canonical 空间顶点坐标，$w_i \in \mathbb{R}^J$ 为 LBS 权重（$J$ 个骨骼关节）。
- **面片** $f_{\theta,j} = (r_{\theta,j}, s_{\theta,j}, c_{\theta,j}, \{\Delta_{j,k}\}_{k=1}^3)$，其中 $r_{\theta,j} \in so(3)$ 为局部高斯旋转，$s_{\theta,j} \in \mathbb{R}^3$ 为缩放，$c_{\theta,j} \in \mathbb{R}^3$ 为颜色，$\Delta_{j,k}$ 为面片所属三个顶点索引。
- **初始化**：以 SMPL 网格初始化顶点和面片；$r_{\theta,j}$ 初始化为零向量，$s_{\theta,j}$ 初始化为全 1，使初始高斯沿面片法线方向极薄，其在三角形上的投影恰好恢复 Steiner 椭圆。

### 3.2 渲染管线
将最终 RGB 图像 $I$ 分解为：
$$I = I_{\mathrm{GS}} \cdot \mathcal{S}$$
- **伪漫反射图 $I_{\mathrm{GS}}$**：将每个面片的局部 Gaussian 参数（$r_{\theta,j}, s_{\theta,j}$）通过面片几何变换矩阵 $A_j$ 转换到世界坐标系：
  - 均值：$\mu_j = \frac{1}{3}\sum_{k=1}^3 p_{\Delta_{j,k}}^o$（三角形质心）
  - 协方差：$\Sigma_j = A_j (R_j S_j S_j^T R_j^T) A_j^T$
  - 高斯在世界坐标下经标准 Gaussian splatting 光栅化得到 $I_{\mathrm{GS}}$ 与 alpha mask $M$。
- **伪阴影图 $\mathcal{S}$**：通过标准网格光栅化获取法线图 $N_{\mathrm{mesh}}$，经位置编码 $\gamma(\cdot)$ 后输入 $1\times 1$ 卷积网络（Shading$_\theta$）逐像素预测缩放因子：
  $$\mathcal{S} = \mathrm{Shading}_\theta(\gamma(N_{\mathrm{mesh}}))$$

### 3.3 Articulation（正向正向形变）
给定 canonical GoM$^c$ 和新姿态 $P$，输出 observation 空间 GoM$^o$：
1. **非刚性形变（NRDeformer）**：MLP 网络预测每个顶点的偏移量：
   $$p_i^{\mathrm{nr}} = p_{\theta,i}^c + \mathrm{NRDeformer}_\theta(\gamma(p_{\theta,i}^c), P)$$
2. **线性混合蒙皮（LBS）**：将 $p_i^{\mathrm{nr}}$ 变换到 observation 空间：
   $$p_i^o = \mathrm{LBS}(p_i^{\mathrm{nr}}, w_i, P) = \frac{\sum_{j=1}^J w_i^j (R_j^p p_i^{\mathrm{nr}} + t_j^p)}{\sum_{k=1}^J w_i^k}$$
   其中 $P = \{(R_j^p, t_j^p)\}_{j=1}^J$ 为姿态参数。
3. 所有其他属性（$r_{\theta,j}, s_{\theta,j}, c_{\theta,j}$）在 canonical 空间共享，无需额外变换。

### 3.4 姿态精修（Pose Refinement）
仅在新视角合成和训练阶段使用（动画时不需要）：预测关节旋转的 SO(3) 空间修正量 $\xi_j$，更新姿态：
$$\{\xi_j\} = \mathrm{PoseRefiner}_\theta(\{\hat{R}_j^p\})$$
$$P = \{(\hat{R}_j^p \cdot \xi_j, t_j^p)\}$$

### 3.5 训练损失
$$\mathcal{L} = \mathcal{L}_I + \alpha_{\mathrm{lips}}\mathcal{L}_{\mathrm{lips}} + \alpha_M\mathcal{L}_M + \alpha_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}$$
其中正则项：
$$\mathcal{L}_{\mathrm{reg}} = \mathcal{L}_{\mathrm{mask}} + \alpha_{\mathrm{lap}}\mathcal{L}_{\mathrm{lap}} + \alpha_{\mathrm{normal}}\mathcal{L}_{\mathrm{normal}} + \alpha_{\mathrm{color}}\mathcal{L}_{\mathrm{color}}$$
- $\mathcal{L}_{\mathrm{mask}}$：网格轮廓与 GT mask 的 L1 距离
- $\mathcal{L}_{\mathrm{lap}}$：Laplacian 平滑，惩罚顶点偏差
- $\mathcal{L}_{\mathrm{normal}}$：相邻面片法向余弦相似度最大化
- $\mathcal{L}_{\mathrm{color}}$：相邻面片颜色平滑

**GoM Subdivision**：训练时在每条边中点插入新顶点，将每个面片细分为四个小面片，复制面片属性以增强细节（推理时不细分以保速）。

## 实验与结果

### 数据集
- **ZJU-MoCap**（6 个受试者，按 MonoHuman 划分）：用于 novel view / novel pose 合成评估
- **PeopleSnapshot**（4 个受试者，按 InstantAvatar 划分）：评估旋转姿态下的渲染质量
- **YouTube 视频**：定性验证 in-the-wild 视频（用 MediaPipe 提取 mask，PARE 提取 SMPL 姿态）

### 关键数值结果（ZJU-MoCap）

| 方法 | Novel View PSNR | Novel Pose PSNR | 推理时间 (ms/frame) | 内存 (MB) |
|------|----------------|-----------------|---------------------|-----------|
| Neural Body | 28.72 | 28.54 | 212.3 | 16.76 |
| HumanNeRF | 29.61 | 29.74 | 1776.7 | 245.73 |
| MonoHuman | 30.26 | 30.05 | 5970.0 | 280.67 |
| **GoMAvatar (Ours)** | **30.37** | **30.34** | **23.2 (43 FPS)** | **3.63** |

- GoMAvatar 的 PSNR 与最强基线 MonoHuman 持平（novel view 30.37 vs 30.26，novel pose 30.34 vs 30.05），但推理速度**快 257 倍**，内存仅为 **1/77**。
- **几何质量**：Normal Consistency（NC）0.6201（最佳），Chamfer Distance（CD）2.8364（次优，略逊 MonoHuman 2.6303，原因：Gaussian 法线方向有厚度）。

### PeopleSnapshot 结果

| 方法 | PSNR | SSIM | LPIPS | 推理时间 (ms) |
|------|------|------|-------|---------------|
| InstantAvatar | 28.61 | 0.9698 | 0.0242 | 71.26 |
| **GoMAvatar (Ours)** | **30.68** | **0.9767** | **0.0213** | **25.82** |

- 相比 InstantAvatar，PSNR 提升 **+2.07 dB**，SSIM 提升 **+0.0069**，速度提升 **2.76 倍**。

### YouTube 定性结果
- 可正确处理**自相交姿态**（如坐姿、腿交叉），而 HumanNeRF/MonoHuman 因体素反向 LBS 产生手部和腋下的 ghost effect 与 floaters。
- 在 inaccurate pose estimation 条件下仍保持稳定的 novel view synthesis 质量。

## 相关工作脉络

1. **Neural Body（Peng et al., CVPR 2021）**：基于 implicit neural representation + LBS 的单目人体渲染，需数百万参数，渲染慢；GoMAvatar 用显式 GoM 替代，参数减少 47 倍，速度提升 257 倍。
2. **HumanNeRF（Weng et al., CVPR 2022）**：NeRF-based 单目人体，novel view/pose 效果强但推理需 ~1.8 秒/帧，且存在体素化 LBS 导致的 ghost effect；GoMAvatar 用正向 LBS + Gaussian splatting 消除这两类 artifact。
3. **MonoHuman（Yu et al., CVPR 2023）**：当前单目人体 SOTA，渲染质量最优但推理最慢（~6 秒/帧），参数最多（280 MB）；GoMAvatar 以同等画质换取 257 倍加速和 77 倍压缩。
4. **InstantAvatar（Jiang et al., CVPR 2023）**：60 秒快速重建，但推理仍为 71 ms/frame；GoMAvatar 在 PeopleSnapshot 上以更快推理和更高 PSNR 全面超越。
5. **3D Gaussian Splatting（Kerbl et al., SIGGRAPH 2023）**：静态场景实时渲染里程碑；GoMAvatar 将其扩展到可形变人体，核心创新是将 Gaussian 参数绑定到三角面片局部坐标而非世界坐标。
6. **4D Gaussian Splatting（Wu et al., arXiv 2023）**：将时间维度引入 Gaussian；GoMAvatar 通过网格拓扑约束 Gaussian 形变，避免时间正则化的额外学习负担，更适合 avatar 场景。

## 局限性与未来方向

1. **Chamfer Distance 略逊于 NeRF 类方法**：因 Gaussian 在法线方向有厚度，渲染出的 mask 略大于真实网格轮廓，导致网格偏小；未来可通过调整 Gaussian 初始化或引入 surface constraint 改善。
2. **依赖 monocular 视频输入的 pose/mask 精度**：YouTube 实验中使用 PARE 和 MediaPipe 估计姿态和 mask，精度不足时仍会产生 artifact；结合更优的 monocular pose estimation 可进一步提升 in-the-wild 效果。
3. **未支持复杂服装/头发动态**：当前 GoM 表示以 SMPL 为基础，对宽松衣物、长发等非刚体形变建模有限。
4. **Subdivision 提升几何但增加推理耗时**：训练时 subdivision 将推理时间从 17.5ms 增至 23.2ms；如何在更高分辨率下保持实时性有待优化。
5. **仅 scene-specific 建模**：本文遵循 Prior 做法针对单个受试者训练，未探索跨受试者的 generalizable avatar 表示。

## 研究启发与可借鉴点

1. **Gaussian-on-Mesh 的混合思路可迁移至其他可形变对象建模**：如动物、面部表情等，将 Gaussian 参数绑定到局部三角坐标系的思想具有通用性，可扩展到 non-human articulated objects。
2. **伪漫反射/伪阴影分解（$I = I_{\mathrm{GS}} \cdot \mathcal{S}$）**是一种简洁有效的视图依赖着色建模方案，无需显式光源估计即可捕获 shading 变化，可推广至 general scene rendering。
3. **正向 articulation + 非刚性残差（LBS + NRDeformer）**替代反向体素映射的思路，可消除 implicit field 方法的 ill-posed 问题，值得在其他 dynamic reconstruction 任务中复现。
4. **GoM Subdivision 机制**（面片属性复制 + 网格加密）是一种轻量级 detail enhancement 策略，推理时不需细分即可在训练阶段获得更高频几何细节。
5. **43 FPS / 3.63 MB 的效率目标**为后续 real-time digital human 研究树立了新的效率基准，可与 NeRF 类方法形成鲜明的 quality-speed-memory 三角权衡参考点。

## 关键术语表

**Gaussians-on-Mesh（GoM）**：在每个三角面片上绑定一个 3D Gaussian 的混合 3D 表示，Gaussian 参数定义在面片局部坐标系，随网格形变自适应变化。
**Linear Blend Skinning（LBS）**：标准的骨骼驱动蒙皮变形方法，通过加权融合多个骨骼的旋转和平移变换顶点坐标。
**Pseudo Albedo Map（$I_{\mathrm{GS}}$）**：由 Gaussian splatting 渲染出的伪漫反射颜色图，不含视图相关阴影。
**Pseudo Shading Map（$\mathcal{S}$）**：从网格法线图经轻量卷积网络预测的伪阴影缩放图，编码视图相关光照效应。
**Non-rigid Deformer（NRDeformer）**：预测 canonical 顶点到 non-rigidly transformed canonical 空间的逐顶点偏移量的 MLP 网络。
**Steiner Ellipse**：三角形内切椭圆的最大面积内切椭圆，GoMAvatar 初始化时使高斯在三角形上的投影恰好恢复 Steiner 椭圆。
**Pose Refinement**：训练和 novel view 合成阶段用于校正 monocular pose estimation 误差的 SO(3) 空间修正模块。
**GoM Subdivision**：训练时通过在边中点插入新顶点将三角面片四等分，复制面片属性以提升几何分辨率。

## 可复现要素

- **数据集**：ZJU-MoCap（公开）、PeopleSnapshot（公开）、YouTube 视频（需自行获取）
- **代码/权重**：项目页面 https://wenj.github.io/GoMAvatar/（论文未明确声明开源状态）
- **关键超参**：$\alpha_{\mathrm{lips}}, \alpha_M, \alpha_{\mathrm{reg}}, \alpha_{\mathrm{lap}}, \alpha_{\mathrm{normal}}, \alpha_{\mathrm{color}}$（论文中仅以符号表示，具体数值见附录/supplementary，正文未给出）
- **训练初始化为 SMPL 网格**，$r_{\theta,j}$ 初始化为零，$s_{\theta,j}$ 初始化为全 1
- **推理设备**：NVIDIA A100 GPU（论文基准测试设备）
