---
title: "ASH-Animatable-Gaussian-Splats-for-Efficient-and-Photoreal-H"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_ASH_Animatable_Gaussian_Splats_for_Efficient_and_Photoreal_Human_Rendering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:20:27"
field: "可动画人体神经渲染"
keywords: ["3D Gaussian Splatting", "Animatable Human Rendering", "Real-time Neural Rendering", "Dynamic Human Avatar", "UV Texture Parameterization", "Pose-conditional Rendering"]
innovations: ["将3D Gaussian参数化在可变形模板网格的UV纹理空间，实现实时高保真人体渲染", "运动感知纹理驱动的双2D卷积解码器，将pose→Gaussian映射转化为2D-to-2D图像翻译", "分阶段warmup训练策略，固定Gaussian位置生成pseudo ground truth解决动态序列收敛难题"]
benchmarks: ["DynaCap", "Novel View Synthesis", "Novel Pose Synthesis"]
---

# 论文速读：ASH-Animatable-Gaussian-Splats-for-Efficient-and-Photoreal-H

## 一句话总结
ASH将3D Gaussian Splatting引入可动画人体渲染，通过将Gaussian参数参数化在可变形模板网格的UV纹理空间，利用2D卷积网络实现实时（~30fps）、高保真的人体渲染，在DynaCap数据集上显著超越现有实时方法DDC，并与非实时方法HDHumans相当甚至更优。

## 研究问题与动机
- **核心问题**：如何在保持实时渲染（≥30fps）的前提下，生成高质量、可任意骨骼姿态驱动的可动画人体数字孪生（digital human avatar）。
- **现有方法不足**：
  - 显式网格方法（如DDC）：渲染速度快但细节与照片级真实感受限，模板网格分辨率成为瓶颈。
  - 混合NeRF方法（如HDHumans、Neural Actor）：质量更高但体积渲染需要大量MLP查询，单帧渲染耗时数秒，无法实时。
  - 原始3D Gaussian Splatting：专为静态场景设计，直接扩展到动态人体面临3D空间参数量爆炸（需>20,000 Gaussians）与计算复杂度高两个挑战。
  - 多数动态NeRF工作仅支持回放同一序列，不支持任意新姿态的控制。

## 核心贡献（创新点）
1. **提出ASH框架**：首次将3D Gaussian Splatting适配到可动画人体场景，在保持实时渲染的同时实现照片级真实感。
   - 与原始Gaussian Splatting的本质区别：原始方法处理静态场景，ASH通过模板网格绑定与2D纹理空间参数化解决动态人体 posed-dependence 建模。
2. **Gaussian在可变形网格上的2D纹理空间参数化**：将N个Gaussian附着到人形模板网格的UV纹理空间，每个texel对应一个Gaussian，使参数量固定且可用2D卷积高效学习。
   - 与naive 3D映射的本质区别：避免在3D空间直接学习pose→Gaussian参数的高维映射，转化为2D-to-2D图像翻译问题，规避3D架构难以扩展的缺陷。
3. **运动感知纹理驱动的双解码器架构**：设计运动感知法向量纹理（T_n,f）和位置纹理（T_p,f），分别通过几何解码器E_geo和外观解码器E_app预测Gaussian的shape参数与SH系数。
   - 与HDHumans等的本质区别：HDHumans联合优化implicit field和template mesh，ASH完全在2D纹理空间学习，无需逐点MLP查询，实现实时。
4. **分阶段训练策略（Warmup + Final）**：Warmup阶段固定Gaussian位置、优化其余参数作为pseudo ground truth，再端到端训练主损失，有效解决长序列动态训练的收敛难题。
   - 与标准3DGS训练的本质区别：标准3DGS依赖grad-cut初始化+位置优化，ASH需保持跨帧一致性，故固定位置、移除splitting/merging操作。

## 方法详解
### 整体流程
输入：骨骼运动序列$\bar{\theta}_f \in \mathbb{R}^{k \times D}$（滑动窗口，含根平移归一化）+ 相机参数$C_c$；输出：实时渲染图像$I'_{f,c}$。

### 1. Animatable Template（可动画模板网格）
- 使用Habermann et al. [13]的character model，先在unposed canonical space中通过learned embedded deformations（[58]）和per-vertex displacements得到$\bar{V}_f$。
- 再通过Dual Quaternion skinning [23]从canonical pose到posed space得到$V_f = M(\theta_f)$。

### 2. Animatable Gaussian Textures（可动画Gaussian纹理）
- 固定数量$N$个Gaussian作为texels存储在模板网格UV空间：$\{\mathcal{G}_i\}_{N} = (\bar{\mu}_{uv,i}, \bar{d}_{uv,i}, q_{uv,i}, s_{uv,i}, \alpha_{uv,i}, \eta_{uv,i}) \in \mathbb{R}^{N \times 62}$。
- **位置参数化**（公式4）：canonical位置通过双线性插值得到：$\bar{\mu}_{uv,i} = w_{a,i}\bar{V}_{f,j} + w_{b,i}\bar{V}_{f,k} + w_{c,i}\bar{V}_{f,l}$，其中$w$为barycentric权重。
- **Skinned到posed space**（公式5）：$\mu_{uv,i} = T_{uv,i}(\bar{\mu}_{uv,i} + \bar{d}_{uv,i})$，$T_{uv,i}$为DQ skinning变换矩阵，$\bar{d}_{uv,i}$为learned per-texel offset（捕获non-rigid motion-dependent deformation）。

### 3. Motion-Aware 2D Convolutional Decoders（运动感知卷积解码器）
- **运动感知纹理**：从posed/deformed mesh $V_f$通过inverse texture mapping计算normal map $T_{n,f}$和position map $T_{p,f}$。
- **Geometry Decoder** $E_{geo}$（公式6）：$E_{geo}(T_{n,f}, T_{p,f}) = (\bar{d}_{uv,i}, s_{uv,i}, q_{uv,i}, \alpha_{uv,i})$，预测形状相关参数。
- **Appearance Decoder** $E_{app}$（公式7）：$E_{app}(T_{n,f}, T_{p,f}, \Phi_f) = \eta_{uv,i}$，预测SH系数；$\Phi_f$为通过浅层MLP编码的全局外观特征（处理captured space中的空间变化光照）。

### 4. 渲染流程
1. 从canonical space将Gaussian splat通过DQ skinning warping到posed space。
2. 按原始3D Gaussian Splatting方法投影到image space：计算投影后协方差$\Sigma_{i,c} = J_c C_c R_i S_i S_i^T R_i^T C_c^T J_c^T$（公式2）。
3. 按深度排序后alpha blending合成像素颜色：$c_p = \sum_{j \in N_p} H(\eta_i, d_p) \alpha'_j \prod_{k=1}^{j-1}(1-\alpha'_k)$（公式3）。

### 5. 训练策略
- **Warmup Stage**：采样t帧，独立优化3D Gaussian参数$\{\mathcal{G}'_i\}$（固定位置$\mu''_{uv,i}$，移除splitting/merging），作为pseudo ground truth。预训练损失：$\mathcal{L}_{pre} = \mathcal{L}_2(\{\mathcal{G}'_i\}, \{\mathcal{G}''_i\})$（公式8）。
- **Final Training**：全序列端到端训练，主损失为pixel-wise L1 + SSIM：$\mathcal{L}_{main} = \lambda_{pix}\mathcal{L}_1(I_{f,c}, I'_{f,c}) + \lambda_{str}\mathcal{L}_{ssim}(I_{f,c}, I'_{f,c})$，其中$\lambda_{pix}=0.1, \lambda_{str}=0.9$（公式9）。

## 实验与结果
### 数据集
- **DynaCap** [13]：公开多视角视频数据集，选取tight和loose两种服装的两位subject。保留4个view用于novel-view评估，testing split motion用于novel-pose评估。
- **自建数据集**：120相机系统，25fps，训练序列27000帧，测试序列7000帧，包含跳舞、慢跑、跳跃等日常动作。

### 评估基线
- **实时方法**：DDC [13]
- **非实时方法**：TAVA [29]、Neural Actor (NA) [33]、HDHumans [15]
- **评估指标**：PSNR、LPIPS，1K分辨率，每10帧平均。

### 主要结果
**Novel View Synthesis（表1）**：
| 方法 | RT | Tight PSNR | Tight LPIPS | Loose PSNR | Loose LPIPS |
|------|-----|------------|-------------|------------|-------------|
| TAVA | × | 24.61 | 62.26 | 27.31 | 37.55 |
| NA | × | 30.33 | 23.71 | 25.30 | 50.01 |
| HDHumans | × | 30.98 | 15.09 | 29.24 | 15.79 |
| DDC | √ | 31.21 | 22.56 | 28.10 | 31.68 |
| **Ours** | **√** | **35.84** | **11.92** | **35.47** | **8.30** |

**Novel Pose Synthesis（表2）**：
| 方法 | RT | Tight PSNR | Tight LPIPS | Loose PSNR | Loose LPIPS |
|------|-----|------------|-------------|------------|-------------|
| TAVA | × | 28.30 | 37.47 | 26.31 | 50.11 |
| NA | × | 28.78 | 25.78 | 25.03 | 44.20 |
| HDHumans | × | 28.17 | 20.69 | 26.71 | 22.75 |
| DDC | √ | 27.77 | 30.16 | 26.43 | 32.22 |
| **Ours** | **√** | **28.90** | **22.83** | **27.12** | **20.22** |

- **最强结果**：Novel View下Loose Outfit PSNR = 35.47，较HDHumans（29.24）提升+6.23 dB，较DDC（28.10）提升+7.37 dB；LPIPS 8.30 vs HDHumans 15.79，下降47%。
- **实时性**：推理约29.64 fps，满足实时要求。

### Ablation（表3，Loose Outfit）
- w/o motion condition（w/o mot.）：PSNR仅27.19，LPIPS 33.16，证明motion conditioning至关重要。
- w/o displacement（w/o disp.）：PSNR 33.21，LPIPS 17.34，证明learned per-texel offset对non-rigid deformation有效。
- Resolution：256为最佳trade-off；128导致明显模糊，512与256效果相近但计算开销大增，破坏实时性。

## 相关工作脉络
1. **DDC (Deep Dynamic Characters, [13])**：同为实时方法，采用mesh-based显式表示（embedded graph + dynamic texture maps），ASH在2D纹理空间学习Gaussian参数，质量远超DDC（LOOSE PSNR +7.37 dB）。
2. **HDHumans ([15])**：hybrid方法，联合优化implicit field和template mesh，单帧渲染需数秒；ASH在同等质量下实现实时，速度与质量兼顾。
3. **Neural Actor (NA, [33])**：将parametric human body mesh的texture map作为local pose feature；不适用于宽松服装，ASH通过Gaussian displacements有效处理loose clothing。
4. **TAVA ([29])**：在canonical space用implicit field表示shape/appearance/skinning weights，需iterative root finding canonicalization，速度慢且对DynaCap复杂运动泛化差（PSNR 24.61/27.31）。
5. **3D Gaussian Splatting ([24])**：原始静态场景表示；ASH的核心创新在于将其适配到dynamic animatable human，通过UV绑定和2D卷积实现pose-conditioned学习。
6. **Dynamic NeRF工作（D-NeRF [47], Nerfies [42], HyperNeRF [43]等）**：多数仅支持sequence replay，不支持arbitrary pose control；ASH明确支持user-controlled skeletal pose输入。

## 局限性与未来方向
- **模板网格不更新**：当前ASH不动态更新底层deformable template mesh的几何结构，限制了对极端形变或大位移服装的表征能力。
- **固定Gaussian数量**：N固定导致细节分辨率受限于纹理分辨率（256），无法像原始3DGS那样通过adaptive density control自动调整。
- **单角色建模**：当前方法为person-specific，需per-subject训练；未探索generalizable multi-subject representation。
- **未来方向**：探索Gaussian splatting直接改进3D mesh几何；尝试将warmup策略推广到其他动态神经渲染任务；考虑稀疏相机或非标定输入下的泛化能力。

## 研究启发与可借鉴点
1. **"3D问题2D化"的参数化思想**：将3D Gaussian参数绑定到UV纹理空间，将高维3D-to-3D映射降维为2D-to-2D图像翻译，这一思路可迁移到其他需要pose-conditioned 3D表示的任务（如hand avatar、面部avatar）。
2. **Motion-aware texture作为pose encoder**：用normal map + position map编码骨骼运动到纹理空间，比直接使用skeletal pose vector更利于2D卷积捕捉局部空间关系，可借鉴到任何基于mesh的appearance synthesis任务。
3. **Warmup pseudo-ground-truth训练策略**：先独立优化3DGS参数作为teacher signal，再训练decoder的网络，这一两阶段策略对任何需要从sparse/noisy supervision学习complex 3D representation的任务都有参考价值。
4. **固定Gaussian数量+texel参数化**：避免了原始3DGS的grad-cut初始化和密度控制，在动态场景中保持跨帧一致性更稳定，适合需要时序一致性的应用。
5. **与团队方向的结合机会**：可将ASH的2D纹理空间Gaussian参数化与自团队关注的"few-shot novel view synthesis"或"monocular human reconstruction"结合——探索单目/稀疏视角下如何初始化或约束UV空间Gaussian参数。

## 关键术语表
**3D Gaussian Splatting**：一种基于3D各向异性Gaussian椭球的显式场景表示方法，通过fast differentiable rasterization实现实时高保真渲染，替代NeRF的MLP volume rendering。
**Animatable Gaussian Splats**：将Gaussian参数（位置、旋转、缩放、不透明度、SH系数）绑定在可变形模板网格的UV texel上，使其能随骨骼pose驱动而变形。
**Dual Quaternion Skinning (DQ Skinning)**：一种基于对偶四元数的skin weighting方法，相比传统线性blend skinning能更好地保持体积，避免"candy-wrapper"效应，常用于角色动画。
**Motion-Aware Textures**：从posed mesh反投影得到的normal map ($T_{n,f}$) 和position map ($T_{p,f}$)，作为2D卷积网络的pose条件输入，编码骨骼运动导致的表面形变信息。
**Canonical Space vs Posed Space**：Canonical space为未pose deformed的统一参考空间（模板网格初始空间）；Posed space为经DQ skinning变形后的实际姿态空间。
**Embedded Deformations (ED)**：[58]提出的non-rigid变形方法，通过预设basis functions学习embedded deformation field，用于在canonical space中捕获clothed human的非刚性形变。
**Spherical Harmonics (SH)**：用于编码Gaussian的view-dependent颜色（appearance），ASH中使用3阶SH（48个系数）表征view-dependent appearance。
**LPIPS (Learned Perceptual Image Patch Similarity)**：基于深度特征的距离度量，比PSNR更能反映人类感知质量，值越低表示渲染图像与ground truth越接近。

## 可复现要素
- **数据集**：DynaCap [13]（公开）；自建120相机多视角数据集（论文未声明开源，仅展示qualitative结果）。
- **代码**：论文未明确声明开源状态（CVPR 2024，截至知识截止未见到官方release），实践中需参考补充材料或联系作者确认。
- **权重**：论文未声明开源。
- **关键超参**：
  - Gaussian数量N：由UV texel覆盖三角形数量决定，纹理分辨率256。
  - 损失权重：$\lambda_{pix}=0.1, \lambda_{str}=0.9$。
  - Warmup帧数t：论文未给出具体数值（Sec 3.3提及"sample t frames evenly"）。
  - SH阶数：3阶（48维）。
  - 解码器架构：基于U-Net [49]的2D卷积网络。
  - 推理帧率：~29.64 fps。
