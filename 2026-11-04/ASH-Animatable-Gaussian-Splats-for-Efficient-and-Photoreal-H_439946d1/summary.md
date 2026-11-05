---
title: "ASH-Animatable-Gaussian-Splats-for-Efficient-and-Photoreal-H"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_ASH_Animatable_Gaussian_Splats_for_Efficient_and_Photoreal_Human_Rendering_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:19:41"
field: "可动画人体实时神经渲染"
keywords: ["3D Gaussian Splatting", "Animatable Human Rendering", "Real-time Rendering", "Dynamic Humans", "Texture-space Parameterization"]
innovations: ["将3D Gaussian附着于可变形模板网格的UV纹理空间，以2D-to-2D卷积实现姿态驱动的高斯参数预测", "分阶段训练策略（warmup固定位置伪GT + 最终端到端图像损失）保障长序列动态外观收敛"]
benchmarks: ["DynaCap", "Novel View Synthesis", "Novel Pose Synthesis"]
---

# 论文速读：ASH-Animatable-Gaussian-Splats-for-Efficient-and-Photoreal-Human-Rendering

## 一句话总结
ASH 提出将 3D Gaussian Splatting 附着于可变形模板网格的 UV 纹理空间，把骨骼姿态到高斯参数的映射转化为 2D-to-2D 图像翻译任务，从而在 ~30 fps 下实现照片级真实感、可自由控制姿态的人体数字分身实时渲染。

## 研究问题与动机
1. **NeRF 类混合方法渲染慢**：HDHumans、Neural Actor 等方法依赖逐射线 MLP 采样，单帧需数秒，无法满足实时交互需求。
2. **原始 3D Gaussian Splatting 仅适用于静态场景**：不能直接建模骨骼驱动下的动态人体外观与几何形变。
3. **显式网格方法（如 DDC）质量受限**：受模板网格分辨率制约，宽松衣物褶皱等精细运动依赖细节恢复不足。
4. **3D 空间直接学习高斯参数扩展性差**：人体高保真渲染通常需要 >20,000 个 Gaussian splats，在 3D 中逐点学习姿态→参数映射计算开销巨大。

## 核心贡献（创新点）
1. **首次将 Gaussian Splatting 扩展至可动画人体实时渲染**：与现有离线 NeRF 混合方法不同，ASH 在保持照片级质量的同时实现 ~30 fps 实时推理。
2. **Gaussian 附着于可变形模板网格的 2D 纹理参数化**：每个 texel 对应一个 3D Gaussian（共 N×62 维），将高斯数量固定，规避了原始 Gaussian Splatting 动态增生/分裂的 3D 扩展难题。
3. **运动感知 2D 卷积解码器**：把姿态编码为法线纹理 $\mathbf{T}_{\mathrm{n},f}$ 与位置纹理 $\mathbf{T}_{\mathrm{p},f}$，通过轻量 2D Conv 网络（$\mathcal{E}_{\mathrm{geo}}$、$\mathcal{E}_{\mathrm{app}}$）实现姿态→高斯参数的 2D-to-2D 翻译，避免 3D 架构的扩展瓶颈。
4. **分阶段训练策略**：先对固定位置 Gaussian 拟合伪 GT 以初始化解码器权重（warmup），再在完整序列上用图像级 L1+SSIM 损失端到端训练，保障长序列动态外观收敛。

## 方法详解
- **可变形模板网格**：采用 Habermann et al. [13] 的 character model，在 unposed-canonical space 中先做 skeletal-motion-dependent embedded deformation 与 per-vertex displacement，得到 $\bar{\mathbf{V}}_f$；再通过 Dual Quaternion skinning [23] 得到 posed 顶点 $\mathbf{V}_f$。
- **Gaussian 参数化（texel-level）**：每个 texel $i$ 存储 $(\bar{\mu}_{\mathrm{uv},i}, \bar{\mathbf{d}}_{\mathrm{uv},i}, \mathbf{q}_{\mathrm{uv},i}, \mathbf{s}_{\mathrm{uv},i}, \alpha_{\mathrm{uv},i}, \eta_{\mathrm{uv},i}) \in \mathbb{R}^{62}$。基位置 $\bar{\mu}_{\mathrm{uv},i}$ 由 canonical 三角形顶点的重心插值（式 4）获得；实际 posed 位置通过 $\mu_{\mathrm{uv},i} = \mathbf{T}_{\mathrm{uv},i}(\bar{\mu}_{\mathrm{uv},i} + \bar{\mathbf{d}}_{\mathrm{uv},i})$（式 5）映射到 posed 空间，其中 $\bar{\mathbf{d}}_{\mathrm{uv},i}$ 为可学习的 motion-aware offset。
- **运动感知纹理**：从 posed 网格 $\mathbf{V}_f$ 经逆纹理映射得到法线纹理 $\mathbf{T}_{\mathrm{n},f}$ 和位置纹理 $\mathbf{T}_{\mathrm{p},f}$，将 3D 骨骼姿态 $\bar{\pmb{\theta}}_f$ 编码为 2D 图像。
- **几何解码器** $\mathcal{E}_{\mathrm{geo}}(\mathbf{T}_{\mathrm{n},f}, \mathbf{T}_{\mathrm{p},f}) = (\bar{\mathbf{d}}_{\mathrm{uv},i}, \mathbf{s}_{\mathrm{uv},i}, \mathbf{q}_{\mathrm{uv},i}, \alpha_{\mathrm{uv},i})$（式 6）。
- **外观解码器** $\mathcal{E}_{\mathrm{app}}(\mathbf{T}_{\mathrm{n},f}, \mathbf{T}_{\mathrm{p},f}, \Phi_f) = \eta_{\mathrm{uv},i}$（式 7），其中 $\Phi_f$ 为通过浅层 MLP 编码的全局外观特征（捕捉场景内空间变化的光照）。
- **训练损失**：warmup 阶段用 $\mathcal{L}_{\mathrm{pre}} = \mathcal{L}_2(\{\mathcal{G}'_i\}, \{\mathcal{G}''_i\})$（式 8）；最终训练用 $\mathcal{L}_{\mathrm{main}} = 0.1 \cdot \mathcal{L}_1 + 0.9 \cdot \mathcal{L}_{\mathrm{ssim}}$（式 9），对多视图 RGB 图像进行监督。

## 实验与结果
- **数据集**：DynaCap [13]（选取 tight/loose 两类服装 subject，留出 4 个相机视图做 novel-view 评估，测试集序列做 novel-pose 评估）；自采 120 相机同步序列（25 fps，训练 27,000 帧，测试 7,000 帧）。
- **基线**：TAVA [29]、Neural Actor (NA) [33]、HDHumans [15]、DDC [13]（唯一同档位实时方法）。
- **评估指标**：PSNR、LPIPS，1K 分辨率，每 10 帧采样平均。
- **主要结果**：
  - Novel View（Tight）：ASH 35.84 / 11.92，较 DDC（31.21 / 22.56）提升 +4.63 dB PSNR、-10.64 LPIPS；亦优于 HDHumans（30.98 / 15.09）。
  - Novel View（Loose）：ASH 35.47 / 8.30，大幅领先 DDC（28.10 / 31.68）与 HDHumans（29.24 / 15.79）。
  - Novel Pose（Tight）：ASH 28.90 / 22.83，PSNR 最优。
  - Novel Pose（Loose）：ASH 27.12 / 20.22，PSNR/LPIPS 双优。
- **速度**：推理约 29.64 fps，满足实时标准。
- **消融**：去除 motion condition（w/o mot.）导致严重伪影；去除 motion-aware offset（w/o disp.）损失细节；纹理分辨率降至 128 导致显著退化，512 收益有限但计算开销剧增，故取 256 为最优折中。

## 相关工作脉络
1. **DDC [13]**：实时网格方法，用可微渲染学习非刚性形变与动态纹理图；ASH 在同等实时性下渲染质量大幅提升，尤其在宽松衣物褶皱恢复上优势明显。
2. **HDHumans [15]**：离线混合方法，联合优化隐式场与模板网格；ASH 以同等或更优质量实现实时渲染，打破 quality-speed 权衡。
3. **Neural Actor [33]**：利用 parametric body mesh 纹理图作为局部姿态条件；无法泛化到 loose outfit，ASH 通过运动感知 offset 与 2D Conv 解码器弥补此缺陷。
4. **TAVA [29]**：纯隐式 canonical-space 表示，通过迭代求根 canonicalize；ASH 避免昂贵体积渲染，以显式 Gaussian splatting 替代。
5. **3D Gaussian Splatting [24]**：静态场景实时渲染基石；ASH 将其扩展至动态可控人体，核心挑战在于姿态条件化与 UV 参数化设计。

## 局限性与未来方向
- 当前 ASH **不更新底层可变形模板网格**，几何细节完全依赖 Gaussian offset；未来可能探索 Gaussian 反哺 3D mesh 几何的直接优化。
- 依赖多视图同步相机系统（DynaCap 为 16 相机，自采为 120 相机），**数据采集成本高**，限制开放场景应用。
- 宽松衣物动力学（如裙摆大幅摆动）虽有所改善，但对极端形变或物理耦合场景的泛化仍需验证。
- 训练需长时间序列（数千至数万帧），**训练周期较长**。

## 研究启发与可借鉴点
1. **2D 纹理空间参数化 3D 高斯**的思路可迁移至其他动态对象（如动物、流体表面）的实时渲染，核心是"把 3D 参数学会 UV 上"这一降维策略。
2. **motion-aware offset（per-texel canonical offset）**的设计对处理非刚性形变有通用价值，可作为类似 avatar 系统的标准模块。
3. **分阶段 warmup 训练**（固定位置拟合伪 GT→端到端图像损失）对训练动态 Gaussian 场景具有参考价值，能缓解长序列训练不收敛问题。
4. **法线+位置纹理联合编码姿态**比直接使用骨骼关节角度更利于 2D Conv 捕获局部外观变化，该表示形式可复用到其他姿态条件化渲染任务。

## 关键术语表
**3D Gaussian Splatting**：用一组 3D 高斯分布（位置、协方差、不透明度、球谐系数）表示场景，通过仿射投影与 alpha blending 实时合成图像的渲染技术。
**Animatable Avatar**：接受外部骨骼姿态驱动、实时生成任意新视角照片级图像的可控数字人体角色。
**Dual Quaternion Skinning**：基于对偶四元数的骨骼蒙皮变形算法，相比传统线性混合蒙皮更能保持物体体积、避免夹角处夹逼效应。
**Canonical Space**：去除当前骨骼姿态影响的归一化参考空间，用于统一存储几何与外观特征，再经正向蒙皮映射至 posed space。
**Spherical Harmonics (SH)**：用于编码 Gaussian 颜色随观测方向变化的正交函数基，ASH 中每个 texel 存储 48 维 SH 系数以刻画 view-dependent 外观。
**LPIPS**：Learned Perceptual Image Patch Similarity，基于深度特征（VGG/ResNet）的感知相似度度量，值越低表示视觉越接近。

## 可复现要素
- **数据集**：DynaCap [13]（公开），自采 120 相机序列（论文未声明公开）。
- **代码/权重**：论文未明确提及开源状态。
- **关键超参**：纹理分辨率 256；$\lambda_{\mathrm{pix}}=0.1$，$\lambda_{\mathrm{str}}=0.9$；热身阶段采样 $t$ 帧（论文未给具体数值）；高斯数量固定、位置热身阶段冻结。
