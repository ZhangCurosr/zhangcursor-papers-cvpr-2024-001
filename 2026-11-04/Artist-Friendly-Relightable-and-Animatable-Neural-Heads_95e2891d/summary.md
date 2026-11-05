---
title: "Artist-Friendly-Relightable-and-Animatable-Neural-Heads"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Artist-Friendly_Relightable_and_Animatable_Neural_Heads_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:20:24"
field: "神经辐射场与数字人渲染"
keywords: ["neural avatar", "relightable rendering", "animatable head", "MVP", "nearfield lighting", "volumetric primitives"]
innovations: ["在MVP架构中引入per-primitive局部光照和视角方向条件，实现动态头部的近场光照与视角重渲染", "纯表情驱动的外观解耦设计，支持艺术家创作的新表情泛化", "仅用稀疏LED灯条阵列（32条）即可训练高质量relightable+animatable神经头部"]
benchmarks: ["TRAvatar*", "novel light directions", "novel performances", "nearfield lighting", "environment map relighting"]
---

# 论文速读：Artist-Friendly-Relightable-and-Animatable-Neural-Heads

## 一句话总结
本文提出了一种基于MVP（体素基元混合）架构的神经头部avatar方法，通过引入per-primitive局部光照和视角方向条件，首次实现了同时支持动画驱动与任意环境光/近场光照重新渲染的动态神经头部，且仅需低成本稀疏LED灯条阵列即可完成训练。

## 研究问题与动机
- **动态与光照的双重控制需求**：现有神经avatar方法通常只能解决运动可控（animatable）或光照可控（relightable）之一，缺乏同时支持两者的高保真方案。
- **传统方法在复杂材质上的局限**：mesh-based方法难以处理头发、眼睛等高光/半透明结构及大遮挡（如牙齿、舌头），而原始NeRF无法建模动态场景。
- **昂贵采集设备的依赖**：现有relightable方法多依赖高密度light stage，硬件成本高；本文希望用廉价的稀疏LED灯条阵列实现高质量结果。
- **远场/近场光照与视角的统一建模**：现有方法难以同时支持远距离环境贴图光照与近距离点光源，以及广角/近景视角变化。

## 核心贡献（创新点）
1. **Relightable MVP架构**：在MVP基础上扩展appearance branch，引入per-primitive局部光照方向$\mathbf{l}_k$和视角方向$\mathbf{v}_k$作为额外条件输入，使动态头部支持任意新光照和视角。
2. **近场光照与视角建模能力**：通过逐基元计算局部方向向量并在卷积网络各层拼接，支持nearfield点光源和nearfield相机视角（如dolly-zoom效果），这是原MVP无法实现的。
3. **低成本采集方案**：采用10相机+32个可独立控制LED灯条的稀疏阵列（继承自ReNeRF设置），以OLAT序列录制动态表演，无需昂贵的dense light stage。
4. **纯表情驱动、无外观烘焙**：mesh encoder仅从tracked geometry提取latent expression vector $\mathbf{z}$，不含外观信息，使推理时可使用未见过的艺术家创作表情。

## 方法详解
**整体框架**：基于Lombardi et al. [25]的MVP架构，包含mesh encoder、geometry branch（沿用原MVP）、illumination-modulated appearance branch三个可训练模块。

**Mesh Encoder**：修改自Deep Appearance Model编码器，移除texture分支，仅从tracked 3D face mesh提取256维latent expression vector $\mathbf{z}$，后接1-layer MLP将其映射为$256 \times 8 \times 8$特征图。

**Geometry Branch**：完全沿用原MVP几何解码器，输入$\mathbf{z}'$，输出$N^2=16384$个3D基元的平移$\mathbf{t}_k$、旋转$\mathbf{R}_k$、非均匀缩放$\mathbf{s}_k$，基元排列于$N \times N$网格并通过guide mesh UV参数化关联。

**Illumination-Modulated Appearance Branch（核心创新）**：
- 对每个基元$k$，先通过geometry branch获取世界坐标变换，再计算：
  - 局部光照方向：$\mathbf{l}_k = \mathbf{p}_{olat} - \mathbf{R}_k \cdot \mathbf{t}_k$
  - 局部视角方向：$\mathbf{v}_k = \mathbf{p}_{cam} - \mathbf{R}_k \cdot \mathbf{t}_k$
- 将$\{\mathbf{v}_k\}$和$\{\mathbf{l}_k\}$拼接为6通道图像$\mathbf{I} \in \mathbb{R}^{6 \times (NM) \times (NM)}$，在每个convtranspose层前双线性下采样后拼接到中间特征。
- 网络结构：7层convtranspose（kernel=4, stride=2, padding=1），分辨率从$8 \times 8$逐步放大至$1024 \times 1024$，输出通道依次为256→128→128→64→64→32→48（48=rgb×$M_z$）。
- Opacity $\alpha$由独立branch预测，不依赖光照/视角条件。

**训练数据**： interleaved OLAT（one-light-at-a-time）与full-on帧序列，光照顺序使相邻OLAT帧差异最大以最大化数据效率；平均每人约2700张/相机，训练帧约1800帧。

**损失函数**：
$$\mathcal{L} = \lambda_{pho}\mathcal{L}_{pho} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{vol}\mathcal{L}_{vol} + \lambda_{kld}\mathcal{L}_{kld} + \lambda_{mat}\mathcal{L}_{mat}$$
其中新增matting loss：$\mathcal{L}_{mat} = \text{MAE}(\mathcal{M}, \tilde{\alpha}(\Theta))$，$\mathcal{M}$由MODNet提取。权重：$\lambda_{pho}=1.0, \lambda_{geo}=10.0, \lambda_{vol}=0.01, \lambda_{kld}=0.001, \lambda_{mat}=0.1$。优化器Adam，lr=0.0001，每人在A6000 GPU训练200k iter（约2天）。

## 实验与结果
**数据集与设置**：3个subject，每人约1800训练帧（10相机×OLAT序列），输入分辨率下采样至1024×768。

**基线对比**：由于同时支持relightable+animatable的开源方法极少，复现了TRAvatar*（基于TRAvatar [51]的lighting branch）作为唯一对比基线。

**定量结果（Table 1）**：
| 场景 | 指标 | Ours | TRAvatar* |
|------|------|------|-----------|
| Novel light (S1) | PSNR↑ | 32.19 | 29.68 |
| Novel performance (S1) | PSNR↑ | 39.75 | 28.86 |
| Novel light+perf (S1) | PSNR↑ | 31.21 | 28.68 |
| Novel light (S2) | PSNR↑ | 32.20 | 30.13 |
| Novel performance (S2) | PSNR↑ | 32.37 | 29.52 |
| Novel light+perf (S2) | PSNR↑ | 29.11 | 28.13 |
| Novel light (S3) | PSNR↑ | 33.73 | 30.65 |
| Novel performance (S3) | PSNR↑ | 31.28 | 30.65 |

Ours在全部测试场景下均显著优于TRAvatar*，PSNR提升幅度在1~11 dB之间，SSIM/LPIPS/MAE均更优。

**定性结果**：
- 支持novel view + novel expression + novel point light联合渲染（Fig. 3）。
- 支持任意environment map照明，可从OLAT训练数据泛化到密集采样的半球方向（Fig. 4–5）。
- 支持nearfield点光源（Fig. 6）和nearfield相机（dolly-zoom，Fig. 7）。

## 相关工作脉络
1. **原始MVP [25]**：基于体素基元混合的动态神经头部表示，本文以此为基础扩展relightability，本质区别是原MVP仅条件于全局view direction，无法支持光照变化。
2. **ReNeRF [50]**：静态神经场景的relightable NeRF，使用OLAT MLP+spherical codebook，但仅支持静态场景；本文将其思想扩展到动态头像。
3. **TRAvatar [51]**（同期工作）：也扩展MVP支持relighting，但采用固定basis线性光照分支，无法插值/外推新光照方向，也不能建模nearfield效果。
4. **DRAM [2]**：基于VAE的deep relightable appearance model，需tracked mesh和average texture输入，仅支持静态expression的自由视角relighting。
5. **NeRSemble [19] / Neural Volumes [24]**：动态神经头部/人体表示，支持novel view synthesis但无relighting能力。
6. **NeLFs [43] / NRTFs [26]**：静态场景的neural light-transport field方法，支持relighting但无法处理动态。

## 局限性与未来方向
- **仅覆盖头部正面**：未实现360°完整头部avatar，导致发际线和颈部边界出现artifact。
- **快速运动模糊问题**：若训练数据含运动模糊，重建结果会变模糊。
- **极端表情外推能力有限**：未见过的极端表情泛化不佳，但作者认为增加训练数据多样性可缓解。
- ** gaze和hair运动不可控**：输入为mesh，无法单独控制眼神和头发动态。
- **未来方向**：可扩展至完整360°avatar、引入物理驱动的发丝/眼部渲染、支持更多fine-grained控制。

## 研究启发与可借鉴点
1. **Per-primitive局部方向条件化**：将光照/视角方向在基元级别计算并逐层注入卷积网络，是一种通用且高效的nearfield效应建模范式，可迁移至人体avatar、手势等其他体积表示任务。
2. **表情与外观解耦设计**：encoder仅从几何提取expression code，不含appearance，使推理时可完全自由创作新表情而不受训练数据外观分布限制，这一设计对艺术友好型avatar具有重要参考价值。
3. **低成本OLAT采集策略**：相邻帧采用差异最大的光照顺序以最大化数据效率，配合full-on帧辅助mesh tracking，为实际拍摄提供了可复用的工程方案。
4. **Matting loss替代背景建模**：在纯黑背景下用MODNet提取alpha mask并加入MAE matting loss，避免了原MVP中复杂的背景foreground分离网络，简化了训练流程。

## 关键术语表
**MVP (Mixture of Volumetric Primitives)**：一种混合神经渲染表示，将场景建模为大量带颜色/透明度体素网格的3D基元集合，由guide mesh驱动形变。
**OLAT (One-Light-At-a-Time)**：一次仅点亮一个光源的采集方式，用于分解光照分量以实现事后重新光照。
**Relightable**：指神经avatar支持在推理时将渲染结果置于新的光照环境（环境贴图或点光源）下。
**Animatable**：指神经avatar支持由表达式参数（blendshapes/rig controls）驱动生成新的面部形变。
**Nearfield lighting**：近距离点光源照明，光照方向随基元位置变化显著，不同于远距离平行光假设。
**Dolly-zoom**：摄影技巧，摄像机靠近主体的同时缩小焦距（增大FOV），产生背景压缩的视觉错觉。
**Guide mesh**：与UV参数化关联的粗糙表面网格，用于初始化基元空间布局并驱动其形变。

## 可复现要素
- **数据集**：论文未公开，使用自采集的3个subject数据（10相机+32 LED灯条阵列）；训练数据约1800帧/人。
- **代码**：论文未提及开源状态。
- **关键超参**：$N^2=16384$个基元，$M=8$体素分辨率；Adam lr=0.0001；batch size=12；200k迭代；输入分辨率1024×768；权重$\lambda_{pho}=1.0, \lambda_{geo}=10.0, \lambda_{vol}=0.01, \lambda_{kld}=0.001, \lambda_{mat}=0.1$。
