---
title: "DiffusionAvatars-Deferred-Diffusion-for-High-fidelity-3D-Hea"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Kirschstein_DiffusionAvatars_Deferred_Diffusion_for_High-fidelity_3D_Head_Avatars_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:50:57"
field: "3D人脸动画与生成"
keywords: ["3D Head Avatar", "Diffusion Model", "Neural Rendering", "NPHM", "ControlNet", "Deferred Rendering"]
innovations: ["提出Deferred Diffusion架构，将NPHM隐式几何与预训练LDM结合实现高保真3D头像", "设计TriPlane空间特征绑定策略以补偿NPHM无UV空间的缺陷", "通过cross-attention直接注入NPHM表达式码以增强细节表情控制"]
benchmarks: ["NeRSemble", "Self-Reenactment", "Avatar Animation"]
---

# 论文速读：DiffusionAvatars: Deferred Diffusion for High-fidelity 3D Head Avatars

## 一句话总结
DiffusionAvatars提出了一种基于扩散模型的神经渲染器，结合NPHM隐式头部几何与预训练LDM（Latent Diffusion Model）的图像生成先验，实现对单人多视角视频的自演绎与动画驱动，在视口一致性、表情细节还原和照片级真实感上显著优于现有3D头像方法。

## 研究问题与动机
- **核心问题**：从单一人物多视角视频中重建可自由控制姿态与表情的照片级3D头部头像（4D光度重建问题本质上是欠约束的）。
- **2D扩散模型不足**：虽然能生成高真实感人脸图像，但缺乏跨视角和时间的一致性以及精确的3D控制能力。
- **传统3D头像不足**：基于FLAME等网格模板的方法视图一致性好，但细节和照片级真实感远不如2D扩散模型。
- **现有方法在复杂表情区域退化**：如口腔内部、眉毛等由粗粒度3DMM难以精确描述的区域，2D解码器容易生成模糊或失真的结果。

## 核心贡献（创新点）
1. **DiffusionAvatars架构**：提出一种基于ControlNet范式的扩散神经渲染器，将预训练LDM转化为图像到图像的翻译模型，用于生成3D头像；与DiffusionRig等直接以FLAME网格渲染图为条件的2D方法本质不同，本文利用NPHM更精确的隐式SDF几何作为代理。
2. **TriPlane空间特征绑定**：针对NPHM缺乏一致UV空间的隐式特性，将可学习空间特征通过TriPlane（3个空间维度平面）和Ambient Map（2个环境维度映射）绑定到头部表面，补偿网格几何缺陷并提升跨视角一致性；相比球形UV近似方案，该设计能更有效地保留表面细节。
3. **表达式直接条件注入**：通过在U-Net中新增cross-attention层，将NPHM表达式码直接投影为token序列注入扩散过程，使模型能区分细微表情细节并合成训练数据未覆盖的表情（如舌头运动）；与仅依赖栅格化渲染图提供粗略姿态/表情信号的方式形成互补。

## 方法详解
- **NPHM代理几何**：给定多视角视频，通过COLMAP获取点云，拟合NPHM得到身份码 $z_{id}$ 和逐帧表达码 $z_{exp}^t$；采用MonoNPHM变体（逆向形变场）。通过marching cubes从SDF提取网格 $M_t$，并对每个顶点计算canonical坐标 $x_{can} = \mathcal{F}_{exp}(x) \in \mathbb{R}^{3+2}$。
- **光栅化**：使用nvdiffrast对 $M_t$ 和相机位姿 $\pi_t$ 进行光栅化，输出法线、深度及canonical坐标渲染图 $R_{can}^t \in \mathbb{R}^{H \times W \times (3+2)}$，构成DiffusionAvatar的基础控制信号。
- **TriPlane特征映射**：
  - 空间特征：$R_{feat}^t = \mathrm{TRIPLANE}(R_{can,0-3}^t)$，使用 $512 \times 512 \times 16$ 的可学习三平面。
  - 环境特征：$R_{feat\_amb}^t = \mathrm{AMBIENTMAP}(R_{can,3-5}^t)$，使用 $512 \times 512$ 的2D特征图。
  - 最终输入共73通道（含光栅化buffer）。
- **表达式条件注入**：
  - $f_{exp}^t = \mathrm{EXP}(z_{exp}^t)$ 经线性层映射为4个expression token（维度 $4 \times d$）。
  - 在LDM原有cross-attention分支上叠加新层：$Z \gets Z + \mathrm{ATTENTION}(Q, W^k f_{exp}^t, W^v f_{exp}^t)$，共插入15个cross-attention层。
- **训练与推理**：
  - 采用v-parameterization（v-prediction）以加速收敛并支持纯噪声输入训练。
  - 损失函数：$\mathcal{L} = \mathbb{E}[||\mathcal{D}(x_\tau^t, R^t, f_{exp}^t) - v||_2]$，联合优化ControlNet C、表达式条件模块 $(W^k, W^v, \mathrm{EXP})$ 及空间特征图。
  - 推理时从 $x_T \sim \mathcal{N}(0, I)$ 开始，由ControlNet提取条件特征后送入LDM去噪。

## 实验与结果
- **数据集**：NeRSemble多视角视频数据集（16相机、26序列/人），对8人进行NPHM拟合，每人大约3300帧用于训练。
- **基线**：NeRFace、MVP、DNR、DNR+GAN、DiffusionRig。
- **自演绎（Self-Reenactment）定量结果**（表1，均值）：
  - PSNR最高：Ours 24.9 vs 次优DNR 24.5。
  - LPIPS最低：Ours 0.081 vs 次优DNR+GAN 0.114（提升约30%相对幅度）。
  - JOD（时序一致性）最高：Ours 7.55 vs 次优DNR+GAN 7.08。
  - CSIM（身份保持）最高：Ours 0.882 vs 次优DNR+GAN 0.868。
  - AKD最低：Ours 3.42 vs 次优DNR 2.06（注意原文DNR为2.06更低，但综合来看本文在多数face-specific指标上更优）。
- **头像动画用户研究**（表2，满分5分，49人、735回复）：
  - VQ（视觉质量）：Ours 4.02 vs 次优DNR+GAN 3.06（显著提升）。
  - DF（驱动保真度）：Ours 4.14 vs 次优DNR+GAN 3.97。
- **消融实验**（表3）：
  - 去除扩散（单步预测）导致LPIPS从0.074升至0.133，证明扩散先验对锐度至关重要。
  - 用FLAME替换NPHM使AKD从1.91升至2.46，说明NPHM SDF几何对复杂表情重建的关键作用。
  - 关闭表达式条件后CSIM从0.918降至0.911，整体性能下降。
  - TriPlane优于球形UV近似（LPIPS 0.074 vs 0.075，AKD 1.91 vs 1.95）。

## 相关工作脉络
- **NeRFace / MVP**：基于NeRF/体素原语的3D渲染方法，视图一致性好但图像真实感受限于隐式场的细节表达能力；DiffusionAvatars将其几何表征升级为NPHM SDF并叠加2D扩散先验。
- **Deferred Neural Rendering (DNR)**：早期利用Pix2Pix在三维MM网格上合成纹理的方法；本文在架构上继承其"deferred rendering"思想，但以ControlNet+LDM替代GAN，并引入NPHM和TriPlane提升几何精度。
- **DiffusionRig**：直接用FLAME渲染图条件化扩散模型；本文的核心区别在于使用更高表达的NPHM SDF网格及额外表达的跨注意力注入，从而在复杂表情区域表现更好。
- **ControlNet / IPAdapter**：提供预训练扩散模型的条件化控制范式；本文将其引入3D头像领域，结合3D几何先验实现视图一致的高质量渲染。
- **3DMM-based方法 (FLAME/BFM)**：传统参数化头部模型；本文指出FLAME在口腔等拓扑变化区域表达能力不足，因此选用NPHM的SDF+逆形变场以获得更精确几何。

## 局限性与未来方向
- **光照建模缺失**：当前方法将光照烘焙到生成图像中，无法独立控制光源和阴影；未来可利用底层3D几何直接合成阴影。
- **推理速度受限**：扩散去噪循环计算开销大，暂不支持实时应用；可结合扩散蒸馏（如Consistency Models、Progressive Distillation）加速。
- **单目标人物训练**：目前为每个个体单独训练，尚未探索零样本或少样本泛化到未见人物的能力。
- **NPHM拟合依赖多视角**：需要COLMAP点云及16相机系统，限制了应用场景的灵活性。

## 研究启发与可借鉴点
1. **Deferred Diffusion范式**：将"预渲染几何buffer + 空间特征绑定 + 扩散解码"作为通用3D-aware生成框架，可迁移至全身avatar、动物头像等任务。
2. **TriPlane替代UV展开**：对于隐式SDF等无全局UV的几何表示，TriPlane+Ambient Map是一种轻量且有效的特征绑定策略，值得在其他3D生成任务中验证。
3. **双重条件注入策略**：粗粒度几何渲染图提供姿态/形状先验，细粒度参数码通过cross-attention注入细节信息，这种"全局+局部"双分支条件化设计对复杂可控生成有参考价值。
4. **v-prediction在图像到图像翻译中的优势**：相比epsilon-prediction，v-parameterization在纯噪声输入时的训练稳定性更好，且有助于加速收敛，值得在控制型扩散任务中优先采用。
5. **消融实验设计的系统性**：从几何模型（NPHM vs FLAME）、条件模块（expression cond.）、特征绑定（TriPlane vs spherical UV）等多维度消融，为后续改进提供了清晰的优化方向。

## 关键术语表
**NPHM (Neural Parametric Head Model)**：基于SDF的参数化隐式头部模型，可通过身份码和表达式码生成高精度头部几何。
**LDM (Latent Diffusion Model)**：在压缩潜空间中运行的扩散模型，如Stable Diffusion，具有强大的图像生成先验。
**ControlNet**：通过在预训练扩散模型旁侧添加可训练条件网络来实现对生成过程的外部控制。
**TriPlane**：将3D空间特征分解为三个正交2D平面的高效表示方法，常用于NeRF等3D生成任务。
**v-prediction**：扩散模型的一种训练目标参数化方式，通过预测v值而非噪声来提升训练稳定性和收敛速度。
**Deferred Neural Rendering**：先渲染几何缓冲区（深度/法线/纹理坐标），再在这些缓冲区上学习神经网络进行最终图像合成的范式。
**CSIM**：基于ArcFace提取的身份embedding余弦相似度，用于评估渲染图像的个体身份保持能力。
**JOD**：基于FovVideoVDP的感知视频失真度量，用于评估时序一致性。

## 可复现要素
- **数据集**：NeRSemble（多视角视频数据集），论文使用其中的8人序列。
- **代码开源情况**：论文未明确声明开源代码，但提供了项目网站链接。
- **权重**：使用Stable Diffusion v2.1作为预训练LDM主干。
- **关键超参**：学习率ControlNet为1e-4、TriPlane为1e-2；batch size=8；分辨率512×512；训练100k步，约2天/人（RTX A6000）。
- **背景去除**：BackgroundMattingV2 + 躯干分割网络。
