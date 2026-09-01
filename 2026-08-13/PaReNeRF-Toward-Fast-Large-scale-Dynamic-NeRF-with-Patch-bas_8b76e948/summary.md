---
title: "PaReNeRF-Toward-Fast-Large-scale-Dynamic-NeRF-with-Patch-bas"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Tang_PaReNeRF_Toward_Fast_Large-scale_Dynamic_NeRF_with_Patch-based_Reference_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:41:24"
field: "神经辐射场与大场景重建"
keywords: ["NeRF", "动态场景重建", "大规模神经渲染", "参考解码器", "patch采样", "光流", "结构相似度"]
innovations: ["基于光流和结构相似性的先验信息搜索方法，加速训练阶段参考信息查找", "patch-based采样替代随机光线采样，结合低分辨率渲染与参考解码器，同步提升训练/推理速度及渲染质量", "轻量级参考解码器融合已知视图信息，有效补偿低分辨率渲染的精度损失"]
benchmarks: ["KITTI", "VKITTI2"]
---

# 论文速读：PaReNeRF-Toward-Fast-Large-scale-Dynamic-NeRF-with-Patch-based-Reference

## 一句话总结
本文针对大规模动态场景NeRF重建训练时间过长、推理速度过慢的问题，提出了一种基于参考解码器（reference decoder）的快速动态NeRF框架PaReNeRF，通过patch-based采样加速训练、基于光流与结构相似性的先验信息搜索，以及轻量级参考解码器融合已知视图信息，在KITTI和VKITTI2数据集上相比SOTA方法SUDS将训练时间降低60%、推理时间降低87%，同时PSNR提升约13%。

## 研究问题与动机
- **大规模动态场景重建的计算瓶颈**：自动驾驶场景数据稀疏且轨迹单一，现有SOTA方法（如SUDS、Mars）对约9秒视频的训练时间超过2天，推理耗时也较长，难以满足实际仿真需求。
- **低分辨率渲染的精度损失**：Unisim等方法通过渲染低分辨率特征图并依赖CNN上采样来加速训练和推理，但重建质量明显下降。
- **先验信息利用不足**：现有稀疏视角NeRF方法（如pixelNeRF、NeRF-SR）主要面向静态小场景，在动态场景的稀疏性和单轨特性下不适用。
- **光线采样效率低**：传统随机光线采样在训练时每次需加载大量数据，导致数据读取时间占比高。

## 核心贡献（创新点）
1. **提出了基于光流和结构相似性的先验信息搜索方法**，利用视频帧间的光流信息缩小搜索区域，显著提升训练阶段参考信息查找效率。
2. **用patch-based采样替代随机光线采样**，结合低分辨率特征图渲染与CNN上采样，大幅减少数据读取次数和ray查询次数，从而加速训练和推理。
3. **设计了轻量级参考解码器（reference decoder）**，将已知视图的高分辨率参考信息融入低分辨率体渲染特征图的解码与上采样过程，有效弥补精度损失。
4. **在KITTI和VKITTI2数据集上实现了速度-质量的综合提升**，训练时间降低60%、推理时间降低87%、PSNR提升13%，全面超越SUDS等基线方法。

## 方法详解
**基线方法回顾**：以SUDS为基础，SUDS将场景分解为静态分支（static branch）、动态分支（dynamic branch）和远距离分支（far-field branch），分别建模静态地形、瞬态/动态物体和远景/天空。

**三个核心改进模块**：

1. **Patch-sampling based radiance fields**：
   - 将随机光线采样改为patch采样，若batch包含N条光线，设patch尺寸为h×w，则只需加载数据N/(h×w)次而非N次，大幅减少数据读取时间。
   - 渲染低分辨率特征图（414×125）而非原图（1242×375），结合3x上采样进一步加速推理。

2. **Prior information searching module**：
   - **训练阶段**：对随机采样的patch P，利用已有2D光流fl_ij，计算patch像素坐标(i,j)加上光流值后的范围，确定前后帧中的相似区域Ref_f和Ref_b；在这些区域内逐像素滑动patch，计算与目标patch的结构相似度（SSIM），取最大SSIM对应的patch作为参考patch RP。
   - **推理阶段**：计算体渲染特征图F与所有已知视图RGB图像的SSIM，选择SSIM最高的图像作为参考图像RI。

3. **Reference decoder结构**：
   - 基于轻量级超分网络CARN（Cascading Residual Network）设计，包含：
     - 特征编码器（feature encoder）：提取参考图像RI的隐藏特征
     - CARN特征编码器：对低分辨率特征图F和融合特征进行编码，通过局部和全局级联模块整合多层信息
     - 上采样模块：3×1卷积 + PixelShuffle完成3x上采样
     - 特征解码器：将最终特征图渲染为RGB图像
   - 公式：rf: (F ∈ R^{Hf×Wf×Nf}, R ∈ R^{H×W×3}) → I ∈ R^{H×W×3}

**优化目标**：沿用SUDS的多种损失（重建损失、形变损失、光流损失、动静分离损失、阴影损失），仅将L2光度损失中的渲染RGB改为参考解码器输出的重建图像，并以patch-wise方式计算损失。训练125,000次迭代，batch含4096条光线、16个16×16 patch，学习率5×10⁻³衰减至5×10⁻⁴。

## 实验与结果
**数据集**：KITTI和VKITTI2，训练序列最多90个时间点（9秒），图像尺寸1242×375，双视角。

**评估任务**：图像重建（image reconstruction）和新视角合成（novel view synthesis, NVS）；NVS设置含75%、50%、25%训练比例。

**主要结果（KITTI图像重建）**：
- PSNR：PaReNeRF 32.642 vs SUDS 28.31，提升约13%
- SSIM：0.933 vs 0.870
- 训练时间：26h vs 64h，降低60%
- 推理时间：1.61s vs 12.86s，降低87%

**VKITTI2图像重建**：
- PaReNeRF PSNR 30.894 vs SUDS* 30.853，训练时间26h vs 64h，推理时间1.35s vs 11.89s

**NVS结果**：在不同训练比例下，PaReNeRF均取得最佳PSNR/SSIM/LPIPS，且在训练视角稀疏时相比SUDS有效减少鬼影artifact。

**消融实验**：
- Patch采样单独使用：训练加速（36h），但精度下降（PSNR 27.04）
- 加入Encoder-Decoder：推理大幅加速（1.75s），精度恢复（29.31）
- 加入Reference Decoder：精度大幅提升（33.23@250K / 32.64@125K），综合最优

## 相关工作脉络
- **SUDS [48]**：当前大规模动态场景SOTA，采用三分支hash table结构，支持丰富监督信号，但训练/推理耗时长；本文以其为基线，在速度和精度上同时超越。
- **UniSim [60]**：采用低分辨率特征图+CNN上采样加速，但encoder-decoder结构损失重建精度；本文在低分辨率渲染基础上引入参考解码器补偿质量。
- **Mars [55]**：模块化实例感知模拟器，训练同样耗时超过2天；本文方法在相同任务上实现数量级的速度提升。
- **NSG [32] / PNF [22]**：基于场景图/语义对象的动态场景NeRF，需人工3D边界框和全景标注，泛化受限；本文无需额外标注。
- **pixelNeRF [61] / NeRF-SR [49]**：稀疏视角先验利用方法，面向静态小场景；本文将其思想扩展到大规模动态场景，并设计光流+SSIM搜索策略适应动态特性。
- **Block-NeRF [44] / Mega-NeRF [47] / BungeeNeRF [56]**：大规模静态场景NeRF；本文聚焦动态场景，解决静态方法在动态数据上的模糊和artifacts问题。

## 局限性与未来方向
- **未实现真正实时**：尽管训练和推理速度大幅提升，作者自述距离真正的real-time training and rendering仍有挑战。
- **参考信息搜索依赖光流精度**：光流估计在纹理缺失或大位移区域可能不准确，影响参考patch选取质量。
- **仅验证于KITTI和VKITTI2**：在更复杂场景（如nuScenes、Waymo）上的泛化能力有待验证。
- **参考解码器引入额外参数**：虽然轻量化，但仍增加了模型复杂度，在极端资源受限场景下可能成为瓶颈。
- **未来方向**：探索更高效的参考搜索策略、扩展至城市级大规模场景、结合更多模态输入（如LiDAR、语义标签）进一步提升性能。

## 研究启发与可借鉴点
1. **Patch-based采样策略可迁移**：将随机光线采样改为patch采样以减少数据读取开销的思路，可推广至其他NeRF变体或神经辐射场相关任务。
2. **光流辅助的先验搜索机制**：利用时序光流缩小参考图像搜索空间，兼顾效率与准确性，可借鉴于视频级NeRF、4D重建等时序场景。
3. **轻量级参考解码器架构**：基于CARN的轻量级超分网络融合参考信息的思路，可在低资源渲染、实时novel view synthesis中复用。
4. **低分辨率渲染+参考补偿的设计范式**：通过降分辨率加速计算、再用参考信息补偿质量损失，是平衡速度与精度的有效范式，可与扩散模型、NeX等结合探索。
5. **多比例训练集评估NVS**：采用75%/50%/25%不同训练视角比例的评估设置，系统性地检验方法在数据稀疏场景下的鲁棒性，值得在类似研究中借鉴。

## 关键术语表
**NeRF（Neural Radiance Field）**：通过MLP隐式编码连续场景辐射场的神经渲染方法，实现照片级真实感新视角合成。

**NFF（Neural Feature Field）**：NeRF的广义形式，将3D点和视角映射为几何特征和特征描述子，NeRF可视为其特例。

**SUDS（Scalable Urban Dynamic Scenes）**：当前大规模动态场景NeRF的SOTA方法，将场景分解为静态、动态、远景三分支hash table结构。

**Patch-based Sampling**：将随机光线采样改为patch级别采样，减少数据读取次数，加速训练但降低采样随机性。

**Optical Flow**：视频中相邻帧间像素的运动矢量场，用于估计像素在时间维度上的位移，本文用于缩小参考区域搜索范围。

**Structural Similarity (SSIM)**：衡量两图像结构相似度的指标，本文用于匹配参考patch/图像与目标patch的相似度。

**Reference Decoder**：基于CARN的轻量级网络，融合已知视图的参考信息对低分辨率体渲染特征图进行解码和上采样。

**Novel View Synthesis (NVS)**：从已知视角生成新视角图像的任务，是NeRF类方法的核心应用之一。

## 可复现要素
- **数据集**：KITTI [12] 和 VKITTI2 [2]，论文使用与 prior works 相同的子序列（最多90时间点，1242×375分辨率）
- **代码**：论文未提及代码开源情况
- **权重**：论文未提及权重是否公开
- **关键超参**：迭代次数125,000；batch size 4096 rays（16个16×16 patch）；渲染分辨率414×125，目标分辨率1242×375（3x上采样）；Adam优化器，初始学习率5×10⁻³，衰减至5×10⁻⁴
