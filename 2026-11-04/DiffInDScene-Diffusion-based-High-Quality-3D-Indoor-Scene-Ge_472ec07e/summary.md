---
title: "DiffInDScene-Diffusion-based-High-Quality-3D-Indoor-Scene-Ge"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Ju_DiffInDScene_Diffusion-based_High-Quality_3D_Indoor_Scene_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:23:38"
---

# 论文速读：DiffInDScene-Diffusion-based-High-Quality-3D-Indoor-Scene-Ge

## 一句话总结
DiffInDScene 提出了一种稀疏级联扩散框架，能在粗到细的多尺度上高效生成高质量的房间级 3D 室内场景 TSDF 体素；该框架还可作为通用后处理模块，利用粗略 Occupancy 条件精化多视角立体（MVS）重建结果，其生成网格质量在用户研究与客观指标上均超越 ScanNet Ground Truth。

## 研究问题与动机
- **核心问题**：如何在三维空间中直接从零开始，高效生成结构完整、细节丰富且尺度一致的房间级 3D 室内场景几何？
- **现有方法不足**：
  1. 扩散模型在 2D 图像与单对象 3D 生成上成功，但房间级场景体素尺度巨大，直接训练计算与显存成本呈指数级增长（InstantNGP 指出普通 3D 场景中仅约 2.57% 体素有效）。
  2. 现有 3D 场景合成工作多聚焦于场景图/布局生成并检索现成对象，无法同步输出精细连续几何；Text2Room 等方法虽能重建房间级几何，但结果常碎片化、畸变严重。
  3. 多视角立体（MVS）重建（如 NeuralRecon）在迭代融合过程中会丢失大量 Mesh 细节，原始网格质量距真实标注仍有差距。

## 核心贡献（创新点）
1. **稀疏级联扩散框架**：通过三阶段粗到细流程（Coarse Occupancy → Fine Occupancy → Detail TSDF）仅在稀疏有效体素上去噪，节省两个数量级的计算与显存开销，突破了房间级场景直接扩散的算力瓶颈。
2. **多尺度 Patch-VQGAN Occupancy 解码器**：学习层级离散 latent 嵌入并预测对应分辨率的二进制占据掩码，为后续扩散阶段提供结构引导，实现“先定格局后填细节”的解耦生成。
3. **随机 TSDF 融合算法**：受 KinectFusion 增量对齐启发，将确定性平均融合替换为扩散过程中的随机采样融合，在保留局部分布方差的同时保证跨重叠 Crop 的全局一致性。
4. **MVS 重建通用精化器**：将 Stage 3 扩散模型剥离为独立后处理模块，以粗糙 Occupancy 为条件对 NeuralRecon 等重建结果进行扩散精化，打通了“从零生成”与“重建优化”的任务边界。

## 方法详解
- **级联三阶段扩散**：Stage 1 从高斯噪声生成最低分辨率 Occupancy Latent $z^{(1)}$；Stage 2 以 $z^{(1)}$ 为条件生成更高分辨率 $z^{(2)}$；Stage 3 联合 $z^{(1)}, z^{(2)}$ 生成最终 TSDF 体积 $x_0$。每阶段配备独立的稀疏扩散模型，去噪仅作用于 Occupancy Mask $M$ 标记的体素。
- **稀疏去噪建模**：遵循 DDPM 前向加噪 $q(v_t|v_{t-1}) = \mathcal{N}(v_t; \sqrt{1-\beta_t}v_{t-1}, \beta_t\mathbf{I})$ 与反向去噪 $p_\theta(v_{t-1}|v_t, y, M)$，去噪网络 $\epsilon_\theta$ 基于 TorchSparse 实现稀疏卷积与注意力，训练损失为掩码加权 MSE：$L_{\text{diff}} = \mathbb{E}[M \odot ||\epsilon - \epsilon_\theta(v_t, y, M, t)||_2^2]$。
- **Patch-VQGAN 多尺度学习**：编码器 $E$ 将 TSDF 逐级量化为离散 Latent $(z^{(1)}, z^{(2)})$，解码器 $G$ 将其还原为 Occupancy Mask 及 TSDF 值。训练时随机裁剪 $96\times96\times96$ 立方体，总损失 $L = L_{\text{rec}} + \lambda_1 L_{\text{vq}} + \lambda_2 L_{\text{GAN}}$，其中 $L_{\text{rec}}$ 包含 TSDF 的 L1 重建与 Mask 的 BCE 损失。
- **随机 TSDF 融合**：推理时将全局空间划分为 $K$ 个重叠 3D Crop，同步进行扩散。在时间步 $t$，对覆盖体素 $\mathbf{p}$ 的所有 Crop 随机均匀采样一个索引 $k$ 更新全局场：$x_t(\mathbf{p}) \sim \mathcal{N}(\mu_t^k(\mathbf{p}), \Sigma_t^k(\mathbf{p})), \; k=\text{RandomSelect}(\mathcal{G}(\mathbf{p}))$，避免平均融合导致的分布方差坍缩。
- **MVS 精化扩展**：给定 MVS/NeuralRecon 输出的粗糙 Occupancy 作为条件，直接复用 Stage 3 扩散器对 TSDF Crop 进行协同扩散与随机融合，输出优化后的精细网格。

## 实验与结果
- **数据集与基线**：生成任务使用 3D-FRONT（筛选 5913 个尺寸 $<512\times512\times128$ 场景）；精化任务使用 ScanNet 测试集，基线包括 Text2Room、Text2Room+Poisson、NeuralRecon (NR)、NR+Lap、SimpleRecon (SR)。
- **生成任务量化**：Mesh 质量三项指标（Aspect Ratio、Circularity、Shape Regularity）均值与方差全面优于基线；Shape Regularity mean 达 0.816，较 Text2Room (0.716) 提升约 14%；用户研究中 Completeness 得分 4.856（满分5）、Perceptual 4.836，均显著领先（Text2Room 分别为 2.532 / 2.472）。
- **精化任务量化**：<30° Normal Error mean 降至 5.05°（NR 为 6.45°，降幅约 21.5%），<30° ratio 提升至 59.27%（NR 为 52.88%）；用户研究 Overall 得分 87.14，反超 ScanNet GT 的 85.62，细节、平面平整度、边缘锐度三项均获最高排名。
- **效率对比**：稀疏扩散 TFLOPs 从 3.290 降至 0.008，batch=1 时 GPU 显存占用 11.8GB（稠密需 22.8GB），参数量保持一致。
- **消融结论**：稀疏策略在保持画质前提下大幅降低资源消耗；随机融合在视觉一致性与分布保真度上显著优于独立分块生成与平均融合。

## 相关工作脉络
1. **DDPM 与 3D 扩散**：既往扩散模型主要集中于 2D 图像或单对象 Voxel/Point Cloud 生成；本文首次将稀疏扩散推广至房间级连续 TSDF 场，核心差异在于引入 Coarse-to-Fine 级联与体素稀疏性约束以应对尺度爆炸。
2. **3D 场景布局合成**：如 [3,8,10,31] 等方法学习场景图并检索对象，仅输出离散布局；本文端到端输出连续几何场，同步完成布局与精细表面生成。
3. **Text2Room [15]**：当前最相关基线，依赖 2D 文生图模型迭代重建，易产生碎片化与结构畸变；本文在纯 3D 空间直接扩散，结构完整性与几何连贯性显著提升。
4. **KinectFusion [17]**：经典增量 TSDF 融合算法采用确定性平均融合；本文将其思想迁移至扩散逆过程，以随机采样替代平均，避免分布方差被平滑坍塌。
5. **NeuralRecon [41] / MVS 重建**：纯重建管线侧重实时性与一致性；本文不竞争重建精度，而是将其输出作为条件注入扩散先验，证明生成模型可作为通用几何后处理优化工具。

## 局限性与未来方向
- **对象生成缺失**：当前框架主要生成房间主体几何（墙体/地面/天花板），未直接在 TSDF 空间中合成室内家具或独立物体，纹理需依赖 DreamSpace 等外部模块后处理。
- **条件控制有限**：正文未深入展开文本/语义/布局的条件引导机制，仅 supplementary 提及 Sketch-to-Scene，大规模复杂户型的稳定可控生成仍待探索。
- **推理步数与融合开销**：随机融合需同步维护多个重叠 Crop 的扩散状态，推理时间随 Crop 数量线性增长，未来可结合渐进采样或稀疏注意力进一步加速。
- **未来方向**：融合多模态条件（文本/布局草图/点云先验）实现可控生成；将家具实例生成纳入统一级联流程；扩展至室外场景或动态 4D 场景；优化融合策略以降低采样步数。

## 研究启发与可借鉴点
- **“低有效体素率”提示的稀疏化范式**：将场景稀疏性直接转化为计算优势（稀疏卷积+掩码损失），为城市级、建筑级大规模 3
