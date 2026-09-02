---
title: "Score-Guided-Diffusion-for-3D-Human-Recovery"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Stathopoulos_Score-Guided_Diffusion_for_3D_Human_Recovery_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:16:18"
field: "3D人体恢复与重建"
keywords: ["3D Human Mesh Recovery", "Diffusion Model", "Score Guidance", "Inverse Problem", "Human Pose Estimation", "SMPL"]
innovations: ["利用扩散模型score guidance在隐空间迭代精化人体参数，无需任务特定重训练", "提出通用框架统一处理单帧拟合、多视图重建和时序精化三类逆问题", "DDIM反演-引导采样循环设计，首次使扩散模型有效替代传统优化拟合"]
benchmarks: ["3DPW", "EMDB 1", "Human3.6M", "Mannequin Challenge"]
---

# 论文速读：Score-Guided-Diffusion-for-3D-Human-Recovery

## 一句话总结
ScoreHMR 利用扩散模型的 score guidance 机制，在隐空间中对初始回归估计进行迭代精化，将人体模型与图像观测（如 2D 关键点）对齐，从而解决 3D 人体网格恢复中的逆问题。该方法无需针对下游任务重新训练，在单帧拟合、多视图重建和视频序列恢复三个场景中均超越所有优化基线。

## 研究问题与动机
- **单目 HMR 中图像-模型对齐困难**：现有前馈回归方法（如 HMR 2.0）虽能达到较高 3D 重建精度，但在单目场景下难以使 SMPL 模型与图像观测（2D 关键点）精确对齐。
- **传统优化方法存在多重缺陷**：基于优化的人体拟合方法（如 SMPLify）依赖手工设计的能量函数，易陷入局部最优、对初始化敏感且速度慢。
- **回归与优化的协同不足**：现有"回归+优化精化"方案仍需多项先验项才能获得合理解，且优化过程本身仍面临多个局部极小值。
- **扩散模型在 3D 人体恢复逆问题中尚未被探索**：扩散模型已广泛应用于文本到图像生成及运动生成，但用于解决 3D 人体姿态与形状估计逆问题的研究仍是空白。

## 核心贡献（创新点）
1. **提出 ScoreHMR，以 score guidance 驱动扩散模型隐空间中的迭代精化**：将扩散模型作为学习的人体参数先验，通过数据似然梯度引导去噪过程实现模型-图像对齐，无需任务特定重训练。
2. **三种统一应用无需额外训练**：通过替换不同的 guidance loss（关键点重投影、跨视图一致性、时序一致性），ScoreHMR 分别适用于单帧拟合、多视图无标定重建、视频序列时序精化。
3. **首次提升 SOTA 前馈系统的 3D 姿态性能**：在单帧模型拟合设置下，ScoreHMR 是唯一能同时提升 HMR 2.0 3D 姿态精度的方法（3DPW 上 PA-MPJPE 从 54.3mm 降至 51.1mm）。
4. **全面超越所有优化基线**：在三个设定和多个数据集上均优于 SMPLify、ProHMR-fitting、VIBE-opt 等传统优化方法。
5. **极高的运行效率**：仅需 1.5 分钟处理完整 Mannequin Challenge（20K 帧）、14 分钟处理 3DPW 测试集（35K 帧），远快于传统迭代优化。

## 方法详解

**整体框架**：给定输入图像 $I$ 和从回归网络获得的初始 SMPL 估计 $\mathbf{x}_{reg} = \{\theta_{reg}, \beta_{reg}\}$，以及额外观测 $\mathbf{y}$（如 2D 关键点），ScoreHMR 通过以下步骤精化估计：

**1. DDIM 反演（Inversion）**：将回归估计 $\mathbf{x}_{reg}$ 通过确定性 DDIM 反演映射到噪声水平 $\tau$ 处的隐变量 $\mathbf{x}_\tau$：
$$\mathbf{x}_{t+1} = \sqrt{\alpha_{t+1}} \hat{\mathbf{x}}_0(\mathbf{x}_t) + \sqrt{1-\alpha_{t+1}} \epsilon_\phi(\mathbf{x}_t, t, I)$$
验证表明反演-采样回环的重建误差小于 $10^{-3}$ 每维度。

**2. Score Guidance 引导的去噪采样**：理想情况下需使用后验分数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | I, \mathbf{y})$，但其似然项无解析形式。在观测噪声为高斯的假设下，近似为：
$$\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|I, \mathbf{x}_t) \simeq -\rho \nabla_{\mathbf{x}_t} \|\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\|_2^2$$
修改后的噪声预测为：$\epsilon'_\phi = \epsilon_\phi + \rho\sqrt{1-\alpha_t} \nabla_{\mathbf{x}_t} \|\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\|_2^2$

**3. 迭代循环**：执行"DDIM 反演 → 引导采样"循环，直至指导损失 $\mathcal{L}_g = \|\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\|_2^2$ 的相对变化低于阈值 $\lambda_{thr}$。

**模型架构**：仅建模 SMPL 姿态参数（144 维 6D 旋转表示），使用 3 个 MLP block，输入为噪声样本 $\mathbf{x}_t$、正弦编码的 timestep $t$ 和来自预训练回归网络的特征 $c$。图像特征骨干 $g$ 保持冻结。

**三种应用对应的 Guidance Loss**：
- **单帧拟合**：$\mathcal{L}_{repr} = \mathbf{y}_{conf} \|\Pi_K(W\mathcal{M}(\hat{\mathbf{x}}_0(\mathbf{x}_t), \beta) + \gamma) - \mathbf{y}_{kp}\|_2^2$
- **多视图精化**：$\mathcal{L}_{MV} = \sum_n \|\hat{\mathbf{x}}_{0,b}^{(n)}(\mathbf{x}_t^{(n)}) - \bar{\mathbf{x}}_{0,b}\|_2^2$（跨视图一致性）
- **时序精化**：$\mathcal{L}_{temp} = \sum_{n=2}^N \|\hat{\mathbf{x}}_0^{(n)}(\mathbf{x}_t) - \hat{\mathbf{x}}_0^{(n-1)}(\mathbf{x}_t)\|_2^2$（时序平滑）

**训练**：使用 Human3.6M、MPI-INF-3DHP、COCO、MPII 上的伪标注（SPIN 或 EFT 生成），标准 DDPM 训练损失 $\mathcal{L}_{DM}$。

## 实验与结果

**数据集**：3DPW（单帧/时序）、EMDB 1（时序）、Human3.6M（多视图）、Mannequin Challenge（多视图）。

**评估基线**：LGD、LFMM、SMPLify、ProHMR-fitting、VIBE-opt；初始回归源为 ProHMR 和 HMR 2.0。

**单帧模型拟合（Table 2）**：
- HMR 2.0 + ScoreHMR-b：3DPW PA-MPJPE = **51.1mm**（优于 HMR 2.0 的 54.3mm，提升 3.2mm）；EMDB 1 = **76.6mm**（优于 HMR 2.0 的 78.7mm）
- 唯一能提升 HMR 2.0 姿态精度的方法；SMPLify 在 HMR 2.0 上反而恶化至 60.1mm
- 全面超越所有优化基线（ProHMR-fitting 3DPW 为 55.1mm，ScoreHMR-b 为 51.1mm）

**多视图精化（Table 3）**：
- HMR 2.0 + ScoreHMR-b：H36M MPJPE = **44.7mm**（优于 HMR 2.0 的 52.8mm）；Mannequin PA-MPJPE = **79.1mm**
- ScoreHMR 持续优于 ProHMR-fitting，因扩散模型能联合优化全身姿态（包括全局朝向），而 ProHMR-fitting 仅更新身体姿态

**时序精化（Table 4）**：
- HMR 2.0 + ScoreHMR-b：3DPW PA-MPJPE = **50.5mm**，Acc Err = **11.1 mm/s²**；EMDB 1 PA-MPJPE = **75.3mm**，Acc Err = **11.9 mm/s²**
- 相比 ProHMR-fitting（Acc Err 14.1），ScoreHMR 在 3DPW 上加速误差降低 **21.3%**，EMDB 1 上降低 **40.5%**
- 全面超越 VIBE-opt 和 ProHMR-fitting

## 相关工作脉络
1. **HMR / PyMAF / PARE / HMR 2.0**（回归派）：直接从前向网络预测 SMPL 参数；ScoreHMR 不替代它们，而是作为通用精化工具叠加在任意回归输出之上。
2. **SMPLify / LGD / LFMM**（优化派）：手工设计能量函数迭代拟合；ScoreHMR 以数据驱动的 score guidance 替代传统优化，避免局部极小值和超参调优。
3. **ProHMR-fitting**（回归+优化协同）：在回归输出基础上进行优化精化；ScoreHMR 同样在此框架下工作，但用扩散 score guidance 取代了传统优化，效果更优。
4. **Diffusion Posterior Sampling / Pseudo-Inverse Guided Diffusion**（图像逆问题）：此前在图像修复、超分等 2D 任务中应用 score guidance；本文首次将其引入 3D 人体恢复这一参数化逆问题。
5. **Human Motion Diffusion Model**（运动生成扩散模型）：关注文本驱动的运动生成，非逆问题求解；ScoreHMR 解决的是从图像观测恢复 3D 参数的逆向推理问题。

## 局限性与未来方向
- **仅建模姿态参数（pose）**，未包含形状参数（shape），尽管作者提到可扩展且补充材料中有实验，但主实验中shape参数未参与精化。
- **依赖伪标注质量**，训练使用 SPIN/EFT 生成的伪 GT，标注误差可能影响扩散先验的准确性。
- **依赖预训练回归网络提供初始估计和图像特征**，无法从零开始工作。
- **2D 关键点检测质量直接影响最终结果**，在遮挡或低质量检测场景下性能受限。
- **超参数（如 $\rho$、$\lambda_{thr}$）需要适当调整**，不同场景下的鲁棒性需进一步验证。

## 研究启发与可借鉴点
1. **Score-guided diffusion 作为通用迭代精化框架**：可将此范式迁移到其他 3D 视觉逆问题，如人手网格恢复、人脸重建、物体位姿估计等，替代传统优化方法。
2. **DDIM 反演-采样循环的设计**：通过将回归估计精确反演到扩散隐空间再引导采样，实现了"以回归为起点、以扩散为先验"的精化流程，思路优雅且可复用于其他参数化模型恢复任务。
3. **模块化 guidance loss 设计**：通过替换不同的观测投影算子 $\mathcal{A}(\cdot)$ 和 loss 形式，同一扩散模型可适配单帧/多视图/时序三种场景，为多模态人体理解提供了统一框架。
4. **将扩散模型视为"学习到的参数先验"**：替代了传统方法中手工设计的正则项（如 pose prior、shape prior），避免了超参调优和局部极小问题，这一思想可推广到任意参数化模型拟合。
5. **与团队方向的结合机会**：可探索将 ScoreHMR 与团队现有的手/脸联合恢复、或动态场景人体估计工作结合，利用其时序一致性指导提升视频序列的 3D 人体恢复质量。

## 关键术语表
**ScoreHMR**：Score-Guided Human Mesh Recovery 的缩写，本文提出的基于扩散模型 score guidance 的 3D 人体网格恢复方法。
**SMPL**：Skinned Multi-Person Linear model，参数化人体模型，包含 24 维姿态参数（6D 表示共 144 维）和 10 维形状参数。
**DDIM**：Denoising Diffusion Implicit Models，确定性扩散采样方法，支持隐变量的精确反演，使回归估计能映射到扩散隐空间。
**Score Guidance**：利用观测似然的梯度 $\nabla_{\mathbf{x}} \log p(\mathbf{y}|\mathbf{x})$ 引导扩散去噪过程，使生成结果与观测数据对齐。
**PA-MPJPE**：Procrustes-Aligned Mean Per-Joint Position Error，对预测关节进行刚体对齐后的平均关节误差（mm）。
**Inverse Problem（逆问题）**：从观测数据 $\mathbf{y}$ 反推隐藏参数 $\mathbf{x}_0$ 的问题，其中正向映射 $\mathbf{y} = \mathcal{A}(\mathbf{x}_0) + \eta$ 不可逆或难以求逆。
**HMR 2.0**：Humans in 4D，基于 Transformer 的全量 3D 人体恢复 SOTA 方法，能处理异常姿态。
**ProHMR**：Probabilistic Human Mesh Recovery，结合概率建模的 HMR 方法，本文主要用作回归初始源和对比基线。

## 可复现要素
- **数据集**：Human3.6M（公开）、MPI-INF-3DHP（公开）、COCO（公开）、MPII（公开）、3DPW（公开）、EMDB（公开）、Mannequin Challenge（公开）
- **代码/权重**：项目网站 https://statho.github.io/ScoreHMR 提供代码和模型
- **关键超参**：T=1000 步扩散、6D 旋转表示、3 个 MLP block、正弦编码 timestep 注入、冻结图像特征骨干、伪 GT 来源为 SPIN/EFT；具体数值见 supplement
- **初始回归网络**：ProHMR、PARE、HMR 2.0（均公开）
