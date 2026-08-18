---
title: "Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Bl"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:12"
field: "扩散模型推理加速"
keywords: ["Diffusion Models", "Inference Acceleration", "Block Caching", "Scale-Shift Adjustment", "DPM-Solver", "Latent Diffusion"]
innovations: ["提出Block Caching技术复用去噪步间冗余层计算，实现1.5x-1.8x加速", "设计基于累积L1_rel阈值的自动缓存调度算法", "引入轻量级scale-shift adjustment校正缓存特征错位"]
benchmarks: ["COCO FID", "LDM-512", "EMU-768", "PartiPrompts", "OUI Prompts"]
---

# 论文速读：Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Block-Caching

## 一句话总结
本文提出**块缓存（Block Caching）**技术，通过分析扩散模型去噪网络内部各层块的时间变化特性，复用相邻去噪步之间冗余的层计算结果，在保持甚至提升图像质量的同时实现 **1.5×–1.8×** 推理加速。

## 研究问题与动机
1. **扩散模型推理成本高**：生成单张图像需多次前向传播应用大型 U-Net 去噪网络，延迟限制实际部署。
2. **现有加速手段 treat network as black box**：改进求解器（DDIM/DPM-Solver）和蒸馏方法主要减少去噪步数或训练小模型，但未利用网络内部冗余计算。
3. **层块输出存在显著冗余**：研究发现不同 Spatial Transformer 块输出随时间步变化平滑、步进变化小，大量计算可被复用。
4. **直接缓存引入伪影**：缓存会导致特征图与当前时间步的"期望"特征不对齐，需额外机制校正。

## 核心贡献（创新点）
1. **首次系统分析 U-Net 内部块的时间变化特性**：揭示块输出随去噪步平滑变化、不同块存在差异化变化模式、多数步进变化量极小这三个关键观察，与仅关注输入/输出图像的已有工作形成本质区别。
2. **提出 Block Caching 技术**：在残差块级别缓存上一跳的块输出供后续步复用，而非简单减少去噪步数；与减少步数的方法本质区别在于"用更多步但更少计算"换取更高细节质量。
3. **设计自动缓存调度算法**：基于累积相对 L1 变化量 $\sum \mathrm{L1}_{rel}(i,t) \leq \delta$ 自动决定每个块何时刷新缓存，无需人工设定，与手工调度或固定跳过策略本质不同。
4. **引入轻量级 Scale-Shift Adjustment**：在缓存位置添加 timestep 相关的逐通道缩放和平移参数，以蒸馏方式微调校正特征错位，避免朴素缓存产生的局部伪影，与无校正的缓存方案本质不同。

## 方法详解
1. **块变化分析**：定义相对绝对变化度量 $\mathrm{L1}_{rel}(i,t)=\frac{||C_i(x_t,s_t)-C_i(x_{t-1},s_{t-1})||_1}{||C_i(x_t,s_t)||_1}$，用于刻画块 $i$ 在步 $t$ 的输出变化幅度；通过随机 prompt 和 seed 平均得到统计规律。
2. **Block Caching 机制**：对 Spatial Transformer 块（计算最贵的部分）应用缓存；当累积变化超过阈值 $\delta$ 时刷新，否则复用之前步的计算结果 $C_i(x_{t_a},s_{t_a})$；公式约束 $\sum_{t=t_a}^{t_b-1}\mathrm{L1}_{rel}(i,t)\leq\delta<\sum_{t=t_a}^{t_b}\mathrm{L1}_{rel}(i,t)$。
3. **自动调度策略**：预先在随机 prompt 集上计算各块 $\mathrm{L1}_{rel}$ 统计量，选定 $\delta=0.5$ 生成缓存调度表（如图4所示），不同位置块具有不同缓存生命周期。
4. **Scale-Shift Adjustment**：在每个接收缓存输入的块后添加 timestep 相关的逐通道缩放 $s$ 和平移 $b$（形状 $N\times C$），通过 linear layer 从 timestep embedding 预测；采用 teacher-student 蒸馏训练：学生模型开启缓存，教师模型关闭缓存但使用学生中间步结果作为输入（避免轨迹漂移），优化 $15\text{k}$ 步，冻结原始权重。
5. **计算成本**：scale-shift 参数量极小，推理时无额外延迟；训练需 8×A100 GPU，耗时 12–48 小时（取决于模型和步数）。

## 实验与结果
1. **模型与设置**：LDM-512（900M 参数，Shutterstock 重训练）与 EMU-768（2.7B 参数）；bfloat16 精度；Nvidia A100 GPU；classifier-free guidance=5.0；求解器 DPM 和 DDIM。
2. **LDM-512 定量结果（Table 1）**：
   - 20 步 + 缓存 vs 14 步基线（同延迟）：**FID 15.95 vs 17.15**，速度提升 **1.65×**（含 SS）。
   - 50 步 + 缓存 vs 30 步基线（同延迟）：**FID 15.15 vs 17.44**，速度提升 **1.79×**。
   - 全配置（缓存+SS）甚至超越原始 20 步和 50 步基线的 FID。
3. **EMU-768 人评结果（Table 2）**：12 名标注员、1320 票，缓存方法在 DPM/DDIM 各配置下均显著优于同延迟基线（胜率 27.8%–34.7%，负率仅 17.9%–23.5%）。
4. **阈值消融**：$\delta=0.5$ 在加速比与质量间取得最优权衡（约 1.5× 加速）；$\delta$ 过大导致质量下降。
5. **ResBlock 不可缓存**：仅缓存 ResBlock 提速不足 5% 且质量劣化明显，证明缓存策略需针对高计算成本且变化平滑的块。

## 相关工作脉络
1. **Improved Solvers（DDIM, DPM-Solver）**：减少所需去噪步数，但仍 treat U-Net as black box；本文与之正交，可结合使用。
2. **Distillation（Progressive Distillation, Guidance Distillation）**：训练学生模型减少 NFE；需重新训练且可能无法处理 negative/composite prompts；本文仅微调少量额外参数，保持原始权重不变。
3. **Consistency Models / Latent Consistency Models**：单步或少步生成；需要专门训练；本文方法可应用于任意已有扩散模型无需重新训练架构。
4. **Classifier-free Guidance**：本文沿用标准 guidance=5.0，不重复提出新机制。
5. **定位差异**：本文从"网络内部分析→冗余识别→缓存复用"出发，区别于主流"步数压缩→黑盒优化"思路，开辟了推理加速的新维度。

## 局限性与未来方向
1. **仅缓存 Spatial Transformer 块**：ResBlock 因变化模式不平滑而不适用，可能还有优化空间。
2. **需要额外的 scale-shift 微调阶段**：虽参数少但仍需 12–48 小时训练，对快速迭代场景有一定门槛。
3. **调度基于统计平均**：自动调度依赖随机 prompt 集上的统计，极端 prompt 下的缓存行为未充分验证。
4. **未来方向**：探索更激进的缓存策略（如跨非相邻步缓存）、扩展至视频/3D 生成、与 solver 和蒸馏方法联合优化。

## 研究启发与可借鉴点
1. **"网络内部分析发现冗余"的研究范式**：不局限于端到端加速，而是深入模型内部结构寻找计算冗余，可迁移至 transformer、Mamba 等序列模型的推理优化。
2. **自动调度 + 轻量校正的组合策略**：先用统计方法确定"何时跳过计算"，再用极简参数（scale-shift）纠正偏差，是一种高效的"近似+修正"框架，可复用于其他迭代生成过程。
3. **teacher-student 蒸馏用于缓存对齐**：固定教师权重、仅微调附加参数的做法资源友好，可用于任何需要"修正近似计算偏差"的场景。
4. **与 solver 正交可叠加**：本文方法与 DPM/DDIM 等求解器无冲突，可与团队现有 solver 优化工作结合产生组合增益。

## 关键术语表
**Block Caching**：在扩散模型去噪过程中，对变化微小的网络块复用前一时间步的计算输出，避免冗余前向传播。
**Scale-Shift Adjustment**：在缓存块后添加 timestep 相关的逐通道缩放和平移参数，通过蒸馏微调校正缓存引入的特征错位。
**L1_rel**：相对绝对变化度量，用于量化每个时间步块输出的变化幅度，作为缓存决策的依据。
**Caching Schedule**：根据累积 L1_rel 阈值自动确定的各块缓存生命周期表，决定何时刷新缓存。
**DPM-Solver**：将去噪过程建模为 ODE 的高级求解器，可在约 10 步内生成高质量图像。
**Classifier-free Guidance**：不依赖分类器、通过条件/无条件预测联合实现引导的扩散模型采样技术。
**Spatial Transformer Block**：U-Net 中执行自注意力与交叉注意力的模块，计算成本最高且变化最平滑，是缓存的主要对象。
**ResBlock**：U-Net 中执行卷积操作的残差块，负责局部细节生成，变化模式不适合缓存。

## 可复现要素
- **数据集**：LDM-512 使用 Shutterstock 内部数据重训练；评估使用 COCO 子集（去除人脸）及 PartiPrompts/OUI Prompts；论文未提及训练集公开情况。
- **代码/权重**：项目页面 fwmb.github.io/blockcaching；论文未明确声明代码是否开源；模型权重为 Meta 内部模型。
- **关键超参**：阈值 $\delta=0.5$；classifier-free guidance=5.0；bfloat16 精度；scale-shift 训练 15k 步；加速比约 1.5×–1.8×。
