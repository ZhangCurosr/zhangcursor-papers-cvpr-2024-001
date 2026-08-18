---
title: "Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Bl"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:23:08"
field: "扩散模型推理加速"
keywords: ["Diffusion Models", "Inference Acceleration", "Block Caching", "Latent Diffusion", "Image Generation"]
innovations: ["提出 Block Caching 技术复用 SpatialTransformer 块的冗余计算，实现 1.5x-1.8x 加速", "设计 Scale-Shift Adjustment 机制通过蒸馏微调消除缓存伪影", "基于 L1_rel 自动推导 per-block 缓存调度策略"]
benchmarks: ["COCO subset (face-removed)", "OUI Prompts", "PartiPrompts"]
---

# 论文速读：Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Block-Caching

## 一句话总结
本文提出 Block Caching 技术，通过复用扩散模型去噪过程中空间 Transformer 块的冗余计算，实现 1.5×-1.8× 推理加速；配合轻量级 scale-shift 调整机制，在相同计算预算下可执行更多去噪步数，获得比朴素减少步数更高的图像质量。

## 研究问题与动机
- **扩散模型推理成本高**：图像到图像网络需多次前向传播逐步从噪声细化图像，延迟制约大规模应用。
- **现有加速方法将网络视为黑盒**：改进求解器（DPM、DDIM）和蒸馏技术主要关注步数压缩，未挖掘 U-Net 内部冗余。
- **层输出变化具有时变平滑性**：作者观察到不同 block 的输出在去噪过程中变化平滑、存在 distinct pattern，且步间变化量往往很小，存在大量可复用计算。
- **残差结构支持细粒度缓存**：U-Net 中各 block 基于残差连接设计，可在 block 级别缓存输出而不破坏信息流。

## 核心贡献（创新点）
- **提出 Block Caching 机制**：在扩散模型推理中缓存空间 Transformer 块的输出并在后续步复用，直接减少冗余计算。
- **设计 Scale-Shift Adjustment**：针对缓存导致的特征不对齐，为每个接收缓存输入的 block 添加 timestep 依赖的缩放与偏移参数，用蒸馏式微调消除伪影。
- **提出自动缓存调度策略**：基于相对 L1 变化量 $\mathrm{L1}_{\mathrm{rel}}$ 自动计算每个 block 的缓存时机与刷新阈值 δ，无需人工设计。
- **在两种主流架构上验证通用性**：在 LDM-512（900M 参数）和 EMU-768（2.7B 参数）上均实现 1.5×-1.8× 加速，且 FID 和人类评估均优于等延迟基线。

## 方法详解
**分析观察**：定义 block $i$ 在第 $t$ 步的相对绝对变化量：
$$
\mathrm{L1}_{\mathrm{rel}}(i, t) = \frac{||C_i(x_t, s_t) - C_i(x_{t-1}, s_{t-1})||_1}{||C_i(x_t, s_t)||_1}
$$
其中 $C_i$ 为 block 的计算部分（不含残差）。观察到三个现象：(1) block 输出随时间平滑变化；(2) 不同位置 block 呈现不同的变化模式（如高层/低分辨率 block 在去噪末期变化更大，深层 block 在初期变化更大）；(3) 大多数步间变化量很小。

**Block Caching**：对 SpatialTransformer block 进行缓存，当累积变化 $\sum_{t=t_a}^{t_b-1} \mathrm{L1}_{\mathrm{rel}}(i, t) \leq \delta$ 时复用先前缓存输出，超过阈值 δ 后重新计算。δ 控制缓存激进程度。

**Scale-Shift Adjustment**：在缓存 block 的输出上加 timestep 依赖的逐通道 scale 和 shift 参数（形状 $N \times C$），由 timestep embedding 预测。优化时将带缓存模型作为 student、不带缓存模型作为 teacher，student 轨迹输入 teacher 获取每步中间结果作为蒸馏目标，仅优化 scale-shift 参数，冻结原模型权重。

## 实验与结果
- **数据集**：COCO subset（去除人脸）用于 FID 评估；OUI Prompts 和 PartiPrompts 用于人类评估。
- **基线模型**：LDM-512（900M）和 EMU-768（2.7B），求解器 DPM 和 DDIM。
- **LDM-512 定量结果**（Tab. 1）：DPM 20步缓存+FID 15.95，比同速度基线（14步，FID 18.67）提升显著，速度 1.65×；50步缓存+FID 15.18，速度 1.82×。DDIM 下类似趋势。
- **人类评估**（Tab. 2）：12位独立标注员，1320票，缓存方案在所有配置下优于等延迟基线，胜率最高达 54.3%（DPM 50步）。
- **最佳阈值**：δ = 0.5 在加速与画质间取得最优平衡，可实现 1.5× 加速且 FID 提升。
- **ResBlock 不可缓存**：缓存 ResBlock 仅获 5% 加速但质量显著下降，因其负责局部细节生成。

## 相关工作脉络
- **Improved Solvers（DPM、DDIM）**：通过改进采样算法减少所需步数，但仍将网络视为黑盒；本文方法与求解器正交，可结合使用。
- **Distillation（Progressive Distillation、Guidance Distillation）**：通过知识蒸馏压缩模型或步数；本文方法无需重新训练完整模型，仅需微调少量附加参数。
- **Consistency Models / Latent Consistency Models**：single-step 生成方法；本文方法保持原模型完整性，不改变生成轨迹本质。
- **Classifier-free Guidance**：论文使用 guidance=5.0，属标准配置，非本文贡献。
- **U-Net 架构分析**：之前工作多关注整体性能，本文首次细粒度分析 per-block 时变行为模式。

## 局限性与未来方向
- **仅缓存 SpatialTransformer，未缓存 ResBlock**：ResBlock 对局部细节至关重要，缓存效果差，未来可探索其他加速 ResBlock 的方式。
- **scale-shift 调优耗时**：15k 迭代在 8×A100 上需 12-48 小时，对快速实验构成一定门槛。
- **阈值 δ 需手动调参**：虽自动调度但仍依赖 δ 选择，不同模型/分辨率可能需要不同最优值。
- **仅验证 LDM 和 EMU**：未在其他架构（如 DiT、PixArt）上广泛测试。

## 研究启发与可借鉴点
- **内部分层分析思维**：从 per-block 时变行为挖掘冗余，而非仅调整外部求解器，为扩散模型加速提供新视角。
- **蒸馏式微调技巧**：student-teacher 同权重共享、仅优化少量附加参数的设计，兼顾效率与兼容性，可迁移至其他模型压缩场景。
- **缓存调度与质量权衡量化**：用累积 L1 变化作为缓存刷新判据，提供了可量化的质量-速度权衡控制方法。
- **与求解器正交**：Block Caching 可与任何现有 solver 结合，为研究者提供即插即用的加速组件。

## 关键术语表
- **Block Caching**：在扩散模型去噪过程中，复用先前步骤中 SpatialTransformer block 的输出以跳过冗余计算的技术。
- **Scale-Shift Adjustment**：为缓存输入添加 timestep 依赖的逐通道缩放和平移参数，通过蒸馏微调消除缓存引入的伪影。
- **L1_rel**：block 输出的相对 L1 变化量，用于量化相邻步间特征变化程度，作为缓存调度依据。
- **Caching Schedule**：基于阈值 δ 自动确定的每个 block 何时刷新缓存、何时复用的调度策略。
- **SpatialTransformer Block**：U-Net 中包含自注意力与交叉注意力的计算块，是主要计算开销来源，也是本文缓存对象。
- **DPM-Solver**：将去噪过程建模为 ODE 的高阶求解器，可在约 10 步内生成高质量图像。
- **Classifier-Free Guidance**：无需分类器的条件引导技术，通过联合训练条件与无条件分支实现生成控制。

## 可复现要素
- **数据集**：COCO subset（FID 评估）、OUI Prompts、PartiPrompts（人类评估）——论文未明确公开数据集链接，但 COCO 为标准公开数据集。
- **代码/权重**：项目页面 fwmb.github.io/blockcaching，论文未明确声明 GitHub 仓库。
- **关键超参**：δ = 0.5（最佳阈值）；guidance = 5.0；bfloat16 精度；scale-shift 优化 15k 迭代，8×A100。
- **实验硬件**：Nvidia A100 GPU。
