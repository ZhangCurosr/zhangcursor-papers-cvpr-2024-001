---
title: "On-Exact-Inversion-of-DPM-Solvers"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:15:07"
---

# 论文速读：On-Exact-Inversion-of-DPM-Solvers

## 一句话总结
本文针对快速DPM-Solvers（一阶DDIM与高阶DPM-Solver++）难以精确反演初始噪声的难题，提出了基于隐式Backward Euler方法的精确反演算法，并结合高阶项近似与解码器梯度优化，显著降低了图像/噪声重建误差，支撑了水印分类与保背景图像编辑等下游应用。

## 研究问题与动机
- **快采样导致反演失效**：DPM-Solvers将去噪步数从1000压缩至10-50步，破坏了朴素DDIM反演所依赖的“相邻步噪声估计几乎相同”的假设，导致重建畸变。
- **现有精确反演方法局限**：基于不动点迭代（FPI）的方法（如Pan et al. [21]）在classifier-free guidance (CFG) > 1时因算子扩张性而发散；而EDICT等可逆采样方法仅支持特定生成器，无法反演标准DDIM/DPM-Solver输出。
- **高阶多步法不可逆**：DPM-Solver++等线性多步法依赖历史状态，反向过程中未来状态未知，缺乏系统的反演理论框架。
- **潜空间信息瓶颈**：LDM使用自编码器压缩图像，直接用编码器反推潜码会引入不可消除的重建误差下界，制约下游编辑保真度。

## 核心贡献（创新点）
- **提出基于Backward Euler的DDIM精确反演算法（Alg. 1）**：将反演建模为隐式方程求解，通过梯度下降或前向步法迭代优化，从根本上避免FPI在大CFG下的收敛失败问题。
- **提出高阶DPM-Solvers的高阶项近似反演算法（Alg. 2）**：针对多步法历史状态缺失问题，用细粒度朴素反演粗估历史潜变量，再将高阶项视为常数代入Backward Euler求解，首次实现10步快采样器的可逆性。
- **设计解码器精确反演模块**：以编码器输出为初值，通过梯度下降最小化像素空间重构损失，打破传统`D(E(x))`的信息损失下界。
- **应用层验证闭环**：系统证明方法在像素空间DPM与LDM上均能显著降低NMSE，并成功拓展至Tree-Ring水印检测/分类、无原始轨迹的保背景编辑等任务。

## 方法详解
- **一阶DDIM反演（Algorithm 1）**：初始化采用朴素DDIM反演，随后对每一步执行Backward Euler隐式求解：$z'_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \hat{z}_{t_{i-1}} - \alpha_{t_i}(e^{-h_i} - 1) z_\theta(\hat{z}_{t_{i-1}}, t_{i-1})$。通过梯度下降 $\nabla_{\hat{z}_{t_{i-1}}} \|\hat{z}_{t_i} - z'_{t_i}\|_2^2$ 或前向步法 $\hat{z}_{t_{i-1}} \leftarrow \hat{z}_{t_{i-1}} - \rho(z'_{t_i} - \hat{z}_{t_i})$ 迭代更新，直至收敛。理论证明表明FPI要求去噪网络满足严格Lipschitz条件，而大CFG会破坏该条件；梯度下降/前向步法可通过降低学习率保证收敛稳定性。
- **高阶DPM-Solvers反演（Algorithm 2）**：对于DPM-Solver++(2M)，高阶项依赖未知的 $x_{t_{i-2}}$。方法先用细粒度朴素DDIM反演估计 $\hat{y}_{t_{i-1}}, \hat{y}_{t_{i-2}}$ 作为替代，然后将高阶项近似为 $\frac{z_\theta(\hat{y}_{t_{i-1}}, t_{i-1}) - z_\theta(\hat{y}_{t_{i-2}}, t_{i-2})}{2r_i}$ 并固定为常数，再代入Backward Euler迭代求解主潜变量。
- **Decoder Inversion**：针对LDM，定义 $\mathcal{D}^\dagger(x) = \arg\min_z \|x - \mathcal{D}(z)\|_2^2$，以 $\mathcal{E}(x_0)$ 为起点执行梯度下降，获得精确潜码后再进入扩散反演流程。
- **统一框架**：将采样过程视为ODE离散轨迹，反演过程视为对应的逆时隐式求解，两者在数学上严格对偶（见Table 1）。

## 实验与结果
- **数据集与设置**：ImageNet64像素空间DPM、标准LDM；采样步数DDIM 50步 / DPM-Solver++(2M) 10步；CFG分别设为1.0（无条件）与3.0（条件）；评估指标为噪声/图像NMSE、LPIPS、SSIM。
- **重建结果**：Alg. 1与Alg. 2在像素空间与潜空间上均显著优于朴素DDIM反演与FPI。朴素反演即使增至1000步也会误差饱和，而隐式方法持续收敛；在CFG=3.0的LDM上FPI噪声重建表现极差，本文方法仍保持低NMSE。
- **水印应用**：在Tree-Ring Watermark任务中，Alg. 2配合10步DPM-Solver++(2M)实现最佳分类性能，使原本仅能检测的水印升级为可区分的三类版权追踪。
- **保背景编辑**：在Patashnik et al. [23] 300张图像实验中（CFG=7.5），Alg. 1估计轨迹后的背景NMSE为12.8±2.0 (×10^-3)，显著优于朴素反演（30.4±4.8）与加解码器朴素反演（18.4±2.0），逼近Oracle（11.0±1.7）。
- **效率**：LDM环境下平均运行时间：朴素50步3s / 1000步59s，FPI 32s，Alg. 1 79s，Alg. 2 159s，计算开销有所上升但仍属可用范围。

## 相关工作脉络
- **Naïve DDIM Inversion**：显式逆推，假设 $\epsilon_\theta(x_{t_i}) \approx \epsilon_\theta(x_{t_{i-1}})$；本文指出其本质是Backward Euler的一阶近似，仅适用于大步数低精度场景。
- **EDICT / Bi-directional Integration (Wallace et al. [34], Zhang et al. [40])**：通过修改采样器结构（可逆耦合/双向积分）保证可逆性；局限在于无法反演标准DDIM/DPM-Solver生成的图像。
- **AIDIE / FPI-based Inversion (Pan et al. [21])**：首个针对标准DDIM图像的精确反演，但依赖不动点迭代，CFG>1时因Lipschitz常数超限而失效；本文方法在数值稳定性与高阶泛化上全面超越。
- **DPM-Solver++ (Lu et al. [14, 15])**：基于指数积分器与线性多步法的快采样器；本文首次建立其可逆性理论，填补了“10步快采样可精确反演”的研究空白。
- **GAN/Normalizing Flow Inversion**：借鉴StyleGAN中潜空间梯度优化的思想，将其迁移至扩散模型解码器对齐，打破VQVAE/AE的固有误差下限。

## 局限性与未来方向
- 计算耗时显著高于朴素反演（约10-30倍），在高实时性场景受限。
- 假设已知生成prompt（尤其LDM），未解决提示词与初始噪声的联合估计问题。
- 在加速调度器与LDM中的“精确”实为“near-exact”，受限于离散化误差与自编码器信息损失。
- 未来可探索自适应步长/迭代次数调度、prompt-噪声联合反演、以及向SDE采样器的理论推广。

## 研究启发与可借鉴点
- **隐式迭代替代显式逆推**：将扩散反演建模为隐式方程求解（Backward Euler + 梯度下降）可有效规避大CFG下的发散问题，该数值稳定性策略可迁移
