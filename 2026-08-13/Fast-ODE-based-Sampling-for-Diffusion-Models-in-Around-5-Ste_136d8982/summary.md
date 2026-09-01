---
title: "Fast-ODE-based-Sampling-for-Diffusion-Models-in-Around-5-Ste"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:54:07"
field: "扩散模型快速采样"
keywords: ["扩散模型", "快速采样", "ODE求解器", "知识蒸馏", "少步生成"]
innovations: ["基于二维子空间几何性质的学习型均值方向求解器AMED-Solver", "可插拔插件AMED-Plugin适配任意ODE求解器", "5 NFE下达到FID 6.61的CIFAR-10生成SOTA"]
benchmarks: ["CIFAR-10 32x32", "ImageNet 64x64", "FFHQ 64x64", "LSUN Bedroom 256x256", "Stable Diffusion 512x512"]
---

# 论文速读：Fast-ODE-based-Sampling-for-Diffusion-Models-in-Around-5-Steps

## 一句话总结
本文提出 AMED-Solver（近似均值方向求解器），基于采样轨迹几乎位于二维子空间的几何观察，通过知识蒸馏学习每个采样步的中间时间步和缩放因子，在仅 5 次函数评估（NFE）下实现高质量扩散模型采样，并将该方法扩展为可插拔插件 AMED-Plugin。

## 研究问题与动机
- 扩散模型采样通常需数百至数千步，计算成本高；现有快求解器（如 DPM-Solver、UniPC）将 NFE 降至 20 左右，但在极小 NFE（~5）下样本质量急剧退化。
- 现有高阶 ODE 数值方法（如 Taylor 展开、多项式外推）本质上仍会产生截断误差，无法在少步数下保持高精度。
- 蒸馏类方法（如 Consistency Models）虽可实现 1 NFE 生成，但需大量预生成数据或复杂训练流程，且样本质量无法随 NFE 单调提升。
- 单步求解器（如 DDIM、EDM、DPM-Solver-2）实现简单、无需历史梯度记录，但在小 NFE 下质量退化比多步求解器更严重。

## 核心贡献（创新点）
1. **提出 AMED-Solver**：利用采样轨迹几乎位于二维子空间的几何性质，将积分用均值定理近似，通过学习中间时刻与缩放因子消除截断误差。与数值方法本质区别：不使用预设的多项式/Taylor 近似，而是端到端学习最优梯度方向。
2. **提出 AMED-Plugin**：将 AMED 思想泛化为可插拔模块，适用于任意 ODE 求解器，仅需少量训练开销且采样开销可忽略。
3. **实验验证与 SOTA 结果**：在 CIFAR-10、ImageNet 64×64、FFHQ、LSUN Bedroom 及 Stable Diffusion 上系统验证，在 ~5 NFE 下取得求解器类方法的最佳 FID 分数。
4. **揭示采样轨迹的二维子空间几何性质**：通过 PCA 分析证明高维采样轨迹可用两个主成分解释方差，为均值方向学习方法提供理论支撑。

## 方法详解
**关键几何观察**：对 1k 条采样轨迹进行 PCA 分析，投影误差（使用两个主成分）≤ 8%，方差几乎完全由两个主成分解释。高维空间（如 3×256×256=196608 维）中的轨迹动态几乎可由二维子空间描述。

**核心公式**：基于均值定理的思想，寻找中间时间步 $s_n \in (t_n, t_{n+1})$ 和缩放因子 $c_n$，使得：
$$\epsilon_\theta(\mathbf{x}_{s_n}, s_n) \approx \frac{c_n}{t_n - t_{n+1}} \int_{t_{n+1}}^{t_n} \epsilon_\theta(\mathbf{x}_t, t) dt$$
从而得到单步更新公式：
$$\mathbf{x}_{t_n} \approx \mathbf{x}_{t_{n+1}} + c_n(t_n - t_{n+1})\epsilon_\theta(\mathbf{x}_{s_n}, s_n)$$
当 $s_n = \sqrt{t_n t_{n+1}}$ 且 $c_n = 1$ 时等价于 DPM-Solver-2。

**AMED Predictor（$g_\phi$）**：极轻量网络（仅 9k 参数），输入为 U-Net 瓶颈特征 $\mathbf{h}_{t_{n+1}}$ 及时刻 $t_{n+1}, t_n$，输出 $r_n$（控制 $s_n = t_n^{r_n} t_{n+1}^{1-r_n}$）和 $c_n$。结构为：通道平均池化 → 全连接层 → 与时间嵌入拼接 → 全连接层 + sigmoid。

**训练方式**：基于知识蒸馏，从当前步前向传播生成学生轨迹和教师轨迹（教师使用更多 NFE），沿时间反向逐步更新 $\phi$，总损失为两轨迹间 L2 距离。训练仅需 2-8 分钟（小数据集）至 1-3 小时（大场景）。

**Analytical First Step (AFS)**：利用初始噪声方向与梯度方向近似一致的性质，在第一步直接使用 $\mathbf{x}_{t_N}$ 作为方向，节省一次 NFE 评估。

**AMED-Plugin**：对于已有 ODE 求解器，将预测的 $(s_n, c_n)$ 融入中间步计算，总 NFE 变为 $2(N-1)$（应用 AFS 后为奇数）。此外，在小分辨率数据集上还可选训练时间缩放因子 $\{a_n\}$ 以扩展解空间。

## 实验与结果
- **数据集**：CIFAR-10 32×32、FFHQ 64×64、ImageNet 64×64、LSUN Bedroom 256×256、Stable Diffusion（512×512）。
- **基线方法**：DDIM、EDM、DPM-Solver-2、DPM-Solver++(3M)、UniPC、iPNDM（order 4）。
- **评测指标**：FID（50k 图像，Stable Diffusion 用 30k 图像+MS-COCO prompt）。
- **关键结果（5 NFE）**：
  - CIFAR-10：AMED-Plugin 于 iPNDM 达 **FID=6.61**（对比 iPNDM 13.59，提升 6.98；对比 DPM-Solver++(3M) 24.97，提升显著）
  - ImageNet 64×64：**FID=10.74**（AMED-Solver），AMED-Plugin 为 13.83（iPNDM 为 17.17）
  - FFHQ 64×64：**FID=12.49**（AMED-Plugin on iPNDM，提升 4.68）
  - LSUN Bedroom 256×256：**FID=13.20**（AMED-Solver）
  - Stable Diffusion（NFE 16/20）：FID 13.96/13.24，优于 DPM-Solver++(2M) 的 14.84/14.58
- **最强结果**：CIFAR-10 32×32 在 5 NFE 下 FID=6.61，为当前求解器类方法最优。

## 相关工作脉络
1. **DDIM**（Song et al., 2021）：将扩散过程视为确定性强度的 ODE，采用 Euler 离散化；AMED-Solver 在形式上扩展了其单步框架，但通过学习型系数替代固定 Euler 步长。
2. **DPM-Solver 系列**（Lu et al., 2022/2023）：基于 Taylor 展开的高阶 ODE 求解器（DPM-Solver-1/2/3 及 DPM-Solver++）；AMED-Solver 摒弃了预设多项式近似，改为学习最优方向和位置，在 5 NFE 下超越其性能。
3. **PNDM / iPNDM**（Liu et al., 2022; Zhang & Chen, 2023）：多步求解器，利用历史梯度信息；AMED-Plugin 以极低额外开销显著提升了 iPNDM 在小 NFE 下的表现，说明单步+学习型修正可比肩多步方法。
4. **EDM**（Karras et al., 2022）：系统分析扩散模型设计空间并提出改进 Euler 方法；本文在其框架基础上引入学习型修正。
5. **Consistency Models / 蒸馏方法**（Salimans & Ho, 2022; Song et al., 2023）：构建噪声到数据的直接映射；AMED 仍保持 ODE 求解性质，无需重新训练基础模型，且支持 NFE 连续可调的质量提升。
6. **Timestep Aligner**（Xia et al., 2023，同期工作）：也关注时间步对齐问题；本文 AMED-Plugin 在此基础上同时学习方向和缩放，扩展性更强。

## 局限性与未来方向
- **时间调度敏感性**：现有实验表明固定时间调度（Uniform/Polynomial/logSNR）难以在所有场景通用，AMED-Plugin 仅部分缓解该问题。
- **教师求解器选择的影响**：消融实验显示学生与教师求解器越相似效果越好，但未深入研究最优教师配置。
- **大分辨率下的多步变体限制**：在 LSUN Bedroom 256×256 上 AFS 与多步求解器配合效果不佳，需进一步分析。
- **未来方向**：探索更自适应的时间调度设计；将 AMED 思想拓展至条件生成、视频生成及 3D 扩散模型；研究无需蒸馏的训练方式。

## 研究启发与可借鉴点
1. **几何结构先验驱动方法设计**：利用采样轨迹的低秩/低维子空间性质指导求解器设计，这一思路可迁移至其他迭代生成模型（如流模型、Rectified Flow）。
2. **知识蒸馏 + 数值求解器的融合范式**：将蒸馏用于学习 ODE 求解器的超参数（时间步、缩放因子）而非直接预测样本，大幅降低训练成本，适用于各类扩散采样场景。
3. **轻量级预测网络嵌入现有框架**：AMED Predictor 仅 9k 参数即可带来显著增益，提示研究者可在不改变主干网络的前提下，通过小模块注入先验知识。
4. **AFS 技巧的工程价值**：利用首步梯度与噪声方向近似一致的几何性质节省 NFE，该技巧可复用于其他少步采样场景。
5. **可与团队方向结合的机会**：若团队关注视频生成或长序列扩散，AMED 的学习型时间步预测机制可有效应对多帧间的轨迹变化；同时可探索将 AMED 与 Rectified Flow 等新兴框架结合。

## 关键术语表
- **PF-ODE（Probability Flow ODE）**：与扩散 SDE 共享边缘分布的概率流常微分方程，常用于确定性采样。
- **NFE（Number of Function Evaluations）**：采样过程中 U-Net 评估次数，直接反映采样速度。
- **AMED-Solver**：本文提出的单步 ODE 求解器，通过学习中间时间和缩放因子逼近均值方向。
- **AMED-Plugin**：AMED 思想的插件化扩展，可应用于任意 ODE 求解器。
- **AFS（Analytical First Step）**：利用初始噪声与梯度方向近似一致的性质，省去第一步 NFE 的技巧。
- **FID（Fréchet Inception Distance）**：衡量生成图像与真实图像分布差异的常用指标，越低越好。
- **知识蒸馏（Knowledge Distillation）**：用高 NFE 教师轨迹指导低 NFE 学生轨迹的训练策略。
- **logSNR 时间调度**：以对数信噪比为均匀间隔的采样时刻安排方式。

## 可复现要素
- **数据集**：CIFAR-10、FFHQ 64×64、ImageNet 64×64、LSUN Bedroom；公开数据集。
- **代码**：已开源，见 https://github.com/zjupi/diff-sampler
- **权重**：使用预训练的 EDM / Stable Diffusion v1.4 权重；AMED Predictor 权重随代码一同提供。
- **关键超参**：AMED Predictor 参数量 9k；训练图像数 10k； polynomial schedule $\rho=7$；M（教师内部步数）=1（DPM-Solver-2）或 2（其他）；距离度量 L2 norm。
- **硬件**：单张 NVIDIA A100 GPU。
