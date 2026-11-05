---
title: "Fast-ODE-based-Sampling-for-Diffusion-Models-in-Around-5-Ste"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:51:18"
field: "扩散模型加速采样"
keywords: ["扩散模型", "快速采样", "ODE求解器", "知识蒸馏", "低NFE生成"]
innovations: ["基于二维子空间几何性质的均值方向学习求解器", "可即插即用的AME-Plugin插件框架"]
benchmarks: ["CIFAR-10", "ImageNet 64x64", "FFHQ 64x64", "LSUN Bedroom 256x256", "Stable Diffusion"]
---

# 论文速读：Fast-ODE-based-Sampling-for-Diffusion-Models-in-Around-5-Steps

## 一句话总结
本文提出 AMED-Solver 及其插件 AMED-Plugin，利用"采样轨迹几乎位于二维子空间"的几何性质，通过知识蒸馏学习均值方向与缩放因子，消除数值离散误差，在仅 5 次函数评估（NFE）下实现扩散模型的高质量生成。

## 研究问题与动机
- 扩散模型采样本质是求解概率流 ODE，现有高阶数值求解器（如 DPM-Solver、UniPC）在极小 NFE（≈5）下截断误差显著，导致样本质量急剧下降。
- 单步求解器（如 DDIM、EDM）实现简单但同样受限于固定梯度方向，NFE 减少时 FID 恶化严重；多步求解器虽性能更好，但需记录历史梯度，推理开销更大。
- 现有蒸馏方法虽能实现 NFE=1，但需大量预生成数据或昂贵训练成本，且无法保证随 NFE 增加质量提升，也难以进行似然评估。
- 作者观察到采样轨迹在嵌入高维空间的二维子空间中近似线性，启发利用均值定理直接学习最优中间时刻与梯度缩放，从而"绕过"数值离散误差。

## 核心贡献（创新点）
- **AMED-Solver**：基于轨迹二维子空间性质的单步 ODE 求解器，通过蒸馏学习中间时刻 $s_n$ 与缩放因子 $c_n$，使每步梯度直接指向真实解方向，从设计上消除截断误差。与 DPM-Solver 固定启发式规则的本质区别在于系数由数据驱动自适应学习。
- **AMED-Plugin**：将 AMED 思想泛化为插件框架，可附加于任意已有 ODE 求解器（如 iPNDM、DPM-Solver++），以极小训练开销（9K 参数、数分钟至数小时）显著提升其极低 NFE 下的性能。与直接修改求解器公式的本质区别在于无需重写原有求解逻辑。
- **几何性质验证**：首次在 CIFAR-10、FFHQ、ImageNet、LSUN Bedroom 等多数据集上系统验证采样轨迹的 PCA 相对投影误差≤8%、方差可由前两个主成分完全解释，为均值方向学习的可行性提供理论依据。

## 方法详解
- **二维子空间观察**：对 1k 条采样轨迹做 PCA，结果表明即使在高维图像空间（3×32×32=3072 至 3×256×256=196608），轨迹动态几乎可由两个主成分描述，投影误差不超过 8%。
- **均值定理近似**：将 PF-ODE 的精确解 $\mathbf{x}_{t_n} = \mathbf{x}_{t_{n+1}} + \int_{t_{n+1}}^{t_n} \epsilon_\theta(\mathbf{x}_t, t) dt$ 近似为 $\mathbf{x}_{t_n} \approx \mathbf{x}_{t_{n+1}} + c_n(t_n - t_{n+1}) \epsilon_\theta(\mathbf{x}_{s_n}, s_n)$，其中 $s_n \in (t_n, t_{n+1})$ 为中间时刻，$c_n$ 为缩放因子，二者由 AMED 预测器 $g_\phi$ 学习。
- **预测器设计**：$g_\phi$ 是一个仅含 9K 参数的浅层 MLP，输入为 U-Net 瓶颈特征 $\mathbf{h}_{t_{n+1}}$ 及当前/下一时刻 $t_{n+1}, t_n$，输出 $r_n$（通过 $s_n = t_n^{r_n} t_{n+1}^{1-r_n}$ 计算中间时刻）和 $c_n$（经 sigmoid 压缩）。
- **蒸馏训练**：使用 NFE 翻倍的教师轨迹 $\{\mathbf{y}_{t_n}\}$ 与 NFE 减半的学生轨迹 $\{\mathbf{x}_{t_n}\}$ 进行知识蒸馏，沿时间反向逐步传播梯度（$N-1$ 次 backprop），损失函数为 L2 距离 $d(\mathbf{x}_{t_n}, \mathbf{y}_{t_n})$。
- **解析第一步（AFS）**：利用大 $t_N$ 时梯度方向与 $\mathbf{x}_{t_N}$ 几乎一致的性质，第一步直接使用 $\mathbf{x}_{t_N}$ 作为方向，省去一次 NFE，对低分辨率数据集几乎无质量损失。
- **时间缩放扩展**：参考 TiMA 工作，对小分辨率数据集额外学习时间缩放因子 $\{a_n\}$，用 $\epsilon_\theta(\mathbf{x}_{s_n}, a_n s_n)$ 替代 $\epsilon_\theta(\mathbf{x}_{s_n}, s_n)$，进一步扩展解空间。

## 实验与结果
- **数据集**：CIFAR-10 32×32、FFHQ 64×64、ImageNet 64×64、LSUN Bedroom 256×256 及 Stable Diffusion 512×512。
- **评估指标**：FID（50k 图像）；Stable Diffusion 使用 30k MS-COCO prompt。
- **最强结果**：
  - **CIFAR-10 5 NFE**：AMED-Plugin on iPNDM 达 FID=**6.61**，相较 iPNDM baseline（13.59）提升 **6.98**；AMED-Solver 达 7.59，超越所有单步与多步求解器。
  - **ImageNet 64×64 5 NFE**：AMED-Plugin on iPNDM 达 FID=**10.74**，相较 baseline（17.17）提升 **6.43**。
  - **FFHQ 64×64 5 NFE**：AMED-Plugin on iPNDM 达 FID=**13.20**，相较 baseline（18.99）提升 **5.79**。
  - **Stable Diffusion 512×512 12 NFE**：AMED-Plugin on DPM-Solver++(2M) 达 FID=**14.84**，相较 baseline（15.99）提升 **1.15**。
- **消融结论**：教师求解器应与学生在采样过程上相似时效果最佳；AMED-Plugin 对不同时间调度均稳定提升；教师求解器选择上 EDM 配合 DPM-Solver-2 学生效果最佳。

## 相关工作脉络
- **DDIM [43]**：一阶欧拉离散化 PF-ODE 的单步求解器，是本工作对比的基线，AMED-Solver 在 5 NFE 下 FID 从 49.66 降至 7.59，展现巨大优势。
- **EDM [18]**：带缩放的高阶单步求解器，在大 NFE 下表现优异但在极小 NFE 下退化严重（5 NFE FID=97.67），AMED-Solver 通过自适应方向学习弥补此缺陷。
- **DPM-Solver / DPM-Solver-2 [25, 26]**：泰勒展开构建的单步/多步高阶求解器，AMED-Plugin 可无缝集成至 DPM-Solver++(3M) 上，5 NFE FID 从 24.97 降至 7.68（CIFAR-10）。
- **iPNDM [23, 51]**：改进型伪数值多步求解器，在低 NFE 下表现最优，AMED-Plugin 在此基础上进一步提升，体现插件方案的通用性。
- **UniPC [52]**：统一预测-校正框架，AME-Plugin 同样可对其有效增强，表明该方法独立于求解器具体结构。
- **TiMA [49]**：同时期提出的时间步对齐方法，AMED-Plugin 与之结合可额外学习时间缩放因子 $\{a_n\}$，进一步扩展解空间。

## 局限性与未来方向
- **时间调度敏感性**：不同求解器对时间调度偏好差异大，且无单一固定调度适用于所有 NFE 情况，AMED-Plugin 仅部分缓解而非根本解决该问题。
- **多步求解器在大分辨率下的限制**：在 LSUN Bedroom 256×256 上使用 AFS 对多步求解器效果不佳，说明插件设计在不同分辨率场景下需针对性调整。
- **未来方向**：需更深入理解采样轨迹几何形状以设计自适应时间调度；探索更高维子空间利用或非线性流形建模；考虑向视频生成等时序任务扩展。

## 研究启发与可借鉴点
- **几何先验驱动算法设计**：利用轨迹的隐式低维结构（二维子空间）替代传统数值误差分析，为其他迭代生成模型（如流匹配、一致性模型）的快速采样提供了新的设计范式。
- **浅层蒸馏预测器架构**：仅 9K 参数即可显著提升下游求解器性能，证明"轻量辅助网络 + 知识蒸馏"在采样加速任务中的高效性，可迁移至其他生成模型加速场景。
- **插件化通用框架**：AMED-Plugin 无需修改原求解器代码，仅需额外预测中间时刻与缩放因子，这种"即插即用"思路便于与现有生态（如 Diffusers、ComfyUI）集成。
- **反向时间传播训练策略**：沿时间从 $t_{N-1}$ 到 $t_1$ 逐步 backprop 的离线训练方式，避免了端到端训练的不稳定性，值得在序列决策任务中借鉴。

## 关键术语表
- **PF-ODE（概率流常微分方程）**：与反向 SDE 具有相同边缘分布的确定性 ODE，用于替代随机采样过程以提高效率。
- **NFE（Number of Function Evaluations）**：采样过程中 score 网络（U-Net）的前向评估次数，直接决定生成速度。
- **AMED-Solver**：本文提出的单步 ODE 求解器，通过学习均值方向消除离散化截断误差。
- **AMED-Plugin**：将 AMED 思想泛化为插件，可附加于任意现有 ODE 求解器以提升其极低 NFE 性能。
- **解析第一步（AFS）**：利用大噪声时刻梯度方向与采样点方向一致的几何性质，省去第一次函数评估。
- **知识蒸馏**：训练轻量预测器以模仿高阶/更多步数教师轨迹的采样行为。
- **iPNDM**：改进型伪数值方法求解器，在多项多步求解器中表现最优，常作为 AMED-Plugin 的宿主。
- **logSNR 调度**：按对数信噪比均匀分配时间步的调度策略，优于均匀时间调度。

## 可复现要素
- **数据集**：CIFAR-10、FFHQ、ImageNet 64×64、LSUN Bedroom、MS-COCO（Stable Diffusion 评估）；CIFAR-10、FFHQ、ImageNet、LSUN Bedroom 均公开可用。
- **代码**：已开源，地址 https://github.com/zjupi/diff-sampler。
- **预训练权重**：论文使用 EDM [18] 与 Stable Diffusion [34] 官方权重，已公开。
- **关键超参**：AMED 预测器参数量 9K；训练数据 10k 图像；训练时间 2-8 分钟（CIFAR-10）至 1-3 小时（LSUN Bedroom），单卡 A100；教师 NFE 为学生 2 倍；L2 距离损失；多项式调度 $\rho=7$（DPM-Solver++ 和 UniPC 使用 logSNR）。
