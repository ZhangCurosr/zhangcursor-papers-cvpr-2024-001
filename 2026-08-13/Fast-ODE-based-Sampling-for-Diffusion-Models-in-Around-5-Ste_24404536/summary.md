---
title: "Fast-ODE-based-Sampling-for-Diffusion-Models-in-Around-5-Ste"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:54:36"
field: "扩散模型快速采样"
keywords: ["diffusion models", "ODE solvers", "fast sampling", "NFE reduction", "image generation", "knowledge distillation"]
innovations: ["提出AMED-Solver，利用轨迹二维子空间性质学习预测平均方向，显著降低极小NFE下的截断误差", "设计AMED-Plugin插件，可无侵入式提升任意现有ODE求解器在低步数下的采样质量", "发现并验证采样轨迹近似位于二维子空间的几何规律，为高效求解器设计提供新视角"]
benchmarks: ["CIFAR-10 32x32", "ImageNet 64x64", "FFHQ 64x64", "LSUN Bedroom 256x256", "Stable Diffusion 512x512"]
---

# 论文速读：Fast ODE-based Sampling for Diffusion Models in Around 5 Steps

## 一句话总结
论文提出**AMED-Solver**，一种单步ODE求解器，利用采样轨迹近似位于二维子空间的几何性质，通过均值定理学习预测中间时间步与梯度缩放因子，大幅降低极小NFE下的离散化误差。该方法可封装为**AMED-Plugin**插件增强任意现有ODE求解器，在NFE≈5时达到Solver-based方法的最优性能。

## 研究问题与动机
- **扩散模型采样速度瓶颈**：扩散模型需多次迭代去噪，传统方法依赖1000步左右，实际应用受限。
- **现有快速求解器在极小NFE下质量骤降**：单步求解器（如DDIM、EDM）和高阶多步求解器（如DPM‑Solver‑2、UniPC）在NFE≈5时截断误差显著，导致FID急剧恶化。
- **蒸馏类方法成本高且缺乏灵活性**：一步到一步映射的蒸馏方法（如Consistency Models）需大量预生成图像或修改训练流程，且增加NFE未必提升质量。
- **求解器设计仍未充分挖掘采样轨迹的几何结构**：现有数值方法（如梯形法则、泰勒展开）基于固定公式，未能自适应调整中间步位置与梯度方向。

## 核心贡献（创新点）
1. **提出AMED‑Solver单步ODE求解器**：通过知识蒸馏训练浅层预测器，学习每一步的中间时间步 \(s_n\) 与缩放因子 \(c_n\)，直接逼近积分均值方向，消除截断误差。
2. **发现采样轨迹的二维子空间几何性质**：对多条采样轨迹进行PCA分析，证明轨迹变异主要由前两个主成分解释（相对投影误差≤8%），为均值定理的应用提供理论依据。
3. **设计AMED‑Plugin插件框架**：将AMED思想泛化为通用插件，可叠加于任意ODE求解器（如iPNDM、DPM‑Solver++），以极小额外计算开销提升其采样质量。
4. **在NFE≈5时达到Solver-based SOTA**：在CIFAR‑10、ImageNet 64×64、FFHQ、LSUN Bedroom等多个数据集上，AMED‑Plugin配合iPNDM/DPM‑Solver‑2均取得最优FID，显著优于同类方法。

## 方法详解
- **几何观察**：对1k条采样轨迹进行PCA，发现前两个主成分已能解释绝大部分方差，相对投影误差始终低于8%，表明轨迹近似位于二维子空间。
- **均值定理近似**：连续解满足 \(\mathbf{x}_{t_n} = \mathbf{x}_{t_{n+1}} + \int_{t_{n+1}}^{t_n} \epsilon_\theta(\mathbf{x}_t, t) dt\)。利用轨迹的低维特性，寻找 \(s_n \in (t_n, t_{n+1})\) 与 \(c_n \in \mathbb{R}\) 使
  \[
  \epsilon_\theta(\mathbf{x}_{s_n}, s_n) \approx \frac{c_n}{t_n - t_{n+1}} \int_{t_{n+1}}^{t_n} \epsilon_\theta(\mathbf{x}_t, t) dt,
  \]
  进而得到离散更新公式：\(\mathbf{x}_{t_n} \approx \mathbf{x}_{t_{n+1}} + c_n (t_n - t_{n+1}) \epsilon_\theta(\mathbf{x}_{s_n}, s_n)\)。
- **AMED预测器**：轻量网络 \(g_\phi\)（仅9k参数），输入为U‑Net在 \(t_{n+1}\) 处的瓶颈特征 \(\mathbf{h}_{t_{n+1}}\) 及当前/目标时间步 \((t_{n+1}, t_n)\)，输出 \(r_n\) 与 \(c_n\)，中间时间步由 \(s_n = t_n^{r_n} t_{n+1}^{1-r_n}\) 计算。
- **知识蒸馏训练**：以高阶求解器（如EDM、DPM‑Solver‑2）生成的轨迹为教师轨迹，学生轨迹由AMED预测器驱动，最小化两轨迹在每一步的L2距离：\(\mathcal{L}_{t_n}(\phi) = d(\Phi_s(\cdot), \mathbf{y}_{t_n})\)，其中 \(d\) 为L2范数，训练仅用10k图像，耗时2分钟至3小时（单卡A100）。
- **分析首步（AFS）优化**：在第一步利用 \(\epsilon_\theta(\mathbf{x}_{t_N}, t_N)\) 与 \(\mathbf{x}_{t_N}\) 方向几乎相同的特性，直接以 \(\mathbf{x}_{t_N}\) 作为初始梯度方向，节省一次NFE。
- **时间缩放因子扩展**：参考iPNDM，可额外学习时间缩放因子 \(a_n\)，用 \(\epsilon_\theta(\mathbf{x}_{s_n}, a_n s_n)\) 替代 \(\epsilon_\theta(\mathbf{x}_{s_n}, s_n)\)，进一步扩展解空间。

## 实验与结果
- **数据集**：CIFAR‑10 32×32、FFHQ 64×64、ImageNet 64×64、LSUN Bedroom 256×256、Stable Diffusion 512×512。
- **评估基线**：DDIM、EDM、DPM‑Solver‑2、DPM‑Solver++(3M)、UniPC(3M)、iPNDM(4阶)。
- **核心结果（FID，越低越好）**：
  - **CIFAR‑10**（5 NFE）：AMED‑Plugin on iPNDM = **6.61**，较iPNDM原始13.59提升**6.98**；AMED‑Solver自身达7.59。
  - **ImageNet 64×64**（5 NFE）：AMED‑Plugin on iPNDM = **10.74**，较iPNDM原始17.17提升**6.43**。
  - **FFHQ 64×64**（5 NFE）：AMED‑Plugin on iPNDM = **10.74**，较iPNDM原始18.99提升**8.25**。
  - **LSUN Bedroom 256×256**（5 NFE）：AMED‑Plugin on DPM‑Solver‑2 = **13.20**，较DPM‑Solver‑2原始42.41提升**29.21**。
  - **Stable Diffusion 512**（8 NFE）：AMED‑Plugin on DPM‑Solver++(2M) = **18.92**，较基线21.33提升**2.41**。
- **结论**：AMED‑Solver在极小NFE下彻底扭转单步方法质量劣化趋势，AMED‑Plugin可稳定提升多种现有求解器，在NFE≈5区间达到Solver‑based方法的最优水平。

## 相关工作脉络
1. **DDIM**（Song et al., 2021）：一阶欧拉离散PF‑ODE，速度慢；AMED‑Solver保留单步形式，但通过学习方向逼近高阶精度。
2. **DPM‑Solver系列**（Lu et al., 2022/2023）：利用泰勒展开构造高阶单步/多步求解器，固定中间步位置（\(r=0.5\)）；AMED通过预测 \(r_n\) 自适应选择中间步。
3. **PNDM / iPNDM**（Liu et al., 2022; Zhang & Chen, 2023）：多步指数积分器，需存储历史梯度；AMED仅用单步，但通过学习达到类似多步的调整能力。
4. **EDM**（Karras et al., 2022）：优化时间调度与单步二阶格式；AMED可嵌入EDM作为插件，进一步提升极低NFE性能。
5. **知识蒸馏加速方法**（Salimans & Ho, 2022; Song et al., 2023）：训练新模型直接映射噪声到数据，成本高昂；AMED仅训练9k参数的轻量预测器，不改变底层扩散模型。
6. **UniPC**（Zhao et al., 2023）：统一预测‑校正框架，多步外推；AMED以单步预测器实现类似效果，计算开销更低。

## 局限性与未来方向
- **对时间调度仍敏感**：不同求解器、数据集的最优调度策略存在差异，固定调度难以全局最优；AMED可部分自适应，但未完全解决调度设计问题。
- **插件可能引入额外NFE**：AMED‑Plugin每步需两次U‑Net评估（总NFE = 2(N‑1)），在极端低预算下可能不如原生单步方法。
- **未来方向**：探索更精细的采样轨迹几何建模，设计自适应时间调度方案；将AMED思想拓展至视频生成、音频生成等其他模态；结合一致性模型等最新进展进一步优化NFE‑质量权衡。

## 研究启发与可借鉴点
1. **几何先验驱动学习方法设计**：利用采样轨迹的低维嵌入性质，将高维ODE积分近似为二维空间内的均值方向预测，大幅降低学习难度。
2. **知识蒸馏用于轻量预测器训练**：以9k参数网络学习高层求解器的行为，只需少量训练图像（10k），即可显著提升采样效率。
3. **插件化架构增强现有方法**：AMED‑Plugin证明了“即插即用”改进的可行性，可无缝集成到DPM‑Solver、iPNDM等成熟框架中。
4. **分析首步（AFS）节省计算资源**：在极小NFE场景下，利用噪声与梯度方向的近似同向性免去第一次网络评估，对资源受限应用有直接价值。
5. **系统性的消融与调参分析**：论文详细对比了教师求解器、时间调度策略的影响，为后续研究提供了可靠的基准实验设计模板。

## 关键术语表
- **ODE求解器**：用于数值求解概率流常微分方程（PF‑ODE）的迭代算法，决定扩散模型采样的速度与质量。
- **NFE（Number of Function Evaluations）**：采样过程中U‑Net模型评估的次数，通常与采样步数直接相关，是衡量采样速度的核心指标。
- **FID（Fréchet Inception Distance）**：衡量生成图像与真实图像分布相似度的常用指标，值越低表示质量越高。
- **AMED‑Solver**：作者提出的单步ODE求解器，通过学习预测每一步的中间时间步与梯度缩放因子来近似积分均值方向。
- **AMED‑Plugin**：将AMED思想泛化为可附加于任意ODE求解器的轻量插件，以极小额外开销提升采样质量。
- **知识蒸馏**：训练轻量学生模型模仿教师模型输出的技术，此处用于训练9k参数的AMED预测器。
- **时间调度（Time Schedule）**：定义采样过程中各离散时间步 \(t_n\) 的分配策略，如多项式调度、logSNR调度等。
- **瓶颈特征（Bottleneck Feature）**：U‑Net编码器末端输出的中间表示，被AMED预测器用作预测中间步与时序因子的输入。

## 可复现要素
- **数据集**：CIFAR‑10、FFHQ、ImageNet 64×64、LSUN Bedroom均为公开数据集；Stable Diffusion使用v1.4官方权重。
- **代码**：项目代码已开源（https://github.com/zjupi/diff-sampler）。
- **权重**：预训练扩散模型权重来自官方发布；AMED预测器权重未单独提供，但网络结构清晰（2个全连接层+池化）。
- **关键超参数**：
  - 多项式调度参数 \(\rho = 7\)（默认）；
  - 训练图像数：10k；
  - AMED预测器参数量：9k；
  - 训练时长：CIFAR‑10约2‑8分钟，LSUN Bedroom约1‑3小时（单卡A100）；
  - 损失函数：L2距离；
  - 教师求解器：DPM‑Solver‑2或EDM（AMED‑Solver）；同类型求解器（AMED‑Plugin）；
  - 插值步数 \(M\)：DPM‑Solver‑2取1，其余取2。
