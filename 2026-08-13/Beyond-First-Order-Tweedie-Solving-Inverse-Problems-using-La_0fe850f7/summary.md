---
title: "Beyond-First-Order-Tweedie-Solving-Inverse-Problems-using-La"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:03:45"
field: "扩散模型逆问题求解"
keywords: ["逆问题求解", "潜在扩散模型", "Tweedie近似", "二阶近似", "图像去污", "文本引导编辑"]
innovations: ["提出STSL采样器，用代理损失函数实现仅需Hessian迹估计的二阶Tweedie近似", "设计DDIM前向初始化策略减少离散化误差，实现50步高效反演", "首次提出含污损图像的端到端文本引导编辑框架STSL-CAT"]
benchmarks: ["FFHQ-1K", "ImageNet-1K", "COCO 2017 Validation"]
---

# 论文速读：Beyond-First-Order-Tweedie-Solving-Inverse-Problems-using-La

## 一句话总结
论文提出 STSL（Second-order Tweedie sampler from Surrogate Loss），一种基于二阶 T weedie 近似的潜在扩散模型（LDM）采样器，通过引入代理损失函数实现高效反向过程，将神经函数评估（NFE）减少 4×–8×，并首次实现了含真实污损图像的文本引导图像编辑。

## 研究问题与动机
- 基于 LDM 的逆问题求解器（如 PSLD、P2L）依赖一阶 Tweedie 近似（用条件期望替代条件分布），会引入回归均值偏差，导致重建质量受限。
- 二阶 Tweedie 近似理论上能缓解偏差，但需要完整 Hessian 矩阵或 Jacobian，计算复杂度高达 $\mathcal{O}(d^2)$，使标准反向扩散过程不可行。
- 现有图像编辑方法（如 NTI）在处理含运动模糊、低分辨率、噪声等真实世界污损的输入时效果显著下降，缺乏端到端的去污-编辑联合框架。
- 需要一种既高效（接近一阶计算开销）又能提供二阶精度的采样器，以在实际应用中平衡重建质量与计算成本。

## 核心贡献（创新点）
1. **提出 STSL 采样器**：设计了基于代理损失函数的反向漂移更新规则，仅需估计 $\text{Trace}(\nabla^2 \log p_t)$，用单次随机投影实现二阶近似，避免全 Hessian 计算。
2. **理论下界保证**：证明代理损失 $\hat{\mathcal{L}}$ 构成 $\log p_{T-t}(\mathbf{y}|Z_t)$ 的下界（Theorem 4.4），且梯度近似分解为测量一致性项与 Hessian 迹修正项，为一阶方法提供了严格改进依据。
3. **高效初始化策略**：用 DDIM 前向过程将反演起点设为 $Z_0 \sim p_T(Z_0|\mathcal{E}(\mathbf{A}^T\mathbf{y}))$，减少从纯高斯初始化的 $\mathcal{O}(de^{-2T})$ 离散化误差，使 50 步采样即可达到高质量重建。
4. **首次实现含污损图像的文本引导编辑**：提出 STSL-CAT 两阶段框架（STSL 反演 + Cross-Attention-Tuning），在清除污损的同时保留原图语义内容，超越 NTI 在真实污损图像上的编辑性能。

## 方法详解
### 3.1 背景与问题形式化
- 线性逆问题：$\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}$，目标是从后验 $p_0(X_0|\mathbf{y})$ 采样。
- 后验 SDE 漂移项分解为 $\nabla \log p_{T-t}(Z_t) + \nabla \log p_{T-t}(\mathbf{y}|Z_t)$，其中 $p_{T-t}(\mathbf{y}|Z_t)$ 对 $t>0$ 不可直接计算。
- 一阶 Tweedie 近似：$p_{T-t}(\mathbf{y}|Z_t) \approx p_{T-t}(\mathbf{y}|\bar{Z}_T)$，其中 $\bar{Z}_T = \mathbb{E}[Z_T|Z_t]$，由 Proposition 4.1 给出均值与协方差公式。

### 3.2 STSL 核心设计
**代理损失函数**（Eq. 4）：
$$
\mathcal{L}(\mathbf{y}, Z_t) = \lambda\|\mathbf{y} - \mathbf{A}\mathcal{D}(\bar{Z}_T)\|_2^2 + \frac{\eta}{d}\mathbb{E}_{\epsilon}\left[\epsilon^T\left(\nabla\log p_{T-t}(Z_t+\epsilon) - \nabla\log p_{T-t}(Z_t)\right)\right]
$$
- 第一项：测量一致性，强制重建结果满足观测方程。
- 第二项：Hutchinson 估计器近似 Hessian 迹，引入曲率信息以缓解一阶偏差。

**反向过程更新**（Algorithm 1）：
1. **初始化**：$\vec{Z}_0 = \mathcal{E}(\mathbf{A}^T\mathbf{y})$，经 DDIM 前向得到 $Z_0 \sim p_T(Z_0|\vec{Z}_0)$。
2. **每步内层循环**（$K=5$ 次随机平均）：
   - 采样 $\epsilon \sim \mathcal{N}(0,I)$，计算 $\bar{Z}_T$。
   - 梯度下降：$Z_t \leftarrow Z_t - \lambda\nabla\mathcal{L}(\mathbf{y}, Z_t)$。
3. **外层 DDIM 反向步**：$Z_{t+1} \leftarrow \text{DDIM}(Z_t)$。
4. 最终输出 $\mathcal{D}(Z_T)$。

**理论依据**（Theorem 4.4）：
- $\hat{\mathcal{L}}(\mathbf{y}, Z_t) \leq \log p_{T-t}(\mathbf{y}|Z_t)$，梯度近似为：
$$
\nabla\hat{\mathcal{L}} \simeq -\lambda\nabla\|\mathbf{y} - \mathbf{A}\mathcal{D}(\bar{Z}_T)\|_2^2 - \gamma\nabla\text{Trace}(\nabla^2\log p_{T-t}(Z_t))
$$
- 当 $\gamma=0$ 时退化为 PSLD 的一阶更新，证明 STSL 是其严格推广。

### 3.3 图像编辑扩展：STSL-CAT
- **反演阶段**：用 STSL 在 50 步内从污损图像 $\mathbf{y}$ 恢复干净潜变量。
- **Cross-Attention-Tuning (CAT)**：在 CAC 控制的反向步后，追加一次 STSL 代理损失梯度更新（Eq. 5），以消除编辑过程中残留的污损 artifact。
- **对比损失**：添加 $\mathcal{L}_{ViT}$ 项（$\nu/d \cdot \mathcal{L}_{ViT}$）以保持原图语义结构，编辑初期（前 30 步）用测量梯度，后期切换为对比梯度。

## 实验与结果
### 数据集与任务
- **FFHQ-1K**（512×512，1000 张测试集）、**ImageNet-1K**（512×512，1000 张）、**COCO 2017 val**（消融实验）。
- 逆任务：运动去模糊、高斯去模糊、超分辨率（4×、8×）、随机 inpainting（40% 缺失）、盐 Pepper 去噪（2%）。

### 主要结果（Table 1, FFHQ）
| 方法 | SR(8×) LPIPS↓ | Motion Deblur LPIPS↓ | Gaussian Deblur LPIPS↓ |
|------|---------------|----------------------|------------------------|
| **STSL (Ours)** | **0.335** | **0.321** | **0.308** |
| P2L [10] | 0.381 | 0.395 | 0.382 |
| PSLD [43] | 0.402 | 0.408 | 0.371 |
| DPS [8] (PDM) | 0.538 | 0.556 | 0.694 |

- STSL 在 8× 超分上较 P2L 提升 **5%** 绝对 LPIPS；ImageNet 上同样全面领先。
- 视觉质量（Figure 2）：STSL 生成更锐利细节，无 PSLD/P2L 的过平滑或伪影。

### 效率对比（Table 2）
| 方法 | 步数 | NFEs | 耗时(s) |
|------|------|------|---------|
| **STSL** | **50** | **250** | **45** |
| P2L | 1000 | 2000 | 500 |
| PSLD | 1000 | 1000 | 194 |

- NFE 减少 **4×（vs PSLD）**、**8×（vs P2L）**，因 NFE 是后验采样的最贵开销。

### 消融实验（Table 3–4）
- **偏差分析**（Table 3）：STSL-biased（仅单步梯度、无 Hutchinson 估计）在 Gaussian Deblur 上 LPIPS 为 0.456，显著劣于 STSL 的 0.380，验证二阶项必要性。
- **随机平均步数 $N$**（Table 4a）：$N=2$ 时 LPIPS 0.403 优于 $N=1$ 的 0.408；$N=5$ 时 50 步 DDIM 已足够。
- **Hessian 系数 $\eta$**（Table 4b）：$\eta=0.02$ 最优（LPIPS 0.388），$\eta=0$ 退化为一阶近似。
- **对比损失系数 $\nu$**（Table 4c）：$\nu=2$ 最佳（LPIPS 0.392）。
- **DDIM 步数**（Table 4d）：50 步即可，更多步数收益边际。

### 图像编辑（Table 4f）
- 在含 SR(8×) 污损的 ImageNet 狗图像上执行 "dog→cat" 编辑：
  - NTI CLIP acc: 70.00%
  - **STSL-CAT CLIP acc: 93.00%**（相对提升 **32%**）
  - NTI-CAT（干净图）：96.00%，证明 CAT 机制本身有效。

## 相关工作脉络
- **DPS [8]**：像素空间扩散逆问题求解器，使用一阶 Tweedie 近似；STSL 将其思想迁移至 LDM 潜空间并引入二阶修正。
- **PSLD [43]**：首个在 Stable Diffusion 潜空间应用一阶 Tweedie 的 LDM 逆问题求解器；STSL 在其基础上增加 Hessian 迹估计项，理论更严格。
- **P2L [10]**：在 PSLD 基础上加入文本 embedding 优化；STSL 无需调优文本 embedding，仅靠潜变量更新即达更好效果。
- **LDPS/GML-DPS [43]**：PSLD 的变体，分别引入 gluing objective 和广义版本；STSL 统一了代理损失框架，计算更轻量。
- **NTI [35]**：零样本图像编辑基线，通过 null-text inversion 对齐真实图像；STSL 指出 NTI 无法处理污损，提出 STSL-CAT 联合去污-编辑流程。
- **Tweedie 二阶近似前作 [5, 32]**：需完整 Jacobian 或 Hessian，复杂度 $\mathcal{O}(d^2)$；STSL 用单次随机投影估计迹，将开销降至 $\mathcal{O}(1)$。

## 局限性与未来方向
- **SSIM 指标偏低**：作者承认 SSIM 易将高频伪影误判为"锐利"，而惩罚模糊，导致部分场景 SSIM 不如 P2L（但 LPIPS 更优）。
- **仅支持线性逆问题**：当前框架针对 $\mathbf{y}=\mathbf{A}\mathbf{x}+\mathbf{n}$，非线性算子（如相位恢复、CT 重建）需扩展 likelihood 建模。
- **超参数 $\eta, \nu, \lambda$ 需调优**：虽声称对不同数据集鲁棒，但未提供自动调度策略。
- **未讨论扩散步数 $T$ 过小的极端情况**：50 步已显著优于 1000 步，但更激进压缩（如 20 步）的性能未测试。
- **编辑任务仅验证文本引导**：未探索多模态编辑（如深度、边缘条件）或区域选择编辑。

## 研究启发与可借鉴点
1. **Hessian 迹的随机投影估计可复用于其他扩散模型任务**：只需 score network 的输出，无需额外训练，适合任何需要曲率信息的逆问题求解器设计。
2. **代理损失函数设计思路**：将 likelihood 下界转化为可微损失 + 正则项形式，为其他"不可 tractable 似然"的生成模型后验采样提供通用范式。
3. **两阶段去污-编辑框架**：STSL-CAT 将逆问题求解器与编辑模块解耦，可借鉴到其他含噪图像的生成任务（如视频修复、医学图像增强）。
4. **DDIM 前向初始化策略**：从 $\mathcal{E}(\mathbf{A}^T\mathbf{y})$ 出发可减少离散化误差，适用于所有需要少步采样的 LDM 逆向应用。
5. **对比损失 $\mathcal{L}_{ViT}$ 的结合时机**：早期用测量梯度、后期切换至对比梯度，这种分段策略可迁移至其他感知质量优化的生成任务。

## 关键术语表
- **Tweedie 公式**：从高斯噪声模型的后验分布中提取均值与协方差的统计恒等式，一阶用期望、二阶引入 Hessian 曲率项。
- **STSL**：Second-order Tweedie sampler from Surrogate Loss，本文提出的二阶近似潜在扩散采样器。
- **代理损失（Surrogate Loss）**：$\mathcal{L}(\mathbf{y}, Z_t)$，结合测量一致性项与 Hessian 迹估计项的可微损失，用作反向过程梯度来源。
- **Hutchinson 估计器**：用随机向量 $\epsilon$ 估计矩阵迹的无偏估计，$\text{Trace}(H) \approx \mathbb{E}[\epsilon^T H \epsilon]$，此处用于估计 Hessian 迹。
- **Cross-Attention-Tuning (CAT)**：在 CAC 编辑控制后追加一次 STSL 梯度更新，以消除残留污损 artifact 的后期精炼机制。
- **NFE（Neural Function Evaluations）**：扩散模型中 score network 的调用次数，决定采样效率的核心瓶颈指标。
- **后验采样（Posterior Sampling）**：从条件分布 $p(X_0|\mathbf{y})$ 生成样本的过程，是扩散逆问题的核心目标。
- **LDM（Latent Diffusion Model）**：在压缩潜空间运行扩散过程的生成模型（如 Stable Diffusion），相比像素空间更高效。

## 可复现要素
- **数据集**：FFHQ-1K、ImageNet-1K（ctest10k 子集）、COCO 2017 val——均为公开数据集。
- **代码**：论文未提供开源链接，需联系作者获取（GitHub 未提及）。
- **权重**：使用预训练 Stable Diffusion v1.5（公开可用）。
- **关键超参**：$T=50$ 步、$K=5$ 次随机平均、$\lambda=\mathcal{O}(1/\sigma_y^2)$、$\eta=0.02$、$\nu=2$（消融实验推荐值）；具体数值见附录 B.1（论文未完整列出）。
- **硬件**：单张 NVIDIA A100 GPU。
