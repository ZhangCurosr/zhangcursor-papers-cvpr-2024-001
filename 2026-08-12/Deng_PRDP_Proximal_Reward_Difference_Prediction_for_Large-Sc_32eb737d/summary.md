---
title: "PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Deng_PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:18:31"
field: "扩散模型对齐与微调"
keywords: ["diffusion model", "reward finetuning", "reward difference prediction", "proximal policy optimization", "RLHF", "black-box reward", "generative alignment"]
innovations: ["将RLHF目标等价转化为监督回归的Reward Difference Prediction目标，首次实现100K+大规模稳定微调", "提出基于PPO近端裁剪的步级更新策略，解决扩散模型奖励微调的训练不稳定问题"]
benchmarks: ["HPDv2", "Pick-a-Pic v1", "HPSv2", "PickScore", "LAION Aesthetic"]
---

# 论文速读：PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models

## 一句话总结
PRDP 首次实现了在超过 10 万条 prompt 的大规模数据集上稳定地对扩散模型进行黑盒奖励微调；其核心是将 RLHF 目标等价转化为一个监督回归目标（Reward Difference Prediction），并通过近端更新与在线优化实现稳定训练，从而克服了现有 RL 方法在大尺度下的不稳定性。

## 研究问题与动机
- **扩散模型最大似然目标与下游任务不对齐**：预训练的扩散模型（如 SD v1.4）在生成审美偏好图像或训练时未见过的新颖物体组合时表现不佳。
- **现有 RL 方法无法扩展到大规模 prompt**：DDPO 和 DPOK 等基于策略梯度的方法虽然在小规模（400 条以内）上有效，但策略梯度方差高，导致大规模训练时严重不稳定，甚至产生无意义噪声。
- **KL 正则化不可解析计算**：与语言模型不同，扩散模型的 KL 散度涉及高维图像空间的不可 tractable 积分，因此需要替代目标进行优化。
- **离线数据集泛化能力不足**：仅依赖预生成数据集进行微调（如 supervised finetuning）会限制生成质量的上限，需要在线优化机制。

## 核心贡献（创新点）
- **提出了 Reward Difference Prediction（RDP）目标**：将 RLHF 最大化目标转化为等价的最优解的监督 MSE 回归目标（预测去噪轨迹生成的图像对之间的奖励差值），从而绕开高方差的策略梯度。
- **证明了 RDP 与 RL 目标的等价性**：理论证明获得完美奖励差值预测的扩散模型恰好是 RL 目标的最大化器（$\pi_\theta = \pi_{\theta^*} \iff \mathcal{L}(\theta) = 0$）。
- **设计了在线优化算法**：采用快照式策略（每 K 步更新参考模型 $\theta_{\text{old}}$），确保采样分布始终覆盖当前模型的输出空间，显著优于纯离线方案。
- **提出近端裁剪更新（Proximal Updates）**：借鉴 PPO 思想，对去噪每一步的对数概率比施加裁剪 $[-\epsilon, \epsilon]$，防止模型偏离过快导致训练崩溃。
- **首次实现 100K+ prompt 的稳定黑盒奖励微调**：在 HPDv2 和 Pick-a-Pic v1 上验证，PRDP 大幅超越 DDPO，且 KL 正则化可有效缓解 reward hacking。

## 方法详解
**Reward Difference Prediction（RDP）目标推导：**

从 RLHF 下界目标出发：
$$\max_{\pi_\theta} \; \mathbb{E}_{\mathbf{x}_0, \mathbf{c}}\left[r(\mathbf{x}_0, \mathbf{c}) - \beta \,\mathrm{KL}[\pi_\theta(\bar{\mathbf{x}}|\mathbf{c}) \| \pi_{\mathrm{ref}}(\bar{\mathbf{x}}|\mathbf{c})]\right]$$

其最优解满足：
$$\pi_{\theta^*}(\bar{\mathbf{x}}|\mathbf{c}) = \frac{1}{Z(\mathbf{c})} \pi_{\mathrm{ref}}(\bar{\mathbf{x}}|\mathbf{c}) \exp\left(\frac{r(\mathbf{x}_0, \mathbf{c})}{\beta}\right)$$

对两个同 prompt 的去噪轨迹取对数比之差消去 $Z(\mathbf{c})$，得到：
$$\Delta\hat{r}_{\theta^*}(\bar{\mathbf{x}}^a, \bar{\mathbf{x}}^b, \mathbf{c}) = \frac{\Delta r(\mathbf{x}_0^a, \mathbf{x}_0^b, \mathbf{c})}{\beta}$$

其中 $\hat{r}_\theta(\bar{\mathbf{x}}, \mathbf{c}) = \log\frac{\pi_\theta(\bar{\mathbf{x}}|\mathbf{c})}{\pi_{\mathrm{ref}}(\bar{\mathbf{x}}|\mathbf{c})}$，$\Delta$ 表示两轨迹之差。

**RDP 损失函数（MSE）：**
$$\mathcal{L}(\theta) = \mathbb{E}\left[\left\|\Delta\hat{r}_\theta(\bar{\mathbf{x}}^a, \bar{\mathbf{x}}^b, \mathbf{c}) - \Delta r(\mathbf{x}_0^a, \mathbf{x}_0^b, \mathbf{c})/\beta\right\|^2\right]$$

**近端裁剪更新：** 将 $\hat{r}_{\theta,t}$ 按去噪步骤分解，对每步应用裁剪：
$$\hat{r}_{\theta,t}^{\mathrm{clip}} = \mathrm{clip}\!\left(\hat{r}_{\theta,t},\, \hat{r}_{\theta_{\mathrm{old}},t} - \epsilon,\, \hat{r}_{\theta_{\mathrm{old}},t} + \epsilon\right)$$
最终损失取裁剪与非裁剪 MSE 的最大值：
$$l_\theta \leftarrow \max\!\left(l_\theta, \, l_\theta^{\mathrm{clip}}\right)$$

**在线优化流程（Algorithm 1）：** 每个 epoch 先用 $\theta_{\mathrm{old}}$ 采样一批去噪轨迹和对应奖励，然后在 K 步内用当前模型参数进行梯度更新，完成后快照更新 $\theta_{\mathrm{old}}$。

## 实验与结果
- **数据集**：小规模 45 条 prompt（"A painting of a ⟨animal⟩"）；大规模 HPDv2 训练集 100K+ prompt；Pick-a-Pic v1 训练集用于多奖励微调。
- **基线**：SD v1.4（预训练）、DDPO（最强 RL 基线）。
- **评估指标**：HPSv2、PickScore、Aesthetic 奖励分数。

**小规模结果（Table 1）：** PRDP 与 DDPO 表现接近且均显著优于 SD v1.4（HPSv2: 0.3471 vs 0.3398；PickScore: 0.2700 vs 0.2664），验证了 PRDP 与 RL 方法具有同等的奖励最大化能力。

**大规模结果（Table 2）：** DDPO 在所有设置上均明显劣于 SD v1.4（如 HPSv2 unseen photo 仅 0.2093 < SD 0.2750），而 PRDP 全面提升：
- HPSv2 Seen HPDv2：SD 0.2685 → **PRDP 0.3175**（提升 +18.3%）
- PickScore Seen HPDv2：SD 0.2092 → **PRDP 0.2424**（提升 +15.9%）
- 各 unseen 子集（Pick-a-Pic v1、Animation、Concept Art 等）PRDP 均稳定领先，提升幅度约 10%~42%。

**多奖励微调：** 组合奖励 PickScore=10, HPSv2=2, Aesthetic=0.05，在复杂 unseen prompt 上生成质量显著优于基线（Figure 1, 11-15）。

**在线 vs 离线优化（Figure 7）：** 在线优化约 10 epoch 即达到离线方法同等水平，并持续提升。

## 相关工作脉络
- **DDPO（Black et al., 2024）**：基于 PPO 的策略梯度方法，在 400 prompt 上稳定，但大规模下崩溃；PRDP 将其转化为等价监督目标以解决稳定性问题。
- **DPOK（Fan et al., 2023）**：拟合价值函数估计期望奖励，在 200 prompt 上效果有限；PRDP 同样处理大规模场景但无需价值网络。
- **DPO（Rafailov et al., 2023）**：语言模型中将 RLHF 转化为监督分类的经典工作；PRDP 的核心思想受其启发，但针对扩散模型的 denoising trajectory 结构重新推导了等价回归目标。
- **RAFT（Dong et al., 2023）**：基于离线高奖励样本的 supervised finetuning，受限于离线数据集覆盖；PRDP 通过在线采样克服这一局限。
- **Diffusion-DPO（Wallace et al., 2024，同期工作）**：同样将 DPO 适配到扩散模型，但从离线偏好数据学习，不涉及在线奖励查询；PRDP 支持通用黑盒奖励函数。
- **DRaFT（Clark et al., 2024）**：针对可微奖励直接回传梯度；PRDP 适用于任意黑盒奖励（无需可微性），二者形成互补。

## 局限性与未来方向
- **训练效率**：每个 prompt 需采样多张图像并计算双轨迹的完整去噪路径，计算开销高于直接梯度方法；大规模训练需 1000 epoch。
- **KL 超参数敏感**：β 需人工调参，过大会抑制奖励提升，过小可能导致 reward hacking。
- **仅针对文本条件扩散模型**：当前方法针对 text-to-image 设定，未扩展到视频、3D 或其他模态。
- **近端裁剪的步级粒度**：当前在每个去噪步独立裁剪，可能不是最优的稳定化策略，后续可探索轨迹级别的裁剪。
- **并行采样成本**：100K prompt 大规模训练需要大量 GPU 资源进行在线推理采样。

## 研究启发与可借鉴点
- **RL → 监督学习的范式转换**：受 DPO 启发，将难以优化的 RL 目标转化为等价监督回归目标是一个通用思路，可迁移到其他生成模型（如视频生成、3D 生成）的 reward finetuning。
- **近端裁剪技巧的跨域适用**：PPO 风格的裁剪策略从离散策略扩展到连续去噪轨迹的概率比，该设计可直接借鉴到任意基于 score 的生成模型对齐任务。
- **在线快照机制避免分布漂移**：每隔 K 步更新参考模型再采样，这一策略在扩散模型的 reward finetuning 中尤为关键，可用于防止模型退化。
- **KL 正则化缓解 reward hacking**：对于仅基于图像无文本条件的奖励模型（如 Aesthetic score），引入 KL 约束可有效保留文本对齐能力，是通用的防 overfitting 手段。
- **配对样本的奖励差值设计**：通过同一 prompt 的两个样本对消去归一化常数 $Z(\mathbf{c})$，是一种优雅的消除不可 tractable 配分函数的技巧。

## 关键术语表
- **Reward Difference Prediction（RDP）**：一种监督回归目标，让扩散模型预测两个生成图像之间的奖励差值，与 RLHF 目标等价。
- **Proximal Updates**：借鉴 PPO 的裁剪策略，限制每步去噪概率比的变化范围，确保训练稳定性。
- **Black-box Reward**：不需可微的奖励函数（如 HPSv2、PickScore），PRDP 无需对奖励模型求梯度。
- **Denoising Trajectory**：从噪声到图像的全程去噪路径 $\mathbf{x}_{0:T}$，PRDP 在 trajectory 级别定义概率比。
- **KL Regularization**：惩罚微调后的模型偏离预训练模型的分布，防止 reward hacking 并保留生成多样性。
- **RLHF（Reinforcement Learning from Human Feedback）**：在语言模型中广泛使用的对齐技术，通过奖励模型 + 策略梯度微调实现。
- **DDPO**：将 PPO 应用于扩散模型 reward finetuning 的代表性工作，PRDP 的主要对比基线。
- **HPDv2 / Pick-a-Pic v1**：大规模人类偏好数据集，分别含 100K+ prompt 和高质量图文对，用于本文大规模实验。

## 可复现要素
- **预训练模型**：Stable Diffusion v1.4（开源，HuggingFace 可获取），微调 UNet 全量参数。
- **奖励模型**：HPSv2 [53]、PickScore [22]、LAION Aesthetic Score（均为开源）。
- **数据集**：HPDv2 训练集（100K+ prompt）、Pick-a-Pic v1 训练集（来源于作者分享的 DRaFT 提示词）；评估用 HPDv2 测试集（Animation/Concept Art/Painting/Photo 各 800 条）和 Pick-a-Pic v1 测试集（500 条）。
- **采样设置**：DDPM sampler，50 步去噪，classifier-free guidance scale = 5.0。
- **超参数**：大规模训练 1000 epoch，每 epoch 64 prompts × 8 images；小规模训练 100 epoch，每 epoch 32 prompts × 16 images；KL 正则化系数 β = 10；近端裁剪步级范围 ε（论文未明确给出具体数值，仅以符号表示）；每 epoch K 步梯度更新（未给出具体值）。
- **代码/权重**：论文主页 https://fdeng18.github.io/prdp，论文声明代码将在录用后公开。
