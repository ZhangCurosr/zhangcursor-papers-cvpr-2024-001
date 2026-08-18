---
title: "PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Deng_PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:17:07"
---

# 论文速读：PRDP: Proximal Reward Difference Prediction for Large-Scale Reward Finetuning of Diffusion Models

## 一句话总结
PRDP 将扩散模型的强化学习奖励微调目标转化为等价的监督回归目标（Reward Difference Prediction），并结合近端裁剪与在线快照优化，首次实现了在十万级提示词数据集上稳定、可扩展的黑盒奖励微调，显著提升了复杂未见提示词的生成质量。

## 研究问题与动机
- **大规模 RL 微调极不稳定**：现有基于策略梯度（如 DDPO、DPOK）的扩散模型奖励微调方法在仅数百条提示词下尚可运行，但一旦扩展至 10 万级以上提示词，高方差梯度会导致训练迅速发散，生成质量崩塌。
- **离线数据覆盖不足**：直接沿用 DPO 的离线固定数据集采样方式，无法跟踪持续更新的生成分布，导致优化偏离实际目标。
- **奖励黑客（Reward Hacking）风险**：纯黑盒奖励最大化易使模型利用奖励函数的漏洞，产出高分但语义错乱或图文不对齐的图像。
- **缺乏可微假设**：多数实用奖励模型（如 HPSv2、PickScore）为黑盒/不可微，无法通过反向传播直接优化去噪网络，亟需一种不依赖奖励梯度的替代优化范式。

## 核心贡献（创新点）
1. **提出 RDP（Reward Difference Prediction）监督回归目标**：通过构造同提示词下两条去噪轨迹的概率比差值，消去不可计算的配分函数，将 RLHF 目标转化为 MSE 回归任务，并严格证明其最优解与原始 RL 目标完全等价。
2. **设计近端裁剪（Proximal Updates）机制**：借鉴 PPO 思想，对每步去噪轨迹的 log 概率比施加 $[-\epsilon, \epsilon]$ 裁剪，最终 loss 取裁剪前后 MSE 的最大值，从根本上抑制单次参数更新幅度过大导致的训练不稳定。
3. **开发在线快照优化算法**：每轮 epoch 保存模型快照 $\pi_{\theta_{old}}$ 进行轨迹采样，积累 $K$ 步梯度后再刷新快照，确保训练样本始终来自当前动态分布，显著提升泛化能力。
4. **首次实现十万级提示词的稳定黑盒微调**：突破 DDPO 在大规模数据集上的崩溃瓶颈，在 HPDv2 与 Pick-a-Pic v1 上均取得显著增益，尤其在复杂未见提示词上展现卓越生成质量。

## 方法详解
- **目标转化**：从 KL 正则化奖励最大化目标出发，推导最优轨迹分布满足 $\log(\pi_{\theta^*}(\bar{\mathbf{x}}|\mathbf{c}) / \pi_{ref}(\bar{\mathbf{x}}|\mathbf{c})) = r(\mathbf{x}_0,\mathbf{c})/\beta - \log Z(\mathbf{c})$。对同一提示词 $\mathbf{c}$ 下的两条轨迹 $\bar{\mathbf{x}}^a, \bar{\mathbf{x}}^b$ 作差，消去 $\log Z(\mathbf{c})$，得到 $\Delta \hat{r}_\theta = \Delta r / \beta$。
- **RDP 损失函数**：定义 $\hat{r}_\theta(\bar{\mathbf{x}},\mathbf{c}) = \log(\pi_\theta(\bar{\mathbf{x}}|\mathbf{c})/\pi_{ref}(\bar{\mathbf{x}}|\mathbf{c}))$，优化 $\mathcal{L}(\theta) = \mathbb{E}[\|\Delta \hat{r}_\theta(\bar{\mathbf{x}}^a,\bar{\mathbf{x}}^b,\mathbf{c}) - \Delta r(\mathbf{x}_0^a,\mathbf{x}_0^b,\mathbf{c})/\beta\|^2]$。完美预测时 $\mathcal{L}=0$，对应 RL 最优解。
- **在线优化（Algorithm 1）**：初始化 $\pi_\theta \leftarrow \pi_{ref}$；每 epoch 冻结快照 $\pi_{\theta_{old}}$，从其中采样 $N$ 个提示词各 $B$ 条去噪轨迹，计算奖励；随后执行 $K$ 步梯度下降更新 $\theta$，循环直至训练结束。
- **近端裁剪**：将 $\hat{r}_\theta$ 按去噪步 $t$ 分解为 $\sum_t \hat{r}_{\theta,t}$，对每步单独裁剪：$\hat{r}_{\theta,t}^{clip} = \mathrm{clip}(\hat{r}_{\theta,t}, \hat{r}_{\theta_{old},t}-\epsilon, \hat{r}_{\theta_{old},t}+\epsilon)$。最终 loss 取 $l_\theta^{unclip}$ 与 $l_\theta^{clip}$ 的最大值，保证优化方向有上界且不会过度偏离当前分布。
- **KL 正则化作用**：公式中的 $\beta$ 控制对参考模型的偏离程度，实验表明合理设置（如 $\beta=10$）可在提升美学质量的同时强制保持图文语义对齐，有效抑制 reward hacking。

## 实验与结果
- **小规模验证（45 prompts）**：在 HPSv2 与 PickScore 奖励下，PRDP 得分分别为 0.3471 与 0.2700，略优于 DDPO（0.3398, 0.2664）与基础 SD v1.4，验证了 RDP 与 RL 目标的等价性。
- **大规模训练（HPDv2 >100K prompts）**：DDPO 在大规模场景下完全失效（如 Photo 类得分仅 0.2093，低于基础模型），而 PRDP 在 Seen 提示词上 HPSv2 达 0.3175（提升 +18.3%），在所有 Unseen 类别（Animation 0.3223、Concept Art 0.3175、Painting 0.3172、Photo 0.3159）均大幅超越基础模型与 DDPO。
- **多奖励联合微调（Pick-a-Pic v1）**：加权组合 PickScore=10, HPSv2=2, Aesthetic=0.05，PRDP 在复杂未见提示词上生成细节更丰富、构图更合理，视觉质量显著领先。
- **消融结论**：在线优化在约 10 epoch 内即追平离线优化，并持续增益；KL 正则化是防止模型无视文本提示、盲目刷分的关键；近端裁剪直接决定了大规模训练的存活率。

## 相关工作脉络
- **RLHF / DPO（语言模型）**：DPO 将语言模型的 RL 目标转为监督分类，本文受其启发推广至扩散模型，但需额外处理轨迹级分布积分与不可解析 KL 的推导难题。
- **DDPO / DPOK**：首个将 PPO 引入扩散模型奖励微调的工作，在小规模（400/200 prompts）有效，但策略梯度高方差导致大规模崩溃，本文直接解决其扩展性瓶颈。
- **RAFT / DRaFT**：RAFT 通过离线数据重采样迭代；DRaFT 针对可微奖励直接反向传播。PRDP 兼容黑盒/不可微奖励，且无需外部重采样流水线，工程更简洁。
- **Diffusion-DPO（CVPR 2024，同期）**：同样借鉴 DPO 思路，但依赖大规模离线偏好数据构建固定对偶；PRDP 采用在线分布跟踪，更适合持续微调与动态 reward 场景。
- **视觉奖励模型（HPSv2, PickScore, LAION Aesthetic）**：作为通用黑盒/白盒评价信号，验证了方法在不同奖励函数与领域下的泛化能力。

## 局限性与未来方向
- 论文未显式讨论的潜在局限：每 epoch 需重新生成整批去噪轨迹，计算与显存开销随 prompt 数量线性增长，大规模训练时间成本较高。
- 性能仍受限于底层奖励模型的准确性；若奖励模型存在系统性偏差，PRDP 虽受 KL 约束，仍可能放大该偏差。
- 未来可探索将价值函数估计引入轨迹采样以降低方差，或与可微奖励（如 DRaFT）结合形成混合优化框架，进一步提升大规模微调的效率与上限。

## 研究启发与可借鉴点
1. **“RL 目标等价监督化”范式可迁移**：将高方差策略梯度转化为等价回归/分类任务，是稳定生成模型微调的通用路径，可直接复用于视频生成、3D 生成、语音合成等轨迹型生成任务。
2. **近端裁剪在分布空间的应用**：本文将对动作空间的 PPO 裁剪平移至“去噪轨迹 log 概率差”，为其他依赖采样器的生成模型提供了可直接套用的稳定性保障模块。
3. **在线快照优化优于纯离线 DPO**：周期性刷新采样分布能有效避免分布外退化，该设计对任何需要动态探索的生成微调框架均有借鉴价值。
4. **KL 正则化是黑盒奖励微调的标配**：实验清晰
