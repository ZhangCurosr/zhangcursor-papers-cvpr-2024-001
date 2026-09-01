---
title: "PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-R"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Deng_PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:41:16"
field: "生成模型对齐与微调"
keywords: ["Diffusion Models", "Reward Finetuning", "RLHF", "Policy Gradient", "Proximal Policy Optimization", "Text-to-Image"]
innovations: ["将RLHF目标转化为等价的监督回归目标RDP", "首次实现10万+提示词规模扩散模型稳定奖励微调", "结合在线优化与近端更新机制解决策略梯度高方差问题"]
benchmarks: ["HPDv2", "Pick-a-Pic v1", "HPSv2", "PickScore"]
---

# 论文速读：PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-Reward-Finetuning-of-Diffusion-Models

## 一句话总结
本文提出了**PRDP（近端奖励差预测）**，首次实现了在**超过10万提示词**的大规模数据集上对扩散模型进行稳定的黑盒奖励微调。核心思想是将RLHF目标转化为等价的监督回归目标（RDP），配合在线优化与近端更新机制，解决了现有基于策略梯度的方法（如DDPO）在大尺度训练中的不稳定性问题。

## 研究问题与动机
1. **目标不对齐**：扩散模型的MLE训练目标与下游应用（如美学偏好、新颖构图）存在偏差。
2. **RL方法难以扩展**：DDPO和DPOK等基于策略梯度的RL微调方法在小规模（400提示词）上表现尚可，但在大规模（>100K提示词）训练时因高方差导致训练不稳定，最终生成无意义噪声或低质量图像。
3. **奖励黑客问题**：直接最大化奖励可能导致模型利用奖励模型的不准确性，忽视文本-图像对齐。
4. **缺乏可微分性**：真实世界奖励模型通常不可微分，无法使用直接反向传播的微调方法。

## 核心贡献（创新点）
1. **提出RDP目标**：将RLHF目标重构为监督回归目标（预测去噪轨迹的奖励差），并证明其最优解与原RL目标完全一致。
2. **首次大规模稳定微调**：在超过10万提示词数据集上实现稳定黑盒奖励微调，解决了DDPO等基线方法在大尺度训练中的崩溃问题。
3. **在线优化策略**：借鉴在线RL思路，每K步快照更新采样分布，相比离线数据集采样显著提升生成质量。
4. **近端更新机制**：引入类PPO的clip操作，限制概率比变化范围，大幅提升训练稳定性。
5. **KL正则化缓解奖励黑客**：通过理论框架内嵌KL散度惩罚，在提升奖励得分的同时保持文本-图像对齐。

## 方法详解
1. **RDP目标推导**：
   - 从KL正则化奖励最大化目标出发：$\max_{\pi_\theta} \mathbb{E}[r(x_0,c) - \beta \cdot \text{KL}[\pi_\theta(\bar{x}|c) || \pi_{\text{ref}}(\bar{x}|c)]]$
   - 推导最优解形式：$\pi_{\theta^*}(\bar{x}|c) = \frac{1}{Z(c)} \pi_{\text{ref}}(\bar{x}|c) \exp(r(x_0,c)/\beta)$
   - 通过取两个同提示词的去噪轨迹的概率比对数差，消去不可计算的配分函数$Z(c)$

2. **损失函数**：
   - $\mathcal{L}(\theta) = \mathbb{E}_{\bar{x}^a, \bar{x}^b, c} \| \Delta\hat{r}_\theta(\bar{x}^a, \bar{x}^b, c) - \Delta r(x_0^a, x_0^b, c)/\beta \|^2$
   - 其中$\hat{r}_\theta(\bar{x}, c) = \log \frac{\pi_\theta(\bar{x}|c)}{\pi_{\text{ref}}(\bar{x}|c)}$是可学习的奖励估计

3. **在线优化**：
   - 每K次梯度更新后保存模型快照$\pi_{\theta_{\text{old}}}$
   - 使用当前快照重新采样去噪轨迹，确保样本分布与当前模型匹配

4. **近端更新**：
   - 借鉴PPO的clip机制，限制$\hat{r}_{\theta,t}$相对于$\hat{r}_{\theta_{\text{old}},t}$的变化范围
   - 损失取unclipped和clipped两者的最大值，确保优化有上界

5. **算法流程**（Algorithm 1）：
   - 初始化：$\pi_\theta \leftarrow \pi_{\text{ref}}$
   - 每epoch：采样N个提示词，每个提示词采样B个去噪轨迹
   - 计算所有轨迹对的奖励差
   - K次梯度更新后更新快照

## 实验与结果
1. **小规模验证**（45个提示词）：
   - HPSv2奖励：PRDP得分0.3471 vs DDPO 0.3398（SD v1.4基线0.2855）
   - PickScore奖励：PRDP得分0.2700 vs DDPO 0.2664（基线0.2179）
   - 验证PRDP与DDPO在奖励最大化能力上相当

2. **大规模训练**（10万+提示词，HPDv2）：
   - HPSv2奖励：
     - HPDv2训练集：PRDP 0.3175 vs DDPO 0.2464 vs SD v1.4 0.2685
     - Pick-a-Pic v1测试集：PRDP 0.3050 vs DDPO 0.2501
     - HPDv2动画类别：PRDP 0.3223 vs DDPO 0.2673
   - PickScore奖励：
     - HPDv2训练集：PRDP 0.2424 vs DDPO 0.2032
     - 测试集各类别PRDP均显著优于DDPO（DDPO在照片类别跌至0.1780）

3. **多奖励微调**（Pick-a-Pic v1）：
   - 组合奖励：PickScore=10, HPSv2=2, Aesthetic=0.05
   - PRDP在复杂 unseen prompts上展现出优越生成质量

4. **在线vs离线优化对比**：
   - 在线优化在约10个epoch后超越离线优化，并持续提升

5. **KL正则化效果**：
   - β=10时PRDP成功保持文本-图像对齐，而DDPO无KL正则化时产生reward hacking（忽略提示词生成相似图像）

## 相关工作脉络
1. **RLHF（Ouyang et al., 2022）**：语言模型的强化学习人类反馈对齐，PRDP的思想来源之一。
2. **DPO（Rafailov et al., 2023）**：将RLHF转化为监督分类目标，PRDP的核心启发。
3. **DDPO（Black et al., 2024）**：基于PPO的扩散模型RL微调，PRDP的主要对比基线。
4. **DPOK（Fan et al., 2023）**：引入价值函数估计的扩散模型RL方法，小规模稳定但扩展性差。
5. **DRaFT（Clark et al., 2024）**：基于可微分奖励的直接微调，PRDP适用于不可微分黑盒奖励。
6. **Diffusion-DPO（Wallace et al., 2024）**：同期工作，将DPO适配到扩散模型，但依赖离线偏好数据。

## 局限性与未来方向
1. **依赖奖励模型质量**：RDP假设奖励模型准确，若奖励模型存在系统性偏差，优化结果会继承这些偏差。
2. **超参数敏感**：KL正则化强度$\beta$和clip范围$\epsilon$需要仔细调优。
3. **计算开销**：每步需要采样多对轨迹并计算奖励，相比直接微调计算成本更高。
4. **未来方向**：与Diffusion-DPO等方法结合、扩展到其他模态（视频/3D）、研究自适应$\beta$策略。

## 研究启发与可借鉴点
1. **RL→监督学习的范式转换**：将策略梯度目标转化为等价监督目标的思想可迁移至其他生成模型（如流模型、GFlowNet）的对齐任务。
2. **近端更新的通用性**：clip机制可有效稳定其他基于轨迹的生成模型优化过程。
3. **在线采样策略**：定期快照重采样的设计平衡了样本多样性和分布匹配，值得在类似场景复用。
4. **KL正则化的防御价值**：在奖励微调中内嵌KL惩罚可缓解reward hacking，提升实用安全性。
5. **多奖励组合微调**：加权组合多个奖励模型可综合提升不同维度的生成质量，为多目标对齐提供思路。

## 关键术语表
**PRDP**：Proximal Reward Difference Prediction，近端奖励差预测，本文提出的大规模稳定奖励微调方法。
**RDP**：Reward Difference Prediction，奖励差预测，将RLHF目标转化为监督回归的核心目标函数。
**RLHF**：Reinforcement Learning from Human Feedback，人类反馈强化学习，语言模型对齐的经典范式。
**DDPO**：Denoising Diffusion Policy Optimization，基于PPO的扩散模型RL微调方法，PRDP的主要基线。
**HPSv2**：Human Preference Score v2，评估文本-图像对齐质量和美学偏好的奖励模型。
**PickScore**：基于用户偏好的图像-文本匹配质量评估模型。
**奖励黑客（Reward Hacking）**：模型利用奖励模型的漏洞或局限性，生成高分但实际质量低下的输出。
**近端更新（Proximal Update）**：借鉴PPO的clip机制，限制模型更新幅度以保持训练稳定性。

## 可复现要素
- **数据集**：HPDv2（训练集10万+提示词）、Pick-a-Pic v1
- **预训练模型**：Stable Diffusion v1.4（公开）
- **奖励模型**：HPSv2、PickScore、LAION Aesthetic Score（均有公开版本）
- **代码开源**：论文提供了项目主页 https://fdeng18.github.io/prdp，但**论文未明确声明代码开源**
- **关键超参数**：
  - 小规模：100 epochs，batch 32 prompts × 16 images
  - 大规模：1000 epochs，batch 64 prompts × 8 images
  - KL正则化系数β=10（小规模），clip范围ε未明确给出
  - 评估：50 denoising steps，classifier-free guidance scale=5.0
