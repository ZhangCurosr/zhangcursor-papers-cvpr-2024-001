---
title: "Diffusion-Model-Alignment-Using-Direct-Preference-Optimizati"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:23:19"
field: "生成模型对齐"
keywords: ["扩散模型", "偏好优化", "DPO", "文本到图像生成", "对齐学习"]
innovations: ["将DPO方法适配到扩散模型，推导基于ELBO的可微偏好优化损失", "证明AI反馈可替代人类偏好训练扩散模型", "3.5B参数DPO微调模型击败6.6B完整SDXL pipeline"]
benchmarks: ["PartiPrompts", "HPSv2", "Pick-a-Pic v2"]
---

# 论文速读：Diffusion Model Alignment Using Direct Preference Optimization

## 一句话总结
本文提出Diffusion-DPO方法，将直接偏好优化(DPO)拓展至扩散模型，通过人类偏好数据直接微调扩散模型的对齐目标，无需训练独立奖励模型或强化学习，在SDXL-1.0上实现了超越基线及更大规模增强版本的SOTA效果。

## 研究问题与动机
- **问题**：大语言模型(LLM)已广泛采用RLHF/DPO等偏好对齐技术，但文生图扩散模型的对齐研究仍处于初级阶段，缺乏有效的偏好学习方法。
- **现有方法不足**：
  1. 当前主流做法仅依赖手工筛选的高质量图像进行监督微调，或依赖美学分类器筛选数据，缺乏细粒度偏好建模。
  2. 像素级梯度优化方法(如DRaFT、AlignProp)存在模式崩溃，且仅支持有限的反馈类型。
  3. RL-based方法(DPOK、DDPO)无法稳定泛化到开放词汇场景，性能随提示词数量增加而下降。

## 核心贡献（创新点）
1. **首次提出Diffusion-DPO框架**：将DPO思想迁移至扩散模型，通过ELBO推导可微偏好优化目标，避免显式训练奖励模型和强化学习步骤。
2. **解决扩散模型似然不可 tractable 问题**：利用前向扩散过程近似反向路径分布，将期望外推为单步采样，实现高效梯度下降训练。
3. **开放词汇设置下实现SOTA对齐效果**：在PartiPrompts和HPSv2基准上，DPO微调的SDXL-base(3.5B参数)击败完整SDXL-pipeline(6.6B参数，含refiner)。
4. **证明AI反馈训练的有效性**：使用PickScore、HPSv2等预训练评分模型构造伪标签进行训练，效果接近人类偏好，为规模化对齐开辟路径。

## 方法详解
**核心思想**：将DPO的奖励重参数化适配到扩散模型的似然框架中。

**关键推导步骤**：
1. **扩散模型似然定义**：引入隐变量 $x_{1:T}$，定义整条路径的奖励 $R(c, x_{0:T})$，使得 $r(c, x_0) = \mathbb{E}_{p_\theta(x_{1:T}|x_0,c)}[R(c, x_{0:T})]$。

2. **ELBO边界优化**：用联合KL散度替代边缘KL散度，得到路径级优化目标：
   $$\max_{p_\theta} \mathbb{E}[r(c, x_0)] - \beta \mathbb{D}_{KL}[p_\theta(x_{0:T}|c) \| p_{ref}(x_{0:T}|c)]$$

3. **Jensen不等式外推**：对反向传播路径取随机时间步 $t$，将期望转移为单步分布比值：
   $$L_{DPO-Diffusion} \leq -\mathbb{E}\left[\log \sigma\left(\beta T \log \frac{p_\theta(x_{t-1}^w|x_t^w)}{p_{ref}(x_{t-1}^w|x_t^w)} - \beta T \log \frac{p_\theta(x_{t-1}^l|x_t^l)}{p_{ref}(x_{t-1}^l|x_t^l)}\right)\right]$$

4. **前向过程近似**：用前向噪声分布 $q(x_t|x_0)$ 近似反向采样，最终导出可计算损失：
   $$L(\theta) = -\mathbb{E}\left[\log \sigma\left(-\beta T \omega(\lambda_t)\left[(\|\epsilon^w - \epsilon_\theta(x_t^w,t)\|^2 - \|\epsilon^w - \epsilon_{ref}(x_t^w,t)\|^2) - (\text{同理}^l)\right]\right)\right]$$

**损失含义**：鼓励模型在去噪过程中对偏好样本($x^w$)的预测误差减小幅度大于劣势样本($x^l$)，$\beta$控制正则化强度。

## 实验与结果
**数据集**：Pick-a-Pic v2，包含851,293对Crowdsourced偏好标注，58,960个唯一提示词。

**评估基准**：
- PartiPrompts (1,632个提示词)
- HPSv2 (3,200个提示词)
- 评估维度：通用偏好(Q1)、视觉吸引力(Q2)、提示词对齐度(Q3)，5位标注者多数投票

**主要结果**：
| 对比对象 | PartiPrompts通用偏好胜率 | HPSv2通用偏好胜率 | 备注 |
|---------|------------------------|------------------|------|
| SDXL-base vs DPO-SDXL | 70.0% | 64.7% | 同参数量对比 |
| SDXL-(base+refiner) vs DPO-SDXL | 69% | 64% | 3.5B vs 6.6B参数 |
| DPO-SDXL vs SDXL-base (People类别) | 73.4% | - | 人物生成优势显著 |
| DPO-SDXL vs SDXL-(base+refiner) (People类别) | 67.2% | - | 虽略降但仍领先 |

**超参数**：SDXL采用 $\beta=5000$，SD1.5采用 $\beta=2000$；有效batch size=2048；AdamW/Adafactor优化器。

**AI反馈实验**：使用PickScore伪标签训练SD1.5，PartiPrompts胜率从59.8%提升至63.3%，验证了非人类反馈训练的有效性。

## 相关工作脉络
1. **DPO (Rafailov et al., 2023)**：本文方法的核心基础，将RLHF简化为直接偏好优化分类目标，本文将其适配至扩散模型域。
2. **DDPO / DPOK (Black et al., 2023; Fan et al., 2023)**：基于RL的扩散模型对齐方法，支持有限词汇但无法泛化至开放词汇，本文方法在泛化性和稳定性上更优。
3. **DRaFT / AlignProp (Clark et al., 2023; Prabhudesai et al., 2023)**：使用像素级梯度优化，存在模式崩溃风险且仅支持简单审美反馈，本文方法无此问题。
4. **DOODL (Wallace et al., 2023)**：推理时迭代优化单张图像，增加推理开销，本文方法训练阶段完成对齐、推理零额外成本。
5. **Emu (Dai et al., 2023)**：基于精选高质量图文对微调，缺乏细粒度偏好建模，本文使用成对偏好数据更精细地捕捉人类偏好分布。

## 局限性与未来方向
- **离线训练限制**：当前为离线算法，未探索在线学习机制，未来可结合在线偏好收集进一步提升性能。
- **个人/群体偏好适配**：当前为通用对齐，尚未探索针对个人或特定群体偏好的定制化微调。
- **数据质量瓶颈**：依赖Crowdsourced偏好数据，标注噪声可能影响上限，数据清洗和自动化标注是重要方向。
- **伦理风险**：网络收集数据的版权和内容安全问题需进一步关注（论文Supp.S7讨论）。

## 研究启发与可借鉴点
1. **ELBO边界技巧迁移**：利用证据下界将不可tractable的边缘似然转化为可计算的联合分布优化，该思路可推广至其他隐变量模型的对齐任务。
2. **AI反馈替代人类标注**：证明预训练评分模型可生成有效伪偏好数据，为低成本大规模对齐提供可行路径，值得在更多生成模型上验证。
3. **参数量效率优势**：3.5B参数的DPO模型击败6.6B的完整pipeline，提示在资源受限场景下优先选择偏好微调而非模型扩大。
4. **多目标联合优化潜力**：可同时训练CLIP、Aesthetic、PickScore等多维度偏好，平衡不同优化目标（如文中提到Aesthetics训练牺牲部分CLIP对齐）。
5. **隐含奖励模型提取**：DPO训练过程中隐式学习到的奖励函数可直接用于偏好分类评测，无需额外训练，可简化下游应用管线。

## 关键术语表
**Diffusion-DPO**：将直接偏好优化适配到扩散模型的对齐方法，通过ELBO推导可微损失函数。

**Pick-a-Pic**：包含85万对图像偏好的开源数据集，由Web应用用户标注，用于训练和评估扩散模型对齐。

**DPO (Direct Preference Optimization)**：绕过RLHF中显式奖励建模和强化学习的直接偏好优化方法，通过分类损失直接优化策略。

**ELBO (Evidence Lower Bound)**：用于处理隐变量模型边缘似然的下界估计，本文用于桥接扩散路径分布与观测图像分布。

**HPSv2 (Human Preference Score v2)**：衡量文生图模型人类偏好得分的基准数据集和评分模型。

**PartiPrompts**：Google Parti图像生成模型的评测基准，包含1,632个复杂提示词用于评估生成质量。

**SDXL (Stable Diffusion XL)**：Stability AI开发的开源文生图模型，本文在其base版本(3.5B参数)上进行DPO微调。

**AI Feedback (RLAiF)**：使用预训练奖励模型生成伪偏好标签替代人类标注进行对齐训练的方法。

## 可复现要素
- **数据集**：Pick-a-Pic v2（公开，huggingface可获取）
- **代码**：论文未明确声明开源，但提及使用公开SDXL-1.0 base模型
- **权重**：未明确公开，需自行复现训练
- **关键超参**：
  - $\beta \in [2000, 5000]$（SD1.5用2000，SDXL用5000）
  - 有效batch size: 2048
  - 优化器: AdamW (SD1.5) / Adafactor (SDXL)
  - 学习率: $\frac{2000}{\beta} \times 2.048 \times 10^{-8}$
  - 线性warmup: 25%
  - 训练分辨率: 固定方形分辨率
