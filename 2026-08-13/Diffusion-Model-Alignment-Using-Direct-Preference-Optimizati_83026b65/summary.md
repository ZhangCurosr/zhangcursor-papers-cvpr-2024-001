---
title: "Diffusion-Model-Alignment-Using-Direct-Preference-Optimizati"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:50:24"
field: "文本到图像生成与对齐"
keywords: ["Diffusion-DPO", "Direct Preference Optimization", "diffusion model alignment", "human preference learning", "SDXL", "RLHF alternative"]
innovations: ["首次将 DPO 框架推广至扩散模型并推导可微损失", "在 SDXL-1.0 上实现超越完整 base+refiner pipeline 的开放词汇对齐效果", "验证 AI 反馈（PickScore/HPSv2）可替代人工标注进行高效扩散模型微调"]
benchmarks: ["PartiPrompts", "HPSv2", "Pick-a-Pic v2"]
---

# 论文速读：Diffusion-Model-Alignment-Using-Direct-Pference-Optimization

## 一句话总结
本文提出 **Diffusion-DPO**，首次将 LLM 领域的直接偏好优化（DPO）方法迁移至文本到图像扩散模型，通过在 851K 对人工标注的偏好数据上直接优化去噪网络，使 SDXL-1.0 base 模型在视觉吸引力与文本对齐度上显著超越完整 SDXL pipeline（含 6.6B 参数的 refinement 模块）。

## 研究问题与动机
- 现有主流文本到图像扩散模型（如 SDXL）通常仅采用单阶段 web-scale 预训练，缺乏类似 LLM 的两阶段"预训练 + 对齐"范式。
- 已有扩散模型对齐方法（如 DRaFT、AlignProp）依赖像素级奖励梯度直接更新生成模型，容易出现模式坍塌，且难以稳定泛化至开放词汇场景。
- RL-based 方法（DDPO、DPOK）虽能提升奖励分数，但在 prompt 数量增加时性能急剧下降，且训练不稳定。
- 人类偏好数据（如 Pick-a-Pic）已有规模化的成对标注，但如何高效利用此类数据进行端到端对齐仍缺乏通用框架。

## 核心贡献（创新点）
- **将 DPO 推广至扩散模型**：通过引入证据下界（ELBO）与 Jensen 不等式，将离散时间扩散路径上的不可 tractable 似然转化为可微损失。
- **推导闭式 Diffusion-DPO 损失**：得到仅依赖单步加噪/去噪误差差值的 Bradley-Terry 分类目标，避免显式训练奖励模型。
- **实证 SDXL 对齐效果**：在 PartiPrompts 上以 69% 胜率优于 SDXL-(base + refiner)，同时参数量仅为其 53%。
- **验证 AI 反馈可行性**：使用 PickScore、HPSv2 等预训练评分器生成伪标签进行训练，效果可与真人标注媲美，为后续规模化提供路径。
- **揭示隐式奖励建模能力**：DPO-SDXL 在 Pick-a-Pic 验证集上的二元偏好分类准确率（72.0%）超越现有识别模型。

## 方法详解
- **问题设定**：给定参考模型 $p_{\mathrm{ref}}$ 生成的成对图像 $\{x_0^w \succ x_0^l \mid c\}$，学习策略 $p_\theta$ 使其更符合人类偏好。
- **理论桥梁**：利用扩散模型的路径分布 $p_\theta(\pmb{x}_{0:T}|c)$ 替代难以计算的边缘似然 $p_\theta(\pmb{x}_0|c)$，并将 KL 正则项扩展为路径联合分布上的上界。
- **核心推导**：对期望项应用 Jensen 不等式后，将时间步 $t$ 采样均匀分布，最终损失简化为：
  $$
  L(\theta) = -\mathbb{E}\left[\log \sigma\left(-\beta T \omega(\lambda_t)\left[\|\epsilon^w-\epsilon_\theta(\pmb{x}_t^w,t)\|^2 - \|\epsilon^l-\epsilon_\theta(\pmb{x}_t^l,t)\|^2 - (\text{ref 对应项})\right]\right)\right]
  $$
- **训练机制**：对每对 $(x_0^w, x_0^l)$ 随机采样时间步 $t$ 与噪声 $\epsilon$，通过前向加噪得到 $\pmb{x}_t$，计算当前网络与参考网络在相同噪声下的 MSE 差异，激励胜图像去噪更准确。
- **超参设置**：$\beta \in [2000, 5000]$，学习率采用逆缩放形式 $\frac{2000}{\beta} \cdot 2.048 \times 10^{-8}$，配合 25% linear warmup。

## 实验与结果
- **数据集**：Pick-a-Pic v2，851,293 对偏好样本，58,960 个唯一 prompt。
- **评估基准**：PartiPrompts（1632 prompt）、HPSv2（3200 prompt），采用 5 人 crowdworker 多数投票。
- **核心结果**：
  - DPO-SDXL vs. SDXL-base：PartiPrompts Q1 胜率 70.0%，HPSv2 胜率 64.7%。
  - DPO-SDXL vs. SDXL-(base + refiner)：PartiPrompts Q1 胜率 69%，HPSv2 胜率 64%。
  - 在 People 细分类别上以 67.2% 胜率超越 base+refiner（对比 base 为 73.4%）。
  - HPSv2 reward 得分 28.16，登顶 leaderboard。
- **AI 反馈实验**：基于 PickScore 伪标签训练 SD1.5，PartiPrompts Q1 胜率从 59.8% 提升至 63.3%。

## 相关工作脉络
- **DDPO / DPOK**：基于 RL 在有限词汇表上优化扩散模型，本文方法在开放词汇与稳定性上优于它们。
- **DRaFT / AlignProp**：通过可微奖励梯度直接微调生成器，易出现模式坍塌；本文通过偏好分类损失规避该问题。
- **DOODL**：仅在推理时对单张图像进行 latent optimization，不改变模型参数；本文是端到端训练。
- **Emu**：依赖人工精选的高质量图文对进行 SFT，缺乏利用大规模噪声偏好数据的机制。
- **传统 RLHF**：需额外训练奖励模型并进行 PPO 等 RL 优化，计算成本高；DPO 框架通过重参数化去除显式奖励网络。

## 局限性与未来方向
- 作为离线算法，尚未探索在线交互学习场景。
- 对高置信度奖励信号（如 Aesthetic）的优化可能牺牲其他维度（如 CLIP 对齐）。
- 偏好数据来源于 SDXL-beta 与 Dreamlike，可能存在分布偏差。
- 个人/群体个性化对齐尚未充分研究。
- 数据清洗与规模化标注仍是影响最终性能的关键瓶颈。

## 研究启发与可借鉴点
- **DPO 框架的可迁移性**：去噪误差差值构造的偏好损失可推广至视频生成、3D 生成等时序扩散任务。
- **隐式奖励建模**：无需额外训练奖励网络即可从 DPO 目标中提取偏好判别信号，降低系统复杂度。
- **AI 反馈替代人力标注**：使用成熟评分器（PickScore、HPSv2）构建伪偏好数据，可作为数据清洗或冷启动对齐的有效手段。
- **逆缩放学习率设计**：$\beta$ 与 learning rate 的联动机制值得在其他 preference learning 任务中复现。
- **SFT 前处理的边界**：当 base 模型质量高于训练数据时，直接 SFT 反而有害，提示需审慎选择预对齐策略。

## 关键术语表
- **Diffusion-DPO**：将 DPO 目标函数适配到扩散模型的去噪网络，利用 ELBO 推导可微损失。
- **Direct Preference Optimization (DPO)**：绕过显式奖励建模，直接在成对偏好数据上优化生成策略的 RLHF 替代方法。
- **Evidence Lower Bound (ELBO)**：用于近似难计算对数似然的下界，此处用于处理扩散路径积分。
- **Pick-a-Pic**：包含 851K 对人工标注偏好的开放图文数据集，用于训练与评估扩散模型对齐。
- **Bradley-Terry 模型**：将成对比较转化为概率选择的统计模型，构成 DPO 损失的理论基础。
- **PartiPrompts / HPSv2**：用于评估文本到图像生成质量的两大 benchmark，前者侧重多样 prompt，后者为自动偏好评分基准。

## 可复现要素
- **数据集**：Pick-a-Pic v2 已公开，含 851,293 对标注。
- **代码/权重**：论文提供 Hugging Face 模型卡与 GitHub 仓库链接（具体见原文 supplement）。
- **关键超参**：SDXL $\beta=5000$，SD1.5 $\beta=2000$；batch size=2048；A100 GPU 16 卡；Adafactor/AdamW 优化器。
