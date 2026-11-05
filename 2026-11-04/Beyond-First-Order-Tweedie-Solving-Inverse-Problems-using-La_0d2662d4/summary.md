---
title: "Beyond-First-Order-Tweedie-Solving-Inverse-Problems-using-La"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:57:03"
---

# 论文速读：Beyond-First-First-Order-Tweedie-Solving-Inverse-Problems-using-La

## 一句话总结
本文提出 STSL（Second-order Tweedie sampler from Surrogate Loss），通过在潜扩散模型中构建融合观测一致性项与 Hessian 迹近似的代理损失，实现了高效且无偏的二阶 Tweedie 后验采样；该方法将线性逆问题的扩散步数从 1000 步降至 50 步，在 FFHQ 与 ImageNet 上显著超越 PSLD 与 P2L，并首次支持对含真实腐蚀图像的文本引导高保真编辑。

## 研究问题与动机
- **核心问题**：如何利用预训练的潜扩散模型（LDM）高效求解线性逆问题 $ \mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n} $，即从后验分布 $ p_0(X_0 | Y=\mathbf{y}) $ 中采样恢复原始图像。
- **一阶 Tweedie 的固有偏差**：现有 SOTA 求解器（PSLD、P2L）依赖 Tweedie 一阶矩近似条件期望 $ \mathbb{E}[X_0|X_t] $，该 Jensen 间隙会导致生成结果回归均值，丧失高频细节并引入伪影。
- **二阶近似的计算不可行性**：理论上二阶修正可消除偏差，但需计算得分函数 Jacobian 或 Hessian（$ \mathcal{O}(d^2) $ 复杂度），在潜空间高分维Setting下使标准逆向扩散过程无法运行。
- **真实腐蚀图像编辑的失效**：现有文本引导编辑方法（如 NTI）依赖空文本嵌入对齐输入，面对模糊、低分辨率等现实退化时分布偏移严重，且传统 Cross-Attention Control 难以兼顾局部编辑与全局内容保持。

## 核心贡献（创新点）
1. **提出 STSL 二阶 Tweedie 采样器**：设计代理损失函数，仅需一阶梯度（去噪器输出）即可估计 Hessian 迹，从理论上给出对后验对数似然的严格下界，本质区别于一阶方法的均值回归偏差。
2. **改进初始化与随机平均优化**：放弃标准高斯初始化，改用 Encoder 输出经 DDIM 正向过程得到的潜变量起始以降低离散化误差；每扩散步执行 K 次 proximal gradient 随机平均，有效累积期望估计。
3. **首次实现含腐蚀图像的高效文本编辑**：将 STSL 的反转能力与 Cross-Attention Tuning (CAT) 串联，在 50 步内同步完成去腐蚀与语义编辑，解决现有方法在真实退化输入上编辑失败
