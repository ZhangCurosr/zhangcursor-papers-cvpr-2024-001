---
title: "Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:50:56"
---

# 论文速读：Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners

## 一句话总结
本文提出了一种免训练的跨模态扩散生成框架，利用预训练 ImageBind 模型作为潜空间对齐器，通过梯度回传在去噪过程中实时引导视觉与音频生成轨迹，首次在开放域实现了文本可控的视频-音频联合生成，并显著提升了 V2A、I2A、A2V 及 Joint-VA 四类任务的视听对齐质量。

## 研究问题与动机
- **单模态生成割裂现实需求**：现有主流工作专注独立生成视频或音频，缺乏跨模态协同能力，难以支撑影视工业等需视听同步创作的场景。
- **两阶段流水线存在累积误差**：先 T2V 再 V2A 或先 T2A 再 A2V 的组合方案依赖的专用跨模态模型能力有限，往往局限于特定下游领域且生成质量不佳。
- **联合生成任务缺乏语义控制**：既有 Joint-VA 工作（如 MM-Diffusion）仅支持无条件生成且受限于小域数据，无法根据自由文本提示进行开放域创作。
- **从头训练巨型多模态模型成本过高**：现成单模态扩散模型（AudioLDM、AnimateDiff 等）已具备强大生成能力，核心瓶颈在于如何以低成本桥接它们，而非重新训练。

## 核心贡献（创新点）
- **免训练跨模态生成新范式**：提出通过共享隐式语义空间桥接预训练单模态扩散模型，无需额外训练开销即可实现视听双向条件生成。
- **扩散潜空间对齐器（Diffusion Latent Aligner）**：将 ImageBind 的多模态嵌入距离转化为扩散去噪梯度的指导信号，类比重 Classifier Guidance 机制实现实时跨模态对齐。
- **双重/三角形损失设计**：针对音频语义信息稀疏的缺陷，引入文本提示作为第三锚点构建双重或三角形距离约束，有效缓解单向引导失效问题。
- **引导式提示词调优（Guided Prompt Tuning）**：在推理阶段直接对文本 embedding 进行梯度更新，补偿跨模态语义偏差，改善音频到视频生成时的时序一致性与内容对齐。
- **首个开放域文本引导联合生成**：系统性覆盖 V2A、I2A、A2V 与 Joint-VA 四项任务，在多种数据集上验证了方法的通用性与 SOTA 性能。

## 方法详解
- **潜变量清洁预测**：在扩散第 $t$ 步，由噪声预测网络输出 $\hat{\epsilon}$，通过公式 $\tilde{z}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} z_t - \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}} \hat{\epsilon}$ 反推当前步对应的清洁潜变量，无需对下游多模态模型进行噪声扰动重训练。
- **多模态梯度引导**：将 $\tilde{z}_0$ 与条件模态 $x^{M_2}$ 分别输入 ImageBind 编码器，计算嵌入空间距离作为对齐损失：$\mathcal{L} = 1 - \mathcal{F}(E^{M_1}(\tilde{z}_0), E^{M_2}(x^{M_2}))$，其中 $\mathcal{F}$ 为余弦相似度。通过链式法则回传梯度至 $z_t$ 并更新：$\hat{z}_t = z_t - \lambda_1 \nabla_{z_t} \mathcal{L}$，逐步偏转去噪轨迹。
- **Dual/Triangle Loss**：为缓解音频语义不完整问题，A2V 任务联合视觉与文本双约束：$\mathcal{L}_{a2v} = \mathcal{F}(e_v, e_a) + \mathcal{F}(e_v, e_p)$；V2A 任务对称设计；Joint-VA 任务构建三角形损失强制三者互相贴近：$\mathcal{L}_{\text{joint-va}} = \mathcal{F}(e_v, e_p) + \mathcal{F}(e_v, e_a) + \mathcal{F}(e_a, e_p)$。
- **Guided Prompt Tuning**：在去噪开始前 detach prompt embedding $y$，保留从 $y$ 到多模态损失的计算图，反向更新文本嵌入：$\hat{y} = y - \lambda_2 \nabla_y \mathcal{L}$
