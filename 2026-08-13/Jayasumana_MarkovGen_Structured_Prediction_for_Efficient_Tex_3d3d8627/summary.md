---
title: "MarkovGen: Structured Prediction for Efficient Text-to-Image Generation"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Jayasumana_MarkovGen_Structured_Prediction_for_Efficient_Text-to-Image_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:38:58"
field: "文本到图像生成与高效推理"
keywords: ["Text-to-Image Generation", "Markov Random Field", "Structured Prediction", "Token-based Generation", "Muse", "Mean-field Inference", "Efficient Sampling"]
innovations: ["首次将完全连接MRF引入离散token空间的文本到图像生成，显式建模全局token兼容性", "提出MarkovGen框架，用可微分MRF层替换Muse后期采样步，实现1.5倍加速与质量提升", "端到端训练MRF参数：自监督预训练+KL散度精调模仿目标模型最终分布"]
benchmarks: ["MS-COCO FID", "Parti Prompts Human Evaluation"]
---

# 论文速读：MarkovGen: Structured Prediction for Efficient Text-to-Image Generation

## 一句话总结
本文提出了一种轻量级的结构化预测方法，通过在离散token空间构建完全连接的马尔可夫随机场（MRF）模型，显式建模token间的空间与标签兼容性关系，从而将Muse文本到图像生成模型的推理速度提升1.5倍，同时改善图像质量并减少伪影。

## 研究问题与动机
- **现有方法计算代价高昂**：当前文本到图像生成模型（如扩散模型、自回归模型、Muse等）需要多次迭代采样才能确保图像不同区域既与文本提示对齐，又彼此兼容，这一过程计算成本极高。
- **单次并行解码质量退化严重**：Muse虽然采用并行解码策略（每次预测所有token），但单次推理会导致生成的图像质量显著下降，必须通过渐进式并行解码多次迭代来弥补。
- **独立采样缺乏全局协调**：现有token-based生成方法（如Muse）在每个采样阶段独立地为每个token位置选择值，未考虑不同token间的全局兼容性，导致结构不完整或纹理不一致等问题。
- **已有MRF/CRF方法不适用于离散token空间**：传统图像分割中的CRF依赖高斯空间核函数且仅考虑局部平滑约束，而token空间中的标签兼容性需要完全连接的图结构和数据驱动的兼容性学习。

## 核心贡献（创新点）
- **首次将MRF引入文本到图像生成的结构化预测**：提出基于完全连接MRF的概率图模型，在VQGAN离散token空间中显式建模token的空间位置关系与标签兼容性，通过均值场推理实现全局协调的token选择，区别于Muse等方法的独立token采样。
- **提出MarkovGen框架，以 negligible 代价加速Muse**：用轻量级MRF层替换Muse的最后若干采样步，MRF推理时间仅约0.29ms（相比Muse单步10.40ms），实现整体1.5倍加速的同时提升图像质量。
- **端到端可微分的MRF训练机制**：将MRF推理过程建模为可微分神经网络层，通过KL散度损失让MRF学习复现Muse最终预测结果，两个阶段的MRF模型仅需数小时即可完成训练。

## 方法详解
- **MRF建模框架**：将token图像表示为随机变量集合 $\mathbf{X} = [X_1, \ldots, X_n]$，其中 $n$ 为token网格位置数（base模型为256，SR模型为1024），标签空间为VQGAN词汇表大小 $V=8192$。概率分布采用Gibbs形式 $P(\mathbf{X}=\mathbf{x}) = \frac{1}{Z}\exp(-E(\mathbf{x}))$。
- **能量函数设计**：$E(\mathbf{x}) = \sum_i u_i(x_i) + \sum_i \sum_j p_{ij}(x_i, x_j)$，其中unary项 $u_i(x_i) = -f_i(x_i, y)$ 取自身网络预测logits的负值（反映单点置信度），pairwise项 $p_{ij}(x_i, x_j) = -c(x_i, x_j)s(i,j)$ 分解为标签兼容性矩阵 $\mathbf{W^c}$ 与空间相似性矩阵 $\mathbf{W^s}$ 的乘积。
- **完全连接图结构**：与图像分割CRF不同，本文MRF对所有token对建立边（$256\times256$ 或 $1024\times1024$ 空间矩阵，$8192\times8192$ 标签矩阵），不假设高斯空间核，所有权重通过学习获得。
- **均值场推理（Mean-field Inference）**：近似后验分布 $Q(\mathbf{X}) = \prod_i Q_i(X_i)$，迭代更新 $Q_i(k)$：先与空间邻居聚合（$\mathbf{W^s}$），再与标签兼容聚合（$\mathbf{W^c}$），最后加上unary logits并softmax归一化，直至收敛。
- **可微分训练**：推理过程的所有操作（矩阵乘法、softmax、加法）均可微，通过KL散度损失 $\mathcal{L}_{KL}(Q_{\text{MRF}} \| P_{\text{Muse final}})$ 端到端训练MRF参数；预训练阶段采用掩码token预测的交叉熵损失辅助初始化。

## 实验与结果
- **数据集与模型**：使用Muse官方1.7B参数模型（WebLI数据集训练），MRF在同一数据集上训练；评估基准为MS-COCO 256×256分辨率。
- **FID指标**：MarkovGen达到 **12.28**，优于Full Muse的13.13和Early Exit Muse的14.37，证实质量提升。
- **推理速度**：MRF单步推理仅0.29ms（base和SR相同），整体MarkovGen耗时281.03ms vs Muse 442.05ms，**加速1.5倍**。
- **人类评估**：在Parti prompts数据集上，MarkovGen相比Full Muse获得更高比例的人类偏好评分，且显著优于同速度Early Exit Muse，说明MRF有效修复了提前停止带来的质量损失。
- **采样步数配置**：Base模型保留前20步（共24步），SR模型保留前3步（共8步），后续步骤由MRF替代完成。

## 相关工作脉络
- **Muse [2]**：本文基线模型，采用BERT式编码器进行渐进并行解码，在每一步预测所有masked token并固定高置信度部分；MarkovGen在其基础上用MRF替代最后几步，本质区别在于引入了全局token兼容性约束而非独立采样。
- **DALL-E / Parti [34]**：自回归token生成模型，逐个token生成且依赖历史token；MarkovGen适用于并行解码范式，不改变主干架构即可增效。
- **扩散模型（DDPM, Stable Diffusion [24, 25]）**：通过多步去噪生成图像；MarkovGen针对离散token空间设计，与连续潜空间方法互补而非竞争。
- **完全连接CRF（Krahenbuhl & Koltun [14], Zheng et al. [36]）**：用于图像分割后处理，依赖高斯核空间相似性且关注语义平滑；MarkovGen将其推广至离散token空间，放弃高斯假设并学习完全数据驱动的标签兼容性。
- **Mask-Predict [17] / MaskGIT [3]**：并行解码的掩码生成方法；MarkovGen与之正交，可在并行解码输出之上叠加结构化post-processing。
- **文本生成的CRF解码（Sun et al. [30], Su et al. [29]）**：在序列模型输出上施加CRF约束；MarkovGen将其从一维序列扩展到二维图像token网格，处理全连接图结构而非链式结构。

## 局限性与未来方向
- **MRF未直接 conditioning 于文本提示**：当前文本引导仅通过unary logits间接传入，空间与标签兼容性权重对prompt内容不变；未来可探索text-conditioned CRF使兼容性自适应文本。
- **非联合训练**：Muse骨干与MRF层分阶段训练，unary分布未必最优适配MRF解码；联合微调可能进一步释放性能。
- **推理成本虽低但随分辨率增长**：MRF矩阵大小为 $n^2 \times V^2$，在更高维度token网格（如64×64以上）上内存与计算开销将显著上升，需进一步优化。
- **仅验证于Muse架构**：未在Parti、Cogview等其他token生成模型上验证通用性，泛化能力有待检验。

## 研究启发与可借鉴点
- **结构化后处理可大幅压缩采样步数**：对于多步迭代生成的模型（扩散、masked generative），可在后期用轻量级结构化模型替代部分昂贵步骤，值得推广至其他迭代生成范式。
- **完全连接图+均值场推理的可微分化设计**：将MRF/CRF推理展开为固定迭代次数的神经网络层并支持反向传播，是一种简单有效的"可微分图模型"范式，可迁移至视频生成、3D生成等任务。
- **预训练+精调两阶段策略高效落地**：先以自监督掩码预测预训练MRF参数，再以KL散度精调模仿目标模型行为，仅需数小时即可适配预训练模型，降低部署门槛。
- **与团队方向的结合机会**：若团队关注低延迟图像生成或视频token预测，可将MRF兼容性建模扩展到时序维度（3D MRF）或引入层级结构（多尺度MRF）以提升长序列一致性。

## 关键术语表
- **Markov Random Field (MRF)**：无向概率图模型，通过节点势能（unary）与边势能（pairwise）建模变量间局部依赖，本文用于编码图像token间的空间与语义兼容性。
- **Mean-field Inference**：近似推断方法，用因式分解分布 $Q(\mathbf{X}) = \prod_i Q_i(X_i)$ 逼近真实后验，通过迭代消息传递求解，本文用作MRF的高效近似推理器。
- **Progressive Parallel Decoding**：Muse采用的解码策略，每步并行预测所有masked token并固定高置信度部分，逐步缩小待预测区域直至完整图像生成。
- **VQGAN Token Space**：向量量化变分自编码器（VQVAE）产生的离散视觉词典，图像被划分为patch并以VQGAN码本中的token表示，本文在此离散空间建模兼容性。
- **Unaries**：MRF中每个节点的单点势函数，直接来源于上游Transformer模型的logits输出，反映该位置预测某token的初始置信度。
- **KL Divergence Loss**：相对熵损失，用于衡量MRF输出分布与Muse最终预测分布的差异，本文作为MRF参数的优化目标。

## 可复现要素
- **数据集**：WebLI（训练Muse与MRF）、MS-COCO（FID评估）、Parti Prompts（人类评估）
- **代码开源**：论文未提及代码开源情况
- **模型权重**：使用Muse官方提供的1.7B参数模型
- **关键超参**：base模型token网格16×16（256个位置），SR模型32×32（1024个位置），词汇表大小8192，base MRF保留前20步/共24步，SR MRF保留前3步/共8步，优化器ADAM，预训练掩码比例20%
