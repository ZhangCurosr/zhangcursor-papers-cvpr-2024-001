---
title: "SDPose-Tokenized-Pose-Estimation-via-Circulation-Guide-Self"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chen_SDPose_Tokenized_Pose_Estimation_via_Circulation-Guide_Self-Distillation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:48:10"
field: "人体姿态估计"
keywords: ["人体姿态估计", "自蒸馏", "Transformer", "轻量化模型", "Multi-Cycled Transformer", "知识蒸馏"]
innovations: ["首次发现循环 forward 可在不增加参数的情况下增大 transformer 隐式深度，设计了 MCT 模块", "设计了首个面向 transformer 姿态估计的自蒸馏范式，利用同向量空间特性实现无额外对齐的蒸馏", "通过训练时多轮推理、推理时单轮执行的方式，在参数量不变的情况下显著提升小型模型性能"]
benchmarks: ["MSCOCO val", "MSCOCO test-dev", "CrowdPose test"]
---

# 论文速读：SDPose: Tokenized Pose Estimation via Circulation-Guide Self-Distillation

## 一句话总结
SDPose 提出了一种循环引导的自蒸馏框架，通过 Multi-Cycled Transformer（MCT）模块让 token 多次循环经过相同的 transformer 层以增强模型隐式深度，并设计自蒸馏范式将多轮推理知识提取到单次前向模型中，在参数量和计算量不变的情况下显著提升了小型 transformer 人体姿态估计模型的性能。

## 研究问题与动机
- **小型 transformer 模型欠拟合严重**：现有 SOTA 的 transformer 姿态估计模型（如 ViTPose 等）参数量高达数亿，无法部署到边缘设备；而轻量化模型因参数量小、表达能力有限，容易出现欠拟合，性能显著下降。
- **增大模型深度的直接方法代价高昂**：堆叠更多 transformer 层可以加深模型，但会线性增加参数量和计算量，违背轻量化目标。
- **传统蒸馏方法存在缺陷**：特征蒸馏需要额外操作对齐向量空间，可能导致性能下降；同时需要单独训练一个强大的教师网络，训练成本巨大。
- **现有轻量化方法难以兼顾性能与效率**：如 PPT 虽减少了计算量，但以性能下降为代价；DistilPose 需要额外教师网络；轻量模型间仍有较大的性能提升空间。

## 核心贡献（创新点）
- **首次发现循环 forward 可增大隐式深度而不增加参数**：设计了 Multi-Cycled Transformer（MCT）模块，让 token 多次循环经过相同的 transformer 层，使小型模型等效于更深的 transformer 网络，缓解欠拟合问题。
- **设计了首个面向 transformer 姿态估计的自蒸馏范式**：利用 MCT 中相邻循环的输出处于同一向量空间这一性质，无需额外变换即可实现蒸馏，将多轮推理知识提取到单次前向模型中，实现训练时多轮推理、推理时单轮执行的效率-性能平衡。
- **在多个基准上验证了方法的广泛适用性**：不仅在 MSCOCO 和 CrowdPose 数据集上达到同规模 SOTA，还可迁移到 DeiT-Tiny 图像分类任务，证明方法对 transformer 模型具有通用性。

## 方法详解
- **Multi-Cycled Transformer (MCT) 模块**：基于 TokenPose 的框架，将图像特征经 backbone 提取后划分为 patch 形成视觉 token（VT），并引入 K 个可学习的关键点 token（KT）作为输入。MCT 模块将 token 经过 N 个循环送入相同的 transformer encoder 层，第 i 个循环的输出作为第 i+1 个循环的输入，最终以第 N 轮输出进行预测。这样做在不增加参数的前提下，增加了模型的有效深度（latent depth）。
- **自蒸馏策略**：训练时，令第 i+1 轮的 token 输出指导第 i 轮的 token 输出，利用相同向量空间的特性直接以 MSE 损失进行蒸馏，无需额外的对齐操作。将所有循环的知识逐步蒸馏到第一轮输出，推理时仅需一次前向传递。
- **损失函数设计**：
  - **关键点 token 蒸馏损失**：$L_{kt} = \sum_{i=1}^{N-1} MSE(KT_i, KT_{i+1})$
  - **视觉 token 蒸馏损失**：$L_{vt} = \sum_{i=1}^{N-1} MSE(VT_i, VT_{i+1})$
  - **预测损失**：每个循环的输出均通过预测头得到预测结果 $P_i$，与 ground truth 计算 MSE：$L_{pose} = \sum_{i=1}^{N} MSE(P_i, GT)$
  - **总损失**：$L = L_{pose} + \alpha_1 L_{kt} + \alpha_2 L_{vt}$，其中 $\alpha_1 = \alpha_2 = 5 \times 10^{-6}$
- **可视化分析**：注意力可视化显示第一轮循环中关键点 token 的注意力逐渐收缩至局部，而第二轮循环中注意力扩展至全局，表明多轮循环带来了更丰富的全局信息；参数分布可视化显示 SDPose 的近零参数明显少于 TokenPose，证明参数被更充分地学习。

## 实验与结果
- **数据集**：MSCOCO（train2017 57K 训练，val2017 5K + dev2017 20K 评估，17 keypoints）和 CrowdPose（20K 图像，约 80K 人，14 keypoints，遮挡严重场景）。
- **评估基线**：TokenPose-S/V1/V2/B、OKDHP、PPT、DistilPose、PRTR、Poseur、SimpleBaseline 等。
- **主要结果（MSCOCO val，256×192 输入）**：
  - **SDPose-T**：4.4M 参数、1.8 GFLOPs，AP = 69.7%，相比 TokenPose-S-V1（6.6M，2.4 GFLOPs，AP 69.5%）参数减少 33.3%、计算减少 25.0%、AP 提升 0.2%。
  - **SDPose-S-V1**：6.6M 参数、2.4 GFLOPs，AP = 72.3%，相比 TokenPose-S-V1（同规模）AP 提升 2.8%。
  - **SDPose-S-V2**：6.2M 参数、4.7 GFLOPs，AP = 73.5%，相比 TokenPose-S-V2（同规模）AP 提升 1.7%，**达到同规模轻量模型中的 SOTA**。
  - **SDPose-B**：13.2M 参数、5.2 GFLOPs，AP = 73.7%，相比 TokenPose-B 提升 0.5%。
- **MSCOCO test-dev 结果**：SDPose-S-V2 AP = 73.5%（SOTA 轻量 heatmap 方法），SDPose-Reg AP = 72.1%，相比 DistilPose-S 提升 1.1%。
- **CrowdPose 结果**：SDPose-S-V1 AP = 57.3%（↑1.6%），SDPose-S-V2 AP = 64.5%（↑2.2%）。
- **与回归方法对比**：SDPose-Reg 相比 DistilPose-S 有 1.1% AP 提升，且参数量相同。
- **消融实验**：完整的三项损失（$L_{pose}$、$L_{kt}$、$L_{vt}$）组合达到最佳性能（72.3% AP）；2 循环优于 3 循环，说明过多循环可能丢失局部信息。
- **扩展性**：与 PPT 结合后，在保持 SDPose-S-V1 性能的同时将计算量降至 PPT-S 水平（6.6M 参数、2.0 GFLOPs，AP 72.3%）；迁移到 DeiT-Tiny 分类任务，SDPose 在 ImageNet 上 AP 从 72.2% 提升至 72.7%。

## 相关工作脉络
- **TokenPose**：本文的基础模型，使用可学习关键点 token 与视觉 token 拼接后输入 transformer，SDPose 在其 transformer 模块基础上引入循环机制与自蒸馏。
- **ViTPose**：纯 transformer 架构的 SOTA 姿态估计方法，参数量达 6.32 亿，计算量 122.9B FLOPs，本文聚焦于解决其小规模版本欠拟合问题。
- **DistilPose**：通过模拟 heatmap 损失将教师网络知识蒸馏到回归式学生网络，但需单独训练教师网络；SDPose 通过自蒸馏避免了额外教师网络。
- **PPT**：通过裁剪低注意力 image token 减少计算量，但以性能下降为代价；SDPose 可与 PPT 结合，在同等计算量下显著提升性能。
- **Be Your Own Teacher / Born-Again Neural Network**：早期的自蒸馏工作，分别在空间深度和时间迭代维度上蒸馏；SDPose 是首个将自蒸馏应用于 transformer 姿态估计的研究。
- **OKDHP**：在线知识蒸馏方法，需要额外教师网络；SDPose 以自蒸馏方式避免了教师网络的训练开销。

## 局限性与未来方向
- **循环次数存在最优值**：消融实验表明 2 循环优于 3 循环，过多循环可能导致局部信息丢失，如何自适应选择循环次数值得进一步研究。
- **仅在标准姿态估计数据集上验证**：虽然验证了 MSCOCO 和 CrowdPose，但未见在 3D 姿态估计、多人姿态估计等更复杂场景的验证。
- **自蒸馏的蒸馏方向固定为从后向前**：是否可以从多个方向蒸馏（如双向或多对多）以及蒸馏策略的进一步优化空间尚未探索。
- **与数据增强等训练策略的结合潜力**：论文未讨论与常见训练增强策略（如随机翻转、MixUp 等）的配合效果。

## 研究启发与可借鉴点
- **"相同参数、更多轮次"的思路可迁移**：对于任何 transformer-based 架构，通过循环使用相同层来增加隐式深度，是一种零参数增长的模型增强策略，可推广到分割、检测等任务。
- **自蒸馏避免了额外教师网络开销**：利用相邻轮次输出处于同一向量空间的性质直接蒸馏，无需对齐操作，这一设计简洁高效，值得在其它需要多轮推理的任务中借鉴。
- **循环+蒸馏的解耦设计**：MCT 负责增强表达能力（训练时），自蒸馏负责知识压缩（训练后推理单轮），这种"训练增强、推理不变"的范式对部署敏感的边缘场景具有实用价值。
- **与 PPT 等轻量方法兼容**：本文证明 MCT 可与 token 裁剪等方法结合使用，提示我们可以在不同维度（结构精简 + 循环增强）上叠加轻量化策略。

## 关键术语表
- **SDPose**：一种循环引导的自蒸馏人体姿态估计框架，通过在 transformer 层上循环 forward 并结合自蒸馏来提升小型模型性能。
- **Multi-Cycled Transformer (MCT)**：将 token 多次循环经过相同 transformer 层的模块，在不增加参数的前提下增大模型隐式深度。
- **隐式深度 (Latent Depth)**：定义为一个完整推理过程中实际经过的 transformer 层总深度，MCT 通过循环而非增加层数来提升隐式深度。
- **自蒸馏 (Self-Distillation)**：在同一模型内部，利用较深/较晚阶段的输出指导较浅/较早阶段的 learning，无需额外教师网络。
- **TokenPose**：基于可学习关键点 token 的 transformer 姿态估计基线方法，本文在其基础上进行改进。
- **heatmap-based vs. regression-based**：两种主要输出表示，前者预测热力图后解码坐标，后者直接回归关键点坐标。
- **PPT (Token-Pruned Pose Transformer)**：一种通过裁剪低注意力 image token 来降低计算量的轻量 transformer 姿态估计方法。

## 可复现要素
- **数据集**：MSCOCO（公开）和 CrowdPose（公开），均在论文中声明使用。
- **代码**：论文未明确提及代码开源，但所有实验均在 MMPose 框架下进行（已重新训练和评估）。
- **权重**：论文未提及权重是否开源。
- **关键超参**：训练 300 epochs，Adam 优化器，初始学习率 1e-3，在 epoch 200 和 260 各 decay 10 倍；损失权重 $\alpha_1 = \alpha_2 = 5 \times 10^{-6}$；输入尺寸 256×192；每 GPU 64 样本，8×V100 GPU。
