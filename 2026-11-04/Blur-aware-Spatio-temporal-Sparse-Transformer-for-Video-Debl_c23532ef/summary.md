---
title: "Blur-aware-Spatio-temporal-Sparse-Transformer-for-Video-Debl"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Blur-aware_Spatio-temporal_Sparse_Transformer_for_Video_Deblurring_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:56:57"
field: "视频恢复与增强"
keywords: ["video deblurring", "spatio-temporal transformer", "sparse attention", "bidirectional propagation", "blur map estimation"]
innovations: ["提出无参数模糊图估计方法引导时空稀疏注意力", "设计模糊感知双向特征传播（BBFP）缓解光流误差累积", "在更长时序窗口（T=48-60）下实现高效视频去模糊，超越 SOTA"]
benchmarks: ["GoPro", "DVD"]
---

# 论文速读：Blur-aware-Spatio-temporal-Sparse-Transformer-for-Video-Deblurring

## 一句话总结
论文提出 BSSTNet，通过非可学习的光流估计模糊图，将密集时空注意力转化为稀疏形式，结合模糊图引导的双向特征传播（BBFP）和模糊感知时空稀疏Transformer（BSST），在更长时序窗口下高效恢复视频模糊区域，在 GoPro 和 DVD 数据集上超越 SOTA。

## 研究问题与动机
- **时序窗口受限**：主流 VRT/RVRT 等时空 Transformer 方法因自注意力计算复杂度高，通常只使用 2 帧短时序窗口，无法充分利用长程视频上下文信息。
- **误差累积问题**：基于光流的双向特征传播（如 BasicVSR++）在模糊帧中光流估计不准确，导致传播过程中引入模糊像素并产生误差累积。
- **模糊区域的稀疏性先验**：运动模糊区域在时空维度上天然稀疏，可与清晰区域形成互补，但现有方法未显式利用这一特性。
- **计算效率与性能的平衡**：扩展时序窗口会显著增加显存和 FLOPs，需要设计稀疏策略在保持性能的同时降低计算开销。

## 核心贡献（创新点）
- **提出无参数模糊图估计方法**：通过前后向光流幅值计算模糊图，无需额外训练参数即可提供运动模糊区域的先验指导。与学习型模糊估计方法（如 FlowInfer）的本质区别在于零参数、即插即用。
- **设计 BSST（模糊感知时空稀疏 Transformer）**：将密集时空注意力转化为稀疏形式，Query 选择模糊区域 token、Key/Value 选择清晰区域 token，在相同 FLOPs 下支持更长时序窗口（最大 60 帧）。
- **提出 BBFP（模糊感知双向特征传播）**：利用 sharp map 作为额外条件引导可变形卷积的偏移和掩码生成，避免相邻帧模糊像素的传播污染，区别于标准 Flow-guided Feature Alignment 的全像素对齐。
- **系统级 SOTA 性能**：在 GoPro（PSNR 35.98）和 DVD（PSNR 34.95）上均超越 Shift-Net+、RVRT 等最强基线，同时在 DVD 上比 Shift-Net+ 节省 13 GFLOPs 并提速 1.6 倍。

## 方法详解
**整体架构**：BSSTNet 包含三模块：① Blur Map Estimation（模糊图估计）→ ② BBFP（模糊感知双向特征传播）→ ③ BSST（模糊感知时空稀疏 Transformer）→ 解码器重建。

**模糊图估计（Section 3.2）**：
- 输入降采样视频序列，使用预训练 RAFT 估计前后向光流 $\mathbf{O}_{t \to t+1}$ 和 $\mathbf{O}_{t \to t-1}$
- 模糊图计算：$\hat{\mathbf{B}}_t = \sum_{i=1}^2 ((\mathbf{O}_{t \to t+1})_i^2 + (\mathbf{O}_{t \to t-1})_i^2)$，归一化后得到 $\mathbf{B}_t$，sharp map $\mathbf{A}_t = 1 - \mathbf{B}_t$
- 边界条件：$\mathbf{O}_{1 \to 0} = \mathbf{0}$, $\mathbf{O}_{T \to T+1} = \mathbf{0}$

**BBFP（Section 3.3）**：
- 双向特征传播公式：$\hat{\mathbf{F}}_t^j = \text{BFA}(\hat{\mathbf{F}}_t^{j-1}, \hat{\mathbf{F}}_{t-1}^j, \hat{\mathbf{F}}_{t-2}^j, \text{Warp}(\cdot), \mathbf{O}, \mathbf{A})$
- **BFA（Blur-aware Feature Alignment）**：将 sharp map $\mathbf{A}$ 作为额外条件输入可变形卷积（DCN），同时作为基础掩码加到 DCN mask 上，确保只有清晰区域特征参与传播

**BSST（Section 3.4）**：
- **Token 生成**：soft split 将特征切分为重叠 patch，拼接为 patch embeddings $\mathbf{z}$，经线性层得到 $z^q, z^k, z^v$
- **全局 token**：通过 depth-wise convolution 生成 pooled global tokens $g^k, g^v$，与窗口内 token 拼接扩大感受野
- **空间稀疏**：
  - 对模糊图 max pooling 得到窗口级模糊度 $\mathbf{U}$
  - 阈值判断：$\mathbf{Q}_{t,i,j} = 1$ if $\mathbf{U}_{t,i,j} \geq \theta$（$\theta=0.3$）
  - 拼接 temporal 维度：$\mathbf{S} = \text{Clip}(\sum_t \mathbf{Q}_t, 1)$
  - 仅保留 $\mathbf{S}_{i,j}=1$ 的窗口 token 构成 $\mathbf{I}^q, \mathbf{I}^k, \mathbf{I}^v$
- **时序稀疏（Query）**：对每个空间位置 $(i,j)$，选取 temporal 维度 Top-$K_q$ 最模糊的帧作为 Query
- **时序稀疏（Key/Value）**：选取每隔一帧（stride=2）中最清晰的帧（Top-$K_{kv}$ 最小模糊度），奇偶 BSST 交替选择，减少 50% KV 大小
- **稀疏注意力计算**：对保留的窗口执行 multi-head self-attention，未选中窗口保持标准窗口注意力，soft composition 输出

**损失函数**：L1 loss，Adam 优化器（$\beta_1=0.9, \beta_2=0.999$），初始学习率 $4 \times 10^{-4}$

## 实验与结果
- **数据集**：GoPro（2103 训练 / 1111 测试，1280×720）和 DVD（5708 训练 / 1000 测试）
- **实现细节**：8×A100 GPU，batch size=8，输入 crop 256×256，RAFT 预训练权重冻结；测试时 T=48, $K_q=K_{kv}=24$；训练时 T=24, $K_q=K_{kv}=12$；$\theta=0.3$, patch size=4, stride=2

**主要结果**：
- **GoPro**：PSNR 35.98 / SSIM 0.9792，超越 Shift-Net+（35.88 / 0.9790）0.10 dB
- **DVD**：PSNR 34.95 / SSIM 0.9703，超越 Shift-Net+（34.69 / 0.9690）0.26 dB / 0.0013 SSIM
- **效率对比**（DVD 单帧 256×256）：BSSTNet 133 GFLOPs / 28ms，比 RVRT（88.8 GFLOPs / 23ms）略高但 PSNR 更高（34.95 vs 34.30）；比 Shift-Net+（146 GFLOPs / 45ms）节省 13 GFLOPs 且提速 1.6×

**消融实验**：
- BBFP 贡献：+0.21 dB PSNR / +0.0011 SSIM（Table 5）
- BFA vs SFFA：+0.13 dB（Table 6）
- BSST 贡献：+0.85 dB PSNR（Table 5）
- 稀疏策略对比：Top 50% tokens 达到全量 97% 性能但仅用 43% FLOPs（Table 7）
- 时序长度扩展：T=60 时 SST OOM，BSST 仍稳定运行且 PSNR+0.06 dB（Table 4）

## 相关工作脉络
- **BasicVSR++ [1]**： aggressive 双向传播代表，但光流误差累积问题突出；本文通过 BFA 和模糊图引导规避此问题
- **VRT [11] / RVRT [12]**：时空 Transformer 代表，但受限于 2 帧短窗口；本文 BSST 在同等 FLOPs 下支持更长时序（48-60 帧）
- **Shift-Net+ [10]**：当前 SOTA，使用 grouped spatio-temporal shift；本文在其基础上进一步扩展时序并利用稀疏注意力提升精度
- **STDAN [24] / STFAN [26]**：flow-guided attention 代表；本文 BBFP 额外引入 sharp map 条件，区分于纯光流引导的对齐
- **RAFT [22]**：预训练光流估计器；本文直接复用其权重估计光流，不引入额外训练开销
- **FuseFormer [14]**：soft split + window partition 机制；本文借鉴该设计用于 patch embedding 生成

## 局限性与未来方向
- **光流质量依赖**：模糊图估计依赖 RAFT 光流精度，极端模糊场景下光流可能失效，导致模糊图不准确
- **非可学习模糊图**：当前方法无参数，无法自适应不同模糊类型或风格；可探索轻量级可学习模糊估计模块
- **固定阈值策略**：$\theta$ 和 Top-K 为手动设定，缺乏场景自适应能力
- **单次多帧输入**：当前处理固定长度片段（T=48），对于超长视频的 online 部署尚未验证
- **仅测试静态模糊**：未涉及运动模糊与离焦模糊混合场景或低光照条件

## 研究启发与可借鉴点
- **稀疏注意力设计的通用范式**：Query 选模糊区域、Key/Value 选清晰区域的策略可迁移至视频插值、超分等时序任务
- **无参数先验引导的设计哲学**：利用光流幅值构造模糊图无需额外训练，可作为即插即用模块嵌入其他视频恢复模型
- **时序稀疏 + 空间稀疏的组合策略**：时空双维度稀疏化（Query 时序 Top-K + KV 隔帧采样）的思路适用于任何长序列 Transformer
- **Sharp map 作为 DCN 额外条件**：将二值掩码融入可变形卷积的设计可推广至 video inpainting、frame interpolation 等需要区域选择性对齐的任务
- **效率-性能权衡的分析方法**：Table 4 中不同时序长度的内存/FLOPs/性能三维对比为后续工作提供评估模板

## 关键术语表
- **BSSTNet**：Blur-aware Spatio-temporal Sparse Transformer Network，本文提出的视频去模糊网络
- **BBFP**：Blur-aware Bidirectional Feature Propagation，模糊感知双向特征传播模块
- **BFA**：Blur-aware Feature Alignment，利用 sharp map 引导的可变形特征对齐操作
- **BSST**：Blur-aware Spatio-temporal Sparse Transformer，时空稀疏注意力核心模块
- **BSSA**：Blur-aware Sparse Spatio-temporal Attention，BSST 中的稀疏注意力机制
- **Blur Map / Sharp Map**：基于光流幅值计算的模糊/清晰区域概率图，$\mathbf{A} = 1 - \mathbf{B}$
- **Soft Split**：重叠 patch 分割操作，生成 patch embeddings 的前处理步骤
- **Top-K 稀疏选择**：沿时序维度选取最模糊/最清晰 K 个窗口 token 的策略

## 可复现要素
- **数据集**：GoPro 和 DVD 均为公开基准数据集
- **代码**：论文未提及代码开源状态（URL 指向项目页面 https://vilab.hit.edu.cn/projects/bsstnet，需自行确认）
- **权重**：RAFT 光流估计器使用官方预训练权重；BSSTNet 权重论文未声明开源
- **关键超参**：$\theta=0.3$, patch size $p=4$, stride $s=2$, 测试 T=48/$K_q=K_{kv}=24$, 训练 T=24/$K_q=K_{kv}=12$, batch size=8, lr=$4 \times 10^{-4}$, Adam $\beta_1=0.9, \beta_2=0.999$, L1 loss
