---
title: "Memory-Scalable and Simplified Functional Map Learning"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Magnet_Memory-Scalable_and_Simplified_Functional_Map_Learning_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:02:07"
field: "深度功能映射学习"
keywords: ["functional maps", "shape matching", "deep learning", "memory scalability", "differentiable optimization", "ZoomOut"]
innovations: ["基于 KeOps 的可微分内存可扩展稠密映射计算", "首次将可微分 ZoomOut 精化算法用于训练期自监督一致性损失", "单分支网络无需对线性系统求导即可实现 SOTA 级形状匹配"]
benchmarks: ["FAUST", "SCAPE", "SHREC19", "DT4D-H"]
---

# 论文速读：Memory-Scalable and Simplified Functional Map Learning

## 一句话总结
本文提出一种**内存可扩展、高效的功能映射学习管道**，通过利用功能映射的结构化特性，在不显式存储稠密点对点（p2p）映射矩阵的情况下完成训练与推理，并首次将可微分 ZoomOut 精化算法用于训练阶段，以自监督一致性损失替代传统双分支网络中的谱分支，实现了更简单、更稳定且性能接近 SOTA 的单分支功能映射学习框架。

## 研究问题与动机
- **现有双分支深度功能映射方法**依赖显式计算并存储 $n_2 \times n_1$ 大小的稠密软点对点映射矩阵 $\Pi$，导致**内存占用随顶点数平方增长**，难以扩展到高分辨率网格。
- 为保证谱分支中线性系统可逆，特征维度通常需设为 128 或 256，进一步拖慢计算。
- 对线性系统求解器**求梯度**会引入数值不稳定性（文献[15]已指出）。
- 尽管“proper”功能映射能提升精度，但现有方法为兼顾谱分支与点态分支往往妥协于低特征维度和近似采样，限制了可扩展性与精度。

## 核心贡献（创新点）
1. **内存可扩展的稠密映射计算**：基于 KeOps 库，将 $\Pi\Phi_1$ 的矩阵乘法转化为在线块级 Kernel 计算，**无需显式存储 $n_2 \times n_1$ 矩阵**，内存复杂度降至线性，支持 GPU 加速与自动微分。
2. **可微分 ZoomOut 精化层与训练期一致性损失**：将原 CPU 实现的 ZoomOut 算法适配为 GPU 可微分版本，并在训练中引入初始功能映射与精化后功能映射（取一阶主子矩阵）的 L2 一致性损失，以此作为结构正则化，**替代原有的谱分支**。
3. **首个无需线性系统求导的单分支功能映射学习网络**：结合前述两项技术，仅保留点态特征提取与可微分精化流程，特征维度可降至 $p=32$，显著提升训练稳定性与计算效率，同时在标准基准上取得与双分支方法相当甚至更优的结果。

## 方法详解
### 1. 可扩展稠密映射（Scalable Dense Maps）
- 传统方法先显式计算高斯核矩阵 $\Pi_{ij}=\exp(\delta_{ij})/\sum_k\exp(\delta_{ik})$，再乘以特征矩阵。本文利用公式：
  $$[\Pi\Phi_1]_i = L_i^{-1}\sum_{j} K([F_2]_i,[F_1]_j)[\Phi_1]_j$$
  其中 $K$ 为 RBF Kernel，$L_i$ 为行归一化因子。
- 借助 **KeOps 库**在 GPU 上按块计算 Kernel 值并进行沿 $j$ 维的求和，**始终不将整个 $n_2\times n_1$ 矩阵载入显存**，同时进行前向与反向传播。
- 测试时通过逐行最大索引（等价于最近邻搜索）提取点对点映射，同样可无显式距离矩阵地高效完成。

### 2. 可微分 ZoomOut（Differentiable ZoomOut）
- ZoomOut 算法迭代执行：用当前功能映射 $\mathbf{C}$ 通过 $\Phi_1\mathbf{C}^\top$ 与 $\Phi_2$ 做最近邻得到点态映射，再用该点态映射按式(2)更新功能映射，同时逐步增大谱基底大小 $K$。
- 用上述可扩展稠密映射**替换每次迭代的最近邻查询与矩阵乘**，使整个 ZoomOut 过程可微、内存高效。
- 训练损失中加入精化前后功能映射的一致性项：
  $$L_{\mathrm{consist}}(\mathbf{C}_{\mathrm{init}},\mathbf{C}_{\mathrm{refine}})=\|\mathbf{C}_{\mathrm{init}}-[\mathbf{C}_{\mathrm{refine}}]_{1:K_{\mathrm{init}},1:K_{\mathrm{init}}}\|_2^2$$
  利用 proper 功能映射的一阶主子矩阵性质，保证尺寸对齐。

### 3. 整体流水线
- **特征提取**：采用 DiffusionNet，输出 $F_1\in\mathbb{R}^{n_1\times 32}, F_2\in\mathbb{R}^{n_2\times 32}$。
- **初始软映射**：用 $\sigma=10^{-2}$ 的高斯核生成 $\Pi_{\mathrm{init}}$。
- **精化**：送入可微分 ZoomOut，迭代 10 步，起始 $K_{\mathrm{init}}=30$，步长 10，最终 $K_{\mathrm{final}}=130$。
- **无监督损失组合**：
  - 正交性损失 $L_{\mathrm{orth}}=\|\mathbf{C}_{\mathrm{init}}^\top\mathbf{C}_{\mathrm{init}}-I\|_2^2$（权重 1）
  - 一致性损失 $L_{\mathrm{consist}}$（权重从 $10^{-4}$ 渐进增至 $10^{-1}$）
  - Laplacian 交换性正则项（权重 $10^2$，源自谱分支遗留）
- 优化器：Adam，初始学习率 $10^{-5}$。

## 实验与结果
- **数据集**：FAUST（100 形状）、SCAPE（71 形状）、SHREC19（44 形状）的 remeshed 版本，以及非等距数据集 DT4D‑H（198 train / 95 test）。
- **评估基线**：GeomFmaps、Deep Shells、NeuroMorph、DUO‑FMNet、UDMSM、ULRSSM、AttentiveFMaps、ConsistentFMaps 等双分支/多分支方法。
- **主要结果**（均值测地线误差，$\times100$）：
  - FAUST：本文 **1.9**，优于 AttentiveFMaps（1.9）、ConsistentFMaps（2.3），次优为 ULRSSM（1.6）。
  - SCAPE：本文 **2.4**，优于 ConsistentFMaps（2.4 持平）、AttentiveFMaps（2.1），次优为 ULRSSM（1.8）。
  - SHREC19：本文 **4.2**，显著优于 ConsistentFMaps（3.8）和多数双分支方法。
  - DT4D‑H inter‑class：本文 **4.1**，优于 AttentiveFMaps（14.6）、ULRSSM（4.4）与 ConsistentFMaps（6.1），仅次 UDMSM（4.4）。
- **内存与速度**：在 11K 顶点以上时，AttentiveFMaps 在 24GB 显存下溢出，本文方法保持稳定；100K 顶点时原 ZoomOut CPU 版需 700s，GPU 原版 OOM，本文可扩展版仅需 2.4s，可微分版本 5s。

## 相关工作脉络
- **GeomFmaps [12]**：早期单分支深度功能映射，依赖线性系统求解与谱分支正交损失；本文将其谱分支替换为可微分精化一致性约束。
- **ULRSSM [9] / UDMSM [8]**：双分支网络，均显式存储稠密 p2p 矩阵，内存与可扩展性受限；本文用相同结构原理但以内存高效方式实现，且为单分支。
- **AttentiveFMaps [23]**：引入注意力机制与谱注意力，仍受限于稠密矩阵计算；本文证明即使不加谱分支，通过训练期精化一致性也能获得可比拟甚至更优的结果。
- **ConsistentFMaps [46]**：明确强调 proper 功能映射与双分支必要性；本文在去除谱分支的同时借助精化一致性达到相似保证，证明“properness+精化对齐”可替代谱分支。
- **ZoomOut [30]**：原始为 CPU 后处理算法；本文首次将其改造为可微分 GPU 版本并融入训练循环。
- **可微分精化工作 [23, 26]**：部分集成精化步骤但仍有内存或近似采样瓶颈；本文通过 KeOps 块级 Kernel 计算彻底消除显式矩阵存储，实现完全可扩展。

## 局限性与未来方向
- **依赖谱分解**：需预先计算输入网格的内在 Laplacian 特征函数，大尺寸网格的预处理开销仍然较大。
- **对高度非等距形变的适应性**：ZoomOut 基于近等距假设，若形变包含严重扭曲或部分遮挡，一致性损失的指导作用可能下降。
- **未来方向**：探索其他精化算法的可微分化与训练期集成；研究如何在强非等距或部分匹配场景下保持稳定性；进一步验证一致性损失在其他功能映射管道中的泛化效果。

## 研究启发与可借鉴点
- **KeOps 库在形状匹配中的迁移价值**：任何依赖高斯核相似度或 RBF 核矩阵乘法的几何深度学习模型均可借鉴其块级在线计算策略，实现内存与速度的双重提升。
- **一致性损失促进特征平滑**：本文显示精化一致性损失能间接推动特征空间光滑性，这一现象可推广至其他基于点对应关系的形状学习任务，作为隐式平滑正则。
- **单分支架构简化设计**：证明通过可微分精化与结构对齐约束，可完全替代传统双分支中的谱分支，为后续研究者提供“更轻、更稳定”的网络设计范式。
- **实验设计亮点**：在 100K 顶点密集网格上对比内存与耗时，直观展示可扩展性优势，并为后续工作提供了 dense mesh 评测的参考基准。

## 关键术语表
**Functional Map**：将形状间对应关系表示为两个形状函数空间之间的线性算子，通常以拉普拉斯特征函数为基底。
**Proper Functional Map**：由某个点对点映射通过拉普拉斯基底 pulled-back 得到的功能映射，保证存在底层点态对应。
**Soft Pointwise Map**：基于特征距离的高斯核加权亲和矩阵，作为点对点映射的概率松弛表示。
**ZoomOut**：一种迭代精化算法，通过逐步扩大谱基底并用最近邻查询更新点态映射，从而将粗糙功能映射精细化。
**KeOps**：基于符号矩阵表达的 GPU 加速库，可在不将完整核矩阵载入显存的前提下高效计算 Kernel 操作与梯度。
**Laplacian Commutativity**：功能映射与 Laplacian 算子交换的约束条件，常用于保证映射的正则性与几何保真性。
**Differentiable ZoomOut**：将原 CPU 实现的 ZoomOut 算法改写为全程可微、内存高效的 GPU 版本，可直接嵌入神经网络训练。

## 可复现要素
- **数据集**：FAUST、SCAPE、SHREC19、DT4D‑H 均为公开数据集（remeshed 版本）；SMAL 结果见补充材料。
- **代码**：可扩展稠密映射部分已开源（https://github.com/RobinMagnet/ScalableDenseMaps），完整管道代码（https://github.com/RobinMagnet/SimplifiedFmapsLearning）。
- **关键超参数**：特征维度 $p=32$，Blur 参数 $\sigma=10^{-2}$，初始谱大小 $K_{\mathrm{init}}=30$，ZoomOut 迭代 10 步、步长 10，最终 $K_{\mathrm{final}}=130$；损失权重：正交性 1、一致性 $10^{-4}\to10^{-1}$、Laplacian 交换性 $10^2$；优化器 Adam，学习率 $10^{-5}$。
