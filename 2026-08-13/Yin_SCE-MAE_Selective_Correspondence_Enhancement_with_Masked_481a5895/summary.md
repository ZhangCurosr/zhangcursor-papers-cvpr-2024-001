---
title: "SCE-MAE: Selective Correspondence Enhancement with Masked Autoencoder for Self-Supervised Landmark Estimation"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Yin_SCE-MAE_Selective_Correspondence_Enhancement_with_Masked_Autoencoder_for_Self-Supervised_Landmark_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:52:28"
field: "自监督人脸关键点估计"
keywords: ["Self-Supervised Learning", "Facial Landmark Estimation", "Masked Autoencoder", "MAE", "Correspondence Learning", "Point Cloud", "3D Reconstruction"]
innovations: ["首次将MAE预训练用于自监督人脸关键点估计的第一阶段", "提出CARB模块通过聚类近似和LCR损失选择性增强关键局部对应关系", "设计了结合空间距离与语义排斥的LCR损失函数以精炼特征"]
benchmarks: ["MAFL", "300W", "AFLW_M", "AFLW_R", "AFLW_RC"]
---

# 论文速读：SCE-MAE: Selective Correspondence Enhancement with Masked Autoencoder for Self-Supervised Landmark Estimation

## 一句话总结
本文提出 SCE-MAE 框架，通过结合 Masked Autoencoder (MAE) 作为第一阶段的预训练骨干网络，以及在第二阶段使用对应近似与优化块 (CARB) 选择性增强关键区域的局部对应关系，解决了无标注情况下自监督人脸关键点估计的挑战，在匹配和检测任务上大幅超越现有方法。

## 研究问题与动机
- **核心问题**：如何在无标注数据的情况下，学习到对稀疏且细微的人脸关键点具有高度判别力的局部特征表示，以完成关键点匹配与检测。
- **现有方法不足**：
    1. 现有领先方法（如 CL, LEAD）在第一阶段采用 instance-level 的自监督学习 (SSL) 范式（如 MOCO, BYOL）预训练骨干网络，此类方法旨在学习类别无关的全局特征，未能显式优化子图像/像素级的区域特征，因此产生的初始局部特征判别力不足。
    2. 第二阶段方法（CL, LEAD）通过将空间邻近的 feature map 拼接成巨大的 hypercolumn 来聚合上下文信息，计算开销和内存占用巨大。
    3. 现有方法对所有 feature descriptor 对进行朴素的全局对应关系学习，假设每对特征都有同等重要性，但人脸中非关键点区域（如脸颊、额头）大且均匀，学习其间冗余的对应关系会浪费模型容量，反而可能干扰对关键点对应的学习。
- **本文动机**：鉴于 MIM（掩码图像建模）协议要求从有限上下文重建被掩码区域，天然适合重建稀疏且独特的人脸关键点区域；同时认为选择性优化重要的局部对应关系比全对优化更有效，因此提出 SCE-MAE。

## 核心贡献（创新点）
1.  **首次将 MIM (MAE) 引入自监督人脸关键点估计的第一阶段**。与之前使用 instance-level SSL (MOCO/BYOL) 的方法相比，基于 region-level 的 MAE 通过掩码预测任务，能更自然地学习到适合稠密预测任务的高保真初始关键点表示。
2.  **提出了对应近似与优化块 (CARB)**。与之前方法操作内存密集的 hypercolumn 并学习全对对应关系不同，CARB 首先在 vanilla feature map 上工作，然后通过密度峰值聚类对不重要区域进行近似，最终只直接优化有选择性的关键局部对应关系，显著降低了计算复杂度。
3.  **设计了局部约束排斥损失 (LCR Loss)**。这是一种新颖的损失函数，通过考虑 token 对的空间距离（ locality ）和对应类型（ repellence ）来加权惩罚错误的相关对应，系统性地削弱了关键与非关键区域之间、以及不同关键点之间的无效远距离对应，从而精炼出高质量的特征。

## 方法详解
SCE-MAE 是一个两阶段框架：
1.  **第一阶段：基于 MAE 的自监督预训练**。使用 ViT (如 DeiT-T/S/B) 作为骨干网络，采用 Masked Autoencoder (MAE) 进行预训练。输入图像被分割成 patch，随机掩码大部分 patch，编码器仅处理可见 patch，然后由轻量解码器重建原始图像。此阶段的目标是获得具有良好局部判别力的初始特征图 `f^p`。
2.  **第二阶段：选择性对应增强 (CARB)**。冻结预训练的骨干网络，提取最后一层的 patch embeddings。
    -   **注意力/非注意力分离**：计算 CLS token 与所有 patch token 的相似度，根据相似度得分将 tokens 分为“注意力组”（高相似度，代表关键点区域）和“非注意力组”（低相似度，代表背景或非关键区域）。
    -   **非注意力 Token 聚类**：对非注意力组的 tokens 应用密度峰值聚类算法，选取 top-$K_c$ 个聚类中心作为代表，用其中心 token 近似替代所有同组的非注意力 token，从而大幅减少需要处理的 token 数量。
    -   **Correspondence Approximation and Refinement Block (CARB)**：将聚类中心替换回原位置，形成完整的 2D feature map。该 feature map 经过一个轻量级投影网络 (projector) 得到最终特征。
    -   **局部约束排斥损失 (LCR Loss)**：定义在投影后的特征之上。对于特征图中的任意两个 token $t_i$ 和 $t_j$，其对应概率 $p(t_j|t_i)$ 由 softmax 计算。LCR Loss 旨在最小化那些空间距离远 ($f_{loc}$ 大) 但对应概率强，且属于“应相互排斥”类型 ($\lambda_{rep}$ 高) 的 token 对的正反馈。公式为：$\mathcal{L}_{LCR} = \sum_{t_i \in T} \sum_{t_j \in T} f_{loc}(t_i, t_j) \cdot \lambda_{rep}(t_i, t_j) \cdot p(t_j | t_i; \Phi, x)$。其中 locality 项 $f_{loc}$ 随空间距离增加而增加（log函数），repellence 项 $\lambda_{rep}$ 根据不同 token 对类型赋予不同权重（att-att 和 att-inatt 权重高于 inatt-inatt）。
3.  **推理**：训练完成后，推理时不使用聚类和替换步骤，直接使用原始特征。为保证与其他方法比较的公平性（特征图尺寸），采用 cover-and-stride 技术将特征图 upscale 到输入分辨率。

## 实验与结果
-   **数据集**：
    -   预训练：CelebA (162,770 张人脸图像)。
    -   关键点匹配评估：MAFL (1000对图像，含同身份500对和不同身份500对)。
    -   关键点检测评估：MAFL, 300W, AFLW_M, AFLW_R (注意：论文发现 AFLW_R 测试集标注存在问题，进行了重新标注，得到 $\mathrm{AFLW}_{RC}$)。
-   **评估基线**：DVE [46], ContrastLandmark (CL) [9], LEAD [19]。
-   **主要结果**：
    -   **关键点匹配** (表1)：SCE-MAE 在不同 backbone 大小和特征维度下均大幅超越 SOTA。例如，使用最小 backbone DeiT-T (5.4M参数) 在相同身份匹配上误差为 0.82，不同身份为 2.19；使用 DeiT-B (85.3M参数) 在相同身份匹配上达到 0.27，不同身份为 1.61，相比 CL/LEAD 提升幅度约为 $\sim20\%$ (同身份) 至 $\sim44\%$ (不同身份)。
    -   **关键点检测** (表2)：即使在所有标注样本下，SCE-MAE (DeiT-T) 也在四个数据集上全面超越基线。最好结果 (DeiT-B) 相比之前的 SOTA (LEAD) 在 MAFL, AFLW_M, $\mathrm{AFLW}_{RC}$, 300W 上分别提升了约 9%-15% 的相对性能 (误差百分比降低)。例如，在 300W 数据集上误差从 LEAD 的 4.87% 降至 SCE-MAE (DeiT-B) 的 3.95%。
    -   **少量标注检测** (表3)：在 AFLW_M 上，使用极少量标注 (1, 5, 10, 20, 50, 100个)，SCE-MAE 始终取得最佳性能，平均相对提升 8.6%，最高达 20.1%，且标准差更小，表明鲁棒性更强。
    -   **消融实验** (表4)：验证了各组件的贡献。Baseline (仅 MAE 特征) 已优于 CL/LEAD。加入聚类主要提升同身份匹配，加入 LCR Loss 主要提升不同身份匹配，两者结合效果最佳。
    -   **可视化** (图5)：t-SNE 和 Silhouette Coefficient 分析表明，SCE-MAE 学到的特征聚类效果更好，类内更紧凑，类间更分离。

## 相关工作脉络
1.  **自监督学习 (SSL) 在人脸关键点估计中的应用**：DVE [46], CL [9], LEAD [19]。本文与之不同的定位在于第一阶段采用了 region-level 的 MAE 而非 instance-level 的 MOCO/BYOL，并引入了 selective correspondence 的思想。
2.  **基于等价学习 (Equivalence Learning) 的无监督关键点估计**：Thewlis et al. [44, 45] 提出的 DVE。本文借鉴了其使用无标注数据的思路，但在表征学习方法上采用了更新的 MAE。
3.  **Masked Image Modeling (MIM)**：BEiT [3], MAE [14], SimMIM [50] 等。本文首次将 MIM/MAE 成功应用于人脸关键点估计的第一阶段预训练，论证了其对稠密预测任务的优越性。
4.  **特征对应学习**：CL [9] 和 LEAD [19] 都致力于建立图像内不同区域间的对应关系。本文与之的核心区别在于提出了 **selective** 对应，通过聚类和 LCR 损失避免了对所有特征对（尤其是冗余的非关键区域对）进行优化。
5.  **Hypercolumn 的使用**：CL [9] 和 LEAD [19] 使用 hypercolumn 聚合多层特征。本文明确指出其内存密集型缺点，转而直接在单层的 MAE 输出特征图上进行操作。
6.  **密度峰值聚类**：Long et al. [28] 等。本文将其应用于人脸 patch token 的聚类以区分关键与非关键区域，是一种新颖的应用。

## 局限性与未来方向
-   **局限性**：
    1.  方法依赖于第一阶段的 MAE 预训练质量，其性能上限受限于预训练数据的质量和规模。
    2.  虽然相比之前方法计算效率更高，但两阶段框架本身仍然有一定计算开销，尤其是在第二阶段投影网络的训练。
    3.  实验主要在人脸关键点任务上进行，其在其他生物医学或通用物体关键点估计任务上的泛化能力有待验证。
    4.  推理时使用 cover-and-stride 进行 upsampling 可能引入额外的计算和内存负担。
-   **未来方向**：
    1.  探索更高效的单阶段端到端架构，或将选择性对应的思想更深度地融入预训练过程。
    2.  将框架扩展到非人脸领域，验证其通用性。
    3.  研究更先进的聚类或特征稀疏化方法来进一步优化对应关系的计算。
    4.  结合少量有标注数据，探索半监督或迁移学习场景下的性能提升。

## 研究启发与可借鉴点
1.  **Backbone 预训练策略的选择**：对于稠密预测任务（如关键点检测、分割），应优先考虑 region-level 的 SSL 方法（如 MAE, MIM），而非传统的 instance-level 对比学习方法，因为它们更能学习判别性的局部特征。这是一个重要的设计原则。
2.  **Selective/Attention-based Feature Refinement**：并非所有特征区域都对最终任务有同等贡献。通过初步分析（如相似度、注意力分数）识别重要区域，并集中计算资源优化这些区域的特征表示，可以有效提升效率和方法性能。聚类简化是一种简洁有效的降维手段。
3.  **损失函数的精细化设计**：LCR Loss 将空间几何约束（locality）和语义/任务约束（repellence）结合起来进行惩罚，这种思路可以迁移到其他需要学习局部对应关系或空间一致性的视觉任务中。
4.  **对现有数据集标注质量的审视**：论文发现并重标了 AFLW_R 测试集的问题标注，这提醒研究者在进行基准测试时，应对数据集标注质量保持警惕，必要时进行人工核查和修正，以确保评估的可靠性。
5.  **评估协议的统一与对比**：通过控制 backbone 大小 (DeiT-T/S/B) 和特征维度来进行公平比较，并详细报告不同设置下的结果，这种严谨的实验设计值得借鉴。

## 关键术语表
-   **Self-Supervised Landmark Estimation**：无需人工标注关键点，通过设计预训练任务从图像数据中自动学习关键点位置和表示的任务。
-   **Masked Autoencoder (MAE)**：一种掩码图像建模方法，随机遮挡图像的大部分 patch，训练编码器-解码器结构从可见 patch 重建原始图像，从而学习有效的视觉表征。
-   **Instance-level vs. Region-level SSL**：Instance-level (如 MoCo, BYOL) 关注区分不同图像实例；Region-level (如 MAE) 关注学习图像内部局部区域的特征，更适合稠密预测任务。
-   **Hypercolumn**：将网络不同深度的特征图在通道维度上拼接，形成一个包含多层语义和空间信息的巨大特征向量，用于捕捉细粒度上下文，但内存消耗大。
-   **Correspondence Approximation and Refinement Block (CARB)**：本文提出的第二模块，负责通过对非注意力区域进行聚类近似，并对关键注意力区域进行局部对应关系的精炼。
-   **Locality-Constrained Repellence (LCR) Loss**：本文提出的新损失函数，通过空间距离惩罚和对应类型权重，来削弱错误的关键点间或非关键点间对应，强化有效对应。
-   **Density Peak Clustering**：一种聚类算法，通过计算每个数据点的局部密度和它与更高密度点的距离来识别聚类中心。
-   **Silhouette Coefficient**：用于评估聚类质量的指标，值越高表示簇内越紧密、簇间越分离。

## 可复现要素
-   **数据集**：
    -   CelebA (预训练): 公开。
    -   MAFL, 300W, AFLW_M, AFLW_R: 公开。
    -   论文提供了重新标注的 $\mathrm{AFLW}_{RC}$ 数据集，**代码/数据未明确声明开源**，但通常在 CVPR 论文中若未提及开源链接，则默认为未公开或随补充材料提供。需查阅论文官方项目页面 (若有)。
-   **代码/权重**：论文未明确提供代码仓库链接或预训练权重下载链接。实验部分提及“官方实现”用于对比基线。
-   **关键超参**：
    -   骨干网络：DeiT-T, DeiT-S, DeiT-B (基于 ViT)。
    -   MAE 预训练：400 epochs, batch size 512, learning rate 3e-4, patch size 8。
    -   输入尺寸：resize 到 $136 \times 136$, crop 中心 $96 \times 96$。
    -   注意力比例 $\eta$: DeiT-B 为 0.25, DeiT-T/S 为 0.1。
    -   聚类数量 $K_c$: 4。
    -   LCR 损失权重：$r_{att-att} = 5, r_{att-inatt} = 5, r_{inatt-inatt} = 2$。
    -   检测器回归头：包含卷积块 (产生 I=50 个中间 heatmap) 和线性层，使用 soft-argmax 获取坐标。
    -   特征 upscale 方法：cover-and-stride。
