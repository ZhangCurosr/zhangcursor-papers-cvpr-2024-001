---
title: "SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Aut"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Yin_SCE-MAE_Selective_Correspondence_Enhancement_with_Masked_Autoencoder_for_Self-Supervised_Landmark_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:48:35"
---

# 论文速读：SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Aut

## 一句话总结
本文提出 **SCE-MAE**，一种用于自监督人脸关键点估计的两阶段框架：第一阶段采用区域级 Masked Autoencoder (MAE) 替代实例级对比学习，获取更适配密集预测的初始 patch 特征；第二阶段设计对应近似与优化模块 (CARB) 结合局域约束排斥损失 (LCR Loss)，仅对关键局部对应关系进行选择性精修，从而在无标注数据下显著提升关键点匹配与检测性能。

## 研究问题与动机
- **核心问题**：自监督人脸关键点估计需要在零标注条件下，学习出对稀疏关键点高度局部分化的特征表示，以支持跨图像匹配与定点回归。
- **现有方法缺陷1**：主流 SOTA（如 CL、LEAD）依赖实例级 SSL（MOCO/BYOL），其 pretext task 偏向类别级语义不变性，对“眼/唇/鼻”等密集子区域区分能力不足。
- **现有方法缺陷2**：前两阶段方法需构建内存开销巨大的超列 (hypercolumn) 特征，并在全局空间特征对之间平等施加对应关系监督，参数利用率低且冗余计算严重。
- **动机假设**：人脸关键点区域稀疏且纹理独特，而背景/脸颊区域大面积均匀；MAE 的掩码重建任务天然要求模型从上下文推断缺失的局部细节，更适合生成可用于密集定位的初始终点特征。

## 核心贡献（创新点）
- **首次将 MAE 引入自监督关键点估计的第一阶段**：利用区域级 MIM 预训练替代实例级 SSL，使初始 patch 特征无需额外对比损失即可形成清晰的关键点边界。与 CL/LEAD 的本质区别在于放弃了大负样本对比范式，转而依靠掩码重建强化局部上下文依赖。
- **提出对应近似与优化模块 (CARB)**：通过 CLS token 相似度自动分离关注（attentive）与非关注（inattentive）token，并利用密度峰值聚类将海量非关键 token 压缩为 $K_c$ 个簇中心代理。与之前工作构建 hypercolumn 全量优化的本质区别在于“空间稀疏化+语义分组代理”。
- **设计局域约束排斥损失 (LCR Loss)**：联合空间距离惩罚与对应类型权重，仅对 att-inatt 及 att-att 间的强对应进行梯度回传，直接精修高价值局部对应。与之前对所有 pair 平等监督的本质区别在于引入了几何局部性与语义排他性的双重重力场约束。

## 方法详解
- **第一阶段：MAE 预训练**。输入图像划分为 patch 并加入位置编码，随机 mask 高比例 patch，ViT 编码器仅处理可见 patch，轻量 decoder 通过 `[MASK]` token 重建原始像素。该过程输出 vanilla patch feature map，避免 hypercolumn 的内存负担。
- **关注/非关注分离**：计算 CLS token 与所有 patch token 的相似度 $Sim_{cls} = \mathrm{Softmax}(\frac{K \cdot q_{cls}}{\sqrt{d}})$，按 $\eta$ 阈值划分：Top-$\eta N$ 为 **attentive tokens**（覆盖关键点及重要面部结构），其余为 **inattentive tokens**（覆盖均匀背景/脸颊）。
- **非关注 Token 聚类近似**：对 inattentive tokens 应用密度峰值聚类，计算局部密度 $\rho_i = \exp(-\sum \|t_i-t_j\|^2)$ 与最近高密度距离 $\delta_i$，选取 $\rho_i \cdot \delta_i$ 最高的 $K_c$ 个 token 作为簇中心，丢弃其余 inattentive token，用簇中心统一代理非关键区域。
- **CARB 与 LCR Loss**：将保留的 attentive tokens 与 inattentive 簇中心送入轻量 projector。定义软最大对应概率 $p(t_j|t_i)$ 后，LCR 损失写作：
  $\mathcal{L}_{LCR} = \sum_{t_i \in T} \sum_{t_j \in T} f_{loc}(t_i,t_j) \cdot \lambda_{rep}(t_i,t_j) \cdot p(t_j|t_i)$
  其中 $f_{loc}=\log(\|t_i-t_j\|+1)$ 随空间距离单调递增以惩罚远距离匹配，$\lambda_{rep}$ 按对应类型赋权（$r_{att-att}=5, r_{att-inatt}=5, r_{inatt-inatt}=2$）。最小化该损失可系统性削弱跨区域虚假强对应，强化关键区域间的判别性映射。
- **推理策略**：冻结 backbone 与 projector，跳过聚类流程直接使用原始 MAE 特征；采用 **cover-and-stride** 技术对特征图进行重叠裁剪与步长拼接，在不二次减小 patch size 的前提下扩展空间分辨率，与基线公平对比。

## 实验与结果
- **数据集**：预训练 CelebA（162,770 张）；评估使用 MAFL、300W、AFLW_M、AFLW_R（含原始 $\mathrm{AFLW}_{RO}$ 与作者重新标注的 $\mathrm{AFLW}_{RC}$）。
- **匹配任务（Landmark Matching）**：在 MAFL 测试集 1000 对图像（同/异身份）上评估均方像素误差。DeiT-T（仅 5.4M 参数）已超越 CL/LEAD；DeiT-B 在同身份误差降至 **0.27**，异身份降至 **1.61**，较前 SOTA 提升约 **20%–44%**。
- **检测任务（Landmark Detection）**：全量标注下，DeiT-S 取得 MAFL 2.08、AFLW_M 5.33、$\mathrm{AFLW}_{RC}$ 5.40、300W 3.94 的跨眼距离误差百分比，较前 SOTA 提升约 **9%–15%**，且全程未使用 hypercolumn。
- **低标注鲁棒性**：在 AFLW_M 上测试 1–100 张标注样本，SCE-MAE 平均相对提升 **8.6%**，最高达 **20.1%**，且多次运行的标准差显著低于 CL/LEAD，证明特征学习更稳定。
- **消融与可视化**：Baseline（仅 MAE 特征）已优于 CL/LEAD；加入聚类主要提升同身份匹配；LCR Loss 主要提升异身份匹配；t-SNE 显示本方法 Silhouette Co
