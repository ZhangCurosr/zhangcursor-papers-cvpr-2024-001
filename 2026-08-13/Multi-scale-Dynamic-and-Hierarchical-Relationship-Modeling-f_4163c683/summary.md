---
title: "Multi-scale-Dynamic-and-Hierarchical-Relationship-Modeling-f"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Wang_Multi-scale_Dynamic_and_Hierarchical_Relationship_Modeling_for_Facial_Action_Units_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:14:30"
field: "面部表情与动作单元识别"
keywords: ["Facial Action Unit Recognition", "Spatio-temporal Modeling", "Multi-scale Dynamic", "Hierarchical Relationship", "Graph Attention", "Face Analysis"]
innovations: ["提出MFD模块，首次自适应建模多尺度面部动态以应对不同AU激活范围/幅度的异质性", "提出HSR模块，首次分层建模局部（同区域）与跨区域AU时空关系", "在BP4D和DISFA上达成新的SOTA（66.6%/66.2% F1）"]
benchmarks: ["BP4D", "DISFA"]
---

# 论文速读：Multi-scale-Dynamic-and-Hierarchical-Relationship-Modeling-f

## 一句话总结
论文提出多尺度动态与层级关系建模方法（MDHR），通过**多尺度面部动态建模模块（MFD）**自适应捕捉不同空间尺度下的面部肌肉动态，以及**层级时空AU关系建模模块（HSR）**分层建模同区域与跨区域AU依赖，在BP4D和DISFA数据集上取得AU识别新的SOTA。

## 研究问题与动机
1. **忽略AU层级关系**：现有Transformer和图方法对所有AU对采用相同策略建模空间关系，未显式考虑同/近面部区域AU间关联更强的自然层级结构（Problem 1）。
2. **动态建模缺乏尺度异质性考量**：不同AU激活在运动范围和幅度上存在显著差异（如AU25涉及大范围嘴部变形，AU2仅为细微眉周运动），现有时序模型对微妙肌肉运动不敏感，且未考虑不同空间尺度动态对各AU的贡献差异（Problem 2）。
3. **静态Patch方法与全局方法各有缺陷**：Patch方法忽略其他区域的上下文线索且受 landmark 检测误差影响；全局方法引入无关区域噪声。

## 核心贡献（创新点）
1. **MFD模块**：首次针对每个AU在每个空间尺度上自适应地建模面部动态，解决不同AU激活范围/幅度异质性问题，与TDN等单尺度动态建模方法本质不同。
2. **HSR模块**：首次分层学习局部（同区域）与跨区域时空关系，通过两阶段策略（AFE+Aux局部建模 → GAT跨区域建模）显式利用面部解剖空间分布，区别于ME-GraphAU等全连接AU关系建模。
3. **端到端SOTA**：在BP4D（66.6% F1，ResNet50）和DISFA（66.2% F1，Swin-B）两个标准数据集上均刷新最优记录。

## 方法详解

**整体流程**：给定T帧连续人脸视频，以帧$t$为中心取$k$帧邻域，经Backbone提取$L$层多尺度静态特征$X_l$，送入MFD获取时空特征$G^t$，再经HSR得到层级关系感知AU特征$\hat{V}^t$，最后由TCN+SC预测所有帧的N个AU。

**MFD模块（多尺度面部动态建模）**：
- **多尺度时序差分**：对相邻帧的第$l$层特征做逐点差分$d_l^t = x_l^t - x_l^{t-1}$，经Conv2D统一形状后沿时间轴平均池化得到$\bar{d}_l^t$，汇总前后$k$帧的动态演化。
- **自适应加权**：将L尺度动态特征沿通道拼接，经$1\times1$卷积降维后Softmax得到权重矩阵$w_l^t$，对各尺度动态特征加权聚合：$x_{\mathrm{motion}}^t = \sum_l w_l^t * \bar{d}_l^t$。浅层强调细微运动（眉/颊），深层强调大范围运动（嘴）。
- 最终时空特征$G^t = x_{\mathrm{motion}}^t + x_L^t$（与输出层静态特征相加）。

**HSR模块（层级时空AU关系建模）**：
- **局部关系建模（Stage 1）**：将$G^t$沿高度切分为上/中/下三个略有重叠的面部区域特征$G_{\mathrm{up}}^t, G_{\mathrm{mid}}^t, G_{\mathrm{low}}^t$。每个AU通过专属FE（$1\times1$卷积+GAP）从其对应区域提取局部关系感知特征$v_n^t$。同时加辅助分支训练AU组合预测（区域AUs的所有组合做one-hot分类），损失$\mathcal{L}_{\mathrm{sub}}$为交叉熵。
- **跨区域关系建模（Stage 2）**：以激活AU节点构建图，通过GAT显式建模跨区域AU依赖：边权重$e_{n,m}^t = \mathrm{LeakyReLU}(r^T[W v_n^t \| W v_m^t])$，注意力$\alpha_{n,m}^t = \mathrm{softmax}(e_{n,m}^t)$，更新$\hat{v}_n^t = \phi(\sum_{m \in N_n^t} \alpha_{n,m}^t W v_m^t)$。

**损失函数**：
- AU预测使用**非对称损失**$\mathcal{L}_{\mathrm{AU}}$（式10），通过$w_n$缓解类别不平衡，负样本项动态降权。
- 总体损失$\mathcal{L} = \mathcal{L}_{\mathrm{AU}} + \lambda \mathcal{L}_{\mathrm{sub}}$，$\lambda=0.01$。

## 实验与结果

**数据集**：BP4D（328视频，~14万帧，12 AU，subject-independent 3-fold）、DISFA（27视频序列，~13万帧，8 AU）。

**评估指标**：frame-based F1 score。

| 数据集 | 最优基线 | 本文 (ResNet50) | 提升 |
|--------|----------|----------------|------|
| BP4D | WSRTL 65.9% | **66.6%** | +0.7% |
| DISFA | WSRTL 64.6% | **66.2%** | +1.6% |

- 显著优于所有静态图像方法（如EAC-Net、JAA-Net等）。
- 优于静态AU关系建模SOTA（ME-GraphAU，BP4D 65.5% vs 66.6%，+1.1%）。
- Swin-B backbone在DISFA上AU1达65.4%，显著提升。
- 消融验证：MFD贡献+1.3%，HSR贡献+1.8%，TCN贡献+0.3%，各模块互补。

## 相关工作脉络

1. **ME-GraphAU [29]**：图注意力建模AU关系，但仅考虑空间关系且对所有AU对统一建模，未分层、未考虑跨区域与同区域差异。本文HSR在ME-GraphAU基础上引入层级两阶段策略与自适应边连接。
2. **FAUDT [13] / FAN-Trans [57]**：Transformer风格AU关联网络，自注意力对所有AU对平等建模，忽略面部解剖层级结构。
3. **WSRTL [52]**：弱监督区域与时序学习，使用辅助任务（ROI inpainting + optical flow）编码动态，但仅单一尺度，且未显式建模AU层级关系。
4. **HSTR-Net [47] / STRAL [37]**：时空图方法，先构建空间图再逐AU建模时序，但图结构固定/预定义，且未考虑多尺度动态的异质性。
5. **TDN [56]**：单尺度时序差分网络，本文MFD推广至多尺度并引入自适应加权，更贴合不同AU激活范围差异。
6. **EAC-Net [19] / JAA-Net [38]**：Patch-based方法，依赖landmark裁剪，忽略全局上下文。本文在全脸特征基础上做自适应区域分割，无需精确landmark。

## 局限性与未来方向

1. **面部区域切片策略较简单**：仅按高度等分三个略有重叠的区域（上/中/下），可能无法精细匹配每个AU的实际解剖位置，未来可引入更灵活的区域划分或learnable region partition。
2. **图边连接策略可扩展**：当前跨区域建模基于"激活AU与其他区域AU连接"的启发式策略，未来可探索更先进的动态图边学习方法（如可学习边表示、多层图结构）。
3. **k值固定**：当前实验中k设为1，未系统探索不同时序窗口的影响。
4. **未使用额外训练数据**：论文提到未与使用额外人脸数据集训练的方法（如[56]）对比，尽管仍领先，但在更宽松设置下可能有进一步提升空间。

## 研究启发与可借鉴点

1. **多尺度动态建模思路可迁移**：MFD的"多尺度时序差分+自适应加权"策略适用于任何需要捕捉多尺度运动/动态的任务（如微表情识别、手势识别、视频异常检测），尤其适合目标动态范围差异大的场景。
2. **层级关系建模的两阶段范式**：HSR的"局部→跨区域"分层建模策略可推广至其他多标签关系建模任务（如医学图像标注、行为识别），先建模强相关子组内关系再扩展至全局。
3. **辅助分支强化局部关系编码**：在特征提取层加入区域级AU组合预测辅助任务（Aux），通过多任务学习强制网络编码局部依赖，这一设计可在任何多标签分类任务中复用。
4. **与TCN的解耦组合**：MFD+HSR提取的特征再经独立TCN处理时序，这种"关系建模+时序建模"解耦设计简洁高效，可作为通用架构模板。
5. **可视化可解释性**：自适应权重矩阵可视化（图3）清晰展示了浅层关注细微运动、深层关注大范围运动的规律，为后续研究提供了可复用的可解释性分析手段。

## 关键术语表

**Action Unit (AU)**：面部动作单元，FACS系统中描述原子面部肌肉运动的最小单位，用于客观刻画表情。
**Multi-scale Facial Dynamic Modelling (MFD)**：多尺度面部动态建模模块，通过多尺度时序差分与自适应加权聚合，捕捉不同空间尺度下的面部肌肉动态。
**Hierarchical Spatio-temporal AU Relationship Modelling (HSR)**：层级时空AU关系建模模块，分两阶段建模同区域局部关系与跨区域全局关系。
**AU-specific Feature Extractor (AFE)**：AU专属特征提取器，对每个AU独立设计$1\times1$卷积+GAP，从其对应面部区域切片中提取局部关系感知特征。
**Asymmetric Loss**：非对称损失，通过动态调整负样本权重缓解AU识别中严重的类别不平衡问题。
**Graph Attention Network (GAT)**：图注意力网络，用于跨区域AU关系建模，自适应学习AU节点间的注意力权重并聚合邻居信息。
**Temporal Convolution Networks (TCN)**：时序卷积网络，在HSR输出后独立处理每个AU的时间序列以预测其 occurrence。
**BP4D / DISFA**：两个广泛使用的AU识别基准数据集，分别包含自发3D视频序列和观看视频引发的面部表情序列。

## 可复现要素

- **数据集**：BP4D [59] 和 DISFA [31]，公开可用。
- **代码**：已开源，GitHub: https://github.com/CVI-SZU/MDHR。
- **骨干网络**：ResNet-50 和 Swin-B，均在ImageNet上预训练。
- **人脸预处理**：MTCNN [58] 裁剪对齐至224×224。
- **优化器**：AdamW，$\beta_1=0.9, \beta_2=0.999$，初始学习率0.0001，cosine decay。
- **超参**：$k=1$（邻域帧数），$\lambda=0.01$（辅助损失权重），面部区域沿高度按3/7比例切分。
- **训练**：subject-independent 3-fold交叉验证，NVIDIA A100 GPU，PyTorch实现。
- **更多细节**：见论文Supplementary Material。
