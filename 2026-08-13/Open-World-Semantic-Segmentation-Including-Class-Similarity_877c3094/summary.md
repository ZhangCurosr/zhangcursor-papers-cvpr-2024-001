---
title: "Open-World-Semantic-Segmentation-Including-Class-Similarity"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Sodano_Open-World_Semantic_Segmentation_Including_Class_Similarity_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:41:01"
field: "开放世界语义分割"
keywords: ["open-world semantic segmentation", "anomaly segmentation", "novel class discovery", "class similarity", "out-of-distribution detection"]
innovations: ["双解码器架构联合优化语义分割与异常分割，无需额外训练数据", "特征空间高斯建模+对比/Objectosphere损失实现已知类描述符学习与未知类检测", "基于高斯推断的类别相似度度量替代直接logit比较"]
benchmarks: ["SegmentMeIfYouCan", "BDDAnomaly"]
---

# 论文速读：Open-World-Semantic-Segmentation-Including-Class-Similarity

## 一句话总结
提出一种双解码器卷积神经网络，在无需额外训练数据的前提下，同时实现高精度的封闭世界语义分割、异常分割（区分已知/未知类别）以及新类别发现，并为每个未知类别提供与已知类别的相似度度量。

## 研究问题与动机
1. **封闭世界假设的局限**：现有语义分割模型基于封闭世界假设训练，对训练集中未见过的物体倾向于过度自信地归类为已知类别，无法适应真实开放世界场景。
2. **下游应用需求**：自动驾驶等系统需要识别未见物体并估计其属性（如运动预测、建图过滤），因此不仅需要检测异常，还需要区分不同新类别并提供类别相似度。
3. **现有方法不足**：多数异常检测方法依赖不确定性估计、生成模型重建或额外OoD数据训练，本文希望不依赖这些额外资源，直接在特征空间操作实现开放世界分割。
4. **双重任务整合**：异常分割（binary known/unknown）与新类别发现（distinguish multiple novel classes）通常被分开研究，本文尝试在同一框架中联合解决。

## 核心贡献（创新点）
1. **双解码器特征空间操控架构**：提出基于encoder-decoder的设计，通过特征损失将同类像素推向学习到的类描述符（mean activation vector），并结合高斯建模实现开放世界分割；与仅依赖softmax阈值的方法本质不同。
2. **对比解码器结合Objectosphere损失**：引入对比损失+Objectosphere损失的组合，迫使已知类特征分布在超球面表面、未知类特征压缩至球心，从而实现更鲁棒的异常分割；区别于纯基于softmax熵或能量的方法。
3. **无需额外数据的类别相似度度量**：利用高斯模型的自然推论，为每个未知像素计算与所有已知类别的匹配分数，自动确定最相似已知类别；相比直接用最高激活值或训练专用分类器的方法更可靠。
4. **SOTA性能与多基准验证**：在SegmentMeIfYouCan上以零额外数据获得FPR95第一、mean F1第一；在BDDAnomaly上全面超越基线，同时验证了开放世界分割和类别相似度能力。

## 方法详解

**网络架构**：采用ResNet34编码器（NonBottleneck-1D块替换标准残差块，3×3卷积分解为3×1和1×3序列），后接金字塔池化模块（PPM）扩大感受野；两个解码器共享结构（3个SwiftNet模块+最近邻+深度可分离卷积上采样），并通过跳跃连接融合细粒度特征。

**Semantic Decoder（语义解码器）**：
- 目标：标准语义分割 + 特征空间约束
- 损失函数：$\mathcal{L}_{\mathrm{sdec}} = w_1 \mathcal{L}_{\mathrm{sem}} + w_2 \mathcal{L}_{\mathrm{feat}}$
- $\mathcal{L}_{\mathrm{sem}}$：加权交叉熵损失，权重为类别逆频率
- $\mathcal{L}_{\mathrm{feat}}$：特征损失，将每类true positive像素的pre-softmax特征推向该类上一步累积的均值$\mu_k^{e-1}$，并除以方差$\sigma_k^{e-1}$归一化
- 类描述符：迭代维护每类的均值$\mu_k$和方差$\sigma_k^2$（running average of true positive activations）

**Contrastive Decoder（对比解码器）**：
- 目标：异常分割（已知类→超球面，未知类→球心）
- 损失函数：$\mathcal{L}_{\mathrm{cdec}} = w_3 \mathcal{L}_{\mathrm{cont}} + w_4 \mathcal{L}_{\mathrm{obj}}$
- $\mathcal{L}_{\mathrm{cont}}$：对比损失，使每类像素的平均特征$\bar{f}_k$接近上一步该类均值$\bar{\mu}_k^{e-1}$，同时与其余类区分（温度参数$\tau$）
- $\mathcal{L}_{\mathrm{obj}}$：Objectosphere损失，已知类像素特征范数被推向阈值$\xi$以上，未知类像素特征范数被压缩至0

**Post-Processing（后处理）**：
- 高斯模型构建：基于每类的$\mu_k$和$\Sigma_k = \mathrm{diag}(\sigma_k^2)$建立多元高斯分布
- Semantic分数：$s_k(f_p) = \exp(-\frac{1}{2}(f_p-\mu_k)^\top \Sigma_k^{-1}(f_p-\mu_k))$，取最大值$s(p)$，未知分数$s_{\mathrm{unk},p}^{\mathrm{sem}} = 1 - s(p)$
- Contrastive分数：$s_{\mathrm{unk},p}^{\mathrm{cont}} = \max(0, 1 - \|f_p\|^2/\xi)$
- 融合：$s_{\mathrm{unk},p} = \frac{1}{2}(s_{\mathrm{unk},p}^{\mathrm{sem}} + s_{\mathrm{unk},p}^{\mathrm{cont}})$，超过阈值$\delta$判定为未知
- 新类别发现：未知像素特征与已发现未知类均值比较，距离小于$\eta$则归入同类并更新均值，否则创建新类

**Class Similarity（类别相似度）**：
- 取使$s_k(f_p)$最大的类别$\tilde{k} = \mathrm{argmax}_k s_k(f_p)$作为最相似已知类别
- 注意：不同于直接用最高激活值，而是用高斯分数衡量"落入某类影响区域"的程度

## 实验与结果

**数据集**：SegmentMeIfYouCan（基于Cityscapes，专门针对异常分割设计）、BDDAnomaly（含人工合成未知类Train/Motorcycle/Bicycle）

**评估指标**：
- Pixel-level：AUPR（上升越高越好）、FPR95（下降越低越好）
- Component-level：sIoU、PPV、mean F1
- 已知类：mIoU
- 新类别发现：与各ground truth类overlap的mIoU
- 类别相似度：像素级准确率

**主要结果**：
- **SegmentMeIfYouCan**（Tab.1右）：ContMAV（无OoD数据）获得FPR95 = 3.8%（第一）、PPV = 61.9%（第一）、mean F1 = 63.6%（第一）；AUPR = 90.2%（第二，低于RbA的94.5%但RbA使用OoD数据）
- **BDDAnomaly异常分割**（Tab.2）：AUPR = 96.1%，FPR95 = 6.9%，全面超越MaxSoftmax、Background、MC Dropout等基线
- **已知类mIoU**（Tab.1左）：OW模型70.8% vs CW模型71.1%（Cityscapes），仅下降0.3个百分点
- **新类别发现**（Tab.3）：含feature loss时Train mIoU=62.4%、Motorcycle=62.2%、Bicycle=56.8%，远超Background+cluster基线；去除feature loss后仍显著优于基线
- **类别相似度**（Tab.4）：ContMAV在Motorcycle上准确率58.9%、Train上49.9%，远超baseline（12.5%/9.8%）和仅用MA的变体

**消融实验**（Tab.5、Tab.6）：
- 最佳异常分割组合：feature loss + contrastive decoder + Gaussian post-processing → AUPR 96.1%，FPR95 6.9%
- Gaussian推断明显优于softmax阈值、最大激活值、最小距离等策略
- 类别相似度方面，Gaussian后处理同样最优（Motorcycle 58.9%，Train 49.9%）

## 相关工作脉络
1. **Anomaly Detection/Classification（分类层面）**：MaxSoftmax [22]、Background [6]、MC Dropout [16]等方法通过softmax阈值、背景类或贝叶斯近似检测异常；本文工作延伸至像素级分割，且在特征空间操作而非仅依赖输出层。
2. **Entropy/Energy-based OOD**：Entropy maximization [10]、Energy score [35]等方法试图缓解overconfidence；本文不依赖不确定性估计，而是通过特征空间约束显式建模已知类分布。
3. **Reconstruction-based方法**：Generative models [29,33]、Student-Teacher [3,67,69]利用重建误差检测异常；本文完全不依赖生成模型或额外OoD数据。
4. **Vision-Language Models**：CLIP-based方法 [25,47,48,72]利用图文对齐做异常分割；本文是纯视觉方法，无需大规模预训练。
5. **Open-World Segmentation benchmarks**：SegmentMeIfYouCan [9]和Fishyscapes [6]是主要评测基准；本文在SegmentMeIfYouCan上取得多项第一，且无需额外训练数据。
6. **Novel Class Discovery**：Han et al. [20]等研究在分类层面的新类别发现；本文将其扩展到像素级分割，实现novel class discovery与anomaly segmentation的联合学习。

## 局限性与未来方向
1. **仅支持RGB图像**：方法目前仅针对RGB输入，未探索RGB-D或多模态扩展。
2. **高斯模型假设简化**：协方差矩阵假设为对角阵（各维度独立），可能无法捕捉特征维度间的相关性。
3. **测试时静态发现**：新类别的发现依赖于固定阈值$\eta$，对于连续流式视频输入的自适应更新策略未详细讨论。
4. **类别相似度的地面真值依赖人工映射**：实验中使用手动lookup table定义"摩托车≈汽车"、"火车≈卡车"，主观性较强，缺乏客观评估标准。
5. **计算开销**：双解码器架构相比单解码器 baseline 增加了参数量和计算负担，实时性需进一步验证。

## 研究启发与可借鉴点
1. **特征空间显式建模的思路**：通过累积true positive像素的pre-softmax特征来学习类描述符，比直接操作logits更稳定，可迁移至开放集识别、域适应等任务。
2. **对比损失+Objectosphere的组合策略**：将类间分离（contrastive）与类内范数约束（objectosphere）结合，是一种有效的异常分割设计范式，可推广至其他OOD检测场景。
3. **无额外数据的开放世界学习**：证明不依赖OoD样本或生成模型也能获得强异常分割性能，为资源受限场景提供了可行方案。
4. **高斯后处理的通用性**：基于running mean/variance构建Gaussian模型进行未知性评分，逻辑简洁且可解释，可嵌入到多种segmentation架构中。
5. **双解码器共享Encoder的设计**：两个任务（语义分割+异常分割）通过共享底层特征学习相互促进，这一设计模式值得在多任务开放世界学习中借鉴。

## 关键术语表
**Open-World Semantic Segmentation**：允许测试图像中包含训练时未见类别的语义分割任务，需同时识别已知类和检测未知对象。
**Anomaly Segmentation**：像素级二值分割任务，将图像分为"已知类"和"异常/未知类"两类。
**Novel Class Discovery**：在仅使用已知类标签训练的前提下，将测试时遇到的未知像素聚类划分为不同新类别。
**Feature Loss ($\mathcal{L}_{\mathrm{feat}}$)**：驱使同类像素的pre-softmax特征向量趋近该类历史均值的高斯归一化距离损失。
**Objectosphere Loss**：约束已知类特征范数大于阈值$\xi$、未知类特征范数趋近0的损失函数。
**Contrastive Loss**：基于InfoNCE的形式，使同类平均特征相互靠近、不同类平均特征相互远离。
**Mean Activation Vector (MAV)**：每类true positive像素pre-softmax特征的running average，作为该类在特征空间的描述符。
**SegmentMeIfYouCan**：NeurIPS 2021提出的异常分割 benchmark，基于Cityscapes构建，专门包含prominent未知物体。

## 可复现要素
- **数据集**：Cityscapes（训练）、SegmentMeIfYouCan（评测）、BDDAnomaly（评测+消融）；论文未明确声明代码开源状态，数据集可公开获取
- **代码/权重**：论文未提及开源代码或预训练权重
- **关键超参**：$\xi = 1$、$\delta = 0.6$、$\tau = 0.1$、$\eta = 0.5$；$w_1 = 0.9$、$w_2 = 0.1$、$w_3 = 0.5$、$w_4 = 0.5$；学习率0.004（one-cycle policy）、batch size 8、500 epochs；Adam优化器；数据增强包括随机缩放、裁剪、翻转
