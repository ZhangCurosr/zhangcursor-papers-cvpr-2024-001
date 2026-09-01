---
title: "MFP: Making Full Use of Probability Maps for Interactive Image Segmentation"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Lee_MFP_Making_Full_Use_of_Probability_Maps_for_Interactive_Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:00:24"
field: "交互式图像分割"
keywords: ["interactive image segmentation", "probability map modulation", "click-based segmentation", "gamma correction", "late fusion", "recursive training"]
innovations: ["首次提出概率图调制（gamma校正）增强历史预测信息的传递", "设计晚期融合架构将概率特征与骨干特征有效聚合", "提出递归训练策略实现全链路有序交互信息利用"]
benchmarks: ["GrabCut", "Berkeley", "DAVIS", "SBD"]
---

# 论文速读：MFP: Making Full Use of Probability Maps for Interactive Image Segmentation

## 一句话总结
提出MFP（Making Full Use of Probability maps）框架，首次引入概率图调制（gamma校正）机制，将前一轮预测的概率图中蕴含的目标形状信息有效传递到当前预测，显著提升基于点击的交互式图像分割性能。

## 研究问题与动机
1. 现有点击式交互式分割算法（如SimpleClick、FocalClick等）虽将前一轮概率图作为额外输入，但概率图中蕴含的目标形状等有用信息未能充分传递到当前预测，导致网络难以利用历史预测细节。
2. 之前的BRS/f-BRS类方法通过推理时反向梯度传播来纠正误标注点击，但计算开销较高；而纯前馈方法缺乏对历史概率图的显式增强机制。
3. RITM等早期采用之前掩码的工作依赖随机采样初始点击，导致初始轮次缺乏有序历史概率图，无法对概率调制进行训练，信息利用不充分。

## 核心贡献（创新点）
1. **首次提出概率图调制方案**：通过gamma校正增强概率图中用户指定目标物体的形状细节表示，与已有方法直接输入原概率图的做法本质不同。
2. **设计MFP交互式分割框架**：将调制后的概率图作为额外输入，并通过晚期特征融合（late fusion）将概率相关信息与骨干特征聚合，区别于SimpleClick等直接拼接输入的做法。
3. **提出递归训练策略（Recursive Training）**：从物体中心开始，基于前一轮预测误差位置顺序生成后续点击，实现全链路有序概率图信息的端到端训练，区别于RITM的随机初始点击+迭代采样混合策略。
4. **系统性验证与多骨干适配**：在ResNet-34、HRNet-18、ViT-B三个骨干上实现MFP，在SBD/COCO+LVIS训练数据下的四个基准数据集上均显著优于使用相同骨干的现有算法。

## 方法详解
### 3.1 概率图调制（Probability Map Modulation）
- **核心思想**：利用当前点击位置$u$及其标签$l(u)$，对上一轮概率图$P^{t-1}$进行gamma校正调制，生成$\tilde{P}^{t-1}$。
- **前景点击**（$l(u)=1$，说明上一轮误判为背景）：
  $$\tilde{P}_x^{t-1} = (P_x^{t-1})^{\frac{1}{\gamma}}, \quad x \in M$$
- **背景点击**（$l(u)=0$）：
  $$\tilde{P}_x^{t-1} = (P_x^{t-1})^{\gamma}, \quad x \in M$$
  其中$\gamma \in [1, \Gamma]$，$\Gamma$设为使得点击处概率变为0.99（前景）或0.01（背景）。
- **调制窗口**$M=\{x : \|x-u\| \leq R\}$，半径$R$取$max$（默认100）与到相反类型点击最小距离的一半的较小值。
- **Gamma分配方案**（两种距离度量）：
  - **欧氏距离**（后期点击使用）：$\gamma = \Gamma \cdot (1-\frac{d}{R}) + \frac{d}{R}$，线性衰减。
  - **概率距离**（前期点击使用，$d=(P_x^{t-1}-P_u^{t-1})^2$，以中位数$\bar{d}$归一化）：$\gamma = (\Gamma-1)\cdot\max\{\frac{(\bar{d}-d)^3}{\bar{d}^3},0\}+1$。
- **策略切换**：前$N=7$次点击用概率距离，后续用欧氏距离。

### 3.2 网络架构
- 输入：图像$I$、当前点击图$C^t$、原概率图$P^{t-1}$、调制概率图$\tilde{P}^{t-1}$。
- **骨干特征提取**：将$I、P^{t-1}、\tilde{P}^{t-1}$拼接后通过两个conv块调整分辨率和通道，嵌入后与图像特征相加，送入ResNet-34/HRNet-18/ViT-B，得到$\mathcal{F}_B^t$。
- **概率相关特征提取**：使用两个Xception conv块提取$\mathcal{F}_P^t$。
- **晚期融合**：concat $\mathcal{F}_P^t$与$\mathcal{F}_B^t$，经四个Xception conv块融合后送入分割头，生成当前概率图$P^t$，阈值化得掩码$Y^t$。

### 3.3 递归训练（Recursive Training）
- 第一轮点击：采样于目标物体中心。
- 后续点击：比较网络预测与ground truth，选择最大误差区域中心作为下一点击。
- 每幅图像递归训练至多24次点击。
- 损失函数：归一化focal loss（normalized focal loss），Adam优化（$\beta_1=0.9, \beta_2=0.999$）。

## 实验与结果
### 数据集
- **训练**：SBD（8498 images / 20172 masks）、COCO+LVIS（104K images / 1.6M masks）
- **评测**：GrabCut（50 images）、Berkeley（96 images / 100 masks）、DAVIS（345 frames / 50 videos）、SBD（2857 validation images / 6671 masks）

### 评估指标
- **NoC@85/90/95**：达到指定IoU阈值所需的平均点击次数（目标IoU分别设为85%、90%、95%）
- **mIoU-AUC**：不同点击数下mean IoU曲线的积分

### 主要结果（SBD训练，ViT-B骨干）
| 数据集 | NoC@85 | NoC@90 | NoC@95 |
|---|---|---|---|
| GrabCut | **1.38** | **1.48** | **1.92** |
| Berkeley | **1.39** | **2.17** | **6.18** |
| DAVIS | **3.92** | **5.32** | **11.27** |
| SBD | **3.21** | **5.24** | **11.20** |

- MFP是**唯一**在所有四个数据集上均能以少于4次点击达到mIoU 85%的算法。
- 在12项NoC测试中取得**9项第一**、3项第二。

### 主要结果（COCO+LVIS训练，ViT-B骨干）
MFP在12项测试中获**7项第一**、5项第二，与SimpleClick（ViT-B）相比：
- GrabCut NoC@95：1.70 vs 1.80
- DAVIS NoC@85：3.37 vs 3.66
- SBD NoC@85：3.26 vs 3.43

### 相同骨干公平对比（Table 4，SBD训练）
- 36组对比中MFP获**28组最优**，其余8组差距<0.1。
- ResNet-34版本：GrabCut NoC@85从1.86降至1.70，NoC@95从3.68降至3.00。
- HRNet-18版本：GrabCut NoC@85从1.76降至1.52。
- ViT-B版本：各指标均稳步提升。

### 定性分析（Figure 6）
- SimpleClick等基线在远离点击的区域（如自行车轮）预测不准确，而MFP能正确分割，证明概率调制有效利用了几率图中蕴含的形状细节。

## 相关工作脉络
1. **BRS/f-BRS [14,26]**：推理时通过反向梯度传播优化点击/特征，计算开销大；MFP为纯前馈方法，无需反向传播。
2. **RITM [27]**：首次将上一轮概率图作为网络输入的前馈方法，但概率图未经过调制增强，且训练依赖随机初始点击；MFP进一步提出调制+递归训练。
3. **SimpleClick [21]**：使用ViT骨干的轻量前馈方法，直接将$P^{t-1}$输入网络；MFP在其基础上加入概率调制和晚期融合，性能显著提升。
4. **FocalClick [4] / FocusCut [19]**：在局部窗口内细化预测；MFP在全局概率图层面进行调制，二者思路互补。
5. **EMC-Click [7]**：使用自注意力与相关性模块传播点击信息；MFP通过调制机制实现类似目的，但计算更轻量。
6. **iCMFormer [16]**：跨模态ViT用于交互分割；MFP可在相同ViT-B骨干上取得更好或相当效果，说明调制策略的有效性。

## 局限性与未来方向
1. **单目标分割局限**：当前MFP针对单对象交互分割，对多目标场景需扩展调制与训练策略。
2. **调制超参固定**：$\Gamma、N、R_{max}$等在当前实验中设为固定值，可能需针对不同数据集自适应调整。
3. **仅依赖2D静态图像**：未探索在视频交互分割中的时序一致性利用。
4. **未来方向**：扩展至多目标分割；设计学习型调制器替代手动gamma校正；结合视频帧间一致性；探索在医学图像分割等低数据场景的应用。

## 研究启发与可借鉴点
1. **概率图调制思想可迁移**：gamma校正增强的思路可推广至其他基于历史预测的迭代推理任务（如图像修复、超分辨率），通过调制历史估计结果来提升当前预测质量。
2. **晚期融合优于早期拼接**：将概率相关特征与骨干特征在分割头前融合，比简单在输入端拼接更有效，这一设计原则可复用于其他多源信息融合任务。
3. **递归训练策略的设计启示**：基于误差位置顺序生成点击的递归训练，比随机+迭代混合策略更能利用有序历史信息，该思路可迁移至其他需要序列交互的任务。
4. **双阶段调制策略（概率距离→欧氏距离）**：前期用语义相似性距离、后期用空间距离的策略，对多轮交互系统有参考价值。
5. **可与团队方向结合**：若团队关注低资源分割或序列标注，可将概率调制+递归训练的思想应用于少样本/零样本交互分割，或探索时序视频分割中的概率传播。

## 关键术语表
- **Probability Map Modulation**：对前一轮分割预测的概率图进行gamma校正增强，提升目标形状细节表达的技术。
- **Interactive Image Segmentation**：通过用户逐步点击（前景/背景）引导，迭代更新分割掩码的图像分割任务。
- **NoC（Number of Clicks）**：达到指定IoU阈值所需的平均点击次数，越低表示效率越高。
- **Late Fusion**：在网络末端（分割头前）将不同来源的特征进行融合的策略，相比输入端拼接更能保留语义信息。
- **Recursive Training**：按交互顺序（从初始点击到后续纠错点击）组织训练数据，使模型学习完整的交互序列。
- **Gamma Correction**：源于图像处理的技术，此处用于非线性拉伸/压缩概率值，使前景概率趋近1、背景概率趋近0。
- **Backbone**：分割网络的核心特征提取器，本文测试了ResNet-34、HRNet-18和ViT-B。
- **Normalized Focal Loss**：针对类别不平衡改进的损失函数，本文用于分割头训练。

## 可复现要素
- **数据集**：GrabCut、Berkeley、DAVIS、SBD（均公开）；训练使用SBD、COCO+LVIS（均公开）
- **代码**：已开源 → https://github.com/cwlee00/MFP
- **权重**：论文未明确提供预训练权重下载链接
- **关键超参**：
  - $N=7$（前期/后期调制切换点击数）
  - $R_{max}=100$（调制窗口最大半径）
  - $\Gamma$：使点击处概率变为0.99（前景）/0.01（背景）
  - 优化器：Adam（$\beta_1=0.9, \beta_2=0.999$）
  - 数据增强：随机resize、crop、flip、rotation、brightness
  - 损失函数：normalized focal loss
