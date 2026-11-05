---
title: "FocSAM-Delving-Deeply-into-Focused-Objects-in-Segmenting-Any"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_FocSAM_Delving_Deeply_into_Focused_Objects_in_Segmenting_Anything_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:00:56"
field: "图像语义分割"
keywords: ["交互式分割", "Segment Anything Model", "动态窗口注意力", "ViT适配", "高效推理"]
innovations: ["提出聚焦细化器框架，打破SAM预提取嵌入一次性局限，动态聚焦目标区域", "设计Dwin-MSA，基于预测掩码动态选择计算窗口避免冗余背景处理", "提出P-DyReLU，以像素级Query-Embedding相似度驱动自适应激活实现交互信息深度融合"]
benchmarks: ["SBD", "COD10K", "DAVIS", "MVTec", "Grab-Cut", "Berkeley"]
---

# 论文速读：FocSAM: Delving Deeply into Focused Objects in Segmenting Anything

## 一句话总结
FocSAM针对SAM交互式分割在不稳定样本上的劣化问题，提出一种聚焦细化器（focus refiner），通过动态窗口自注意力（Dwin-MSA）和逐像素动态ReLU（P-DyReLU）动态聚焦目标区域并深度融合初始点击的交互信息，在仅增加约10%计算开销的情况下将SAM性能提升至SOTA，CPU推理时间仅为SimpleClick的5.6%。

## 研究问题与动机
1. **SAM无法高效实现图像级zoom-in策略**：SAM的图像预处理阶段一次性生成全局图像嵌入，交互过程中无法动态放大并重新聚焦目标对象区域，限制了模型对目标对象的空间感知能力。
2. **轻量级解码器难以充分融合交互信息**：为满足实时响应需求，SAM解码器缺乏将少数关键点击信息与图像嵌入深度融合的机制，导致初始点击的引导作用未能充分发挥。
3. **SAM在挑战性样本上表现极不稳定**：如图1所示，在迷彩物体等困难样本上，SAM在第9次点击（IoU=88.59）后再添加一次点击可能导致IoU骤降至12.78，严重制约实际应用场景。
4. **现有SOTA方法推理效率过低**：SimpleClick虽在分割质量上领先，但其CPU端SPC达6.99秒，远慢于SAM的0.35秒，难以满足实时交互式标注需求。

## 核心贡献（创新点）
1. **提出聚焦细化器（focus refiner）框架**：在SAM流水线基础上，于交互早期（第K=2次点击后）插入一次refiner处理，利用当前预测掩码动态调整目标相关的图像嵌入，替代原始全局嵌入供后续交互使用；与SAM的本质区别在于打破了预提取嵌入"一次性"的局限，使嵌入可随交互动态聚焦目标。
2. **设计动态窗口多头自注意力（Dwin-MSA）**：基于先前预测掩码的边界框动态选择与其相交的窗口子集进行注意力计算，避免对无关背景的冗余处理；与现有固定窗口注意力（如Swin Transformer）的本质区别在于窗口选择是数据驱动的、动态变化的。
3. **提出逐像素动态ReLU（P-DyReLU）**：利用SAM解码器生成的点击融合查询嵌入（click-fused query embedding），以像素级相似度方式自适应调节Dwin-MSA中的激活参数，增强目标相关特征并抑制无关背景；与标准ReLU或全局DyReLU的本质区别在于激活系数由像素级Query-Embedding相似度推导而来，实现了交互信息的精细融合。
4. **实现SOTA分割质量与极低CPU推理时间的统一**：在6个数据集上匹配SimpleClick的NoC@90性能，同时CPU端SPC仅为SimpleClick的约5.6%（对象数>10时更低至1.2%）；与SimpleClick的本质区别在于采用SAM预提取嵌入的流水线，避免了每步全模型推理。

## 方法详解
**整体架构**：FocSAM构建于SAM流水线之上，增加一个聚焦细化器（focus refiner），每个对象在交互过程中调用一次。Refiner接收图像嵌入F、上一轮预测掩码M^(K-1)和点击融合查询嵌入q_c^(K-1)，输出细化后的图像嵌入F_r^(K)，替换原始嵌入供后续交互使用。

**Refine Block设计**：Refiner由12个block交替堆叠（6个普通refine block + 6个shift refine block）。每层block共享同一个掩码M^(K-1)，依次经过Dwin/Shift Dwin模块（动态窗口选择）和MSA模块（特征细化与融合）。

**Dwin-MSA**：将图像嵌入F划分为S×S的窗口（S=16），根据掩码边界框筛选与其相交的窗口子集F_W∈R^(M×S×S×256)。采用移位策略（shifting strategy）交替普通Dwin与Shift Dwin，确保边界框内所有patch间充分的信息交换。MSA模块中：① 查询嵌入q_c与窗口特征f做cross-attention得到q_f；② f做self-attention得到f_hat；③ 通过P-DyReLU激活；④ 经DeformConv残差连接输出f_q和q_f（各窗口q_f经平均求和）。

**P-DyReLU**：以DyReLU为骨架，系数由像素级相似度推导：a⁰=b⁰=Expand(f̂·q_f^⊤)（捕获像素级Query-Embedding相似度），a¹=b¹=Expand(AvgPool(f̂))（捕获全局统计量）。四组系数经通道级MLP变换后，按公式9进行逐像素Piecewise Linear激活，实现基于交互信息的自适应特征增强与抑制。

**训练策略**：采用两阶段训练——先微调SAM解码器320k次（batch size=16），再以冻结解码器训练FocSAM 160k次。损失函数采用归一化焦点损失（NFL，来自RITM）+ 辅助点损失（PTL，来自BRS）。训练中使用InterFormer的click simulation策略，图像嵌入预提取并缓存。

## 实验与结果
**数据集**：训练集COCO+LVIS；测试集Grab-Cut、Berkeley、SBD（6671样本）、DAVIS、MVTec（异常检测）、COD10K（2026样本，迷彩目标）；评估指标NoC@90（20次点击内达90% IoU所需平均点击数）和SPC（CPU端每秒点击数）。

**主要结果（Table 1）**：FocSAM-ViT-H在5/6数据集上取得最优NoC@90，其中COD10K达8.91（最优），SBD达4.69（最优），平均NoC为5.38，略优于SimpleClick-ViT-H的5.40。CPU端SPC为0.39（含refiner），仅为SimpleClick-ViT-H（6.99）的约5.6%；当单图对象数>10时进一步降至约1.2%。

**稳定性分析（Figure 4）**：在SBD、MVTec、COD10K上以100次点击评估，FocSAM的∆IoU分布显著右移，表明额外点击导致分割质量大幅下降的样本数量远少于SAM。

**消融实验（Table 2）**：Dwin-MSA和P-DyReLU各自独立贡献相近，组合后在100 clicks@95% IoU指标上优势更显著，说明两者提供的交互信息互补，共同缓解挑战性样本的负面效应。

## 相关工作脉络
1. **SAM** [28]：零样本分割大模型，采用"预提取ViT嵌入+轻量解码器"流水线，但无法动态聚焦目标且解码器融合交互信息能力有限，本文在其基础之上增加refiner进行修正。
2. **SimpleClick** [36]：首个将大型ViT引入交互式分割的方法，采用纯ViT架构逐次全量推理，分割质量优异但CPU推理慢（SPC=6.99），本文方法在匹配其性能的同时将推理速度提升约18倍。
3. **InterFormer** [23]：通过复用图像特征减少冗余的实时交互分割方法，本文采用相似的"冻结encoder+预提取嵌入"思路，但引入了refiner增强动态聚焦能力。
4. **f-BRS** [46]：提出图像级zoom-in策略，通过裁剪目标区域并重采样来动态聚焦，本文的Dwin-MSA在嵌入空间实现了等效的动态聚焦，避免了RoIAlign在ViT嵌入上的插值不适配问题。
5. **FocalClick** [5]：利用焦点区域加权的交互式分割方法，本文与其目标一致（提升聚焦能力），但通过动态窗口注意力而非焦点加权实现。
6. **DyReLU** [6]：输入依赖的动态ReLU激活函数，本文的P-DyReLU将其扩展至像素级，并巧妙利用Query-Embedding相似度驱动激活系数的生成。

## 局限性与未来方向
1. **CPU端refiner引入额外开销**：FocSAM的SPC（0.39）相比原始SAM（0.35）仍有约10%的增长，refiner的设计仍有优化空间。
2. **仅评估了点击提示交互**：论文聚焦于点击-based交互分割，对于框提示、掩码提示等其他prompt类型的适应性未验证。
3. **训练数据偏差担忧**：虽使用COCO+LVIS训练以避免SAM预训练数据（SA-1B含大量自动生成掩码）的偏差，但refiner对解码器输出依赖较强，训练不稳定性需通过两阶段策略缓解。
4. **未来方向**：可探索更高效的refiner结构（如更浅的block数或更小的窗口）；将动态聚焦思想推广至视频交互分割、3D分割等时序/空间扩展场景；P-DyReLU的像素级自适应激活机制也可迁移至其他视觉任务。

## 研究启发与可借鉴点
1. **预提取大模型嵌入+轻量动态refiner的范式**：在ViT等大模型的embedding空间中引入轻量级的动态聚焦模块，是一种兼顾预训练知识复用与任务适配的高效范式，可迁移至其他大模型适配场景。
2. **早期点击的"锚定效应"设计**：利用前2次点击的预测掩码边界框来指导后续嵌入细化，这一"以少导多"的策略对设计高效交互系统具有启发价值——关键交互信号应在最早阶段发挥作用。
3. **Dwin-MSA的动态窗口策略**：基于预测掩码的边界框动态选择计算窗口，避免了对无关背景的计算浪费，这一"任务驱动的计算资源分配"思路可推广至目标检测、实例分割等任务的高效推理。
4. **P-DyReLU的跨模态激活调制**：将Query Embedding与Image Embedding的逐像素相似度作为激活系数的输入，实现了跨模态信息对特征非线性变换的精细控制，该机制可迁移至其他需要细粒度特征选择的多模态融合任务。
5. **稳定性分析的评估范式**：通过∆IoU分布（过滤ΔIoU>-1%的显著劣化）量化交互过程中的稳定性，为交互式模型评估提供了比单一NoC指标更丰富的分析视角。

## 关键术语表
**FocSAM**：本文提出的交互式分割方法，在SAM流水线基础上引入聚焦细化器，动态增强目标相关的图像嵌入并深度融合初始交互信息。

**Dwin-MSA（Dynamic Window Multi-head Self-Attention）**：基于预测掩码边界框动态筛选窗口的多头自注意力模块，避免对无关背景的冗余计算，结合移位策略保证长距离交互。

**P-DyReLU（Pixel-wise Dynamic ReLU）**：以像素级Query-Embedding相似度驱动激活参数的动态ReLU变体，自适应增强目标相关特征并抑制无关背景。

**Focus Refiner**：部署于SAM交互流程中的额外模块，在交互早期（K=2次点击后）对图像嵌入进行一轮动态聚焦优化，输出细化嵌入替代原始嵌入。

**NoC@90（Number of Clicks at 90% IoU）**：评估指标，表示达到90% IoU分割精度所需的平均最少点击次数，值越低性能越好。

**SPC（Seconds Per Click）**：评估指标，表示CPU端每次点击的平均推理耗时（秒），值越低实时性越好。

**Zoom-in策略**：图像级目标聚焦技术，通过裁剪目标区域并放大重采样以增强模型对目标细节的感知，本文在嵌入空间以动态窗口方式实现了等效功能。

**f-BRS（Fast Back-Propagation Refinement Scheme）**：早期交互式分割方法，通过反向传播迭代细化分割结果并提出了图像级zoom-in策略，本文与其在动态聚焦思想上一脉相承但实现路径不同。

## 可复现要素
- **训练数据集**：COCO + LVIS（公开）
- **测试数据集**：Grab-Cut、Berkeley、SBD、DAVIS、MVTec、COD10K（均公开）
- **代码**：已开源，https://github.com/YouHuang67/focsam
- **模型权重**：基于SAM预训练的ViT-Huge backbone（公开），refiner部分代码开源可复现
- **关键超参**：Refiner共12个block（6普通+6移位）、window size S=16、refine step K=2（第2次点击后激活）、嵌入维度256、两阶段训练（解码器微调320k次→refiner训练160k次，batch size=16）、图像尺寸1024×1024、损失函数NFL+PTL、学习率/优化器等细节见supplementary
