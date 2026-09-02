---
title: "MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Zhang_MAPSeg_Unified_Unsupervised_Domain_Adaptation_for_Heterogeneous_Medical_Image_Segmentation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:53:57"
field: "医学图像分析-域适应"
keywords: ["无监督域适应", "医学图像分割", "掩码自动编码器", "伪标签", "联邦学习", "测试时适应", "3D分割"]
innovations: ["提出统一MAPSeg框架，首次系统性解决四种医学图像域偏移", "首次实现统一框架同时适用于集中式、联邦式和测试时三种域适应场景", "设计无需目标标签的模型选择策略Score", "结合3D MAE预训练与掩码伪标签MPL，提升伪标签稳定性"]
benchmarks: ["BCP50婴儿脑MRI", "MMWHS 2017心脏CT-MRI"]
---

# 论文速读：MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling

## 一句话总结
本文提出 MAPSeg，一个统一的无监督域适应（UDA）框架，结合 3D 掩码自动编码器（MAE）与掩码伪标签（MPL）技术，系统解决医学图像分割中的跨序列、跨站点、跨年龄及跨模态域偏移问题，并首次实现该框架在集中式、联邦式和测试时三种场景下的统一应用，在婴儿脑 MRI 和心脏 CT→MRI 分割任务上大幅超越现有 SOTA。

## 研究问题与动机
1. **医学图像存在多种异构域偏移**：同一模态内存在跨序列（T1/T2 对比差异）、跨站点（扫描仪/参数差异）、跨年龄（脑部发育导致尺寸与对比度变化）；不同模态间存在跨模态差异（如 CT 与 MRI 信号本质不同），导致源域训练模型在目标域性能显著下降。
2. **现有方法缺乏统一性与通用性**：先前工作通常针对单一域偏移设计（如基于图像翻译的方法无法处理跨年龄形变），且多数仅适用于集中式场景。
3. **数据共享受法规限制**：HIPAA 和 GDPR 等法规限制多中心数据集中，联邦学习和测试时适应成为更实际的方案，但现有 UDA 方法难以无缝切换。
4. **伪标签易产生漂移**：大域偏移下伪标签可靠性低，缺乏稳定性机制（如全局-局部上下文约束、MAE 预训练先验）易导致误差累积。

## 核心贡献（创新点）
1. **提出统一的 MAPSeg 框架，首次系统性地解决四种医学图像域偏移**：基于 3D MAE + MPL + GLC 的协同设计，区别于以往仅针对单一偏移（如仅跨模态）的方法。
2. **统一框架同时适配集中式、联邦式与测试时三种 UDA 场景**：通过损失函数的分解与重组（Eq.9/10），保持三种场景下可比拟的性能，而多数现有方法仅支持单一场景。
3. **设计 3D 掩码伪标签（MPL）机制，将随机掩码作为强扰动一致性正则化**：与 MIC [31] 等插件式方法不同，MPL 作为独立核心组件，结合 EMA 教师模型显著提升伪标签稳定性。
4. **提出全局-局部特征协作（GLC）模块，利用解剖分布先验增强分割可靠性**：仅在编码器输出层融合全局-局部特征（而非所有层），在降低计算开销的同时抑制伪标签漂移。
5. **设计无需目标标签的模型选择策略 Score = Dice_Src − 0.5 × mean(L_MPL)**：避免现实中不可行的目标域标签验证，实测性能下降仅 0.9 Dice。

## 方法详解
MAPSeg 由三个核心组件构成：

**1. 3D 多尺度掩码自动编码器（MAE）**
- 使用 3D CNN 骨干网络，同时在局部 patch（x，96³  voxels）和下采样全局扫描（X，96³）上进行掩码重建。
- 局部 patch 划分为 8³ 大小的非重叠子 patch，随机掩码 70%；全局扫描同样掩码但子 patch 大小为 4³。
- 损失为掩码区域的均方误差：$L_{MAE} = MSE(x^M_{recon}, x) + MSE(X^M_{recon}, X)$，实现自监督预训练。

**2. 3D 掩码伪标签（MPL）**
- 教师-学生框架：教师模型 $f_\theta$ 对完整目标图像 $x_t$ 生成伪标签 $f_\theta(x_t)$（梯度 detached）；学生模型 $f_\phi$ 在掩码输入 $x_t^M$ 上学习。
- 损失函数：$\mathcal{L}_{MPL} = \mathcal{L}_{Seg}(f_\phi(x_t^M), f_\theta(x_t)) + \beta \mathcal{L}_{Seg}(f_\phi(x_s^M), y_s)$，$\beta = 0.5$。
- 教师参数通过 EMA 更新：$\theta_{t+1} = \alpha \theta_t + (1-\alpha)\phi_t$，根据预训练数据规模动态设置 $\alpha$（大尺度 0.999→0.9999，小尺度 0.99→0.999→0.9999）。

**3. 3D 全局-局部协作（GLC）**
- 编码器 $g$ 分别处理局部 patch $x$ 和全局扫描 $X$，提取局部特征 $\chi_{loc} = g(x)$ 和裁剪上采样后的全局特征 $\chi_{glo} = \text{upsample}(M \odot g(X))$。
- 分割头 $h$ 在隐空间拼接：$f(x) = h(\chi_{loc} \oplus \chi_{glo})$。
- 添加余弦相似度损失防止特征失衡：$\mathcal{L}_{cos}(x,X) = 1 - \frac{\chi_{loc} \cdot \chi_{glo}}{\max(\|\chi_{loc}\|_2, \|\chi_{glo}\|_2, \epsilon)}$。
- 源域总损失：$\mathcal{L}_{GLC}^S = \gamma(\mathcal{L}_{Seg}(f(X_s), Y_s) + \mathcal{L}_{Seg}(f(X_s^M), Y_s)) + \delta(\mathcal{L}_{cos}(x_s, X_s) + \mathcal{L}_{cos}(x_s^M, X_s^M))$，$\gamma=0.05, \delta=0.025$。
- 目标域类似使用伪标签 $f_\theta(X_t)$。

**总损失（集中式）**：$\mathcal{L} = \mathcal{L}_{FSS} + \mathcal{L}_{MPL} + \mathcal{L}_{GLC}$。

**联邦扩展**：服务器仅用源数据（Eq.9），客户端仅用目标数据（Eq.10），每轮后通过 FedAvg 聚合教师参数。

**测试时扩展**：先在源域 warm-up 1000 步，再将 $\phi$ 分发到目标域初始化教师 $f_\theta$，异步自适应。

**模型选择策略**：$Score = Dice_{Src} - 0.5 \times \overline{\mathcal{L}_{Seg}}(f_\phi(x_t^M), f_\theta(x_t))$，无需目标标签。

## 实验与结果
**数据集**：
- 私有婴儿脑 MRI：2,421 scans（2,306 无标注用于预训练），115 标注 scans 用于评估（BCP50、ECHO5、MCRIB10）。
- 公开心脏 CT-MRI：MMWHS 2017，40 scans（20 CT 源 + 20 MRI 目标）。

**评估任务**：跨序列（T1w→T2w）、跨站点（BCP→MCRIB/ECHO）、跨年龄（12-24月→0-6月）、跨模态（CT→MRI）。

**主要结果（集中式 UDA，脑 MRI）**：
| 任务 | MAPSeg Avg Dice | 次优方法 Avg Dice | 提升幅度 |
|------|----------------|------------------|---------|
| Cross-Sequence | 77.7% | DAR-UNet 68.2% | **+9.5** |
| Cross-Site | 73.1% | HRDA 62.7% | **+10.4** |
| Cross-Age | 75.2% | MIC 64.7% | **+10.5** |

**心脏 CT→MRI（Tab.4）**：MAPSeg 平均 Dice 80.3%（无目标标签验证），较前作 SDUDA (73.3%) 提升 **+7.0**，较 FSUDA-V2 (74.8%) 提升 **+5.5**。

**联邦 UDA（Tab.2）**：Fed-MAPSeg 跨序列 69.9%、跨站点 73.6%、跨年龄 71.0%，显著优于 FAT (27.6%/63.8%/69.0%) 和 DualAdapt (28.4%/66.1%/54.8%)。

**测试时 UDA（Tab.3）**：目标域性能下降 <3%，源域下降 1.6%~5.8%（存在一定知识遗忘）。

**消融（Tab.5）**：MAE 预训练贡献最大（MPL 单独 31.6% → MAE+GLC+MPL 75.3%）；掩码率 70%、patch 大小 8³ 为最优。

## 相关工作脉络
1. **AdvEnt [63]**：对抗熵最小化，通过判别器对齐源/目标特征分布；MAPSeg 不使用对抗训练，依赖伪标签与 MAE 自监督先验。
2. **DAFormer [29] / HRDA [30]**：伪标签一致性正则化方法；MAPSeg 引入随机掩码作为更强扰动，并配合 MAE 增强泛化，且在跨序列任务上避免 HRDA 对 pallidum/accumbens 的零分割失败。
3. **MIC [31]**：将掩码图像一致性作为插件增强基线；MAPSeg 将掩码机制内化为 MAE+MPL 的统一框架，而非后加组件。
4. **DAR-UNet [70]**：基于图像翻译的方法；MAPSeg 避免翻译误差（如跨年龄尺寸变化导致的分割错误），性能更稳定。
5. **FAT [50] / DualAdapt [69]**：联邦多目标域适应；MAPSeg 假设客户端完全无标签（更严苛设定），设计更简单的 EMA 教师-学生机制，且专针对 3D 医学图像。
6. **SIFA-V1/V2 [3,4] / SDUDA [12] / FSUDA-V2 [43]**：心脏跨模态 UDA；MAPSeg 在无需目标标签验证的前提下超越这些方法，更具实用性。

## 局限性与未来方向
1. **测试时 UDA 存在源域知识遗忘**：源域性能下降 1.6%~5.8%，需探索抗遗忘机制（如正则化、记忆回放）。
2. **联邦场景假设较理想化**：未考虑客户端数据 Non-IID、通信延迟、客户端异构性等实际挑战。
3. **小数据场景仍需从头预训练 MAE**：虽然仅几十条 scan 即可训练，但大尺度预训练（>2000 scans）才能直接迁移到新目标域。
4. **未探索多源域适应（multi-source UDA）**：现实场景常涉及多个源域，当前框架为单源→多目标。
5. **计算开销较高**：3D 视频重建、GLC 多尺度特征融合增加推理与训练时间，轻量化设计待探索。

## 研究启发与可借鉴点
1. **随机掩码作为强扰动的一致性正则化**：将 MAE 的掩码思想移植到伪标签学习中，为自监督域适应提供新视角，可迁移至自然图像或视频分割任务。
2. **全局-局部特征协作的轻量级设计**：仅在编码器输出层融合上下文而非所有层，在保障性能的同时控制计算开销，值得在高分辨率 3D 分割中借鉴。
3. **无目标标签的模型选择指标**：Score = Dice_Src − λ × mean(L_MPL) 有效指示适应程度，可替代现实中不可行的目标验证，适用于其他 UDA 任务。
4. **EMA 教师模型的动态更新策略**：根据预训练数据规模灵活调整 α，提升小样本场景稳定性，具有普适价值。
5. **统一损失分解适配多场景**：通过 Eq.9/10 将集中式损失拆解为服务器/客户端子损失，为联邦/测试时扩展提供清晰范式。

## 关键术语表
**无监督域适应（UDA）**：利用源域标注数据和无标签目标域数据，使模型适应目标域分布的机器学习技术。
**掩码自动编码器（MAE）**：通过随机掩码部分输入并重建缺失区域，学习鲁棒特征表示的自监督预训练方法。
**掩码伪标签（MPL）**：在教师-学生框架下，对目标域掩码输入生成伪标签并约束一致性，实现域自适应自训练。
**EMA（指数移动平均）**：对模型参数进行滑动平均更新，生成更稳定、泛化更强的教师模型。
**跨序列/跨站点/跨年龄/跨模态域偏移**：分别指 MRI 序列差异、扫描设备差异、发育阶段差异、成像模态差异导致的输入分布变化。
**GLC（全局-局部协作）**：融合局部 patch 特征与全局上下文特征，利用解剖先验增强分割可靠性的模块。
**联邦 UDA**：在数据不出本地的分布式场景下，通过模型参数聚合实现多机构间的域适应。

## 可复现要素
- **代码**：已开源 https://github.com/XuzheZ/MAPSeg/
- **心脏数据集**：MMWHS 2017（公开）
- **脑 MRI 数据集**：BCP50（私有，含 115 标注）、ECHO5（私有）、MCRIB10（部分公开）
- **关键超参**：掩码率 70%，局部 patch 8³，全局 patch 4³，β=0.5，γ=0.05，δ=0.025，EMA α=0.999→0.9999
- **模型架构**：3D ResNet-like 编码器 + DeepLabV3 分割头
- **训练框架**：PyTorch
- **论文未提及**：具体 GPU 型号、学习率 schedule、batch size、optimizer 细节
