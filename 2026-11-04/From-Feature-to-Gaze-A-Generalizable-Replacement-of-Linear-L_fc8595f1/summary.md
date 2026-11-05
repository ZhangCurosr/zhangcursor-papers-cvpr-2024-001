---
title: "From-Feature-to-Gaze-A-Generalizable-Replacement-of-Linear-L"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Bao_From_Feature_to_Gaze_A_Generalizable_Replacement_of_Linear_Layer_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:00:18"
field: "视线估计与域泛化"
keywords: ["gaze estimation", "domain generalization", "feature projection", "geodesic distance", "Isomap", "sphere-oriented training"]
innovations: ["GPM：用测地距离解析性映射替代FC层，仅10参数实现球面对齐", "SOT：通过反向过程优化特征提取器使特征更贴合球面分布", "Isometric Propagator：用MLP参数化Isomap实现可微投影"]
benchmarks: ["ETH-XGaze", "Gaze360", "MPIIFaceGaze", "EyeDiap"]
---

# 论文速读：From-Feature-to-Gaze-A-Generalizable-Replacement-of-Linear-L

## 一句话总结
本文针对视线估计模型跨域泛化能力差的问题，提出了解析性视线泛化框架（AGG），通过将高维图像特征解析性映射到三维空间来替代传统全连接层，在不使用目标域数据的情况下显著提升了跨数据集的视线估计精度。

## 研究问题与动机
- **核心问题**：基于外观的视线估计方法在跨数据集测试时性能显著下降，现有方法多依赖目标域数据进行域适配，难以在实际场景中部署。
- **过拟合根源**：现有方法使用包含数千参数的高维全连接层将512D图像特征映射到3D视线方向，极易过拟合图像中的非视线无关因素（如外观、光照、头部姿态）。
- **物理洞察**：视线方向在物理空间中分布在单位球面上，特征空间中沿流形的测地距离与样本间的视线角度差呈正比关系。
- **任务定位**：本文聚焦于**视线泛化任务**（gaze generalization），即在无任何目标域数据的情况下提升模型跨域性能，比域适配更具挑战性。

## 核心贡献（创新点）
- **GPM（测地投影模块）**：提出通过Isomap算法将高维图像特征解析性投影到三维空间，提取视线主成分（PCG），仅用10个可学习参数完成球面对齐，从根本上缓解过拟合。
- **SOT（球面导向训练）**：设计反向过程优化预训练CNN特征提取器，使提取的特征更符合测地距离与视线角度差的正比关系，无需目标域数据。
- **Isometric Propagator（等距传播器）**：用三层MLP参数化Isomap以解决其无法直接融入反向传播的问题，仅在源域训练阶段使用。
- **系统性实验验证**：在12个跨数据集设置下，AGG稳定提升性能，最大提升幅度达35.79%，超越SOTA视线估计与泛化方法。

## 方法详解
**AGG框架包含两个核心模块：**

**1. Geodesic Projection Module（GPM）**
- **步骤一：特征提取与投影**。冻结预训练的CNN特征提取器 $F_{\theta_1}$，从源域提取512D图像特征 $f_i$，利用Isomap算法通过测地距离将其投影到3D空间，得到 Principle Gaze Feature（PGF）$e_i \in \mathbb{R}^3$：
  $$e_i = \text{Isomap}(f_i)$$
- **步骤二：球面对齐（Sphere Alignment）**。PGF分布在近似球面上，通过旋转和平移将PGF球面与单位视线标签球面对齐：
  $$e'_i = R(e_i - O_c)$$
  其中 $O_c$ 为球心，$R$ 为旋转矩阵。
- **步骤三：欧拉角线性拟合预测视线**。对对齐后的特征计算球坐标，通过线性拟合得到yaw和pitch角：
  $$\theta'_i = k_1 \arctan(x_i^{e'}/z_i^{e'}) + b_1, \quad \psi'_i = k_2 \arcsin(y_i^{e'}) + b_2$$
  共10个可学习参数 $\theta_s = \{O_c, R, k_1, k_2, b_1, b_2\}$，通过最小化预测视线与标签的角偏差优化。

**2. Sphere-Oriented Training（SOT）**
- **理想PGF反算**：利用Sphere Alignment的可逆性，从视线标签反算理想PGF位置 $\hat{e}_i = S A^{-1}(y_i)$。
- **Isometric Propagator（IP）训练**：由于Isomap无法直接反向传播，先用源域2000个样本训练三层MLP（Isometric Propagator $IP_{\theta_3}$）来模拟Isomap：
  $$\min_{\theta_3} \mathcal{L}_1(\text{Isomap}(f_i), IP_{\theta_3}(f_i))$$
- **特征提取器优化**：冻结IP和预训练CNN参数，解冻CNN特征提取器，用IP替代Isomap进行反向传播训练：
  $$\min_{\theta_1} \mathcal{L}_1(\hat{e}_i, IP_{\theta_3}(F_{\theta_1}(x_i)))$$
- **测试阶段**：使用原始Isomap（非IP）进行特征投影，保证更好的泛化能力。

**实现细节**：Adam优化器，学习率 $10^{-4}$，预训练10个epoch，SOT训练10个epoch，IP训练100个epoch，batch size=512，Isomap邻居数=300，无数据增强。

## 实验与结果
- **数据集**：ETH-XGaze（$\mathcal{D}_E$，756k图像）、Gaze360（$\mathcal{D}_G$，101k图像）、MPIIFaceGaze（$\mathcal{D}_M$，45k图像）、EyeDiap（$\mathcal{D}_D$，16k图像）。$\mathcal{D}_M$ 和 $\mathcal{D}_D$ 仅作为目标域。排除 $\mathcal{D}_E \leftrightarrow \mathcal{D}_G$ 跨域设置（误差过大约20°）。
- **基线模型**：ResNet-18、ResNet-50、VGG16。
- **核心结果（Tab. 2）**：
  - 在全部12个跨数据集设置下稳定提升，最大提升达 **35.79%**（ResNet-18，$\mathcal{D}_G \to \mathcal{D}_D$：12.35°→7.93°）。
  - ResNet-18+AGG 在 $\mathcal{D}_E \to \mathcal{D}_M$、$\mathcal{D}_E \to \mathcal{D}_D$、$\mathcal{D}_G \to \mathcal{D}_M$、$\mathcal{D}_G \to \mathcal{D}_D$ 四项中分别达到 7.10°、7.07°、7.87°、7.93°。
- **SOTA对比（Tab. 3）**：
  - 超越 PureGaze、LatentGaze、ADL、Full-Face 等方法。
  - ResNet-50+AGG 在 $\mathcal{D}_E \to \mathcal{D}_M$（5.91°）、$\mathcal{D}_E \to \mathcal{D}_D$（6.75°）、$\mathcal{D}_G \to \mathcal{D}_M$（9.20°）、$\mathcal{D}_G \to \mathcal{D}_D$（11.36°）均取得最优。
- **消融验证**：GPM仅在推理时替换FC层即可提升跨域性能；SOT显著降低PGF球面误差（Sphere Error），使特征分布更均匀地落在球面上。

## 相关工作脉络
- **PureGaze (Cheng et al., AAAI 2022)**：通过净化源域训练中的视线特征提升泛化能力，但未触及特征到视线的解析映射机制，本文在此基础上提出更物理可解释的解析性替换方案。
- **ADL (Kellnhofer et al., ICCV 2019)**：基于对比学习的域适配方法，需目标域数据，本文方法无需目标域数据，适用场景更广。
- **LatentGaze (Lee et al., ACCV 2022)**：通过眼动感知分析潜码操纵实现跨域泛化，依赖数据驱动策略，本文利用测地距离的物理特性提供解析性替代。
- **Rotation Consistency (Bao et al., CVPR 2022)**：利用旋转一致性提升泛化能力，仍需目标域辅助信息，本文完全在源域内完成。
- **传统全连接层回归**：本文指出FC层因参数量大（数千）易过拟合，而GPM仅用10个参数实现解析映射，本质区别在于参数规模与物理可解释性。

## 局限性与未来方向
- **Gaze表示的限制**：当视线用2D欧拉角（yaw, pitch）表示时，投影特征分布在近似2D平面上而非球面，需调整Sphere Alignment算法（作者留作未来工作）。
- **Isometric Propagator的折衷**：IP为MLP结构，虽仅用于训练阶段，但仍存在潜在过拟合风险；若能找到将Isomap融入反向传播的更好方案，性能可能进一步提升。
- **域内精度轻微下降**：为追求跨域泛化，域内精度略有牺牲（约2%-17%），需在特定应用中权衡。
- **未验证的模型架构**：目前仅在ResNet-18/50、VGG16上验证，在其他 backbone（如Transformer）上的适用性待探索。

## 研究启发与可借鉴点
- **物理先验引导特征映射**：将问题的物理约束（视线球面分布）与特征空间的几何性质（测地距离正比于角度差）结合，为其他具有物理意义的回归任务（如姿态估计）提供了可借鉴的解析映射思路。
- **测地距离替代欧氏距离**：在高维特征空间中利用测地距离而非简单L2距离可以更好地保留流形结构，这一思想可迁移到其他视觉表征学习场景。
- **小参数模块替代大参数FC层**：用10参数模块替代数千参数的FC层是一个简洁有效的防过拟合策略，适用于各类高维特征到低维输出的回归任务。
- **离线投影+在线推理分离**：Isomap在测试时离线计算、IP在训练时可微分的分离设计，为无法直接反向传播的几何变换提供了实用解决方案。
- **特征球面化训练目标**：Sphere-Oriented Training 提出的"使特征分布贴合球面"的训练信号，可作为通用的特征正则化手段。

## 关键术语表
- **Gaze Estimation（视线估计）**：从人脸/眼睛图像中直接估计视线方向的技术，分为基于模型的方法和基于外观的方法。
- **Principle Component of Gaze（PCG，视线主成分）**：高维图像特征中与视线方向相关的低维子空间，理论上维度为2（对应视线球面的两个自由度）。
- **Principle Gaze Feature（PGF，视线主特征）**：通过Isomap将高维图像特征投影到3D空间后得到的特征，分布在近似球面上。
- **Geodesic Projection Module（GPM，测地投影模块）**：用测地距离将图像特征解析性映射到3D空间并进行球面对齐的模块，作为FC层的一般化替换。
- **Sphere-Oriented Training（SOT，球面导向训练）**：通过反向过程优化特征提取器CNN，使提取的特征更符合测地距离与视线角度差的正比关系。
- **Isometric Propagator（IP，等距传播器）**：用三层MLP参数化Isomap的代理网络，用于将Isomap融入反向传播训练流程。
- **Sphere Error（球面误差）**：PGF点到球面的距离与球半径之比，用于衡量特征分布的球面化程度。
- **Domain Generalization（域泛化）**：在无目标域数据的情况下提升模型在未见域的泛化性能。

## 可复现要素
- **数据集**：ETH-XGaze、Gaze360、MPIIFaceGaze、EyeDiap（均为公开数据集）
- **代码开源**：论文未提及代码开源情况
- **权重开源**：论文未提及
- **关键超参**：Isomap邻居数=300，预训练10 epoch，SOT训练10 epoch，IP训练100 epoch，batch size=512，学习率=10⁻⁴，Adam优化器，无数据增强，像素归一化到[0,1]
- **实现框架**：PyTorch，Isomap使用Scikit-learn实现
