---
title: "From-Feature-to-Gaze-A-Generalizable-Replacement-of-Linear-L"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Bao_From_Feature_to_Gaze_A_Generalizable_Replacement_of_Linear_Layer_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:43:31"
field: "眼动估计与域泛化"
keywords: ["gaze estimation", "domain generalization", "geodesic projection", "Isomap", "FC layer replacement", "analytical regression", "cross-domain"]
innovations: ["用测地投影模块（GPM）以10参数替代数千参数FC层，从根本上缓解眼动估计跨域过拟合", "提出Isometric Propagator使Isomap可微分，将球面几何约束反向传播到CNN特征提取器", "揭示并验证深度特征测地距离与视线角差成正比的跨域普适规律"]
benchmarks: ["ETH-XGaze", "Gaze360", "MPIIFaceGaze", "EyeDiap"]
---

# 论文速读：From Feature to Gaze: A Generalizable Replacement of Linear Layer for Gaze Estimation

## 一句话总结
本文提出 **AGG（Analytical Gaze Generalization，解析眼动泛化）框架**，通过用测地投影模块（GPM）替代传统全连接层，并将球面导向训练（SOT）融入源域训练，无需任何目标域数据即可显著提升眼动估计模型在未知域上的泛化精度。

---

## 研究问题与动机
1. **跨域性能退化严重**：基于外观的眼动估计方法（appearance-based）在源域训练、目标域测试的场景下误差急剧增大，实用性受限。
2. **FC层过拟合是根因**：现有CNN直接从512维高维图像特征经数千参数的全连接层映射到3D单位向量，大量冗余参数易过拟合到光照、外观、头姿等眼动无关因素。
3. **目标域数据不可得**：传统领域自适应方法（对抗学习、对比学习、协作学习）均需目标域样本进行适配，在真实场景中常不可行；PureGaze [8] 虽不依赖目标域数据，但本文从更本质的几何投影角度重新审视该问题。
4. **缺乏可解释的解析映射**：现有方法依赖纯数据驱动的端到端回归，缺乏对"特征空间几何结构—眼动物理空间"之间对应关系的显式建模。

---

## 核心贡献（创新点）
1. **提出GPM模块，用10参数解析投影替代数千参数FC层**：利用Isomap将高维图像特征沿测地距离投影到3D球面空间，以极小参数量的球面对齐算法完成眼动预测，从根本上阻断过拟合路径。
2. **提出SOT模块，将GPM思想反向融入特征提取器训练**：通过Isometric Propagator（可微分MLP近似Isomap）使测地投影约束可反向传播，持续优化CNN提取特征，使其更符合眼动几何先验。
3. **揭示并验证跨域普适的测地距离—视线角正比规律**：在ETH-XGaze、Gaze360、MPIIFaceGaze、EyeDiap四个数据集上均验证了图像特征测地距离与视线角差异成正比的规律，为该方法的跨域有效性提供理论支撑。
4. **在12个跨域设置上取得SOTA级性能**：最大提升35.79%（D_G→D_D，ResNet-18），且在多组对比中稳定超越PureGaze、ADL、LatentGaze等现有方法，无需目标域数据。

---

## 方法详解

### 4.1 Geodesic Projection Module（GPM）
GPM作为预训练模型最后一层FC的即插即用替代，包含两步：

**Step 1 — 测地投影（Isomap）**：冻结预训练的CNN特征提取器 $F_{\theta_1}$，在源域提取全部512维特征 $\{f_i\}$，用Isomap算法将其投影到3D空间得到 Principle Gaze Feature（PGF）$e_i \in \mathbb{R}^3$：
$$\{e_i\} = \text{Isomap}(\{f_i\})$$
投影后 $e_i$ 近似分布在3D球面上，且球面弧长与视线角差成正比。

**Step 2 — Sphere Alignment（SA）球面对齐**：用10个可学习参数将PGF球面与单位眼动球面对齐：
1. 定位PGF球心 $O_c$，作平移和旋转对齐：$e_i' = R(e_i - O_c)$
2. 计算 $e_i'$ 的Euler角，通过线性拟合预测yaw和pitch：
$$\theta_i' = k_1 \arctan(x_i'/z_i') + b_1, \quad \psi_i' = k_2 \arcsin(y_i') + b_2$$
3. 将Euler角转换回3D单位向量即为最终眼动预测。

10个参数为 $\theta_s = \{O_c(3), R(6), k_1, k_2, b_1, b_2\}$，仅随机选取2000个源域样本优化即可，避免过拟合。测试时目标域特征追加到源域构建的距离矩阵中重新做Isomap，SA参数保持固定。

### 4.2 Sphere-Oriented Training（SOT）
GPM仅作用于推理阶段，SOT进一步将投影约束反向传播到CNN：

1. **Isometric Propagator（IP）训练**：用3层MLP $IP_{\theta_3}$ 在源域上拟合Isomap的输入-输出映射（训练100 epochs，2000样本），最小化 $\mathcal{L}_1(IP_{\theta_3}(f_i), \text{Isomap}(f_i))$。
2. **特征提取器优化**：冻结预训练CNN和IP参数，以SA的逆过程构造理想PGF位置 $\hat{e}_i = SA^{-1}(y_i)$，训练CNN最小化：
$$\arg\min_{\theta_1} \mathcal{L}_1(\hat{e}_i, IP_{\theta_3}(F_{\theta_1}(x_i)))$$
训练10 epochs，batch size=512，Adam lr=1e-4。

**关键设计**：IP仅在源域训练阶段使用；测试时完整恢复Isomap以保证泛化性，IP不参与推理。

### 4.3 实现细节
- 框架：PyTorch；优化器：Adam lr=1e-4
- 预训练：10 epochs（取最后一轮为baseline）
- Isomap邻居数：300（Scikit-learn实现）
- 像素归一化到[0,1]，无数据增强

---

## 实验与结果

### 数据集
- **ETH-XGaze ($\mathcal{D}_E$)**：75.6万张实验室高精度图像，大视野范围；最后5个受试者为测试集
- **Gaze360 ($\mathcal{D}_G$)**：10.1万张户外360°相机图像，仅用正面人脸
- **MPIIFaceGaze ($\mathcal{D}_M$)**：4.5万张笔记本日常使用图像，视野范围较小，仅作为目标域
- **EyeDiap ($\mathcal{D}_D$)**：1.6万张实验室图像，仅作为目标域
- 排除 $\mathcal{D}_E \leftrightarrow \mathcal{D}_G$ 双向设置（跨域误差过大，约20°）

### GPM单独效果（Tab.1）
仅将最后一层FC替换为GPM（不改动CNN），在8个跨域设置中7个提升：
- ResNet-18：$\mathcal{D}_E \to \mathcal{D}_M$ 8.66° → **7.87°**；$\mathcal{D}_E \to \mathcal{D}_D$ 7.76° → 7.72°
- ResNet-50：$\mathcal{D}_E \to \mathcal{D}_M$ 6.92° → **6.56°**；$\mathcal{D}_G \to \mathcal{D}_M$ 8.48° → **8.14°**
- 跨域稳定性显著提高（std普遍降低）

### AGG完整框架效果（Tab.2）
在12个跨域设置中全部稳定提升：
- **最大提升35.79%**：ResNet-18在 $\mathcal{D}_G \to \mathcal{D}_D$ 从12.35°降至**7.93°**
- ResNet-50+AGG在 $\mathcal{D}_E \to \mathcal{D}_M$ 达到 **5.91°**（优于PureGaze的7.08°）
- 域内精度小幅下降属预期（为泛化而牺牲轻微域内拟合）

### 与SOTA对比（Tab.3）
| 方法 | $\mathcal{D}_E \to \mathcal{D}_M$ | $\mathcal{D}_E \to \mathcal{D}_D$ | $\mathcal{D}_G \to \mathcal{D}_M$ | $\mathcal{D}_G \to \mathcal{D}_D$ |
|------|------|------|------|------|
| Full-Face | 12.35 | 30.15 | 11.13 | 14.42 |
| ADL | 7.23 | 8.02 | 11.36 | 11.86 |
| PureGaze | 7.08* | 7.48* | 9.28 | 9.32 |
| **ResNet-18+AGG** | 7.10 | **7.07** | **7.87** | **7.93** |
| **ResNet-50+AGG** | **5.91*** | **6.75*** | **9.20*** | **11.36*** |

AGG在多个设置中超越SOTA，ResNet-50+AGG在4个设置中均取得最优或次优。

---

## 相关工作脉络
1. **PureGaze [8]**：同样针对无目标域数据的泛化任务，但通过"特征纯化"（分类头梯度过滤）实现，本文从几何测地投影角度提出互补思路。
2. **ADL [12] / CA-Net [6]**：传统域适应方法，需要目标域数据训练特定域适配器，本文完全不依赖目标域，适用场景更广。
3. **LatentGaze [18]**：通过潜码解析操作实现跨域，与本文直接操作特征空间几何结构的方法论不同。
4. **LLE [24] / t-SNE [31]**：本文验证了Isomap（保留全局测地距离）优于局部线性嵌入和概率邻域嵌入，后者无法保持球面分布规律。
5. **Lu et al. [22]**：早期工作已发现眼动图像间的测地距离与视线角相关，本文将其扩展到深度特征空间并实现可训练框架。
6. **Few-shot Adaptive Gaze [23]**：通过少样本适配实现个性化，与本文的域泛化（零样本跨域）目标不同但可结合。

---

## 局限性与未来方向
1. **Isometric Propagator（IP）是妥协方案**：IP本身为MLP，虽仅用于训练阶段，但仍是不可微分Isomap的近似替代，存在精度损失风险；论文承认"目前深度学习社区尚无完美可微分Isomap方案"。
2. **2D Euler角表示不兼容**：当眼动用yaw/pitch直接表示时，投影特征分布在近似平面上而非球面，当前SA算法需相应调整，论文留为未来工作。
3. **极端视角下规律减弱**：在Gaze360中，当视线角差超过140°时测地距离与角差的线性关系变模糊，原因是头部侧转导致图像质量下降。
4. **未来方向**：开发更精确的可微分测地映射；适配不同眼动表示形式（2D角、3D向量统一处理）；探索在姿态估计等其它物理回归任务中的迁移。

---

## 研究启发与可借鉴点
1. **几何先验替代参数堆叠**：用测地距离+解析对齐（10参数）替代FC层（数千参数）的设计范式，可直接迁移到姿态估计、旋转估计等具有明确物理度量空间的回归任务。
2. **"不可微分算法→可微分代理→反向传播→测试时还原"的训练范式**：IP的设计思路（用MLP近似复杂数学运算以支持梯度传播，推理时换回精确算法）对图神经网络、几何深度学习等有参考价值。
3. **测地距离正比性验证的可复用实验范式**：Fig.6中绘制"特征距离vs任务变量差异"的散点关系，可作为一种通用的特征空间质量诊断工具，用于检验任何具物理意义的回归任务的特征有效性。
4. **域内精度小幅下降是域泛化的合理代价**：Tab.2中within-dataset精度下降被合理归因于泛化优化，而非缺陷，这一评估视角值得在后续domain generalization工作中借鉴。

---

## 关键术语表
- **AGG（Analytical Gaze Generalization）**：解析眼动泛化框架，通过几何投影替代FC层以提升跨域泛化能力。
- **GPM（Geodesic Projection Module）**：测地投影模块，利用Isomap将高维图像特征解析映射到3D球面空间，以极少参数完成眼动预测。
- **PCG（Principle Component of Gaze）**：眼动主成分，指高维图像特征中包含眼动信息的低维子空间，理论最小维度为2（对应3D单位向量的2个自由度）。
- **PGF（Principle Gaze Feature）**：眼动主特征，经Isomap投影后的3D向量 $e_i$，近似分布在球面上，保留了眼动几何结构。
- **SOT（Sphere-Oriented Training）**：球面导向训练，通过GPM逆过程构造监督信号，优化特征提取器CNN使其输出的特征更符合测地距离—视线角正比规律。
- **Isometric Propagator（IP）**：3层MLP，用于在SOT阶段近似Isomap映射以实现可微分反向传播，仅在源域训练时使用。
- **Sphere Alignment（SA）**：球面对齐算法，通过平移（球心）、旋转和线性缩放（6+2+2=10参数）将PGF球面与单位眼动球面对齐。
- **Domain Generalization（域泛化）**：无需目标域数据，仅通过在源域训练使模型具备对未知域的泛化能力，与域适应（需目标域）相区别。

---

## 可复现要素
- **数据集**：ETH-XGaze、Gaze360、MPIIFaceGaze、EyeDiap（均为公开数据集，论文已说明预处理方式）
- **代码/权重**：论文未提及代码或权重是否开源
- **关键超参**：Adam lr=1e-4；预训练10 epochs；SOT 10 epochs；IP 100 epochs（2000随机样本）；batch size=512；Isomap邻居数=300；像素归一化至[0,1]；无数据增强
- **实现环境**：PyTorch + Scikit-learn
