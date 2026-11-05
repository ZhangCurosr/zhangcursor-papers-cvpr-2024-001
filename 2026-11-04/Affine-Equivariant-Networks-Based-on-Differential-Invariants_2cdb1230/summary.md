---
title: "Affine-Equivariant-Networks-Based-on-Differential-Invariants"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Affine_Equivariant_Networks_Based_on_Differential_Invariants_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:56:25"
field: "等变深度学习"
keywords: ["等变网络", "仿射等变性", "微分不变量", "对称PDE", "SupNorm归一化", "affConv"]
innovations: ["首次无需群离散化/采样构建仿射等变网络", "提出SupNorm归一化微分不变量解决除零问题并提升表达能力", "设计可嵌入现有CNN的InvarLayer模块实现多群统一等变"]
benchmarks: ["affNIST", "Scale-MNIST", "Scale-Fashion", "RS-MNIST", "RS-Fashion"]
---

# 论文速读：Affine-Equivariant-Networks-Based-on-Differential-Invariants

## 一句话总结
本文从对称PDE的视角，基于微分不变量首次构建了无需对群进行离散化或采样的仿射等变网络。通过提出SupNorm归一化微分不变量（SNDI）解决分数多项式导致的除零问题，设计了可直接嵌入现有CNN的InvarLayer，在仿射等变性任务上显著优于现有方法。

## 研究问题与动机
- **仿射等变的计算挑战**：仿射群是高维非紧群，现有群卷积方法需对连续群离散化/采样，导致模型参数量和显存随层数指数增长（如affConv [37]）。
- **采样方案的理论缺陷**：Finzi等 [16] 依赖Haar测度采样，不适用于仿射群；MacDonald等 [37] 在李代数上积分仍依赖采样；Mironenco等 [39] 采用log-normal分布近似GL(n,R)-不变测度，理论上无法保证严格等变性。
- **微分不变量方法的局限**：现有基于偏微分算子（PDO）的方法仅针对欧氏群子群（平移、旋转），未探索其在一般变换群上的潜力 [25, 27, 33, 34]。
- **分数不变量的数值稳定性**：仿射群的微分不变量呈分数多项式形式（如Hessian行列式除以梯度相关项），在图像均匀区域分母趋零，导致数值不稳定。

## 核心贡献（创新点）
- **首次无采样构建仿射等变网络**：从对称PDE视角出发，利用微分不变量直接构造等变算子，避免了群的离散化/采样，突破affConv的网络深度限制。
- **SupNorm归一化微分不变量（SNDI）**：将多项式相对微分不变量按SupNorm归一化，替代经典分数型微分不变量，既消除除零问题又保留更多信息（具有光照不变性）。
- **InvarPDEs-Net架构**：将多层可学习对称PDE迭代过程堆叠为深度等变网络，天然具备残差连接，支持可变通道数。
- **InvarLayer灵活模块**：提取单步迭代过程设计为可自由指定输入/输出通道数的等变层，可作为现有CNN架构的即插即用替换模块。
- **统一框架适用于子群**：框架可扩展至仿射群的连续子群（尺度群、旋转-尺度群、等变仿射群等），实现单一范式覆盖多类非欧几里得群等变性。

## 方法详解
- **对称PDE与等变性的理论联系**：根据微分不变量理论 [42]，PDE $\frac{\partial \tilde{\mathbf{u}}}{\partial t} = \mathbf{H} \circ \hat{\mathbf{\mathcal{T}}}_{FDI}[\mathbf{u}^{(t)}]$ 是G-对称的当且仅当其右侧由群G的基本微分不变量构成。
- **网络迭代表达**：对时间导数采用前向差分离散化，得到第i层更新公式 $\mathbf{u}^{(t_{i+1})} = \mathbf{u}^{(t_i)} + \Delta t_i \cdot \mathbf{h}_{\theta_i} \circ \hat{\mathbf{\mathcal{T}}}_{SNDI}[\mathbf{u}^{(t_i)}]$，其中$\mathbf{h}_{\theta_i}$为MLP（实现为1×1卷积+ReLU），$\hat{\mathbf{\mathcal{T}}}_{SNDI}$为SupNorm归一化微分不变量算子。
- **SNDI构造**：选取一组同权重的多项式相对微分不变量$\mathcal{I}_i$，定义$\mathcal{T}_i = \mathcal{I}_i / \|\mathcal{I}\|_{\sup}$，其中$\|\cdot\|_{\sup} = \sup_{\mathbf{x}} \|\cdot\|_\infty$通过全局Max-Pooling实现。归一化后保留相对关系信息，且满足$\mathcal{T}(c\cdot\mathbf{u}) = \mathcal{T}(\mathbf{u})$（光照不变性）。
- **空间导数计算**：采用高斯导数核估计偏导数（如$f_x(\mathbf{x}_0) \approx \sum_n \partial_x G(\mathbf{x}_n; \sigma) f(\mathbf{x}_n + \mathbf{x}_0)$），实现为可微卷积操作。
- **InvarLayer设计**：移除跳连结构，直接输出$\mathbf{u}_{out} = \mathbf{h}_\theta \circ \hat{\mathbf{\mathcal{T}}}_{SNDI}[\mathbf{u}_{in}]$，支持任意输入/输出通道数$C_1, C_2$。
- **兼容性增强**：BatchNorm、逐点非线性、Dropout、1×1卷积均为等变算子的后复合，保持等变性；全局池化输出不变特征用于分类头。

## 实验与结果
- **尺度等变实验**：Scale-MNIST和Scale-Fashion数据集（缩放因子[0.3, 1]，零填充回原尺寸）。InvarPDEs-Net在Scale-MNIST达**98.30±0.06%**（最优），在Scale-Fashion达**89.62±0.26%**（最优），均超越SE-CNN [41]、SS-CNN [19]等基线，参数量<500k。
- **旋转-尺度等变实验**：RS-MNIST和RS-Fashion数据集（旋转[0, 2π]，缩放[0.3, 1]）。InvarPDEs-Net在RS-MNIST达**95.80±0.09%**（最优，超越RST-CNN [18]的93.19%），在RS-Fashion达**79.48±0.31%**（最优）。InvarLayer调参后RS-MNIST达93.40%。
- **仿射等变实验**：在affNIST数据集上（训练50k MNIST，测试320k仿射扰动图像，OOD设置）。InvarLayer（ResNet-32结构）达**98.45±0.15%**，超越affConv [37]的95.08% **+3.37%**，也超越RU CapsNet [8]的97.69%。**0.76%边际提升**。参数量365k < affConv的373k。
- **主要结论**：所提方法在三类非欧几里得群等变性任务上均取得SOTA或次优结果，仿射等变性提升显著；InvarLayer兼顾性能与灵活性。

## 相关工作脉络
- **群卷积方法**：Cohen & Welling [3] 开创性提出Group CNN；Finzi等 [16] 尝试Lie群采样；MacDonald等 [37] 的affConv在李代数上积分但仍有采样，本文首次完全避免群离散化。
- **可旋转CNNs**：SFCNN [64]、RDCF [2] 等仅支持旋转等变，本文InvarLayer可同时处理旋转-尺度-仿射变换，通用性更强。
- **尺度等变网络**：SEVF [38]、SS-CNN [19]、ScDCFNet [79] 等方法依赖核变换或分解，本文通过微分不变量统一构造，理论更简洁。
- **可学习PDE方法**：Liu等 [33, 34] 提出平移-旋转等变PDE系统，本文将其推广至仿射群等非欧几里得群，扩展了对称PDE的应用边界。
- **胶囊网络**：CapsNet [46]、GE CapsNet [31]、affine CapsNet [21] 等通过路由机制模拟等变性但缺乏严格数学保证，本文提供理论完备的替代方案。
- **微分不变量理论**：Olver [42] 奠定移动标架法理论基础；Wang等 [60] 提出相对仿射微分不变量的实用推导，本文在此基础上解决数值稳定性问题并构建神经网络。

## 局限性与未来方向
- **维度限制**：当前框架针对2D平面图像设计，扩展至3D空间或球面流形需重新推导微分不变量 [37]。
- **泛化能力待验证**：仅在分类任务上验证，在检测、分割、生成等下游任务的适用性未探索。
- **超参数敏感性**：InvarLayer在不同设置下性能波动（如RS-Fashion仅74.51%），调参复杂度较高。
- **计算开销**：SupNorm需全局Max-Pooling，相对局部卷积增加全局通信成本。
- **更多李群扩展**：论文指出微分不变量理论适用于满足正则条件的李群，但具体构造方法尚未系统研究。

## 研究启发与可借鉴点
- **无采样等变性新范式**：绕过群离散化的思路可迁移至其他高维/非紧群（如射影群、共形群），为等变网络设计开辟新路线。
- **归一化稳定技术**：SupNorm归一化思想可推广至其他存在除零风险的等变构造场景，提升数值鲁棒性。
- **模块化层设计**：InvarLayer作为CNN插件的思路可与现有骨干网络（ResNet、ViT）结合，推动等变性在大规模视觉模型中的应用。
- **对称PDE统一框架**：将网络深度对应PDE演化步数，为可解释深度架构提供新视角，值得在动力学系统建模中进一步探索。
- **跨群统一构造**：单一框架覆盖尺度、旋转-尺度、仿射群，证明微分不变量理论的统一能力，可启发其他对称性（如射影等变）的系统性研究。

## 关键术语表
- **Equivariance（等变性）**：输入经群变换后，输出按相同规则变换的性质，即Ψ[g·x] = g·Ψ[x]。
- **Differential Invariant（微分不变量）**：由函数及其导数组成的、在群作用保持不变的量。
- **Relative Differential Invariant（相对微分不变量）**：满足I(g·x, g·u) = w(g)I(x, u)的量，w(g)为权重因子。
- **SupNorm Normalization（SupNorm归一化）**：将相对微分不变量除以其全局无穷范数，构造真正的不变量。
- **Symmetric PDE（对称偏微分方程）**：解集在群作用下封闭的PDE，其右侧必由微分不变量构成。
- **InvarPDEs-Net**：由多层对称PDE迭代堆叠而成的等变网络，自带残差连接。
- **InvarLayer**：提取自PDE迭代的等变模块，支持任意输入输出通道数，可嵌入现有CNN。
- **affNIST**：基于MNIST的仿射扰动数据集，用于评估仿射等变网络性能的标准基准。

## 可复现要素
- **数据集**：Scale-MNIST、Scale-Fashion、RS-MNIST、RS-Fashion、affNIST（均基于公开MNIST/Fashion-MNIST生成，未提及独立开源，但生成方式明确）。
- **代码/权重**：论文未提及代码开源情况。
- **关键超参**：高斯核标准差σ、MLP层数与宽度、步长Δt、训练轮数等论文未详细列出，需参考补充材料（Supplementary Material）。
