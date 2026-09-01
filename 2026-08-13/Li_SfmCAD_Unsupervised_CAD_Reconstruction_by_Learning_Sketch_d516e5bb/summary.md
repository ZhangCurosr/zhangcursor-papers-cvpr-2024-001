---
title: "SfmCAD: Unsupervised CAD Reconstruction by Learning Sketch-based Feature Modeling Operations"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Li_SfmCAD_Unsupervised_CAD_Reconstruction_by_Learning_Sketch-based_Feature_Modeling_Operations_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:00:47"
field: "3D几何重建与CAD逆向"
keywords: ["CAD reconstruction", "sketch-based modeling", "unsupervised learning", "implicit surface", "Sweep/Loft", "Bezier curve", "feature primitive"]
innovations: ["首个无监督支持Extrude/Sweep/Loft/Revolve四种CAD操作的通用草图特征建模重建网络", "结构-细节解耦的Box+Path两阶段粗到细学习策略", "可微分Sweep/Loft算子结合实时采样加速训练"]
benchmarks: ["ABC", "ShapeNet", "Tree", "PartNet"]
---

# 论文速读：SfmCAD: Unsupervised CAD Reconstruction by Learning Sketch-based Feature Modeling Operations

## 一句话总结
本文提出 SfmCAD，一种无监督神经网络，将体素化 3D 形状重建为工业标准的草图特征建模（sketch-based feature modeling）CAD 表示。该方法解耦了 3D 结构与局部几何细节，通过两阶段粗到细学习策略，同时支持 Extrude、Revolve、Sweep、Loft 四种常见 CAD 操作，显著提升了重建精度与模型可编辑性。

---

## 研究问题与动机
1. **现有 CAD 重建方法缺乏精度与可解释性的平衡**：几何原语/隐式场方法精度高但难以编辑；CSG 方法紧凑但操作复杂且不直观。
2. **已有特征建模方法操作类型单一**：如 ExtrudeNet、SECAD-Net 仅支持 Extrude 操作，无法表达复杂曲线结构。
3. **同时学习草图轮廓和扫描路径极为困难**：二维草图细节与三维结构路径耦合训练导致收敛慢、易陷入局部最优。
4. **CAD 模型的可编辑性需求未被充分满足**：工业设计中设计师需要直接调整参数化特征（草图尺寸、路径形状），而非仅获得网格或隐式场。

---

## 核心贡献（创新点）
1. **首个无监督通用草图特征建模 CAD 重建网络**：同时学习 Extrude、Revolve、Sweep、Loft 四种标准 CAD 操作，区别于仅支持单一 Extrude 操作的 SECAD-Net 和 ExtrudeNet。
2. **结构-细节解耦的神经 typed sketch+path 表示**：用三维 Bezier 曲线捕捉整体结构路径，用隐式 2D 草图刻画局部几何细节，两者分离使模型既精确又易于参数化编辑。
3. **两阶段粗到细学习策略**：第一阶段学习 Box+Path 粗略结构（快速收敛），第二阶段学习隐式草图细节（精细化），显著加速训练并提升重建质量。
4. **可微分的 Sweep/Loft 算子**：将隐式草图沿三维路径进行 Sweep 或 Loft 操作的过程完全可微分，支持端到端无监督训练。
5. **跨领域泛化能力验证**：在 CAD 零件（ABC）、通用物体（ShapeNet）、树状结构（Tree）、语义分割零件（PartNet）四类数据上均取得 SOTA 结果。

---

## 方法详解

### 整体框架
SfmCAD 以 $64^3$ 体素网格为输入，输出由 $N_p$ 个特征原语（feature primitives）组成的 CAD 程序，每个原语由一个 typed sketch+path 三元组定义：2D 草图 $Z$、3D Bezier 曲线路径 $\mathcal{C}$、操作类型（Sweep/Loft）。

### 第一阶段：Box+Path 学习（粗结构）
1. **编码器**：3D CNN 将体素编码为全局特征向量 $\mathbf{z}$。
2. **MLP 解码**：将 $\mathbf{z}$ 映射为每个 Bezier 曲线的 4 个控制点 $\{\mathbf{P}_1, \dots, \mathbf{P}_4\}$（三次 Bezier 曲线），以及每个 Box 代理的参数 $\{l_i, w_i, \delta^u, \delta^b\}$（长、宽、起点/终点扭转角）。
3. **Box 代理 SDF 计算**：沿 Bezier 曲线均匀采样 $N_s$ 个中心点 $\mathbf{c}_i$，计算每段高度 $h_i = |\mathbf{c}_{i+1}-\mathbf{c}_i|$、法向 $\mathbf{n}_i$、切向 $\mathbf{t}_i$ 和副法向 $\mathbf{b}_i$，构成一系列刚性 Box 代理的并集。
4. **损失函数**：
   - 重建损失 $\mathcal{L}_B$：预测占用率 $\hat{\mathcal{O}}_B$ 与真实占用率 $\tilde{\mathcal{O}}$ 的 MSE（经可微分 soft-min union 和 tanh 转换）。
   - 平滑正则 $\mathcal{L}_{\mathrm{sm}}$：惩罚相邻法向量夹角过大（防止锐角曲线）。
   - 长宽正则 $\mathcal{L}_{\mathrm{lw}}$：惩罚 Box 尺寸超过阈值 $\Theta$（防止自交）。
   - 总损失：$\mathcal{L}_{\mathrm{box}} = \mathcal{L}_B + \lambda_1 \mathcal{L}_{\mathrm{sm}} + \lambda_2 \mathcal{L}_{\mathrm{lw}}$。

### 第二阶段：隐式草图学习（细细节）
1. **局部特征提取**：MLP 将 $\mathbf{z}$ 映射为 $N_p$ 个局部特征 $\mathbf{z}'_i$。
2. **隐式草图 SDF**：将 $\mathbf{z}'_i$ 与测试点局部坐标拼接后输入 MLP，预测 2D 草图 SDF $\hat{S}^{u}_{{\rm sk},i}$ 和 $\hat{S}^{l}_{{\rm sk},i}$（Sweep 只需一个草图，Loft 需要上下两个草图）。
3. **可微分 Sweep/Loft 算子**：
   - **Sweep**：将路径分解为 $N_s$ 段 Extrude，各段 SDF 取 min。
   - **Loft**：在上下草图间按高度比例线性插值 $\hat{S}^{i_\alpha}_{\rm sk} = (1-\alpha)\hat{S}^l_{\rm sk} + \alpha\hat{S}^u_{\rm sk}$，再应用 Extrude 公式。
4. **实时采样加速策略**：仅在 Base 平面采样网格点 $y_i$，计算其占用率后沿路径复制 $N_s-1$ 次，避免对全部 $N_p \times N_s$ 个点重复计算，训练速度大幅提升。
5. **损失函数**：$\mathcal{L}_{\rm rec} = \mathbb{E}_{x \in \mathcal{X}}[\|\hat{\mathcal{O}}_F - \tilde{\mathcal{O}}\|_2^2]$。

### 关键公式汇总
- **Bezier 曲线**：$\mathcal{C} = \Upsilon(t) = \sum_{i=0}^{n}\binom{n}{i}(1-t)^{n-i}t^i\mathbf{P}_i$
- **Extrude SDF**：$\hat{S}^i_{\rm extrude} = \min(\mathcal{M}_1(\hat{S}^i_{\rm sk}), 0) + \sqrt{\mathcal{M}_2(\hat{S}^i_{\rm sk})^2 + \mathcal{M}_3^2}$
- **Loft SDF**：同 Extrude 形式，草图替换为插值草图
- **Sweep SDF**：$\hat{S}^i_{\rm sweep} = \min_{j}\hat{S}^j_{\rm extrude}$

---

## 实验与结果

### 数据集与基线
- **ABC 数据集**（CAD 零件）：5000 训练 / 1000 测试，选 100 个评估。基线：UCSG-Net、CSG-Stump、ExtrudeNet、SECAD-Net。
- **ShapeNet**（chair/table/display）：各 40 个，共 120 个评估。基线同上。
- **Tree 数据集**（树形分支）：4500 训练 / 500 测试，选 50 个评估。基线：ExtrudeNet、SECAD-Net。
- **PartNet**（语义分割重建）：chair/table/trashcan 三部分，共 90 个完整形状评估。基线：ExtrudeNet、SECAD-Net。

### 主要结果
| 数据集 | 方法 | CD ($\times 10^{-3}$) | ECD ($\times 10^{-2}$) | NC |
|--------|------|---------------------|----------------------|----|
| ABC | **SfmCAD** | **0.395** | **5.038** | **0.919** |
| ABC | SECAD-Net | 0.506 | 7.286 | 0.884 |
| ShapeNet | **SfmCAD** | **0.626** | **14.096** | **0.867** |
| ShapeNet | SECAD-Net | 0.836 | 14.818 | 0.837 |
| Tree | **SfmCAD** | **0.686** | — | **0.722** |
| Tree | SECAD-Net | 2.197 | — | 0.622 |
| PartNet | **SfmCAD** | **1.410** | **0.446** | **0.814** |
| PartNet | SECAD-Net | 1.888 | 0.472 | 0.818 |

**最强结果**：ABC 数据集 CD 达 0.395，相对 SECAD-Net 提升 **22%**；PartNet 相对 SECAD-Net 提升 **25.3%**（CD）。

### 消融实验（PartNet Chair）
| 变体 | CD | ECD | NC |
|------|-----|-----|----|
| 去掉 $\delta^u, \delta^l$ | 4.092 | 0.173 | 0.655 |
| 去掉 $\delta^l$ | 3.367 | 0.139 | 0.714 |
| 去掉 $\mathcal{L}_{\rm lw}$ | 4.804 | 0.164 | 0.685 |
| 去掉 $\mathcal{L}_{\rm sm}$ | 3.268 | 0.154 | 0.732 |
| **完整模型** | **3.142** | **0.133** | **0.776** |

### 实现细节
- 优化器：Adam，lr=1e-4，β=(0.5, 0.99)
- 超参：$\mu=20, \beta=50, \Theta=0.1, \lambda_1=\lambda_2=0.05$
- 训练 500 epoch/阶段，batch=24；测试形状 fine-tune 200 iter/阶段
- 实现：PyTorch，NVIDIA TITAN RTX

---

## 相关工作脉络
1. **CSG 重建方法**（UCSG-Net, CSG-Stump, CAPRI-Net）：用布尔树组合简单几何原语，模型紧凑但精度受限，且缺少对工业 CAD 操作（Sweep/Loft）的支持。
2. **几何原语拟合方法**（Point2Cyl, CPFN, Superquadrics）：直接拟合圆柱/超二次曲面等，只能表达有限类型的基本体素，无法处理自由曲线结构。
3. **Sketch+Extrude 方法**（ExtrudeNet, SECAD-Net）：仅支持 Extrude 操作，对弯曲路径无能为力；本文扩展至 Sweep/Loft/Revolve，表达能力更强。
4. **CAD 生成方法**（DeepCAD, Fusion360, ZoneGraph, SkexGen）：面向从文本/图像生成 CAD 程序，而非从无标注形状重建，目标不同。
5. **B-Rep 学习方法**（BRepNet, ComplexGen, UV-Net）：直接学习边界表示，精度高但输出缺乏参数化特征，难以直观编辑。
6. **特征建模先验方法**（CADOPS-Net, Point2Cyl）：关注生成或仅支持 Extrude；本文是无监督条件下首个支持四种主流 CAD 操作的通用重建网络。

---

## 局限性与未来方向
1. **当前每次重建只能选择 Sweep 或 Loft 一种操作**，虽可同时训练但需手动配置；统一支持混合操作类型是潜在改进方向。
2. **Bezier 曲线阶数固定为三次**，复杂路径可能需要更高阶曲线或分段 Bezier 来提升表达力。
3. **草图仅用 MLP 隐式表示**，未引入显式参数化曲线（如 NURBS/样条），限制了与 CAD 系统的直接兼容性。
4. **未探索生成式应用场景**，作者指出未来可将其拓展为生成型 CAD 设计工具。
5. **树状结构的处理需替换 Box 为 Cylinder**，说明当前 Box+Path 表示对特定形状需手动适配，泛化性有待加强。
6. **未来计划探索 2D 草图 + 3D 零件模板的联合学习**，进一步提升无监督重建效率。

---

## 研究启发与可借鉴点
1. **结构-细节解耦的两阶段学习范式**：先用粗代理快速捕获全局结构（Box+Path），再用隐式网络细化局部细节——此策略可迁移到其他 3D 重建/解析任务中加速收敛。
2. **可微分 Sweep/Loft 算子的设计思路**：将传统 CAD 操作转化为可微分的 SDF 运算，是无监督几何重建中"传统规则嵌入神经网络"的典型范式，值得在其他 CAD 相关任务中复现。
3. **实时采样加速策略**（沿路径复制占用率而非全量重算）：对具有平移/扫掠对称性的任务有普遍适用价值，可作为高效训练技巧复用。
4. **正则化设计**（曲率平滑 $\mathcal{L}_{\rm sm}$ + 尺寸约束 $\mathcal{L}_{\rm lw}$）：有效防止自交和病态形状，适用于所有基于曲线的 3D 生成/重建任务。
5. **跨领域验证策略**：同时验证于 CAD 零件、通用物体、自然结构（树）和分割部件，为方法普适性提供了强有力的证据链，可作为论文实验设计的参考模板。

---

## 关键术语表

**Sketch-based Feature Modeling**：现代 CAD 系统的核心建模范式，通过将 2D 工程草图沿 3D 路径进行拉伸、旋转、扫描或放样等操作来生成 3D 特征实体。

**Typed Sketch+Path Representation**：本文提出的统一表示，将 Extrude/Revolve/Sweep/Loft 四种操作抽象为"带类型的 2D 草图 + 3D 路径"，其中"类型"区分具体操作。

**Box+Path Representation**：第一阶段使用的粗粒度代理，用一系列刚性 Box 沿 Bezier 曲线排列来近似特征原语的 3D 结构。

**Differentiable Sweep/Loft Operator**：将隐式 2D 草图沿 3D 路径进行 Sweep 或 Loft 操作的可微分 SDF 计算过程，使端到端无监督训练成为可能。

**Realtime Sampling Strategy**：仅在 Base 平面采样草图，沿路径复制占用率而非逐点重算，大幅降低第二阶段训练的计算开销。

**Feature Primitive**：由单个草图+路径操作生成的 3D 实体单元，对应现代 CAD 中的一个独立特征（如孔、槽、凸台）。

**SDF (Signed Distance Function)**：符号距离函数，描述空间中任意点到物体表面的带符号距离，用于隐式表示 3D 形状。

**ABC Dataset**：大规模 CAD 模型数据集（A Big Cad Model Dataset），包含 500 万+ 参数化 CAD 模型，广泛用于 CAD 深度学习研究。

---

## 可复现要素
- **代码**：已开源，GitHub: https://github.com/BunnySoCrazy/SfmCAD
- **数据集**：ABC（公开）、ShapeNet（公开）、Tree（作者提供生成代码）、PartNet（公开）
- **权重**：论文未提及预训练权重是否公开
- **关键超参**：$\mu=20, \beta=50, \Theta=0.1, \lambda_1=\lambda_2=0.05$；lr=1e-4，β=(0.5,0.99)；500 epoch/阶段，batch=24
- **环境**：PyTorch，NVIDIA TITAN RTX

---
