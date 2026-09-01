---
title: "SfmCAD: Unsupervised CAD Reconstruction by Learning Sketch-based Feature Modeling Operations"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Li_SfmCAD_Unsupervised_CAD_Reconstruction_by_Learning_Sketch-based_Feature_Modeling_Operations_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:00:51"
field: "3D计算机视觉与图形学"
keywords: ["CAD重建", "草图特征建模", "无监督学习", "Bézier曲线", "隐式神经表示", "可微分SDF", "参数化3D重建"]
innovations: ["首个支持拉伸/扫描/放样/旋转四种操作的无监督通用CAD重建网络", "结构-细节解耦的草图+路径表示，用Bézier曲线编码路径、隐式网络编码2D草图", "两阶段粗到细学习策略配合实时采样加速隐式草图训练"]
benchmarks: ["ABC dataset", "ShapeNet (chair/table/display)", "Tree dataset", "PartNet semantic segmentation"]
---

# 论文速读：SfmCAD: Unsupervised CAD Reconstruction by Learning Sketch-based Feature Modeling Operations

## 一句话总结
SfmCAD 是一种无监督神经网络，将3D体素形状解析为由**草图+路径**参数化表示的现代CAD特征建模操作（拉伸、扫描、放样、旋转），通过两阶段粗到细的学习策略实现高精度、可编辑的CAD重建。

## 研究问题与动机
1. **现有方法精度与可解释性的矛盾**：几何基元/B-rep方法重建细节精细，但模型复杂难编辑；CSG方法紧凑但基本基元（立方体、球体）不足以表达复杂细节，精度受限。
2. **单一操作类型的局限**：现有学习草图特征建模的方法（如SECAD-Net、ExtrudeNet）仅支持**拉伸（extrude）**一种操作，无法表达曲面路径、多截面放样等更丰富的CAD命令。
3. **同时学习草图与路径的挑战**：草图轮廓和3D路径联合学习难度高、耗时长，需要有效的分解训练策略。
4. **缺乏统一的无监督框架**：现有方法大多针对特定操作或有限形状类别设计，缺少能同时处理拉伸、扫描、放样、旋转四种操作的通用无监督CAD重建方法。

## 核心贡献（创新点）
1. **首个支持四种CAD操作的无监督通用网络**：SfmCAD是首个无监督、通用的神经网络，可同时学习拉伸（extrude）、扫描（sweep）、放样（loft）、旋转（revolve）四种现代CAD标准命令，区别于仅支持拉伸的SECAD-Net/ExtrudeNet。
2. **结构-细节解耦的草图+路径表示**：用3D Bezier曲线捕获整体结构路径，用隐式2D草图表征局部几何细节，两者分离使得模型既精确又易于用户编辑，区别于CSG方法的结构-细节混在一起的基元表示。
3. **两阶段粗到细学习策略**：先学习Box+Path粗表示快速捕捉路径和大致形状，再通过隐式草图网络细化细节，显著加速训练收敛，区别于端到端同时学习所有参数的方法。
4. **可微分的多种CAD操作算子**：给出了拉伸、放样、扫描的可微分SDF实现，支持梯度回传到隐式网络，使无监督端到端训练成为可能。

## 方法详解
1. **Typed Sketch+Path 表示**：定义封闭曲线轮廓 $\mathcal{Z}$ 和3D路径 $\mathcal{C}$（用三次Bézier曲线 $\Upsilon(t)$ 参数化，控制点 $\mathbf{P}_i$）。四种操作统一为"沿路径提升2D草图"：拉伸→线性路径的扫描特例；旋转→近似圆周的Bézier曲线；放样→两个草图间的线性插值；扫描→任意路径提升。
2. **Box+Path 粗表示**：将弯曲路径离散为 $N_s$ 个刚性box代理，中心均匀采样Bézier曲线，高度由相邻采样点距离决定，方向由路径切向确定，引入可学习的宽 $w$、长 $l$ 及首尾扭转角 $\delta^u, \delta^b$ 参数化box朝向。
3. **可微分SDF算子**：
   - **Sketch-Extrude**：利用中间函数 $\mathcal{M}_1, \mathcal{M}_2, \mathcal{M}_3$ 组合草图SDF与拉伸高度约束，得到拉伸SDF（公式5）。
   - **Sketch-Loft**：对上下两个草图SDF沿高度线性插值（公式6），代入与拉伸相同的形式得放样SDF（公式7）。
   - **Sketch-Sweep**：将扫描分解为 $N_s$ 段拉伸的SDF取最小值（公式8）。
4. **两阶段训练**：
   - **Stage 1（Box+Path学习）**：3D卷积编码器→MLP输出Bézier控制点和box参数。损失 $\mathcal{L}_{box} = \mathcal{L}_B + \lambda_1\mathcal{L}_{sm} + \lambda_2\mathcal{L}_{lw}$，其中 $\mathcal{L}_B$ 为occupancy MSE，$\mathcal{L}_{sm}$ 惩罚曲线尖锐弯曲，$\mathcal{L}_{lw}$ 惩罚box尺寸过大导致自交。
   - **Stage 2（隐式草图学习）**：将特征 $z$ 映射为 $N_p$ 个局部特征，与点坐标拼接输入隐式草图网络预测SDF值。采用**实时采样策略**：在基础草图平面采样网格点，沿路径复制occupancy值，大幅加速训练（避免对全点集重复计算 $N_p \times N_s$ 次）。
5. **参数化转换**：网络输出的Bézier控制点和隐式草图经可微分操作生成最终参数化CAD模型。

## 实验与结果
**数据集与基线**：
- ABC数据集（5000训练/1000测试）：对比 UCSG-Net、CSG-Stump、ExtrudeNet、SECAD-Net
- ShapeNet（chair/table/display）：同上基线
- Tree数据集（程序化树）：对比 ExtrudeNet、SECAD-Net
- PartNet语义分割重建：对比 ExtrudeNet、SECAD-Net

**主要结果**（越低越好↓，越高越好↑）：

| 数据集 | 方法 | CD (×10⁻³) | ECD (×10⁻²) | NC |
|---|---|---|---|---|
| ABC | SECAD-Net | 0.506 | 7.286 | 0.884 |
| ABC | **Ours** | **0.395** | **5.038** | **0.919** |
| ShapeNet | SECAD-Net | 0.836 | 14.818 | 0.837 |
| ShapeNet | **Ours** | **0.626** | **14.096** | **0.867** |
| Tree | SECAD-Net | 2.197 | — | 0.622 |
| Tree | **Ours** | **0.686** | — | **0.722** |
| PartNet | SECAD-Net | 1.888 | 0.472 | 0.818 |
| PartNet | **Ours** | **1.410** | **0.446** | **0.814** |

- **最强结果**：Tree数据集CD从2.197降至0.686（提升约68.8%）；PartNet相对SECAD-Net CD提升25.3%、ECD提升5.5%。
- SfmCAD在所有数据集上CD、ECD、NC三项指标均全面超越对比方法，验证了草图+路径表示对复杂形状的更强表达能力。

## 相关工作脉络
1. **CSG无监督重建**：UCSG-Net [14]、CSG-Stump [30] — 用基本体素+布尔运算表示形状，模型紧凑但基元类型有限、细节精度不足；SfmCAD用特征建模操作替代，精度更高且支持编辑。
2. **草图+拉伸方法**：ExtrudeNet [31]、SECAD-Net [23] — 仅支持沿直线的拉伸操作；SfmCAD扩展到扫描、放样、旋转，表达能力大幅扩展。
3. **几何基元提取**：CPFN [19]、ParSeNet [37]、Superquadrics [29] — 从点云拟合几何基元，输出离散基元集合而非参数化CAD程序，缺乏可编辑性。
4. **CAD生成模型**：DeepCAD [42]、Fusion360 [41]、SkexGen [44] — 面向参数化CAD**生成**，非**重建/反求**任务，使用不同的损失函数和训练目标。
5. **B-Rep学习**：BRepNet [17]、ComplexGen [9]、CADops-Net [6] — 直接学习边界表示，缺乏特征级别的可解释性和用户友好编辑能力。
6. **点云到圆柱**：Point2Cyl [38] — 仅恢复圆柱形拉伸特征，SfmCAD推广到任意路径和多种操作类型。

## 局限性与未来方向
1. **仅支持两种草图数量的放样**：当前Loft操作限制为两个草图间的插值，无法处理多截面复杂放样。
2. **Sweep与Loft由超参决定**：网络不能自动选择操作类型，需人工预设（尽管附录提到可以同时学习两种操作）。
3. **复杂拓扑结构的限制**：对于具有自交、空洞较多或极复杂分支的结构，Box+Path近似可能不够精确。
4. **作者展望**：结合2D草图和3D零件模板提升重建效率；利用输出的可操纵性扩展至生成式CAD设计。

## 研究启发与可借鉴点
1. **两阶段粗到细策略**：先学习全局结构（Box+Path）再细化局部细节（隐式草图），可有效缓解联合优化难度，该思路可迁移到其他3D形状解析任务。
2. **实时采样加速技巧**：利用扫描操作中草图沿路径的一致性（occupancy值沿路径复制），将逐点SDF计算从 $N_p \times N_s$ 降为仅在一个基准平面采样，值得在其他隐式场方法中借鉴。
3. **多种CAD操作统一为可微分SDF算子**：将拉伸、扫描、放样统一用可微分SDF表达，使不同操作可被同一网络学习，这种统一框架设计对其他领域（如点云到参数化表示）有参考价值。
4. **结构-细节解耦表示**：用Bézier曲线编码全局结构、隐式网络编码局部细节，这种解耦方式有助于提升模型的可解释性和编辑灵活性，可与本团队在可编辑3D生成方向结合。

## 关键术语表
**Sketch-based Feature Modeling**：现代CAD中最常用的建模方法，通过绘制2D草图并执行拉伸、扫描等操作生成3D实体。
**Typed Sketch+Path Representation**：将CAD特征建模操作统一表示为2D草图轮廓沿3D路径生成的参数化形式，"typed"指操作类型。
**Bezier Curve**：参数化曲线 $\Upsilon(t) = \sum \binom{n}{i}(1-t)^{n-i}t^i \mathbf{P}_i$，用于表示3D扫描/拉伸路径。
**Signed Distance Function (SDF)**：空间中每点到物体表面的带符号距离，负值在内部、正值在外部，用于隐式表征3D形状。
**Loft Operation**：在多个截面草图之间进行插值，生成平滑过渡的3D实体。
**Sweep Operation**：将2D草图沿指定3D路径移动，生成广义柱体形状。
**Box+Path Representation**：用沿Bézier曲线分布的一组矩形box代理近似弯曲路径附近的几何形状。
**Realtime Sampling**：在基础草图平面采样后，沿路径复制occupancy值以加速隐式草图网络训练的采样策略。

## 可复现要素
- **数据集**：ABC [16]、ShapeNet [1]、PartNet [48]、程序化树数据集；均使用公开数据
- **代码**：已开源，GitHub: https://github.com/BunnySoCrazy/SfmCAD
- **关键超参**：$\mu=20$（softmin温度）、$\beta=50$（SDF到occupancy转换）、$\Theta=0.1$（box尺寸阈值）、$\lambda_1=0.05$、$\lambda_2=0.05$（正则化权重）
- **训练设置**：Adam优化器，lr=1e-4，beta=(0.5, 0.99)，每阶段500 epoch，batch size=24；测试时每形状fine-tune 200 iter/阶段
- **硬件**：NVIDIA TITAN RTX GPU，PyTorch实现
