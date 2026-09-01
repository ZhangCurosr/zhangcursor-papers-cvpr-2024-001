---
title: "Interactive3D-Create-What-You-Want-by-Interactive-3D-Generat"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Dong_Interactive3D_Create_What_You_Want_by_Interactive_3D_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:59:52"
field: "交互式3D生成"
keywords: ["Interactive 3D Generation", "Gaussian Splatting", "InstantNGP", "User Interaction", "Score Distillation Sampling"]
innovations: ["两阶段框架结合高斯泼溅与InstantNGP实现灵活交互与精细重建", "交互式哈希细化模块解决特征冲突并支持局部细节增强"]
benchmarks: ["CLIP R-Precision", "Cap3D"]
---

# 论文速读：Interactive3D: Create What You Want by Interactive 3D Generation

## 一句话总结
本文提出 Interactive3D，一个两阶段交互式 3D 生成框架：第一阶段利用 Gaussian Splatting 实现添加/删除部件、可变形与刚性拖拽、几何变换及局部语义编辑等直接用户交互；第二阶段通过 NeRF 蒸馏将高斯表示转为 InstantNGP，并引入交互式哈希细化模块以进一步提升细节与几何质量，显著提升了 3D 生成的可控性与生成质量。

## 研究问题与动机
- 现有 3D 生成方法主要依赖初始文本提示或 2D 参考图像，可控性有限，难以实现用户预期的精确定制。
- 基于文本提示的方法缺乏对复杂 3D 结构的特异性表达；基于 2D 图像重建的方法受限于单视角深度信息，易引入伪影且缺乏直接 3D 操纵的灵活性。
- 直接将 2D 控制机制（如 ControlNet）迁移至 3D 面临挑战：3D 条件数据收集困难、缺乏类似 Stable Diffusion 的强大基础模型。
- 用户希望在 3D 生成过程中能够灵活介入，直接对中间生成结果进行多种几何与语义修改，以实现对生成方向的实时引导。

## 核心贡献（创新点）
- **两阶段交互式生成框架**：结合 Gaussian Splatting（便于交互）与 InstantNGP（便于精细几何重建与网格提取），发挥两种表示的互补优势。
- **丰富的 3D 交互操作**：在 Stage I 支持添加/删除部件、可变形拖拽、刚性拖拽、几何变换及局部语义编辑，允许用户在任意生成步骤进行引导。
- **交互式 SDS Loss**：通过自适应相机缩放策略聚焦修改区域，并允许调整去噪步长 t，提升局部优化效率并避免全局结构剧烈变化。
- **交互式哈希细化模块**：针对 InstantNGP 的特征容量有限与哈希冲突问题，设计部分特定（part-specific）的层级哈希表与残差 MLP，实现局部细节增强且不污染已满意区域。
- **显著的生成质量与可控性提升**：在 CLIP R-Precision 上达到 0.94，优于 DreamFusion (0.67) 与 ProlificDreamer (0.83)，且总优化时间仅 50 分钟。

## 方法详解
**整体架构**：分为两个级联阶段，Stage I 基于 Gaussian Splatting 进行交互式生成，Stage II 将高斯表示转换为 InstantNGP 并进行细节细化。

**Stage I：基于 Gaussian Splatting 的交互生成**
- **表示**：3D 对象表示为 N 个高斯椭球 $\mathcal{E} = \{(c_i, o_i, \mu_i, \Sigma_i)\}_{i=1}^N$，其中 $c_i, o_i, \mu_i, \Sigma_i$ 分别表示颜色、不透明度、位置和协方差。将高斯中心集合视为点云 $\mathcal{S} = \{\mu_i\}$ 以便操作。
- **添加/删除部件**：通过合并或移除高斯椭球集合实现部件的增删。部件选择支持两种方式：(1) 基于 2D 渲染图，利用分割模型（如 SAM）获取多视图掩码，投影后确定属于该部件的高斯点；(2) 在 3D 空间中直接选择点。
- **几何变换**：对选定部件 $P$ 构建包围盒 $B$，施加旋转、平移、拉伸等变换 $\tau$，再与原集合拼接：$\mathcal{E}_{\text{trans}} = \tau(B(P)) + \mathcal{E}'$。
- **可变形与刚性拖拽**：受 DragGAN 启发，用户指定源点 $p_s$、目标点 $p_t$ 和局部半径 $r$，激活区域 $P = \{e_i | \|\mu_i - p_s\|_2 \leq r\}$。每步优化前沿 $p_s \to p_t$ 方向施加偏移：$\mu_i' = \mu_i + \alpha \frac{(p_t - p_s)}{\|p_t - p_s\|_2}$。可变形拖拽通过持续更新源点位置实现；刚性拖拽则引入刚性约束损失 $L_{\text{rigid}} = \sum_{i \in I(P)} |\|p_s - p_i\|_2 - \|p_s^* - p_i^*\|_2|$ 保持局部结构不变。拖拽后执行高斯密集化与剪枝以填充空洞。
- **局部语义编辑**：选定部件后输入新文本提示（如“翅膀着火”），通过交互式 SDS Loss 优化该部分以匹配新语义特征。
- **交互式 SDS Loss**：将相机置于修改区域附近进行局部渲染，计算 SDS Loss；允许用户调整去噪步长 t（早期生成用大 t，局部微调用小 t）。

**Stage II：基于 InstantNGP 的细化**
- **NeRF 蒸馏**：将 Stage I 的高斯表示 $\mathcal{E}$ 转换为 InstantNGP 参数 $\mathcal{F}$，通过最小化渲染差异进行蒸馏：$L_{\text{distill}} = \frac{1}{M} \sum_{c \sim C(\theta,\phi)} \|\mathcal{V}(\mathcal{F}, c) - \mathcal{R}(\mathcal{E}, c)\|_1$，其中 $c$ 为相机位姿。
- **交互式哈希细化模块**：
  - 从转换后的 InstantNGP $\mathcal{T} = \{\mathcal{F}, \mathcal{H}\}$ 中提取二值占据网格 $O$（分辨率 $32\times32\times32$）。
  - 给定感兴趣区域 $Q$（由中心 $o$ 和半径 $r$ 定义），求交得到部分占据区域 $O^{\text{part}} = O \cap Q$。
  - 将 $O^{\text{part}}$ 划分为多分辨率网格，建立部分特定的多层哈希表 $\mathcal{H}^{\text{part}}$ 与可学习特征 $\mathcal{F}^{\text{part}}$，映射函数为：
    $$f_k = \begin{cases} H_k^{\text{part}}(p, \mathcal{F}_k^{\text{part}}), & p \in O^{\text{part}} \\ 0, & p \notin O^{\text{part}} \end{cases}$$
  - 新增轻量 MLP 将残差特征映射为颜色与密度残差，叠加到原始渲染结果。此设计避免哈希冲突导致的特征共享退化，并可针对不同复杂度区域自适应分配细节资源。
  - 使用交互式 SDS Loss 对局部区域进行优化。

## 实验与结果
- **数据集与评估**：使用来自 Cap3D 的 50 个提示词进行生成测试；采用 CLIP R-Precision 作为定量评估指标。
- **基线方法**：DreamFusion、ProlificDreamer。
- **主要结果**：
  - Interactive3D 在 CLIP R-Precision 上达到 **0.94**，显著优于 DreamFusion (0.67) 与 ProlificDreamer (0.83)。
  - 平均生成时间仅 **50 分钟**（Stage I 10k 步，Stage II 10k 步），远低于 DreamFusion (1.1h) 与 ProlificDreamer (3.4h)。
- **定性结果**：可视化展示了对动漫角色、龙骑士组合、霸王龙等不同主题的交互编辑（如姿态改变、部件拖拽、细节细化），证明该方法在可控性与生成质量上的优势。
- **消融实验**：
  - **刚性约束损失**：去除后拖拽会导致局部结构变形，加入后能保持刚性运动。
  - **交互式 SDS Loss**：相比无交互版本，在相同步数下获得更好的局部修改效果。
  - **交互式哈希细化模块**：仅使用原始哈希表时细节模糊；添加部分特定哈希映射后，纹理与几何精度大幅提升，且避免了区域冲突。

## 相关工作脉络
- **DreamFusion 系列 (Poole et al., 2023; Wang et al., 2023; Chen et al., 2023)**：基于 SDS Loss 优化 3D 辐射场，但仅依赖初始文本提示，缺乏交互控制能力。
- **ProlificDreamer (Wang et al., 2024)**：改进 SDS Loss 以提升生成质量，但仍限于文本驱动，无法进行局部几何编辑。
- **Shap-E (Jun & Nichol, 2023) / PointE (Nichol et al., 2022)**：前馈式 3D 生成，速度较快但可控性弱，不支持用户介入。
- **DragGAN (Pan et al., 2023)**：在 2D 图像流形上进行交互式点级拖拽，本文将其思想拓展至 3D 高斯表示。
- **3D Gaussian Splatting (Kerbl et al., 2023)**：显式高效 3D 表示，适合交互操作，但难以直接输出高质量网格；本文结合其交互友好性与 InstantNGP 的几何提取优势。
- **InstantNGP (Müller et al., 2022)**：基于多层哈希编码的快速 NeRF 实现，支持细节重建与网格提取，但隐式表示难以直接交互；本文通过部分特定哈希映射解决其冲突问题。

## 局限性与未来方向
- **复杂场景扩展性**：当前交互操作主要针对单一对象或组合部件，对于高度复杂的多人物或多物体场景，交互效率与效果有待验证。
- **依赖 2D 分割模型的局限性**：部件选择若基于 2D 渲染图与 SAM 分割，可能因渲染质量不足或分割误差影响选择准确性；未来可探索更鲁棒的 3D 原生分割方法。
- **实时交互能力**：虽然单次优化总耗时约 50 分钟，但交互过程仍需等待当前优化步骤完成，尚未实现真正的实时反馈。
- **网格提取质量**：Stage II 虽转换为 InstantNGP，但最终网格提取仍依赖 Marching Cubes 等标准方法，可能保留部分伪影；可进一步融合其他几何重建技术。
- **交互操作的泛化性**：当前交互模式（拖拽、删除等）主要针对几何编辑，未来可扩展至材质、光照、动态行为等更丰富的语义控制。

## 研究启发与可借鉴点
- **表示互补的两阶段设计**：将适合交互的显式表示（Gaussian Splatting）与适合精细重建的隐式表示（InstantNGP）串联，兼顾可控性与生成质量，为其他 3D 生成任务提供架构参考。
- **局部交互式优化策略**：通过相机缩放聚焦修改区域、调整去噪步长，实现高效且稳定的局部优化，避免全局重优化带来的时间与质量损耗，可迁移至其他扩散引导的 3D 生成方法。
- **部分特定哈希映射解决冲突**：针对隐式表示中特征共享导致的退化问题，引入区域独立的残差特征与哈希表，既能细化局部细节又不污染全局，该思路可应用于其他基于哈希编码的神经表示。
- **交互操作与生成流程无缝集成**：在优化中途插入用户交互，并通过附加损失（如运动监督、刚性约束）引导优化方向，无需重新训练，为交互式生成系统提供了实用范式。
- **评估指标与效率分析**：结合 CLIP R-Precision 与生成时间进行综合评估，突出方法在可控性提升的同时未显著增加计算负担，值得在类似工作中借鉴。

## 关键术语表
- **Gaussian Splatting**：一种显式 3D 表示方法，用一组可微分的高斯椭球拟合场景，支持高效渲染与直接编辑。
- **InstantNGP**：基于多分辨率哈希编码的神经辐射场加速方法，能在保证渲染质量的同时显著提升训练与推理速度。
- **SDS Loss (Score Distillation Sampling)**：利用预训练 2D 扩散模型的梯度信号来优化 3D 表示，使随机角度渲染的 2D 图像符合文本提示。
- **Interactive Hash Refinement**：针对 InstantNGP 设计的局部细化模块，通过部分特定的哈希表与残差 MLP 增强感兴趣区域的细节。
- **Deformable/Rigid Dragging**：两种 3D 拖拽交互模式，前者允许局部结构平滑形变，后者保持局部形状不变进行整体移动。
- **Semantic Editing**：基于文本提示的局部语义编辑操作，用户可针对选定部件输入新描述以修改其外观特征。
- **CLIP R-Precision**：评估生成图像与文本提示对齐程度的指标，计算在检索结果中相关样本的比例。
- **NeRF Distillation**：将高斯表示的渲染结果作为教师信号，蒸馏训练 InstantNGP 参数，实现表示转换与初步优化。

## 可复现要素
- **数据集**：使用 Cap3D 的 50 个提示词进行评估，Cap3D 数据集公开可用。
- **代码与权重**：论文未明确提及代码开源状态；项目页面为 https://interactive-3d.github.io/，权重未提及。
- **关键超参**：总优化步数 20k（Stage I 10k，Stage II 10k）；占据网格分辨率 $32\times32\times32$；拖拽步长 $\alpha$ 与刚性损失权重为可调超参，具体数值见补充材料。
- **硬件**：实验在 NVIDIA A100 GPU 上进行。
