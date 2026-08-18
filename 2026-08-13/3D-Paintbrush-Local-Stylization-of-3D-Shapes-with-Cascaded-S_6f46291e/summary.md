---
title: "3D-Paintbrush-Local-Stylization-of-3D-Shapes-with-Cascaded-S"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Decatur_3D_Paintbrush_Local_Stylization_of_3D_Shapes_with_Cascaded_Score_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:22:00"
---

# 论文速读：3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation

## 一句话总结
提出了一种直接在网格上运行的文本驱动局部纹理生成方法，通过联合学习显式定位掩码与高分辨率纹理贴图，并设计级联评分蒸馏（CSD）技术充分利用多级扩散模型的监督信号，实现了高精度、高细节、零样本的3D网格局部风格化编辑。

## 研究问题与动机
- **全局编辑主导，局部精细控制缺失**：现有文本驱动3D生成方法（如DreamFusion、Magic3D）多聚焦全局重构，难以在特定语义部件上进行定向修改而不污染整体外观。
- **已有局部编辑方法的两大缺陷**：现有局部编辑工作（如Vox-E、DreamEditor）的定位区域通常粗糙平滑、缺乏高频细节；且多数不原生支持网格格式，无法直接产出标准图形管线所需的UV纹理贴图。
- **级联扩散模型的监督潜力未被挖掘**：当前主流高分辨率2D扩散模型采用级联超分结构，但标准SDS仅利用基础低分辨率阶段，浪费了后续超分阶段的细节生成能力。
- **局部编辑对监督粒度提出双重需求**：小区域编辑既需要基础阶段提供正确的全局语义定位，又需要超分阶段提供刻画细微结构的梯度信号，现有单阶段SDS无法兼顾。

## 核心贡献（创新点）
1. **首个网格原生的文本驱动局部风格化框架**：直接输入网格与文本提示，同步输出显式定位图与高分辨率纹理贴图，无缝对接传统渲染管线，区别于仅在体素或NeRF上工作的局部编辑方法。
2. **级联评分蒸馏（CSD）机制**：首次将SDS推广至级联扩散模型的所有阶段，同时蒸馏多分辨率梯度，实现“全局语义定位”与“局部高频细节”的可控加权平衡。
3. **3D坐标空间神经纹理场表示**：将MLP定义在Mesh的3D表面坐标上（而非2D UV空间），从根本上消除UV接缝处的色彩/法线跳变伪影，并提供天然的超分辨率采样能力。
4. **三分支视觉损失协同优化架构**：设计局部纹理损失、定位边界损失与背景补偿损失，三者通过互补掩码相互约束，迫使定位边缘锐利且纹理严格贴合目标区域。
5. **开放词汇零样本局部编辑与无缝合成**：无需针对特定类别训练，支持自由文本描述任意语义部件；生成的定位掩码可精确叠加多个局部纹理，避免层叠伪影。

## 方法详解
- **神经场表示**：给定网格 $M$ 与文本提示 $y$，通过逆变换UV映射获取3D表面坐标 $\mathbf{x}=(x,y,z)$，输入三个独立的6层MLP（定位 $\mathcal{F}_\theta$、纹理 $\mathcal{F}_\phi$、背景 $\mathcal{F}_\psi$），经位置编码后输出概率值或RGB值，再映射回2D空间得到 $L_{map}$、$T_{map}$、$B_{map}$。
- **三分支视觉损失**：
  1. *局部纹理损失*：用 $L_{map}$ 掩码 $T_{map}$ 得到 $T'_{map}$ 渲染至网格 $M_t$，结合派生提示 $y_t$ 计算CSD梯度，鼓励掩码内生成符合文本的高频纹理。
  2. *定位损失*：按 $L_{map}$ 将黄色叠加到网格得到 $M_l$，使用专门构造的定位提示 $y_l$（沿用3D Highlighter格式）计算CSD梯度，强制定位区域具有明确的视觉边界与语义意义。
  3. *背景损失*：在非定位区域 $(1-L_{map})$ 渲染背景贴图 $B_{map}$，与黄色混合得 $M_b$，使用描述原始物体类别+黄色定位区的提示 $y_b$ 计算CSD梯度，显式要求非编辑区域保留原物特征，抑制定位膨胀。
- **CSD数学形式**：对网格用可微渲染器 $g$ 输出 $N$ 个分辨率图像 $\mathbf{x}=\{x^1,...,x^N\}$。基础阶段 $\phi^1$ 采用标准SDS梯度 $\nabla_{x^1}\mathcal{L}_{SDS}=w(t)(\epsilon_\phi(z_t^1,t,y)-\epsilon)$；对超分阶段 $i>1$，分别在当前与上一级分辨率采样噪声 $\epsilon^i, \epsilon^{i-1}$，构造 $z_t^i=\alpha_t x^i+\sigma_t\epsilon^i$ 与 $z_s^{i-1}=\alpha_s x^{i-1}+\sigma_s\epsilon^{i-1}$，利用 $\phi^i$ 预测噪声并计算 $\nabla_{x^i}=w(t)(\epsilon_{\phi^i}(z_t^i,t,z_s^{i-1},s,y)-\epsilon^i)$。总参数梯度为 $\nabla_\theta\mathcal{L}_{CSD}=\lambda^1\nabla_{x^1}\frac{\partial x^1}{\partial\theta}+\sum_{i=2}^N\lambda^i\nabla_{x^i}\frac{\partial x^i}{\partial\theta}$，跳过U-Net Jacobian直接更新MLP权重，各阶段权重 $\lambda^i$ 可由用户调节以控制细节与全局理解的比重。

## 实验与结果
- **数据集**：使用来自 TurboSquid、ShapeNet、Thingi10k 等多源开源网格（有机/人工类别），覆盖零样本开放词汇评估。
- **基线方法**：定位任务
