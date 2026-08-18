---
title: "CAD-SIGNet-CAD-Language-Inference-from-Point-Clouds-using-La"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Khan_CAD-SIGNet_CAD_Language_Inference_from_Point_Clouds_using_Layer-wise_Sketch_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:22:21"
---

# 论文速读：CAD-SIGNet-CAD-Language-Inference-from-Point-Clouds-using-Layer-wise-Sketch-Instance-Guided-Attention

## 一句话总结
提出了CAD-SIGNet，一种端到端可训练的自回归架构，通过层间交叉注意力与草图实例引导注意力（SGA）机制，从输入点云逐步推断出由草图-拉伸操作组成的CAD设计历史序列，并支持交互式多方案逆向工程。

## 研究问题与动机
- **核心问题**：如何从点云有效学习CAD视觉-语言联合表征，以实现3D逆向工程中的CAD语言推断？
- **现有方法不足**：
  1. **表征学习割裂**：DeepCAD与MultiCAD等前期工作采用两阶段策略，分别学习点云视觉表征与CAD语言表征后再求映射，易产生与推理任务无关的模态特异性特征。
  2. **推理策略非自回归**：现有方法多为前馈（feed-forward）一次性预测完整设计史，无法支持设计师在每一步输入偏好或选择不同设计方案，缺乏交互性。
  3. **草图参数化依赖全局点云**：传统方法直接用全局点云表征推断草图，忽略了实际设计中仅需特定平面交集区域（Sketch Instance）即可参数化草图的事实，引入大量无关噪声。

## 核心贡献（创新点）
1. **首个端到端自回归逆向工程网络**：直接建立点云到CAD语言序列的自回归映射；与DeepCAD/MultiCAD的前馈或两阶段对比学习策略本质不同，实现了视觉-语言表征的联合端到端学习。
2. **层间多模态Transformer块与交叉注意力机制**：每层Block同步处理点云与CAD token嵌入，并通过逐层交叉注意力实现几何信息到语言信息的渐进式传递；与一次性全序列预测的本质区别在于支持逐步条件生成与历史状态维护。
3. **草图实例引导注意力（SGA）模块**：利用已预测的拉伸token定义草图平面与边界框，从点云中筛选出相关子集（Sketch Instance）用于草图token的交叉注意力计算；与全量点云参与注意力的本质区别在于显著降低噪声干扰并提升细粒度草图参数预测精度。
4. **交互式多方案生成与几何择优策略**：基于自回归特性提出混合采样（Hybrid Sampling），可输出多条合理设计路径并通过重建点云Chamfer Distance自动择优；突破了传统单解输出的局限，适配真实设计交互场景。

## 方法详解
- **问题形式化**：输入点云 $\mathbf{X} \in \mathbb{R}^{N \times 3}$，输出CAD设计历史序列 $\mathcal{C}$（由草图与拉伸token交替组成）。目标学习映射 $\Phi$，采用自回归分解 $p_\theta(\mathcal{C}|\mathbf{X}) = \prod_{i=1}^{n_{ts}} p_\theta(t_i | \{t_{j<i}\}, \mathbf{X})$。
- **点云与CAD语言嵌入**：
  - 点云：线性层+ReLU后接两个LFA（Local Feature Aggregation）模块聚合k-NN邻域特征，得到 $\mathbf{F}_0^v \in \mathbb{R}^{N \times d_e}$。
  - CAD语言：将二维草图坐标 $(p_x, p_y)$ 视为单个2D token，其余token补零对齐为矩阵形式；结合one-hot编码、token类型标志 $\mathbf{C}_{\mathrm{type}}$、步骤标志 $\mathbf{C}_{\mathrm{step}}$ 及位置编码 $\mathbf{C}_{\mathrm{pos}}$，经线性层得到 $\mathbf{F}_0^c$。
- **层间多模态Transformer块**：共 $B$ 层。每层对CAD嵌入做多头自注意力（SA），对
