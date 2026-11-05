---
title: "Confronting-Ambiguity-in-6D-Object-Pose-Estimation-via-Score"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Hsiao_Confronting_Ambiguity_in_6D_Object_Pose_Estimation_via_Score-Based_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:21:37"
field: "6D物体位姿估计"
keywords: ["6D位姿估计", "扩散模型", "SE(3)李群", "姿态模糊性", "分数匹配", "对称感知", "李群生成模型"]
innovations: ["首个在SE(3)李群上应用分数扩散模型解决6D位姿模糊性", "提出SE(3)替代Stein分数提升去噪收敛性", "设计基于傅里叶的周期条件机制增强SO(3)分布学习"]
benchmarks: ["SYMSOL", "SYMSOL-T", "T-LESS"]
---

# 论文速读：Confronting-Ambiguity-in-6D-Object-Pose-Estimation-via-Score-Based-Diffusion

## 一句话总结
本文提出首个将分数阶扩散模型应用于SE(3)李群的6D位姿估计方法，通过联合建模旋转与平移的联合分布来解决对称性和遮挡引起的姿态模糊性问题。在SYMSOL-T和T-LESS数据集上均取得SOTA性能。

## 研究问题与动机
- **姿态模糊性本质**：单目RGB图像到6D位姿存在"一对多"映射关系，由物体对称性（如立方体的多个等价旋转）和遮挡引起，导致传统回归方法性能下降
- **现有方案局限**：对称感知标注方法需要手动标注等价视角，对无纹理物体（如杯子把手遮挡时）几乎不可行；隐式概率密度方法（Implicit-PDF、HyperPose-PDF）仅在SO(3)旋转空间建模，未考虑旋转-平移联合分布
- **计算复杂度**：非参数密度估计需要穷举采样整个SO(3)空间，网格搜索精度受分辨率限制，扩展到SE(3)面临内存爆炸
- **扩散模型潜力**：迭代去噪过程可聚焦探索姿态空间，无需显式密度估计，适合处理高维分布和多模态

## 核心贡献（创新点）
- **首个SE(3)扩散模型用于图像域位姿估计**：区别于仅作用于SO(3)或R³×SO(3)的现有工作，直接在SE(3)李群上建模旋转-平移联合分布
- **提出SE(3)上的替代Stein分数公式**：利用多步子采样逼近真实分数，解决SE(3)上左/右雅可比不对称导致收敛困难的问题，比自动微分更高效稳定
- **引入基于傅里叶的条件机制**：针对SO(3)的周期性特征设计cos/sin投影，替代传统scale-bias条件，显著提升神经网络捕获周期分数场的能力
- **构建SYMSOL-T新数据集**：在原始SYMSOL基础上添加随机平移，提供评估SE(3)空间扩散模型的测试基准

## 方法详解
- **李群扰动核**：在SE(3)上定义高斯分布 $\mathcal{N}_{SE(3)}(Y; X, \Sigma) = \frac{1}{\zeta(\Sigma)}\exp(-\frac{1}{2}\mathrm{Log}(X^{-1}Y)^\top \Sigma^{-1} \mathrm{Log}(X^{-1}Y))$，通过Lie代数对数映射度量群元素间距离
- **分数匹配损失**：训练目标 $\mathcal{L}(\theta;\sigma) = \frac{1}{2}\mathbb{E}_{p_{data}}\mathbb{E}_{\tilde{X}}[\|s_\theta(\tilde{X},\sigma) - \nabla_{\tilde{X}}\log p_\sigma(\tilde{X}|X)\|^2]$，其中真实分数 $\nabla_{\tilde{X}}\log p_\sigma(\tilde{X}|X) = -\frac{1}{\sigma^2}\mathbf{J}_r^{-\top}(z)z$
- **替代Stein分数推导**：SE(3)不满足 $\mathbf{J}_l(z) = \mathbf{J}_r^\top(z)$ 性质，但通过多步子采样可使小变换积分逼近 $-\frac{1}{\sigma^2}z$，故用 $\tilde{s}_X(\tilde{X},\sigma) = -\frac{1}{\sigma^2}z$ 替代真实分数训练
- **去噪采样过程**：采用测地随机游走的李群版本 $\tilde{X}_{i+1} = \tilde{X}_i \mathrm{Exp}(\epsilon_i s_\theta(\tilde{X}_i, \sigma_i) + \sqrt{2\epsilon_i}z_i)$，其中 $z_i \sim \mathcal{N}(0,I)$
- **条件生成架构**：图像编码器（ResNet）一次性提取特征 $c$，去噪网络由多个MLP块组成；条件机制采用傅里叶形式 $f_i(x,c) = \sum_{j=0}^{d-1}\mathbf{W}_{ij}(\mathbf{A}_j(c)\cos(\pi x_j) + \mathbf{B}_j(c)\sin(\pi x_j))$，无需额外参数即可增强周期特征学习

## 实验与结果
- **数据集**：SYMSOL（5个对称物体，25万张图像）、SYMSOL-T（添加随机平移）、T-LESS（30个无纹理工业物体，50k合成+37k真实训练，10k测试）
- **评估指标**：SYMSOL/SYMSOL-T使用平均角度误差（度）和平均平移距离；T-LESS使用BOP挑战指标MSPD、MSSD、VSD及对称感知指标R@2/5/10、T@2/5/10
- **SYMSOL结果**：ResNet34平均误差0.42°，ResNet50平均误差0.37°，显著优于Normalizing Flows（0.70°）、HyperPosePDF（1.94°）等基线
- **SYMSOL-T结果**：SE(3)模型在所有5个形状上旋转误差最低（四面体0.59°、立方体0.58°、二十面体0.64°等），优于R³SO(3)变体；二十面体上R³SO(3)失败（29.35° vs 0.64°）
- **T-LESS结果**：SE(3)模型MSPD达93.16%（vs GDRNPP的90.17%，+3.0%）、MSSD 60.17%（+7.1%）、VSD 56.88%（+9.3%）；R@10达94.78%，远超GDRNPP的90.56%
- **推理效率**：SE(3)模型在5步去噪下达250 FPS，10步下161 FPS，满足实时需求

## 相关工作脉络
- **SO(3)扩散模型**：Leach等[32]和Jagvaral等[28]在SO(3)上使用DDPM/SGM，但依赖于无闭式解的IG_{SO(3)}分布；本文将其扩展到SE(3)并解决计算效率问题
- **R³SO(3)参数化扩散**：Urain等[59]和Yim等[68]使用分离扩散，旋转平移独立建模；本文证明联合SE(3)参数化能获得更鲁棒的旋转估计
- **概率位姿估计**：Implicit-PDF[39]和HyperPosePDF[23]在SO(3)上进行非参数密度估计，依赖网格搜索；本文通过扩散模型避免显式密度计算
- **对称感知回归**：GDRNPP[62]、COPE[58]等使用对称标注和几何引导；本文完全无需对称标注，仅依赖RGB图像和GT位姿训练
- **李群生成建模**：De Bortoli等[5]将扩散扩展到黎曼流形；本文专门针对SE(3)设计替代分数和条件机制

## 局限性与未来方向
- **单一模型处理多种形状**可能导致不同姿态分布相互干扰（锥体在ResNet50上性能下降）
- **依赖真值边界框和分割掩码**进行ROI裁剪，限制了在无标注场景的应用
- **未利用3D模型几何信息**，导致平移精度略逊于GDRNPP等使用深度几何引导的方法
- **SE(3)替代分数的近似误差**可能影响极端情况下的去噪质量
- **未来方向**：开发形状自适应的架构、探索无需GT标注的自监督训练、结合3D几何先验进一步提升平移估计

## 研究启发与可借鉴点
- **SE(3)李群上的替代分数策略**可迁移至其他需要联合优化旋转平移的任务（如机器人抓取位姿生成、SLAM初始化）
- **傅里叶条件机制**的周期特性设计对任何在紧致流形（如SO(3)、 quaternion sphere）上的生成任务均有借鉴价值
- **无需对称标注的端到端训练范式**降低了数据收集成本，适用于工业场景中新物体的快速部署
- **分离条件提取与去噪的网络架构**（ResNet一次性编码+多次MLP去噪）有效降低推理延迟，值得在实时系统设计中采用
- **SYMSOL-T数据集的构建思路**可推广至其他需要在连续变换空间评估概率模型的数据集构建

## 关键术语表
**SE(3)李群**：描述三维空间刚体变换（旋转+平移）的6自由度数学结构，元素形式为(R,T)，其中R∈SO(3)，T∈ℝ³
**Score-based diffusion model**：通过学习数据分布的对数梯度（score）来进行去噪采样的生成模型，无需显式概率密度
**Stein分数**：概率密度对数梯度的另一种表述，在扩散模型训练中作为目标信号
**替代分数**：在SE(3)上用简单形式 $-\frac{1}{\sigma^2}z$ 近似真实分数，避免复杂的雅可比逆矩阵计算
**对称感知指标**：考虑物体等价姿态的评估标准，如R@k表示旋转误差在k度内的预测比例
**Mollweide投影**：将球面数据（如SO(3)旋转分布）映射到平面的可视化方法，经度和纬度对应yaw/pitch
**Dynamic Zoom-In**：一种数据增强策略，在保持ROI宽高比的同时随机缩放图像以引入训练随机性

## 可复现要素
- **数据集**：SYMSOL（原数据集公开）、SYMSOL-T（需自行构建，基于SYMSOL添加随机平移）、T-LESS（公开基准）
- **代码开源情况**：论文未明确说明代码开源状态（需查看补充材料或项目主页）
- **权重开源情况**：论文未提及
- **关键超参**：输入尺寸224×224像素、ResNet34/50 backbone、位置编码嵌入、多步去噪（5-100步可选）
- **训练设备**：AMD Ryzen Threadripper 2990WX CPU + RTX 2080 Ti GPU，使用JAX框架
