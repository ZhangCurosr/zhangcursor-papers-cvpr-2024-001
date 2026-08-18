---
title: "Beyond-First-Order-Tweedie-Solving-Inverse-Problems-using-La"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:03:39"
field: "生成模型与逆问题求解"
keywords: ["Latent Diffusion Models", "Inverse Problems", "Tweedie's Formula", "Posterior Sampling", "Image Editing", "Second-order Approximation"]
innovations: ["通过代理损失函数实现高效二阶Tweedie近似，仅需Hessian迹估计将复杂度从O(d²)降至O(1)", "改进初始化策略并结合随机平均迭代，以50步扩散实现高质量逆问题求解，NFEs减少4-8倍", "提出STSL-CAT两阶段框架，首次实现含污损图像的有效文本引导编辑"]
benchmarks: ["FFHQ-1K", "ImageNet-1K", "COCO 2017"]
---

# 论文速读：Beyond-First-Order-Tweedie-Solving-Inverse-Problems-using-Latent-Diffusion

## 一句话总结
本文提出了STSL（Second-order Tweedie sampler from Surrogate Loss），一种高效的后验采样器，通过代理损失函数实现隐式扩散模型中可 tractable 的二阶Tweedie近似，以50步扩散过程完成逆问题求解，神经函数评估次数仅为SoTA方法PSLD的1/4、P2L的1/8，同时在文本引导的含污损图像编辑任务上超越NTI达32% CLIP准确率提升。

## 研究问题与动机
- **一阶Tweedie近似的偏差问题**：现有SoTA求解器（PSLD、P2L）依赖Tweedie一阶矩近似条件期望 $\mathbb{E}[X_0|X_t]$ 来估计似然项，这种回归均值（regression to the mean）操作引入系统性偏差，导致重建图像细节丢失（Jensen's gap）。
- **二阶近似计算复杂度瓶颈**：已有二阶方法需要计算Hessian矩阵或Jacobian，计算复杂度为 $\mathcal{O}(d^2)$，使得标准逆向扩散过程对于后验采样不可行。
- **逆问题与图像编辑的协同困难**：现有编辑方法（如NTI）在干净源图像上表现良好，但面对真实世界污损（模糊、低分辨率、噪声）时无法有效去除残留失真；而逆问题求解器（PSLD/P2L）虽能去污损但需1000步，实用性受限。
- **初始化误差在少步数下放大**：当扩散步数从1000减至50时，从标准高斯分布 $\pi_d$ 初始化的离散化误差 $\mathcal{O}(de^{-2T})$ 会显著影响重建质量。

## 核心贡献（创新点）
1. **提出可计算的二阶Tweedie近似方法**：通过代理损失函数仅需估计Hessian矩阵的迹（而非完整矩阵），以随机投影方式用单次采样近似，将复杂度从 $\mathcal{O}(d^2)$ 降至 $\mathcal{O}(1)$，建立理论下界保证。
2. **设计改进的初始化与迭代优化策略**：将逆向过程初始化从标准高斯分布改为 $Z_0 \sim p_T(Z_0|\mathcal{E}(\mathbf{A}^T\mathbf{y}))$，通过DDIM前向传播获得初始潜变量，减少离散化误差；结合随机平均（stochastic averaging）进行多次近端梯度更新。
3. **端到端图像编辑框架STSL-CAT**：将STSL逆问题求解器（50步）与Cross-Attention-Tuning结合，首次实现从含污损图像出发的文本引导编辑，有效去除残留失真并精确定位编辑区域。
4. **显著提升效率与质量**：在FFHQ、ImageNet、COCO基准上，以50步替代SoTA方法的1000步，神经函数评估减少4-8倍，LPIPS提升约5%绝对值。

## 方法详解
**核心思想**：通过代理损失函数 $\mathcal{L}(\mathbf{y}, Z_t)$ 在逆向扩散过程中注入二阶校正项。

**代理损失函数设计**（公式4）：
$$\mathcal{L}(\mathbf{y}, Z_t) := \lambda\|\mathbf{y} - \mathbf{A}\mathcal{D}(\bar{Z}_T)\|_2^2 + \frac{\eta}{d}\mathbb{E}_{\epsilon\sim\pi_d}\left[\epsilon^T\left(\nabla\log p_{T-t}(Z_t+\epsilon) - \nabla\log p_{T-t}(Z_t)\right)\right]$$
- 第一项为测量一致性损失，确保重建图像与观测值匹配
- 第二项为Hessian迹的随机近似项，通过Hutchinson估计器实现，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 为单次随机采样

**初始化策略**：
- 传统方法：$Z_0 \sim \pi_d$（标准高斯）
- STSL方法：$Z_0 \sim p_T(Z_0|\mathcal{E}(\mathbf{A}^T\mathbf{y}))$，通过DDIM前向过程从 $\vec{Z}_0 = \mathcal{E}(\mathbf{A}^T\mathbf{y})$ 生成

**逆向SDE修正**（公式3）：
$$dZ_t = \left(Z_t + \mathcal{G}(\mathbf{y}, Z_t) + 2\nabla\log p_{T-t}(Z_t)\right)dt + \sqrt{2}d\tilde{W}_t$$
其中漂移项 $\mathcal{G}(\mathbf{y}, Z_t) \approx -\nabla\mathcal{L}(\mathbf{y}, Z_t)$，替代传统单次梯度更新。

**算法流程**（Algorithm 1）：
- 步骤1-2：初始化，从观测值编码后经DDIM前向生成初始潜变量
- 步骤3-10：对于每个扩散时间步，执行K次随机平均迭代，每次迭代：(a) 采样随机噪声 $\epsilon$；(b) 计算去噪后潜变量 $\bar{Z}_T$；(c) 通过代理损失梯度更新 $Z_t$
- 步骤9：DDIM逆向扩散一步

**图像编辑扩展**（STSL-CAT）：
- 两阶段设计：先用STSL逆求解器（50步）恢复干净图像
- 然后结合Cross-Attention-Tuning，在逆向过程中交替执行CAC更新与代理损失梯度更新
- 早期30步使用测量更新，后期使用ViT对比损失保持内容一致性

**理论保证**（Theorem 4.4）：
证明代理损失 $\hat{\mathcal{L}}(\mathbf{y}, Z_t)$ 是 $\log p_{T-t}(\mathbf{y}|Z_t)$ 的下界，且其梯度近似为：
$$\nabla\hat{\mathcal{L}}(\mathbf{y}, Z_t) \simeq -\lambda\nabla\|\mathbf{y} - \mathbf{A}\bar{Z}_T\|_2^2 - \gamma\nabla\text{Trace}(\nabla^2\log p_{T-t}(Z_t))$$

## 实验与结果
**数据集**：FFHQ-1K（1000张，512×512）、ImageNet-1K（1000张，512×512）、COCO 2017验证集（消融实验）

**评测基线**：
- LDM求解器：PSLD [43]、P2L [10]、LDPS [43]、GML-DPS [43]、LDIR [16]
- PDM求解器：DPS [8]、DiffPIR [55]
- 图像编辑：NTI [35]

**主要结果（Table 1）**：
- **SR(×8)**：STSL LPIPS=0.335（FFHQ）vs P2L=0.381、PSLD=0.402；SSIM=91.32 vs P2L=89.14
- **Motion Deblur**：STSL LPIPS=0.321 vs P2L=0.395
- **Gaussian Deblur**：STSL LPIPS=0.308 vs P2L=0.382
- ImageNet上同样全面超越SoTA，SR(×8) LPIPS 0.392 vs P2L 0.441

**效率对比（Table 2）**：
| 方法 | Runtime(s) | NFEs | Steps |
|------|-----------|------|-------|
| STSL | 45 | 250 | 50 |
| P2L | 500 | 2000 | 1000 |
| PSLD | 194 | 1000 | 1000 |

STSL实现4X（vs PSLD）和8X（vs P2L）的NFEs减少。

**图像编辑（Table 4f）**：
- 在含SR×8污损的图像编辑任务上，STSL-CAT CLIP准确率93.00% vs NTI 70.00%，相对提升32%
- 干净图像上，NTI-CAT（本文CAT加入NTI）CLIP准确率为96.00%，与NTI持平

**消融研究（Table 3-4）**：
- STSL-biased（移除Hessian迹估计）在各项指标上明显劣于完整STSL
- 最优超参：$\eta=0.02$、$\nu=2$、K=5次随机平均、50步DDIM

## 相关工作脉络
1. **PSLD [43]**：在Stable Diffusion潜空间使用一阶Tweedie估计量，通过gluing目标优化潜变量；本文STSL在相同框架基础上引入二阶校正，以更少步数获得更高重建质量。
2. **P2L [10]**：在PSLD基础上联合优化文本嵌入和潜变量；本文STSL避免文本嵌入优化，通过二阶校正直接提升重建质量，且仅需50步。
3. **DPS [8]**：像素空间扩散求解器，依赖一阶Tweedie估计；本文在潜空间工作，更高效且精度更高。
4. **Prior等 [5,32]**：二阶Tweedie修正方法需计算完整Hessian或Jacobian；本文仅需迹估计且单次随机采样，复杂度显著降低。
5. **NTI [35]**：通过空文本嵌入优化实现真实图像编辑；本文指出其在污损图像上失效，通过先复原再编辑的两阶段设计解决此问题。
6. **Cross-Attention Control [17]**：基础编辑方法难以同时保持原图内容；本文CAT通过后续STSL后验采样修正保留更多内容信息。

## 局限性与未来方向
- **SSIM指标偏低**：论文承认在某些任务上SSIM不如竞争对手，因SSIM倾向于将高频伪影标记为"锐利"而惩罚模糊。
- **对比损失依赖超参**：对比损失系数ν需仔细调节（消融显示ν=2最优），过大或过小均影响效果。
- **单样本估计的方差**：Hessian迹估计仅用单次随机采样，可能在高维场景带来估计方差。
- **线性逆问题假设**：当前方法主要针对线性观测模型 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}$，非线性逆问题尚未探索。
- **计算资源依赖**：仍需A100 GPU运行，对低资源环境不够友好。

## 研究启发与可借鉴点
1. **Hessian迹的随机近似技术**：通过Hutchinson估计器+随机投影实现$\mathcal{O}(1)$复杂度的二阶信息估计，可迁移至其他需要曲率信息的生成模型优化任务。
2. **代理损失函数的设计范式**：将测量一致性项与正则化项（此处为二阶修正）结合的思路，可用于构建各种逆问题的统一求解框架。
3. **初始化策略的重要性**：从标准高斯改为条件分布初始化，对少步数采样尤为重要；可借鉴此思路改进其他扩散模型加速方法。
4. **两阶段编辑框架**：先复原再编辑的策略解决了现有编辑方法对污损图像鲁棒性差的问题，可扩展至视频编辑、3D生成等任务。
5. **理论与实践的结合**：定理4.4提供了代理损失作为下界的严格证明，为后续研究提供了可信的理论基础，值得在其他扩散模型应用中复现此分析框架。

## 关键术语表
**Tweedie's Formula**：连接后验期望与得分函数的统计恒等式，用于从含噪观测中估计干净信号。
**Second-order Approximation**：通过协方差项（Hessian迹）修正一阶Tweedie估计，减少Jensen gap导致的偏差。
**Surrogate Loss**：代理损失函数，结合测量一致性项和Hessian迹估计项，作为对数似然的可计算下界。
**Hutchinson's Estimator**：通过随机投影高效估计矩阵迹的无偏估计方法，此处用于近似Hessian迹。
**Cross-Attention Control (CAC)**：通过修改文本-图像交叉注意力机制实现编辑控制的扩散模型技术。
**Cross-Attention-Tuning (CAT)**：在CAC更新后通过STSL后验采样进一步精炼潜变量，提升编辑保真度。
**Neural Function Evaluations (NFEs)**：神经网络前向传播次数，衡量采样效率的关键指标。
**Posterior Sampling**：从后验分布$p(X_0|\mathbf{y})$中采样，是逆问题求解的核心目标。

## 可复现要素
- **数据集**：FFHQ-1K、ImageNet-1K、COCO 2017（公开可用）
- **代码开源状态**：论文未明确声明代码开源
- **预训练模型**：使用Stable Diffusion v1.4/v1.5（公开可用）
- **关键超参数**：
  - 扩散步数T=50
  - 随机平均步骤K=5
  - 二阶校正系数η=0.02
  - 对比损失系数ν=2
  -  likelihood强度λ（遵循[8,10,43]惯例）
- **硬件配置**：单张A100 GPU
