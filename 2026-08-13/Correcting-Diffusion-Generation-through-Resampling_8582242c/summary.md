---
title: "Correcting-Diffusion-Generation-through-Resampling"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_Correcting_Diffusion_Generation_through_Resampling_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:15:03"
field: "扩散模型采样与后处理"
keywords: ["diffusion models", "particle filtering", "text-to-image generation", "distribution correction", "missing object error", "image quality improvement", "resampling"]
innovations: ["提出基于粒子滤波的重新采样框架显式修正扩散生成分布差异", "设计判别器与物体检测器结合的混合修正项同时解决缺失物体和质量问题"]
benchmarks: ["MS-COCO", "GPT-Synthetic", "ImageNet-64", "FFHQ"]
---

# 论文速读：Correcting-Diffusion-Generation-through-Resampling

## 一句话总结
本文提出一种基于粒子滤波的重新采样框架，通过显式减少生成图像与真实图像之间的分布差异，有效修正扩散模型在文本到图像生成中的缺失物体错误并提升图像质量。

## 研究问题与动机
- **缺失物体错误（Missing Object Errors）**：扩散模型在文本到图像生成时，有时无法生成文本提示中提及的物体（如"Cat on chair peering over top of table at glass of beverage"中缺少玻璃杯）。
- **低图像质量问题**：生成图像存在人工伪影和非自然扭曲（如缺失左下肢的人物图像）。
- **根本原因**：上述问题的根本原因是扩散生成过程与真实数据分布之间存在**分布差异（distributional discrepancies）**，源于去噪网络的有限表达能力、离散化数值求解ODE/SDE轨迹引入的误差等。
- **现有方法不足**：现有方法（如修改交叉注意力机制、使用判别器校正分数函数）大多未针对分布差异这一根本原因，效果受限；且过度强调提升物体出现率可能导致质量下降。

## 核心贡献（创新点）
1. **提出基于粒子滤波的重新采样框架**：将粒子滤波引入扩散生成过程，在每个去噪步骤通过重新采样显式修正分布差异，而非仅在最终步选择样本或修改分数函数。
2. **设计两种修正项计算方法**：(1) 判别器方法（PF-DISCRIMINATOR），使用少量带caption的真实图像训练条件判别器估计无条件似然比；(2) 混合方法（PF-HYBRID），结合无判别器和预训练物体检测器，分别修正整体分布差异和缺失物体错误。
3. **理论证明与实验验证**：证明该框架理论上可使生成分布收敛到真实分布；在文本到图像、无条件生成和类条件生成任务上均显著优于基线方法。
4. **计算效率优势**：相比D-GUIDANCE在每个去噪步骤评估判别器的方法，本文方法仅在部分步骤进行重采样且无需反向传播，计算成本更低（约0.66倍）。

## 方法详解
**粒子滤波框架概述**：
- 从$X_T$（纯高斯噪声）开始，逆向生成$X_{T-1}, ..., X_0$。
- 每个步骤$t$包含两个子步骤：
  - **提议（Proposal）**：使用扩散模型的去噪转移概率$q(X_t|X_{t+1}, C)$生成提议样本$\tilde{x}_t^{(k)}$。
  - **重采样（Resampling）**：根据权重函数$w(x_{t+1}^{(k)}, \tilde{x}_t^{(k)}|C) = \frac{\phi_t(\tilde{x}_t^{(k)}|C)}{\phi_{t+1}(x_{t+1}^{(k)}|C)}$进行带替换的重采样，保留$\tilde{x}_t^{(k)}$作为新样本$x_t^{(k)}$。
- 关键：修正项$\phi_t(X_t|C)$设计为使最终分布$v(X_t|C) = q(X_t|C)\phi_t(X_t|C)$逼近真实分布$p(X_t|C)$。理想情况下$\phi_t(X_t|C) = p(X_t|C)/q(X_t|C)$。

**判别器方法（Section 3.5）**：
- 使用带caption的真实图像集作为外部指导。
- 训练条件判别器$d(X_t|C; t)$区分真实样本（真实图像加噪）和伪造样本（扩散模型生成后加噪）。
- 利用Goodfellow等人的理论结果，最优判别器$d^*$满足：$\frac{p(X_t|C)}{q(X_t|C)} \approx \frac{d^*(X_t|C; t)}{1 - d^*(X_t|C; t)}$。
- 可直接用于估计$\phi_t$，但判别器可能忽略缺失物体错误而侧重图像质量。

**混合方法（Section 3.6）**：
- 结合无判别器和预训练物体检测器（如DETR）的两类外部指导。
- 根据贝叶斯规则分解条件似然比，聚焦于两项：无条件似然比（图像质量）和物体提及比（缺失物体错误）。
- **无条件似然比**：使用无判别器$d(X_t; t)$估计，类似判别器方法但不依赖text condition。
- **物体提及比**：假设$O_C$各维度条件独立，分解为提及物体和未提及物体两组。
  - 分子$p(O_{Ci}=1|X_t)$近似为：对去噪预测的干净图像$f(X_t)$运行物体检测器得到的出现概率$\hat{p}(O_{Xi}=1|f(X_t))$。
  - 分母$q(O_{Ci}=1|X_t)$近似为：$\hat{p}(O_{Xi}=1|f(X_t)) + \hat{p}(O_{Xi}=0|f(X_t)) \frac{(1-\kappa_{it})\pi_{it}}{(1-\kappa_{it})\pi_{it} + 1 - \pi_{it}}$，其中$\kappa_{it}$为初始生成阶段物体$i$在文本提及时的检测出现率，$\pi_{it}$为超参数。
- 最终$\phi_t(X_t|C) = \frac{p(X_t)}{q(X_t)} \cdot \prod_{i:O_{Ci}=1} \frac{p(O_{Ci}=1|X_t)}{q(O_{Ci}=1|X_t)}$。

## 实验与结果
**数据集**：
- GPT-Synthetic：500个复杂caption，每个含2-5个物体及颜色/空间关系描述。
- MS-COCO验证集：筛选含至少4个物体的复杂描述子集（261个caption）。
- ImageNet-64：类条件生成基准。
- FFHQ：无条件生成基准。

**评估基线**：SD (Stable Diffusion)、SD-GUIDANCE、SPATIAL-TEMPORAL、ATTEND-EXCITE、OBJECTSELECT、TIFASELECT、REWARDSELECT。

**主要结果（MS-COCO）**：
- **物体出现率（Object Occurrence）**：PF-HYBRID达到**68.13%**，比最强基线SD-GUIDANCE提升**5%**。
- **FID（图像质量）**：PF-HYBRID为**24.03**，比SD的30.03提升约6分；PF-DISCRIMINATOR可达**22.91**。
- **图4显示**：采样基线方法（圆形点）普遍优于非采样方法（十字点）；PF-HYBRID是唯一同时实现高物体出现率和低FID的算法。
- **ImageNet-64**：PF方法以4图像生成、165 NFE达到SOTA的**FID = 1.02**，超越使用判别器指导的强基线。
- **FFHQ**：同样取得最低FID。

**主观评估（Table 1）**：在MTurk上100个caption的人为比较中，PF-HYBRID在物体出现率和图像质量上均胜出所有基线。

**消融实验（Table 2）**：
- 移除粒子滤波（仅做最终步选择）会显著降低PF-HYBRID的所有指标，验证粒子滤波过程的重要性。
- 移除判别器（仅用物体提及比）提升物体出现率但增加FID；反之移除物体项则物体出现率下降。
- 混合方法在两者间取得最佳平衡。

## 相关工作脉络
1. **忠实文本到图像生成**：Wu等人（SPATIAL-TEMPORAL）和Feng等人（ATTEND-EXCITE）修改交叉注意力机制分别关注caption中每个名词短语；本文方法不修改模型内部注意力，而是通过采样后处理修正分布。
2. **Karthik等人（TIFASELECT/REWARDSELECT）**：仅在最终步进行样本选择，且不追求逼近真实分布；本文方法在每个去噪步骤都进行重采样，以显式减小分布差异。
3. **粒子滤波在扩散中的应用**：Trippe等人（protein backbone生成）和Wu等人（条件采样）将粒子滤波用于无条件→条件生成或多样性提升；本文用于显式修正分布差异以解决缺失物体和质量问题，目标不同。
4. **判别器引导扩散生成**：Kim等人（D-GUIDANCE）使用判别器修改score function，仍受ODE/SDE离散化误差影响；本文用判别器估计权重进行重采样，受离散化误差影响更小。
5. **Xiao等人（DDGAN）和Kang等人（OGDM）**：训练扩散模型时引入判别器损失以加速收敛或提升小步数质量；本文方法为后处理采样策略，无需重新训练模型。
6. **Corso等人（Particle Guidance）**：通过联合粒子势函数修改score函数以提升多样性；与本文正交，可结合。

## 局限性与未来方向
- **计算开销**：需要生成K个样本并运行物体检测器/判别器，比单路径生成更耗时（尽管计算效率优于D-GUIDANCE）。
- **超参数敏感性**：混合方法中的$\pi_{it}$超参数可能需要调优；初始生成阶段用于估计$\kappa_{it}$引入额外计算。
- **对外部指导的依赖**：需要少量真实图像和预训练物体检测器；在缺乏这些资源的情况下效果可能受限。
- **扩展性**：目前主要针对缺失物体和图像质量两类错误，对于其他不一致性（如错误属性绑定、位置错误）的修正能力待进一步探索。
- **潜在失败案例**：附录F显示方法可能在某些复杂场景下仍失败（如物体重叠、极端视角）。

## 研究启发与可借鉴点
1. **分布差异作为统一问题框架**：将缺失物体、低质量等不同生成错误统一归因于分布差异，为系统设计提供清晰目标（显式缩小$p$与$q$的距离），可迁移至其他生成任务（如视频、3D生成）。
2. **粒子滤波在扩散采样中的灵活应用**：证明在每个去噪步骤插入重采样模块是有效的；可与现有采样器（如Restart sampler、EDM）无缝集成，无需修改基础模型。
3. **混合修正项设计**：结合判别器（全局质量）和物体检测器（细粒度物体存在）的双重指导策略，平衡多目标优化，避免单一信号导致的偏差（如判别器忽略物体）。
4. **计算效率权衡分析**：提出NFE相同下的公平比较框架，并指出NFE不能完全反映计算成本（需考虑判别器前向/反向传播）；这对后续工作的实验设计有参考价值。
5. **开放代码与可复现性**：代码已开源（https://github.com/UCSB-NLP-Chang/diffusion_resampling.git），包含详细实现和消融实验，便于后续研究直接复用或扩展。

## 关键术语表
- **Particle Filtering（粒子滤波）**：一种蒙特卡洛采样框架，通过维护一组带权重的样本（粒子）并迭代进行提议、加权、重采样来近似目标分布。
- **Distributional Discrepancy（分布差异）**：生成图像分布$q$与真实图像分布$p$之间的差异，是扩散生成错误的根本原因。
- **Missing Object Error（缺失物体错误）**：文本到图像生成中，文本提示提及但生成图像中未出现的物体错误。
- **Resampling Weight（重采样权重）**：粒子滤波中用于计算每个粒子被保留概率的权重，本文设计中与分布修正项$\phi_t$相关。
- **Unconditional Likelihood Ratio（无条件似然比）**：真实图像分布与生成图像分布之比$p(X_t)/q(X_t)$，用于修正整体图像质量分布差异。
- **Object Mention Ratio（物体提及比）**：给定 noisy image 下文本提及某物体的条件概率之比，用于修正缺失物体错误。
- **Restart Sampler（重启采样器）**：一种迭代去噪-加噪-去噪的采样策略，本文实验中使用该采样器并插入重采样模块。
- **NFE（Number of Function Evaluations）**：扩散采样中评估去噪网络（U-Net）的次数，常作为计算开销的度量指标。

## 可复现要素
- **数据集**：GPT-Synthetic（Wu et al. [61]引入）、MS-COCO（公开）、ImageNet-64（公开）、FFHQ（公开）。代码仓库包含数据处理脚本。
- **代码**：开源于 https://github.com/UCSB-NLP-Chang/diffusion_resampling.git。
- **权重**：使用Stable Diffusion v2.1-base（公开预训练权重）、DETR with ResNet-50 backbone（公开）、训练判别器所需数据为公开数据集的子集。
- **关键超参**：生成样本数K（实验取5、10、15）；混合方法中$\pi_{it}$为0-1间超参数（论文未指定具体值，需查阅附录或代码）；判别器训练轮次和学习率等细节见Appendix。
