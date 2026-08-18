---
title: "Masked and Shuffled Blind Spot Denoising for Real-World Images"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Chihaoui_Masked_and_Shuffled_Blind_Spot_Denoising_for_Real-World_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:17:45"
field: "自监督单图像去噪"
keywords: ["单图去噪", "盲点去噪", "自监督学习", "相关噪声", "像素洗牌", "自适应遮挡"]
innovations: ["提出自适应遮挡比选择机制，通过双噪声估计差距自动推断噪声相关性水平", "引入局部像素洗牌（LPS）技术从源头破坏相关噪声的空间结构", "在 SIDD/FMDD 等多个真实数据集上达到单图自监督去噪 SOTA"]
benchmarks: ["SIDD Validation", "SIDD Benchmark", "FMDD", "PolyU"]
---

# 论文速读：Masked and Shuffled Blind Spot Denoising for Real-World Images

## 一句话总结
本文提出了 MASH（Masked and Shuffled Blind Spot Denoising），一种针对含相关噪声的真实单图像的自监督去噪方法，通过自动估计噪声相关性水平来选择最优遮挡比，并引入局部像素洗牌技术破坏噪声的空间相关性，在多个真实数据集上达到了优于现有单图方法的 SOTA 性能。

## 研究问题与动机
1. **真实图像中的相关噪声问题**：真实世界图像（如手机拍摄、荧光显微图像）中的噪声往往具有空间相关性（非 i.i.d.），而主流 BSD 方法假设噪声是独立的，难以处理此类数据。
2. **单图去噪的过拟合风险**：单图训练数据极少，网络容易过拟合噪声模式（尤其是相关噪声呈现类纹理图案时），导致去噪性能下降。
3. **固定遮挡比的局限性**：已有 BSD 方法使用固定遮挡策略，无法适应不同图像中未知的噪声相关程度。
4. **现有方法的泛化瓶颈**：在未知噪声分布的数据上，依赖训练数据集的方法泛化能力有限，需要无监督/自监督的单图解决方案。

## 核心贡献（创新点）
1. **揭示了遮挡比与噪声相关性的内在关系**：实验发现低遮挡比有利于 i.i.d. 噪声，高遮挡比有利于高相关噪声，由此提出一种自动估计噪声相关性水平的方法。
2. **自适应盲点遮挡机制**：通过双遮挡比（$\tau^{low}=0.2$、$\tau^{high}=0.8$）训练后计算噪声估计差距 $\varepsilon$，自动选择最优遮挡比，替代了此前固定的遮挡策略。
3. **局部像素洗牌（Local Pixel Shuffling, LPS）**：利用中间去噪结果作为伪干净图像识别平坦区域，在这些区域内对相同颜色强度的像素进行局部随机置换，从源头破坏噪声空间相关性，这是 BSD 框架内首次引入此类去相关技术。
4. **端到端自监督单图去噪新 SOTA**：在 SIDD、FMDD、PolyU 三个真实数据集上，相比同类单图方法取得显著提升（SIDD +~2 dB，FMDD +1.5 dB），并在 FMDD 上超越数据集方法。

## 方法详解
**整体框架**：MASH 基于 BSD（Blind Spot Denoising）框架，核心流程如下：

**步骤一：自适应噪声相关性估计**
- 分别用 $\tau^{high}=0.8$ 和 $\tau^{low}=0.2$ 训练两个盲点模型。
- 利用收敛后的噪声水平估计值计算差距：$\varepsilon = |\hat{\sigma}_{\tau^{high}} - \hat{\sigma}_{\tau^{low}}|$，其中 $\hat{\sigma}_\tau = \sqrt{\frac{1}{HWC}\|f_\theta(\mathbf{m}\odot\mathbf{y}) - \mathbf{y}\|_2^2}$。
- $\varepsilon$ 与噪声相关程度成正比：$\varepsilon$ 小 → 低相关（i.i.d.为主），$\varepsilon$ 大 → 高相关噪声。

**步骤二：最优遮挡比动态选择**
$$\tau^{optimal} = \begin{cases} \tau^{low}, & \varepsilon \leq \varepsilon^{low} \\ \tau^{medium}, & \varepsilon^{low} < \varepsilon < \varepsilon^{high} \\ \tau^{high}, & \varepsilon \geq \varepsilon^{high} \end{cases}$$
默认阈值：$\varepsilon^{low}=1.5$，$\varepsilon^{high}=2.5$。

**步骤三：局部像素洗牌（LPS）**（仅在高相关噪声时启用）
- 利用已训练 $N_1$ 步的中间模型输出 $\hat{\mathbf{y}}$ 作为伪干净图像。
- 计算每个 $s \times s$ 区块内各颜色通道的标准差 $\sigma[\mathbf{i}]$，设定阈值 $\lambda$，将满足 $\sigma[\mathbf{i}] < \lambda$ 的像素归入平坦区域 $\Omega_{const}$。
- 在平坦区域内仅对 $s \times s$ 块内的像素做局部随机置换：$\mathbf{y}^{shuffled} = \mathbf{c}(\mathbf{y}) \odot \Gamma(\mathbf{y}) + (\mathbf{1}-\mathbf{c}(\mathbf{y})) \odot \mathbf{y}$，其中 $\mathbf{c}(\mathbf{y})$ 为平坦区域掩码。
- 使用洗牌后图像作为目标，优化 loss：$L(\theta) = \mathbb{E}_\mathbf{m}[\|( \mathbf{1}-\mathbf{m}) \odot (f_\theta(\mathbf{y}\odot\mathbf{m}) - \mathbf{y}^{shuffled})\|_2^2]$。

**步骤四：集成输出**
最终去噪结果为 $K=10$ 次不同随机遮挡预测的平均：$\hat{\mathbf{y}} = \frac{1}{K}\sum_{p=1}^K f_\theta(\mathbf{m_p}\odot\mathbf{y})$。

**关键超参**：$\tau^{high}=0.8, \tau^{medium}=0.5, \tau^{low}=0.2$，$\varepsilon^{low}=1.5, \varepsilon^{high}=2.5$，$s=4$（洗牌块大小），$N=800$（迭代步数）。

## 实验与结果
**数据集**：SIDD（validation + benchmark，各 1280 个 256×256 patch）、FMDD（512×512 荧光显微图像）、PolyU（100 张 512×512 真实相机图像）。

**对比基线**：
- 单图方法：BM3D、DIP、Self2Self、PD-denoising、NN+denoiser、APBSN-single、ScoreDVI、Baseline（固定 τ=0.5 的 BSD）
- 数据集方法：APBSN、CVF-SID、LUD-VAE
- 监督方法：DnCNN

**主要结果（PSNR/SSIM）**：

| 数据集 | MASH | 最佳对比方法 | 提升幅度 |
|--------|------|-------------|----------|
| SIDD Validation | **35.06 / 0.851** | ScoreDVI (34.75/0.856) | +0.31 dB PSNR |
| SIDD Benchmark | **34.78 / 0.900** | PD-denoising (33.61/0.894) | +1.17 dB PSNR |
| FMDD | **33.71 / 0.882** | ScoreDVI (33.10/0.865) | +0.61 dB，且超越所有单图及数据集方法 |
| PolyU | 37.62 / 0.932 | ScoreDVI (37.77/0.959) | 略低于 ScoreDVI |

- 相比固定 τ=0.5 的 Baseline：SIDD +1.94 dB，FMDD +1.46 dB。
- 自适应遮罩选择准确率：SIDD 88.7%，FMDD 92.4%。
- 消融验证：LPS 带来约 +0.7 dB 增益；自适应遮罩带来约 +1.0 dB 增益。
- 计算效率：参数 0.99M，FLOPs 11.44G，推理时间 75.3s（256×256 输入），优于 Self2Self（3546.5s）和 NN+denoiser（897.6s）。

## 相关工作脉络
1. **Noise2Void (Krull et al., 2019)**：最早提出盲点去噪（单一像素遮挡），MASH 将其推广为任意比例的随机遮挡，并进一步分析了遮挡比与噪声相关性的关系。
2. **Self2Self (Quan et al., 2020)**：使用 dropout 式遮挡对单图训练去噪器，预测被遮挡像素；MASH 继承其思想但提出自适应遮挡策略和局部洗牌增强。
3. **Deep Image Prior (DIP, Ulyanov et al., 2018)**：利用 CNN 归纳偏置从单退化图像恢复，MASH 同样针对单图场景但通过盲点机制避免恒等映射陷阱，且效率远高于 DIP（75s vs 146s）。
4. **PD-denoising (Zhou et al., 2020)**：通过像素洗牌下采样削弱结构化噪声的空间相关性；MASH 的方法更轻量且面向相关高斯噪声而非纯结构化噪声，且不需要多 stride 训练-测试。
5. **APBSN (Lee et al., 2022)**：使用不对称像素下采样（不同 stride）于训练和测试阶段以应对真实噪声；MASH 不依赖下采样操作，转而通过自适应遮罩和局部洗牌解决。
6. **ScoreDVI (Cheng et al., 2023)**：利用 score prior 进行单图去噪；MASH 在 FMDD 数据集上超越 ScoreDVI，且无需额外的 score 估计步骤。

## 局限性与未来方向
1. **计算效率仍有优化空间**：当前 MASH 推理时间约 75s（256×256），相比传统方法（BM3D 仅数十毫秒）仍偏慢；论文指出初始两阶段训练可并行化加速。
2. **阈值超参的通用性待验证**：$\varepsilon^{low}=1.5$、$\varepsilon^{high}=2.5$、$s=4$ 等超参在当前数据集上表现良好，但在更广泛的噪声类型或更低/更高噪声强度下可能需调整。
3. **洗牌技术依赖伪干净图像质量**：LPS 利用中间去噪结果识别平坦区域，若初始去噪误差较大，可能导致错误分区从而影响洗牌效果。
4. **仅针对加性相关噪声建模**：合成噪声采用多元高斯模型，对于泊松噪声、混合噪声或传感器特定噪声模式的适用性尚待验证。

## 研究启发与可借鉴点
1. **遮挡比与噪声相关性的定量关联分析**可作为通用范式：不仅适用于去噪，也可迁移至其他盲点自监督任务（如超分辨率、去模糊）中，通过分析遮挡强度与输入数据相关性的关系来设计自适应策略。
2. **局部像素洗牌的去相关思路**具有广泛可迁移性：任何涉及空间相关噪声/干扰的自监督学习任务（如视频去噪、医学图像增强）均可参考 LPS 思想——利用模型预测结果定义等价类并进行局部置换。
3. **双估计量差异作为数据特性代理指标**的设计巧妙且轻量：用两种极端配置的训练结果差异来估计数据的未知属性（此处为噪声相关性），避免了额外的网络或模块，值得在其他自适应方法中借鉴。
4. **集成多次随机遮挡预测**（Eq.11）作为最终输出的方式与 DropPath/MC Dropout 思想一致，可作为自监督场景下提升泛化性的通用后处理技巧。

## 关键术语表
**Blind Spot Denoising (BSD)**：一种自监督去噪框架，通过在输入中隐藏部分像素并要求网络预测被隐藏像素，利用信号的空间相关性而噪声的独立性实现去噪。
**Local Pixel Shuffling (LPS)**：MASH 提出的去相关技术，在伪干净图像识别的平坦区域内对相同强度像素进行局部随机置换，以破坏噪声的空间相关性。
**Adaptive Masking**：根据估计的噪声相关性水平自动选择最优遮挡比（低/中/高）的机制，替代固定遮挡策略。
**Noise Correlation Gap (ε)**：用高低两种遮挡比下的噪声水平估计值之差来量化输入图像中噪声的相关程度。
**Pseudo-clean Image**：在 LPS 中由中间训练阶段模型输出的去噪图像，用作识别平坦区域和定义洗牌集合的参考。
**SNR (Signal-to-Noise Ratio) 估计**：通过重建残差计算的噪声标准差估计，用于判断噪声水平和指导自适应策略。

## 可复现要素
- **数据集**：SIDD（公开，https://engine.nikdai.com/sidd/）、FMDD（公开）、PolyU（公开）
- **代码**：开源，项目页面 https://hamadichihaoui.github.io/mash（论文声明）
- **权重**：论文未提及预训练权重，网络从头训练
- **关键超参**：$\tau^{high}=0.8, \tau^{low}=0.2, \tau^{medium}=0.5$；$\varepsilon^{low}=1.5, \varepsilon^{high}=2.5$；洗牌块大小 $s=4$；迭代步数 $N=800$，前段 $N_1$ 未明确（需参考补充材料）；集成预测数 $K=10$；优化器 Adam + cosine annealing
- **网络架构**：与 Noise2Noise 相同（UNet-like 结构）
