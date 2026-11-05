---
title: "Efficient-Multi-scale-Network-with-Learnable-Discrete-Wavele"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Efficient_Multi-scale_Network_with_Learnable_Discrete_Wavelet_Transform_for_Blind_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:58:54"
field: "图像恢复与增强"
keywords: ["motion deblurring", "multi-scale network", "wavelet transform", "blind image restoration", "SIMO architecture", "real-world blur"]
innovations: ["提出SIMO多尺度基线消除复杂融合模块", "可学习小波变换节点受完美重构条件约束", "利用方向连续性先验提升真实模糊恢复"]
benchmarks: ["RealBlur-J", "RealBlur-R", "RSBlur", "GoPro"]
---

# 论文速读：Efficient-Multi-scale-Network-with-Learnable-Discrete-Wavele

## 一句话总结
论文提出MLWNet，一种基于单输入多输出(SIMO)架构的多尺度盲运动去模糊网络，结合可学习离散小波变换(LWT)模块，在真实世界去模糊任务上同时实现了高精度与高效率的SOTA性能。

## 研究问题与动机
1. **现有MIMO多尺度方法复杂度高**：传统去粗到精方法需要复杂的多尺度特征融合模块，且在低层网络中引入人为构造的低分辨率图像对，可靠性不足。
2. **多尺度架构细节恢复瓶颈**：自顶向下的语义传递虽然语义精确但空间模糊，导致高频细节恢复能力受限。
3. **真实模糊与合成模糊分布差异**：真实世界模糊具有方向连续性（由相机曝光轨迹决定），而合成模糊轨迹不自然，现有方法在这类差异上利用不足。
4. **频率域方法的局限性**：现有频域方法（如DFT-based）更适合处理突变信号，但对真实模糊的方向性和连续性刻画不够。

## 核心贡献（创新点）
1. **SIMO多尺度基线设计**：与传统MIMO不同，保留原始高分辨率图像作为唯一输入，逐级输出不同尺度清晰图像，消除复杂融合模块，降低计算复杂度。
2. **可学习离散小波变换节点(LWN)**：将2D-DWT以分组卷积形式实现并端到端学习，自适应数据分布，同时捕捉方向连续性（水平、垂直、对角高频）和频率特征。
3. **完美重构条件自监督约束**：通过z-transform导出并约束小波核满足混叠消除和完美重构条件，防止退化为普通分组卷积。
4. **多尺度分层损失设计**：按尺度倒序加权($w_k = 1/k$)的多尺度PSNR损失，避免低尺度输出对高层恢复产生负面影响。
5. **在真实场景的卓越泛化**：在RealBlur-J/R和RSBlur等真实数据集上取得SOTA，且跨数据集泛化能力强。

## 方法详解
**架构设计（Fig.2）**：
- **Encoder阶段**：由Simple Encoder Blocks(SEB)堆叠，自上而下特征提取并下采样。输出：$E_{out}^i = \phi_i(E_{out}^{i-1})$
- **Fusion阶段**：Wavelet Fusion Blocks(WFB)自底向上融合多层语义，实现深度语义与浅层细节的融合。
- **Decoder阶段**：Wavelet Head Blocks(WHB)逐级上采样，最终输出各尺度复原图像。仅最高尺度输出在推理时使用。

**可学习小波变换（LWN）**：
- 将1D-DWT扩展至2D，通过外层积构造4个2D卷积核：
  - $\mathcal{F}_{ll} = \vec{a}_0 \times \vec{a}_0^T$（低频）
  - $\mathcal{F}_{lh}, \mathcal{F}_{hl}$（水平/垂直高频）
  - $\mathcal{F}_{hh}$（对角高频）
- 以小波核$\mathcal{K}_w$拼接后以分组卷积形式应用于输入特征图。
- 前向传播：$X_{in} \in (C, H, W) \rightarrow X_{out} \in (4C, H/2, W/2)$，分离为4个子带。
- 逆小波变换通过转置卷积完成。

**完美重构约束**：
- 引入z-transform导出两个条件：
  1. 混叠消除：$A_0(-z)S_0(z) + A_1(-z)S_1(z) = 0$
  2. 完美重构：$A_0(z)S_0(z) + A_1(z)S_1(z) = 2$
- 转化为可微损失$\mathcal{L}_{wavelet}$，以MSE形式优化。

**损失函数**：
- 多尺度损失：$\mathcal{L}_{multi} = \sum_{k=1}^{K} \frac{1}{k} \mathcal{L}_{psnr}(x_k, y_k)$
- 小波损失：$\mathcal{L}_{wavelet}(\theta_i) = (\sum_{k}^{N-1}(\langle a_0, s_0\rangle_k + \langle a_1, s_1\rangle_k) - \hat{V})^2$
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{wavelet} + \mathcal{L}_{multi}$

## 实验与结果
**数据集**：RealBlur-J/R（真实模糊）、RSBlur（最新真实模糊）、GoPro（合成模糊）

**主要结果**：
| 数据集 | 指标 | 数值 | 对比提升 |
|--------|------|------|----------|
| RealBlur-J | PSNR/SSIM | 33.02/0.933 | +0.91dB vs GRL |
| RealBlur-R | PSNR/SSIM | 40.69/0.976 | +0.49dB vs GRL |
| RSBlur | PSNR/SSIM | 34.94/0.880 | SOTA |
| GoPro | PSNR/SSIM | 33.83/0.968 | 与FFTformer SSIM差距仅0.001 |

**效率**：MLWNet-B在256×256 patch上推理时间仅0.05s，较FFTformer降低60%耗时，GMACs显著更低。

**泛化性**：在RealBlur-J训练、RSBlur测试下PSNR达30.91，优于所有基线。

**消融验证**：
- SIMO > SISO/MIMO（Table 5）
- 加入WFB+WHB+L_wavelet后PSNR提升0.25dB（Table 6）
- 无小波损失时LWN退化为普通分组卷积，无法突破细节恢复瓶颈

## 相关工作脉络
1. **MIMO-UNet (Cho et al.)**：多输入多输出单解码器架构，需人工构建低分辨率图像对；本文SIMO无需额外输入对，结构更简。
2. **FFTformer / DeepRFT (Kong/Mao)**：基于DFT的频域方法，擅长全局频率分析但对突变信号适应性差；本文LWN对小波域突变信号更优。
3. **SDWNet (Zou et al.)**：固定小波变换辅助空间特征；本文小波核可学习且受完美重构条件约束。
4. **GRL / Restormer**：当前SOTA，但参数量和计算开销较大；本文在相近精度下大幅降低推理时间和算力需求。
5. **NAFNet (Chen et al.)**：无激活简单基线；本文在其思想上扩展至多尺度+小波域。
6. **SRN / MSSNet**：经典渐进式去模糊；本文通过SIMO消除了多层间冗余特征融合。

## 局限性与未来方向
1. **合成模糊性能受限**：在GoPro等合成数据集上未达最优，因合成模糊的噪声分布和频率混合模式与真实模糊不同。
2. **噪声敏感性差异**：表7显示随噪声差异增大，GoPro性能快速下降，而RealBlur-J保持稳健，说明对合成噪声分布适配不足。
3. **小波基的选择**：当前使用Daubechies族，未来可探索其他小波基（如Haar、Symlets）或自适应选择。
4. **扩展至其他恢复任务**：SIMO+LWN框架可迁移至去噪、超分等图像恢复任务。

## 研究启发与可借鉴点
1. **SIMO设计范式**：保留原始高分辨率输入、多尺度逐级输出的架构，可推广至任意图像恢复任务，减少融合模块的设计负担。
2. **完美重构条件的自监督应用**：将数学领域的完美重构约束转化为可微损失，为可学习频域变换提供理论保障，避免退化，这一思路可迁移至其他变换域学习。
3. **方向性先验利用**：通过2D-DWT分离4个子带（ll/lh/hl/hh），显式建模模糊的方向连续性，对真实运动模糊场景有直接指导意义。
4. **多尺度加权策略**：按$1/k$倒序加权低尺度损失，避免低分辨率输出干扰高层恢复，这一设计可复用于其他渐进式恢复任务。
5. **效率-精度权衡范式**：证明简化架构+专用模块可超越复杂SOTA方法，为资源受限场景下的模型设计提供新思路。

## 关键术语表
**SIMO**：Single-Input Multiple-Output，单输入多输出架构，本文提出的多尺度基线设计。
**LWN**：Learnable Wavelet Node，可学习小波变换节点，核心模块，以分组卷积形式实现自适应2D-DWT。
**2D-DWT**：Two-Dimensional Discrete Wavelet Transform，二维离散小波变换，将图像分解为低频和三个方向的高频子带。
**完美重构条件**：小波变换中保证原始信号可无损恢复的数学约束，本文转化为自监督损失约束小波核学习。
**混叠消除条件**：下采样过程中避免频率混叠的条件，与完美重构条件共同约束分析/合成滤波器对。
**WFB/WHB**：Wavelet Fusion Block / Wavelet Head Block，分别用于多尺度语义融合和逐级恢复的核心模块。
**RealBlur**：包含RealBlur-J和RealBlur-R两个子集的真实世界运动模糊数据集，标注了模糊轨迹方向。
**RSBlur**：2022年提出的包含极端光照条件的真实模糊数据集，用于评估跨场景泛化能力。

## 可复现要素
- **数据集**：RealBlur-J/R、RSBlur、GoPro（均为公开数据集，需申请下载）
- **代码**：已开源于 https://github.com/thqiu0419/MLWNet
- **超参数**：AdamW优化器(lr=1e-3, cosine annealing)，patch size=256×256，batch size=8，训练600K iterations
- **实现细节**：WFB和WHB中LWN的分组卷积维度、SEB中1×1 conv与3×3 depth-wise conv的配置、小波核长度N需参考代码
