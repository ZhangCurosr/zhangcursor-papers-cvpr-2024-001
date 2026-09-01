---
title: "Efficient-Multi-scale-Network-with-Learnable-Discrete-Wavele"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Efficient_Multi-scale_Network_with_Learnable_Discrete_Wavelet_Transform_for_Blind_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:51:39"
field: "低层视觉-图像恢复"
keywords: ["盲运动去模糊", "多尺度网络", "可学习小波变换", "SIMO架构", "频域处理"]
innovations: ["提出SIMO多尺度基线架构替代MIMO降低计算复杂度", "设计可学习离散小波变换模块增强高频细节恢复", "引入完美重建约束的小波损失防止核退化"]
benchmarks: ["RealBlur-J", "RealBlur-R", "RSBlur", "GoPro"]
---

# 论文速读：Efficient-Multi-scale-Network-with-Learnable-Discrete-Wavele

## 一句话总结
本文提出 MLWNet，一种基于单输入多输出（SIMO）架构的可学习离散小波变换多尺度网络，用于高效盲运动去模糊；通过引入自适应小波变换模块与多尺度损失，在真实场景去模糊任务上达到最优精度与效率。

## 研究问题与动机
1. **现有MIMO多尺度算法复杂度高**：传统多尺度去模糊采用多输入多输出（MIMO）架构，需手动构建低分辨率图像对并引入复杂融合模块，导致计算冗余。
2. **多尺度细节恢复能力不足**：低分辨率层传递的特征"语义精确但空间模糊"，高频细节恢复受限，现有方法缺乏对方向连续性特征的利用。
3. **真实模糊与合成模糊分布差异大**：真实世界模糊具有方向连续性（轨迹局部连续），而合成模糊（如GoPro数据集）存在不连续轨迹和高低频混叠，现有方法在真实场景泛化性不足。

## 核心贡献（创新点）
1. **SIMO多尺度基线架构**：采用单输入多输出策略替代MIMO，保留原始分辨率输入并逐尺度生成恢复图像，消除跨分辨率融合模块的计算复杂度。
2. **可学习小波变换节点（LWN）**：将2D离散小波变换（2D-DWT）与分组卷积结合，自适应学习数据分布与特征空间，分解不同方向与频率分量以提升细节恢复。
3. **完美重建约束的小波损失（L_wavelet）**：引入Z变换条件下的混叠消除与完美重建损失，防止小波核退化为普通分组卷积。
4. **多尺度损失设计（L_multi）**：按分辨率层级分配权重（w_k = 1/k），平衡各尺度监督信号，避免低分辨率输出对最终结果产生负面影响。

## 方法详解
**整体架构（MLWNet）**：采用编码器-解码器结构，分为三个阶段（图2）：
- **Encoder阶段**：由Simple Encoder Block（SEB）堆叠，信息自上而下流动，每个块后下采样一次。公式(1)描述第i层输出。
- **Fusion阶段**：由Wavelet Fusion Block（WFB）构成，信息自下而上融合多尺度语义。公式(2)描述第i层输出。
- **Decoder阶段**：由Wavelet Head Block（WHB）构成，渐进上采样并逐尺度生成去模糊图像。公式(3)描述第i层输出。

**可学习小波变换（LWN）**：
- 通过可学习滤波器向量 $\vec{a_0}$（低通）和 $\vec{a_1}$（高通）的outer product构建2D小波核：$\mathcal{K}_w = \text{cat}(\mathcal{F}_{ll}, \mathcal{F}_{lh}, \mathcal{F}_{hl}, \mathcal{F}_{hh})$（公式5）。
- 前向小波变换将特征从空间域投影到小波域（低频、水平高频、垂直高频、对角高频），经深度可分离卷积（扩张率r）后由可学习逆小波变换还原至空间域输出。

**损失函数**：
- 多尺度损失：$\mathcal{L}_{multi}(x,y) = \sum_{k=1}^{K} w_k \times \mathcal{L}_{psnr}(x_k, y_k)$，其中 $w_k = 1/k$。
- 小波损失：基于完美重建条件（公式6、7），转换为卷积等价形式（公式9）：$L_{wavelet}(\theta_i) = (\sum_k (\langle a_0, s_0 \rangle_k + \langle a_1, s_1 \rangle_k) - \hat{V})^2$。
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{wavelet} + \mathcal{L}_{multi}$。

## 实验与结果
**数据集**：
- 真实模糊：RealBlur-J/R（测试）、RSBlur（训练）
- 合成模糊：GoPro

**评估指标**：PSNR、SSIM、计算复杂度（GMACs）、推理速度

**主要结果**：
- **RealBlur-J**：MLWNet-B达到33.84 dB / 0.941 SSIM，较SOTA GRL提升0.91 dB，推理时间仅0.05s。
- **RealBlur-R**：40.69 dB / 0.976 SSIM，优于GRL-B 0.49 dB。
- **RSBlur训练→RealBlur-J测试**（泛化性验证）：30.53 dB / 0.905 SSIM，超越所有对比方法。
- **GoPro（合成）**：33.83 dB / 0.968 SSIM，与FFTformer（0.001 SSIM差距）相当但推理速度快60%。

**消融实验**：
- SIMO架构较SISO/MIMO在RealBlur-J上PSNR提升0.08-0.18 dB（表5）。
- LWN模块使PSNR提升0.25 dB（表6）。
- 小波损失不可或缺：无L_wavelet时WFB+WHB仅提升0.03 dB，证明小波核会退化为普通卷积。

## 相关工作脉络
1. **MIMO多尺度去模糊**：MIMO-UNet系列[3]采用多输入单编码器模拟多尺度，仍需融合模块；本文SIMO架构更简洁，无需降采样图像对。
2. **单尺度去模糊SOTA**：NAFNet[2]、FFTformer[10]等通过注意力机制提升精度但计算成本高；本文在多尺度框架下实现效率-精度平衡。
3. **频域方法**：DeepRFT[19]、SDWNet[42]引入DFT处理高频成分；本文选择小波变换更适合处理含突变信号的真实图像（纹理边缘）。
4. **小波网络应用**：MLWCNN[16]、自适应小波池化[34]等将小波用于去噪/超分；本文首次将可学习小波应用于盲运动去模糊并与多尺度架构深度融合。

## 局限性与未来方向
1. **合成模糊性能不及真实模糊**：GoPro数据集上未达到最优（与FFTformer差距0.001 SSIM），归因于合成模糊的不自然轨迹和高低频混叠。
2. **噪声敏感性差异**：随着清晰-模糊图像对噪声差异增大，GoPro性能下降快于RealBlur-J，表明对合成噪声建模有限。
3. **小波核长度限制**：公式9中N取值影响频率分辨率，论文未深入讨论最优核长的选择策略。
4. **潜在扩展方向**：可探索小波变换与其他频率域方法（如DFT）的融合；适应更多恢复任务（去噪、去雨）的通用性验证。

## 研究启发与可借鉴点
1. **SIMO架构的轻量化价值**：单输入多输出设计简化了多尺度网络，避免MIMO的重复特征提取与复杂融合，可迁移至图像超分、去噪等任务。
2. **可学习小波变换的工程实现**：通过分组卷积实现小波变换并施加完美重建约束，既保留理论保证又具备端到端学习能力，为频域模块设计提供范式。
3. **多尺度损失权重设计**：按分辨率倒数分配权重（w_k=1/k）缓解低分辨率监督干扰，类似思想可用于渐进式生成任务。
4. **真实/合成模糊差异分析**：论文系统性对比两类模糊的频率特性与噪声分布差异，启发后续工作应在真实数据集上验证泛化性。

## 关键术语表
**SIMO (Single-Input Multiple-Output)**：单输入多输出架构，网络接收原始分辨率图像，在训练时逐尺度生成恢复图像并计算多尺度损失，推理时仅输出最高尺度结果。

**LWN (Learnable Wavelet Node)**：可学习小波变换节点，将2D-DWT与分组卷积结合，通过可学习滤波器自适应分解特征的空间-频率信息。

**Perfect Reconstruction (完美重建)**：小波变换中要求低通/高通滤波器组合能使信号无损重建的条件，本文将其转化为卷积约束损失。

**Z-transform aliasing cancellation (Z变换混叠消除)**：下采样引起的频谱混叠需在频域满足 $A_0(-z)S_0(z) + A_1(-z)S_1(z) = 0$，确保小波核学习符合理论约束。

**Wavelet Loss (小波损失)**：基于完美重建条件构造的自监督损失，防止可学习小波核退化为普通分组卷积。

**GMACs (Giga Multiply-Accumulate Operations)**：十亿次乘加运算数，衡量模型计算复杂度的标准指标。

## 可复现要素
- **数据集**：RealBlur [23]、RSBlur [24]、GoPro [20]（均为公开数据集）
- **代码**：开源，GitHub: https://github.com/thqiu0419/MLWNet
- **预训练权重**：论文未明确声明，代码仓库应包含
- **关键超参**：
  - 优化器：AdamW (α=0.9, β=0.99)
  - 迭代次数：600K
  - 初始学习率：10⁻³（cosine annealing调度）
  - Patch大小：256×256
  - 数据增强：随机翻转、旋转
- **训练环境**：论文未提及具体GPU型号，建议参考代码仓库
