---
title: "A-B-BNN-Add-Bit-Operation-Only-Hardware-Friendly-Binary-Neur"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_AB_BNN_AddBit-Operation-Only_Hardware-Friendly_Binary_Neural_Network_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:18:59"
field: "模型压缩与硬件友好推理"
keywords: ["Binary Neural Network", "Hardware-friendly", "No Multiplication", "BN-Free", "Quantized Activation", "Edge Deployment"]
innovations: ["提出 Mask Layer 吸收 β 乘法并在推理时移除", "设计 Quantized RPReLU 将斜率约束为 2 的整数次幂以位移替代乘法", "实现推理阶段乘法操作数为 0 的 BNN 架构"]
benchmarks: ["ImageNet", "CIFAR-10", "CIFAR-100"]
---

# 论文速读：A-B-BNN-Add-Bit-Operation-Only-Hardware-Friendly-Binary-Neur

## 一句话总结
本文提出了 A&B BNN，一种完全消除推理阶段乘法操作的二进制神经网络架构，通过引入可移除的 mask layer 和量化 RPReLU 结构，将传统 BNN 中的全精度乘法替换为等价的位移/位运算，在 ImageNet 上达到 66.89% Top-1 精度且乘法操作数为 0。

## 研究问题与动机
- **BNN 仍残留乘法操作**：现有先进二进制网络（如 ReActNet、Bi-ResNet）为提升精度引入了缩放因子、BN 层、PReLU、实值残差等技术，但这些技术会带来数百万次全精度乘法（MO），与硬件友好目标冲突。
- **BN 是乘法的主要来源**：BN 层引入 $c \cdot h \cdot w$ 量级乘法，虽可借助 BN-Free 架构消除，但引入的 $\alpha$ 和 $\beta$ 缩放因子仍产生乘法操作。
- **硬件成本敏感场景的需求**：在缺乏内置乘法器的嵌入式/边缘芯片上，每次乘法需多次通信往返，延迟显著；消除乘法可直接部署于纯加法器/位移器硬件。
- **精度-效率权衡不足**：现有无乘法方法（如 AdderNet）因依赖 BN 仍含乘法；已有 BN-Free BNN 虽去除了 BN，但无法彻底消除 $\alpha$、$\beta$、PReLU 带来的乘法。

## 核心贡献（创新点）
1. **Mask Layer 设计**：利用 BNN 符号函数的数学特性，将 $\beta$ 的乘法吸收进 mask layer，训练期间调控梯度传递，推理时直接移除，实现零乘法。
2. **Quantized RPReLU 结构**：将 PReLU 的每通道斜率约束为 2 的整数次幂（可学习），使非线性激活通过位移操作实现，比固定斜率 RLeakyReLU 在 ImageNet 上提升 1.14%。
3. **Bit-Operation 全替换策略**：将 $\alpha$ 乘法替换为左移位（$\alpha = 2^{-2}$），平均池化除法替换为右移位（$2 \times 2$ kernel），结合 mask layer 实现推理阶段乘法操作数降为 0。
4. **硬件效率验证**：在 Xilinx Zynq-7000 FPGA 上综合，位移操作较乘法器节省 31.9% LUTs 和 100% DSPs，PReLU 较量化后节省 43.9% LUTs 和 100% DSPs。

## 方法详解

### 基础架构：BN-Free BNN
采用 [25] 提出的 Scaled Weight Standardization (SWS) + Adaptive Gradient Clipping (AGC) 的无归一化架构，残差块形式为：
$$x_{\ell+1} = x_\ell + \alpha \cdot f_\ell(x_\ell / \beta_\ell)$$
其中 $\beta = \sqrt{\text{Var}(x_{in})}$ 用于方差归一化，$\alpha$ 控制残差方差增长率（通常取小值如 0.2）。

### Mask Layer 移除 $\beta$ 乘法
**原理**：基于 sign 函数的齐次性，对任意 $k > 0$，有 $\text{Sign}(k \cdot x) = \text{Sign}(x)$。因此可在二值化层前引入 mask layer $M(x)$，满足：
$$\text{Sign}(\text{Mask}(x)) = \text{Sign}(\text{Mask}(k \cdot x)) = \text{Sign}(x)$$
将 $\beta$ 的吸收进 mask 函数（采用 Sigmoid$(x, \delta=3)$），前向传播时梯度等价为 $f_A'(x)$，推理时直接删除该层，无需重新计算 $\xi \cdot \beta$（训练后为常数）。

### Quantized RPReLU 消除 PReLU 乘法
标准 RPReLU 负区间计算为 $y_i < 0$ 时输出 $a_i \cdot y_i + \xi_i$，引入乘法。本文将其量化为：
$$f(y_i) = \begin{cases} y_i, & y_i \geq 0 \\ 2^{\text{round}(a_i)} \cdot (y_i + \xi_{i_1}) + \xi_{i_2}, & \text{otherwise} \end{cases}$$
其中 $a_i, \xi_{i_1}, \xi_{i_2}$ 为可学习参数，斜率强制为 $2^k$（整数次幂），负区间乘法替换为位移操作。

### 其他 Bit Operation 替换
- $\alpha$ 乘法：取 $\alpha = 2^{-2}$，通过左移 2 位实现。
- 平均池化：kernel size 固定为 $2 \times 2$，除法替换为右移 2 位。

### 损失函数
使用 Distillation Loss [24]：
$$\mathcal{L}_{\text{Dis}} = -\frac{1}{n}\sum_c\sum_{i=1}^n \rho_c^{\mathcal{R}}(X_i) \times \log\left(\frac{\rho_c^{\mathcal{B}}(X_i)}{\rho_c^{\mathcal{R}}(X_i)}\right)$$
配合两阶段训练策略：第一阶段仅二值化激活，第二阶段联合二值化权重与激活。

## 实验与结果

### 数据集与模型
- **ImageNet**：ReActNet-18/34/A（ResNet-18/34 和 MobileNetv1 的增强版）
- **CIFAR-10/100**：ReActNet-18/A、XnorNet-18、BiResNet-18

### 主要结果

| 模型 | 数据集 | Accuracy | MO |
|------|--------|----------|-----|
| ReActNet-A (A&B BNN) | ImageNet | **66.89%** | **0** |
| ReActNet-34 (A&B BNN) | ImageNet | 65.19% | 0 |
| ReActNet-18 (A&B BNN) | ImageNet | 61.39% | 0 |
| ReActNet-18 (A&B BNN) | CIFAR-10 | **92.30%** | 0 |
| ReActNet-18 (A&B BNN) | CIFAR-100 | **69.35%** | 0 |

**提升幅度**：
- 相比 BN-Free ReActNet-A，减少 14.7M 乘法操作，精度损失仅 1.11%。
- 相比 BN-Free ReActNet-18，减少 4.6M 乘法操作，精度提升 0.29%。
- Quantized RPReLU 相比固定斜率 RLeakyReLU（$2^{-3}/2^{-7}$），在 ImageNet 第二步提升 1.14%。
- $\alpha = 2^{-2}$ 优于 $\alpha = 2^{-3}$，提升 0.25%~1%。

### 硬件开销（FPGA 综合）
- 乘法器：47 LUTs, 12 slices, 4 DSPs
- 位移操作：32 LUTs（↓31.9%）, 11 slices（↓8.3%）, 0 DSPs（↓100%）
- PReLU：57 LUTs, 17 slices, 4 DSPs
- 量化 PReLU：32 LUTs（↓43.9%）, 9 slices（↓47.1%）, 0 DSPs（↓100%）

## 相关工作脉络
- **BNN 基础工作 [18]**：XNOR-Net 首次实现纯逻辑运算推理，但未解决精度瓶颈。
- **Scaling Factor [19] & Real-value Residual [22]**：引入全精度乘法，牺牲硬件友好性换取精度。
- **ReActNet [24]**：提出 RSign/RPReLU 增强非线性，但仍保留乘法操作。
- **BN-Free BNN [25]**：首次将无归一化架构引入 BNN，消除 BN 乘法但引入 $\alpha/\beta$ 乘法，本文在此基础上进一步消除剩余乘法。
- **Normalizer-free 架构 [26, 27]**：Brock 等人提出 SWS+AGC 技术，是本文的基础。
- **AdderNet [44]**：使用 $\ell_1$ 距离替代卷积，但因依赖 BN 仍有乘法，非真正无乘法网络。

## 局限性与未来方向
- **大模型扩展性待验证**：仅在 ReActNet-18/34/A 上验证，未测试更深的 ResNet 变体或 Vision Transformer。
- **$\alpha$ 搜索空间受限**：目前仅测试 $2^{-2}$ 和 $2^{-3}$，更大范围的可能性能未探索。
- **CIFAR 上的非线性增益有限**：ablation 显示小数据集上 quantized RPReLU 效果与固定斜率差异不大，可能因网络复杂度限制。
- **蒸馏依赖全精度教师模型**：当前使用 ReActNet 教师的蒸馏策略，若教师本身含乘法，则训练阶段仍非完全无乘法。

## 研究启发与可借鉴点
1. **Mask Layer 思想可迁移**：利用函数齐次性吸收乘法因子的思路，可推广至其他量化场景（如低比特权重的比例因子吸收）。
2. **量化斜率的离散搜索策略**：将连续参数约束为 $2^k$ 并通过 round 操作可微学习，适用于其他需替换乘法的激活函数设计。
3. **BN-Free 架构的乘法分解**：系统分析各组件的乘法来源并逐类替换，为其他无乘法网络设计提供方法论参考。
4. **硬件感知训练策略**：结合 SWS+AGC 与位移操作约束，实现训练-推理一致的零乘法架构，可延伸至低功耗边缘部署场景。
5. **精度-效率权衡分析框架**：报告 MO 数量与精度损失的对应关系，为后续研究提供可复用的评估基准。

## 关键术语表
- **BNN (Binary Neural Network)**：权重和激活均二值化为 ±1 的神经网络，理论上可完全消除乘法。
- **BN-Free 架构**：去除 Batch Normalization 层、依赖 SWS+AGC 维持训练稳定的网络设计。
- **Mask Layer**：利用 sign 函数齐次性吸收乘法因子的中间层，训练时调控梯度，推理时移除。
- **Quantized RPReLU**：将 PReLU 斜率约束为 2 的整数次幂的可学习激活函数，通过位移实现非线性。
- **MO (Multiplication Operand)**：模型中乘法操作的总次数，本文目标将其降至 0。
- **SWS (Scaled Weight Standardization)**：对权重进行标准化以替代 BN 的归一化技术。
- **AGC (Adaptive Gradient Clipping)**：自适应梯度裁剪，防止未归一化网络训练不稳定。
- **Distillation Loss**：知识蒸馏损失，通过全精度教师引导二进制网络学习。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、ImageNet（公开）
- **代码**：开源，地址 https://github.com/Ruichen0424/AB-BNN
- **权重**：论文未提及是否公开
- **关键超参**：
  - $\lambda$（AGC 阈值）：ImageNet 用 0.02，CIFAR 用 0.001
  - $\alpha$：$2^{-2} = 0.25$（推荐）
  - $\delta$（mask sigmoid 温度）：3
  - 学习率：1e-3，线性衰减
  - weight decay：第一阶段 5e-6，第二阶段 0
  - 训练轮数：ImageNet 128 epochs，CIFAR 256 epochs
  - 优化器：Adam
  - 数据增强：随机裁剪、光照扰动、水平翻转（ImageNet）；随机裁剪、水平翻转（CIFAR）
