---
title: "CAMEL-CAusal-Motion-Enhancement-tailored-for-Lifting-Text-dr"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_CAMEL_CAusal_Motion_Enhancement_Tailored_for_Lifting_Text-driven_Video_Editing_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:50:48"
field: "文本驱动视频编辑"
keywords: ["文本驱动视频编辑", "运动连贯性", "频率分解", "扩散模型", "运动提示", "因果滤波器"]
innovations: ["可学习运动提示（Motion Prompts）：通过直接优化生成无条件embedding作为运动条件注入扩散模型", "因果运动增强注意力（CAM-Attn）：可学习因果滤波器解耦高/低频分量并协同增强运动连贯性", "频率域解耦框架：首次系统性将时间维度频率分解用于文本驱动视频编辑的运动-外观解耦"]
benchmarks: ["TGVE-DAVIS", "TGVE-Videvo", "UMTScore", "CLIP帧一致性"]
---

# 论文速读：CAMEL: CAusal Motion Enhancement tailored for Lifting Text-driven Video Editing

## 一句话总结
本文提出 CAMEL，一种基于频率分解的文本驱动视频编辑方法，通过解耦运动与外观内容，引入可学习运动提示（Motion Prompts）和因果运动增强注意力（CAM-Attn），在保持预训练 T2I 模型外观泛化能力的同时显著提升生成视频的运动连贯性。

## 研究问题与动机
- **运动与外观纠缠问题**：现有方法（Tune-A-Video、Text2Video-Zero 等）将运动模式与外观内容交织建模，导致网络要么过度拟合外观而丢失运动特征，要么只关注运动而牺牲对创造性文本提示的内容泛化能力。
- **传统 Haar 小波窗口过窄**：经典 Haar 小波变换仅用 2 元宽窗口，高维运动模式（如大范围位移、长时间跨度动作）无法被准确捕捉，导致运动连贯性退化。
- **严格空间约束限制自由度**：ControlVideo 等方法依赖深度图或边缘图等预定义运动信号进行空间约束，极大限制了运动动力学的自由度。
- **多对象编辑闪烁问题**：现有方法在多对象场景下（多人、多车辆）产生严重闪烁和不一致，缺乏有效的跨帧时序一致性保障机制。

## 核心贡献（创新点）
1. **可学习运动提示（Motion Prompts）**：通过直接优化生成单一无条件 embedding，以跨注意力方式注入扩散模型潜表示，与 CLIP 文本 embedding 作用类似但专用于运动条件引导——本质区别在于运动提示不依赖外部结构信号，而是从视频模板自身动态中学习。
2. **因果运动增强注意力（CAM-Attn）**：引入可学习的因果运动滤波系数（窗口宽度 σ=11），替代固定 Haar 小波，实现高/低频分量解耦后再加权融合——与已有工作相比，滤波系数端到端可优化且受因果机制约束，更贴合人类对运动模式的感知认知。
3. **解耦-重加权频率框架**：将视频潜表示沿时间维度分解为高频（运动）与低频（外观）两部分，分别增强后重新组合，使运动连贯性与文本泛化能力得以协同提升——这是首次将频率分解思想系统性引入文本驱动视频编辑的任务设定。

## 方法详解
**整体框架**：基于预训练 Stable Diffusion v1.4，冻结 T2I UNet 权重，仅优化 CAM-Attn 模块及运动提示，参数量极少（参数高效微调范式）。

**① Motion Prompt Learning**：初始化一个无条件 embedding $m$，经投影网络 $\tau_{\theta'}$（线性层 + LayerNorm）映射为长度 $N=32$、维度与 CLIP embedding 相同的序列特征 $\tau_{\theta'}(m) \in \mathbb{R}^{N \times d_r}$，在每一推理 timestep 通过最小化重构损失进行直接优化：

$$\mathcal{L} = \mathbb{E}_{z_0, y, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y), \tau_{\theta'}(m))\|_2^2\right]$$

**② Causal Motion Filter**：替代 Haar 小波，设计可学习因果高通滤波器：

$$z_H^\ell = \mathcal{C}_{H,\sigma}(\varphi(z^\ell)), \quad \mathcal{C}_{H,\sigma}(x)[ :, n, :] = \sum_{k=0}^{\sigma} x[ :, n-k, :] \cdot c(k)$$

其中窗口宽度 $\sigma=11$，系数 $\mathbf{c}=(c_0,\cdots,c_{11})$ 可优化，因果结构确保当前帧运动参考历史帧而非未来帧。低频分量通过互补操作获得：$\mathcal{C}_{L,\sigma}(x) = x - \mathcal{C}_{H,\sigma}(x)$。

**③ CAM-Attn 交叉注意力机制**：将解耦的高频分量与运动提示做 cross-attention：

$$Q_H^\ell = \tilde{W}_Q^\ell \cdot z_H^\ell, \quad K_m^\ell = V_m^\ell = \tilde{W}_K^\ell \cdot \tau_{\theta'}(m)$$
$$\hat{z}_H^\ell = \text{Attn}(Q_H^\ell, K_m^\ell, V_m^\ell)$$

最终重加权潜表示：$\text{CAM-Attn} := \hat{z}_H^\ell + \mathcal{C}_{L,\sigma}(\varphi(z^\ell))$。

**④ 推理阶段**：采用 DDIM inversion 获取源视频潜噪声，结合 classifier-free guidance 联合文本与运动条件：

$$\hat{\epsilon}_\theta = w \cdot \epsilon_\theta(z_t, t, \tau_\theta(y), \tau_{\theta'}(m)) + (1-w) \cdot \epsilon_\theta(z_t, t, \tau_{\theta'}(m))$$

## 实验与结果
**数据集**：TGVE-DAVIS（16 视频，来自 DAVIS）+ TGVE-Videvo（37 视频，来自 Videvo），共 53 视频，每视频 32 帧，分辨率 $480\times480$，含 1 个基准 caption 和 4 类创意文本提示（物体编辑、背景改变、风格迁移、多重编辑）。

**评估指标**：UMTScore（视频-文本对齐，以 UMT 替换 CLIP）、帧一致性（跨帧 CLIP embedding 平均余弦相似度）。

**主要结果（UMTScore，TGVE-DAVIS）**：

| 方法 | 物体 | 背景 | 风格 | 多重 |
|------|------|------|------|------|
| Tune-A-Video | 35.02 | 34.98 | 33.31 | 32.76 |
| Text2Video-Zero | 32.69 | 30.93 | 32.03 | 31.58 |
| ControlVideo | 30.98 | 30.96 | 30.66 | 30.35 |
| **CAMEL** | **36.52** | **36.60** | **34.08** | **33.92** |

CAMEL 较 Tune-A-Video 分别提升 **+1.50% / +1.62% / +0.77% / +1.16%**；TGVE-Videvo 上同样全面领先（物体 37.03 vs 35.49，背景 37.54 vs 36.15）。

**帧一致性**：CAMEL 在两个数据集上均达最高分（DAVIS 背景 95.16，Videvo 背景 97.56）。

**消融**：移除因果运动滤波器（Index-3→4）导致物体/背景编辑大幅下降；移除运动提示（Index-2→4）轻微下降；Haar 小波（Index-1）显著劣于因果滤波器（Index-2），验证了可学习宽窗口的必要性。

## 相关工作脉络
1. **Tune-A-Video [33]**：通过参数高效时序注意力微调实现单样本视频生成；CAMEL 与其核心差异在于将运动/外观显式解耦为频率分量分别处理，而非在联合潜空间中隐式学习。
2. **Text2Video-Zero [14]**：零样本视频生成，将帧级 self-attention 重编程为跨帧 attention；CAMEL 引入额外运动条件（motion prompts）进行显式引导，而非完全依赖自注意力重构。
3. **ControlVideo [41]**：依赖深度/边缘图等预定义结构信号控制运动；CAMEL 不依赖外部条件图，运动信息完全从视频模板本身提取。
4. **TokenFlow [6]**：通过中间扩散表示的语义对应保持跨帧一致性；CAMEL 从频率域入手，与 TokenFlow 的像素级对应策略形成互补视角。
5. **AnimateDiff [7]**：训练独立 motion layer 模块；CAMEL 同样参数高效但通过频率分解实现解耦，而非添加额外层级。
6. **Render-A-Video [37]**：使用光流作为先验引导层级跨帧约束；CAMEL 使用因果运动滤波器替代光学流，无需额外的光流估计步骤。

## 局限性与未来方向
- **单 GPU 训练，扩展性待验证**：全部实验在单张 A100 上进行，未探索多卡/高分辨率场景下的扩展性能。
- **固定窗口宽度 σ=11**：超参数 σ 为人工设定，针对不同运动幅度的视频可能需自适应调节，论文未讨论自动搜索机制。
- **依赖预训练 T2I 模型质量**：CAMEL 建立在 SD v1.4 之上，若源模型外观生成能力存在缺陷，CAMEL 无法弥补。
- **仅 53 视频的小规模基准**：测试集来自 LOUVE-TGVE 竞赛，规模有限，泛化到更长视频或更复杂场景需进一步验证。
- **运动提示的单 embedding 假设**：当前每个视频仅学习一个全局 motion prompt，对多主体不同运动模式（如场景中多人各自不同动作）的建模能力存疑。

## 研究启发与可借鉴点
1. **频率分解解耦运动与外观的思路可迁移**：将时空信号分解为高/低频分量分别处理的范式，可推广至图像编辑、3D 生成、视频超分等任务，作为通用正则化手段。
2. **可学习因果滤波器替代固定小波**：用可优化系数窗口替代手工设计的小波基，在时间序列分析、音频处理等领域均有借鉴价值。
3. **Motion Prompts 的概念设计精巧**：以"无条件的文本-like embedding"作为运动条件，类比 CLIP text encoder，实现简单且与现有扩散模型架构无缝集成，可复用于其他条件生成任务。
4. **实验设计完整且消融清晰**：从 Haar 小波→因果滤波器→加入运动提示的逐层消融，逻辑链条严密，值得在类似工作中借鉴。
5. **与多对象编辑的结合**：论文展示了在多对象场景下 CAMEL 的闪烁抑制效果，提示频率域方法在保持全局一致性方面具有潜在优势，可进一步探索多粒度运动分解。

## 关键术语表
- **Text-driven Video Editing**：给定视频模板和编辑文本提示，生成既符合新文本描述又保留原视频运动模式的新视频。
- **Motion Prompts**：可学习的无条件 embedding 序列，经投影网络后作为运动条件注入扩散模型，替代 CLIP 文本 embedding 发挥运动引导作用。
- **Causal Motion Filter（因果运动滤波器）**：可学习系数的因果型高通滤波器（窗口宽度 σ=11），用于沿时间维度解耦高频运动分量，受因果约束避免未来帧泄露。
- **CAM-Attn（Causal Motion-enhanced Attention）**：将解耦的高频运动分量与运动提示做交叉注意力，再与低频外观分量重加权融合的核心模块。
- **UMTScore**：以 UMT（Unmasked Teacher Model）替换 CLIP 的视频-文本对齐评估指标，与人类判断相关性更高。
- **Classifier-free Guidance**：推理时联合文本条件与无条件（仅运动条件）预测进行加权插值，增强生成视频与文本提示的对齐度。
- **Tune-A-Video**：通过单样本视频微调时序注意力层实现文本驱动视频生成的基准方法，CAMEL 与之对比的主要 baseline。

## 可复现要素
- **数据集**：TGVE-DAVIS（16 视频）和 TGVE-Videvo（37 视频），来自 CVPR 2023 LOUVE-TGVE 竞赛，论文未明确公开代码和权重。
- **基座模型**：Stable Diffusion v1.4，预训练权重公开可获取。
- **关键超参**：运动提示长度 $N=32$，滤波器窗口宽度 $\sigma=11$，微调 500 timestep，学习率 $3\times10^{-5}$，batch size=1，DDIM 采样 50 步，A100 单卡。
- **代码开源状态**：论文未声明代码/权重是否开源（*注：实际 GitHub 地址需另行核实*）。
