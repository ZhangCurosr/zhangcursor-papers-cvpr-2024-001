---
title: "Learning Continuous 3D Words for Text-to-Image Generation"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Cheng_Learning_Continuous_3D_Words_for_Text-to-Image_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:16:58"
field: "文本到图像生成与可控合成"
keywords: ["text-to-image generation", "continuous control", "3D attribute learning", "diffusion models", "concept customization", "attribute disentanglement"]
innovations: ["用MLP将连续3D属性映射为token embedding实现细粒度滑杆控制", "两阶段训练解耦物体身份与3D属性", "ControlNet渲染增强+负向提示实现单mesh跨物体泛化"]
benchmarks: ["用户研究（20人，60+问题/设置）", "ControlNet v1.1 基线对比", "Zero-1-to-3 对比"]
---

# 论文速读：Learning Continuous 3D Words for Text-to-Image Generation

## 一句话总结
本文提出 Continuous 3D Words，通过在文本到图像扩散模型中引入特殊 token，用单个 3D 网格配合渲染引擎即可学习光照方向、姿态、相机参数等**连续 3D 属性**的细粒度控制，支持与文本提示词联合使用且推理无额外开销。

---

## 研究问题与动机
1. **文本控制的抽象性局限**：现有 text-to-image 扩散模型仅支持高层语义文本控制，无法精确操控抽象连续属性（如"光照方向旋转 11°"或"鸟翼角度微调"）。
2. **3D 渲染的专业门槛**：3D 渲染引擎虽能实现细粒度物理控制，但构建详细 3D 场景需要专业知识，普通用户难以使用。
3. **离散 token 方案不可行**：对连续属性使用离散 token 需要数百个 token 近似，训练困难且无法支持推理时插值。
4. **训练数据稀缺**：标注精确 3D 属性的真实图像极为稀少，预训练模型对这些属性的知识不够精确。

---

## 核心贡献（创新点）
1. **Continuous 3D Words 概念**：用 2 层 MLP（输入含 positional encoding 的连续属性 → 输出 token embedding）将 3D 属性连续映射到文本 embedding 空间，支持任意滑杆值控制。*与已有工作的本质区别：将属性作为连续函数学习而非离散 token 近似，且支持多属性联合控制与推理时插值。*
2. **两阶段训练策略**：先以 Dreambooth 方式学习物体唯一标识符 [Obj]，再冻结 [Obj] 独立学习各属性 MLP g_φ(a)，防止模型将同一物体的不同属性值编码为不同物体。*与已有工作的本质区别：显式解耦物体身份与属性，使学到的属性可泛化到新物体。*
3. **ControlNet 背景增强**：用预训练 Depth ControlNet（形状变化类属性）和 Lineart ControlNet（光照/阴影类属性）自动生成多样背景图像作为数据增强，防止过拟合渲染引擎的纯白背景。*与已有工作的本质区别：无需真实 3D 场景即可通过轻量渲染 + ControlNet 实现背景多样化。*
4. **推理时负向提示 [Obj]**：将物体标识符作为 negative prompt（替换 classifier-free guidance 中的 null-text），进一步抑制生成图像中包含训练物体，强化属性与身份的解耦。*与已有工作的本质区别：用简单负提示 trick 提升泛化性，无需额外网络结构。*

---

## 方法详解

### 3.1 问题定义
图像 I 是物体 O、类别 C 及多个属性 a_i（形状、材质、旋转/平移、相机参数、形变等）的函数：I = f(a_1, a_2, ..., a_n)。目标是学习属性映射而非物体实例。

### 3.2 连续控制核心公式
传统方法对每个离散值 x 分配 token T_x；本文改为学习连续函数：

$$g_\phi(\mathbf{a}): \mathcal{D} \rightarrow \mathcal{T}$$

其中每个属性 a 先经 positional encoding 升至高频空间，再输入 2 层 MLP 输出 Continuous 3D Word（即 token embedding）。训练目标为：

$$\arg\min_{\theta, \phi} \mathbb{E}\left[\|S_\theta(\hat{I}_{\epsilon, \mathbf{a}}, P(g_\phi(\mathbf{a}))) - I_\mathbf{a}\|_2^2\right]$$

### 3.3 两阶段解耦训练
直接优化式(2)会产生退化解（模型将同一物体不同属性值视为不同物体）。解决方案：

- **Stage 1**：对所有渲染图像使用相同 prompt P([Obj])，学习物体标识符 T_O 及扩散模型参数 θ（Dreambooth 方式）。
- **Stage 2**：冻结 T_O，使用 prompt P([Obj], g_φ(a)) 联合优化 θ 和 MLP 参数 φ，使属性信息被 MLP 捕获而非进入 [Obj]。

### 3.4 ControlNet 增强策略
两种增强路径（见原文 Figure 3）：
- **形状变化属性**（如 wing pose）：渲染 ground-truth depth map → Depth ControlNet 生成多样背景。
- **非形状变化属性**（如 illumination）：渲染无纹理图像 → lineart extractor 提取 sketch → Lineart ControlNet 保留阴影/明暗细节。

### 3.5 推理技巧
- **负向提示 [Obj]**：在 DDIM 采样的每个去噪步，将 classifier-free guidance 中的 null-text embedding 替换为 T_O，抑制训练物体再现。
- **LoRA 微调**：仅微调 denoising U-Net 和 text encoder，模型仅约 6MB，单 A10 GPU（~16GB）即可训练 3–4 小时（15k–20k steps）。

---

## 实验与结果

### 实验设置
- ** backbone**：Stable Diffusion v2.1
- **训练数据**：单个狗 mesh（光照/姿态）、单个动画鸽子 mesh（翅膀姿态/姿态）、五个 Pix3D 椅子 mesh（dolly zoom）
- **基线**：ControlNet v1.1（strength=0.5 / 1.0），从训练集检索对应条件图

### 用户研究（20 名参与者，>60 个问题/设置）

| 控制类型 | ControlNet (1.0) | ControlNet (0.5) | **Ours** |
|---|---|---|---|
| 光照 [☀] | 28.3% | 10.0% | **61.7%** |
| 翅膀姿态 [🦅] | 16.2% | 28.8% | **55.0%** |
| 翅膀姿态 + 姿态 [🦅]+[↩] | 35.0% | 12.5% | **52.5%** |
| 光照 + 姿态 [☀]+[↩] | 15.0% | 32.5% | **52.5%** |
| **平均** | **23.6%** | **21.0%** | **55.4%** |

- **平均排名**：Ours 2.43 ± 0.70，显著优于 ControlNet(1.0) 的 1.84 ± 0.72 和 ControlNet(0.5) 的 1.73 ± 0.70。
- **核心结论**：在所有设置中 Our 方法均为第一名（>50% 偏好率），且无需调参。

### 泛化能力
- 仅用单只狗 mesh 学习的光照 + 姿态，可泛化生成马、出租车、北极熊等不同物体。
- 多属性联合控制（Figure 5）显示：固定一个属性改变另一个不会破坏图像质量。

### 真实图像编辑（Figure 6）
- 先用 Dreambooth 将真实图像编码为稀有 token，再结合 Continuous 3D Words 编辑，可保持原图外观同时修改属性。
- 对比 Zero-1-to-3：本方法无需多步 pipeline（分割→补全→合成），避免误差累积，且支持的属性类型更多。

### 插值能力（Figure 7）
- 连续 MLP 方案可直接插入中间属性值实现平滑过渡；对比 18 个离散 token 方案，离散方案在插值点生成结果错误甚至完全偏离。

### Ablation（Figure 8）
- **去掉两阶段训练**：prompt 为 "truck" 时仍生成狗，身份未解耦。
- **去掉 ControlNet 增强**：背景过拟合纯白渲染图，无法生成真实背景。
- **去掉负向提示 [Obj]**：生成 Truck 时残留狗的形状和阴影，轻微劣化。

---

## 相关工作脉络
1. **ControlNet (Zhang et al., ICCV 2023)**：通过零卷积提供 depth/canny/sketch 等条件控制，但条件图像需逐帧从训练集检索，无法支持推理时连续调节；本文直接在学习的 token 中编码连续属性，推理时无需检索。
2. **Zero-1-to-3 (Liu et al., 2023) / DreamSparse (Yoo et al., 2023)**：专注于视角编辑，依赖大规模 3D 数据集；本文仅需单个 mesh，且支持光照、姿态、dolly zoom 等多种连续属性联合控制。
3. **Textual Inversion (Gal et al., ICLR 2022) / Dreambooth (Ruiz et al., CVPR 2023)**：学习特定物体实例的 token；本文学习的是跨物体可泛化的抽象 3D 属性概念。
4. **ViewNETI (Burgess et al., 2023)**：首个将视角作为概念学习的并发工作；本文认为 2D 扩散模型的 3D 感知远超视角，可联合学习光照+姿态+相机参数等多种 3D-aware 概念。
5. **Custom Diffusion (Kumari et al., CVPR 2023) / Cones2 (Liu et al., 2023)**：多概念个性化生成；本文关注跨物体的通用属性泛化而非多实例个性化。

---

## 局限性与未来方向
1. **风格控制失效**：当 prompt 要求特定画风（如 "Monet painting"）时，模型难以完整呈现风格（Figure 10a）。
2. **属性-形状混淆**：当训练属性与目标物体属性相似时，仍会残留训练物体痕迹（Figure 10b：T-Rex 出现四条腿而非两脚站立，与训练狗的站立姿态混淆）。
3. **单 mesh 的类别局限**：虽能泛化到语义相近物体，但对差异极大的类别（如狗→火车）的泛化能力有限，dolly zoom 训练使用了 5 个椅子 mesh 才获得较好效果。
4. **用户偏好偏差**：用户研究中发现，部分用户会优先选择严格满足条件但物理合理性差的图像（Figure 9），说明自动评估指标仍需完善。

---

## 研究启发与可借鉴点
1. **MLP 映射连续属性到 token 空间**的设计可直接迁移至视频生成、3D 生成等领域，将任何连续参数（时间、温度、运动速度等）编码为可控 token。
2. **两阶段解耦训练策略**（先学身份再学属性）具有普适价值：凡是需要将"共性属性"与"个体身份"分离的任务（如风格迁移、属性编辑）均可借鉴。
3. **ControlNet 渲染增强 pipeline**（depth/lineart → ControlNet 生成多样背景）为少样本 finetuning 提供了低成本的数据增强方案，可在其他个性化生成任务中复用。
4. **负向提示解耦技巧**（将已学 token 作为 negative prompt）实现简单且有效，可作为 finetuning 流程的标准组件推广。
5. **本团队可探索**：将 Continuous 3D Words 与 3D Gaussian Splatting / NeRF 结合，实现文本驱动的真实 3D 场景属性编辑；或将属性 MLP 扩展为条件 diffusion transformer 的 cross-attention 注入，探索更大规模预训练下的属性学习能力。

---

## 关键术语表
- **Continuous 3D Words**：特殊设计的 token embedding，由 MLP 根据连续 3D 属性值动态生成，用于在文本提示中实现细粒度 3D 控制。
- **Positional Encoding**：将连续属性值映射到高维正弦频率空间，使 MLP 能更好地学习连续函数的局部变化。
- **两阶段训练（Two-stage Training）**：第一阶段用 Dreambooth 学习物体唯一标识符，第二阶段冻结标识符学习属性 MLP，实现身份与属性的解耦。
- **ControlNet Augmentation**：利用预训练 ControlNet 将单一渲染图扩展为多样背景和纹理，防止 finetuning 过拟合人工背景。
- **Negative Prompting [Obj]**：推理时将物体标识符作为 negative prompt 输入，抑制生成图像中出现训练物体，强化属性泛化。
- **LoRA（Low-Rank Adaptation）**：低秩适配技术，仅微调 U-Net 和 text encoder 的低秩矩阵，大幅降低显存和训练成本。
- **Dolly Zoom**：推拉变焦效果（背景缩放而主体大小不变），用椅子 mesh 学习并演示。
- **Classifier-Free Guidance**：扩散模型推理时的条件控制机制，本文将其中的 null-text 替换为物体标识符实现负向提示。

---

## 可复现要素
- **数据集**：单只狗 mesh、单个动画鸽子 mesh、Pix3D 五把椅子 mesh；**未公开**，仅作者内部 mesh 文件。
- **代码**：项目主页 https://ttchengab.github.io/continuous3dwords，论文未明确说明 GitHub 仓库链接。
- **权重**：LoRA 权重约 6MB，论文未提供公开下载链接。
- **关键超参**：Stable Diffusion v2.1 backbone；LoRA rank 未提及；训练 15k–20k steps；单 A10 GPU ~16GB 显存；ControlNet v1.1 官方实现；guidance strength 本文无需调参。
- **渲染引擎**：未具体指定（使用常规渲染引擎 + lineart extractor [6]）。
