---
title: "360DVD-Controllable-Panorama-Video-Generation-with-360-Degre"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_360DVD_Controllable_Panorama_Video_Generation_with_360-Degree_Video_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:18:49"
field: "全景视频生成"
keywords: ["panorama video generation", "controllable diffusion model", "text-to-video", "equirectangular projection", "360-degree video"]
innovations: ["提出轻量级360-Adapter实现冻结T2V模型到全景视频的插件式适配", "设计纬度感知损失与潜在旋转/循环填充机制解决ERP投影畸变与首尾不连续问题", "构建WEB360全景视频-文本数据集并引入LLM辅助的多视角文本融合标注策略"]
benchmarks: ["User Preference Study", "WEB360 Dataset"]
---

# 论文速读：360DVD-Controllable-Panorama-Video-Generation-with-360-Degree-Video-Diffusion-Model

## 一句话总结
本文提出了首个面向可控全景视频生成的扩散模型框架 360DVD，通过轻量级 360-Adapter 将冻结的预训练 T2V 模型适配至 ERP 全景域，并设计纬度感知损失与首尾连续性增强技术，结合新构建的 WEB360 数据集实现了高质量、文本与运动条件驱动的全景视频生成。

## 研究问题与动机
- **全景视频捕获成本高昂**：360°×180° 视场角的沉浸式视频通常依赖多鱼眼相机阵列同步录制，时间与算力成本极高，亟需通过文本提示直接生成。
- **现有 T2V 模型存在领域鸿沟**：直接沿用 AnimateDiff 等标准视频扩散模型生成 ERP 视频时，因全景与标准视频在内容分布、运动规律及边界连续性上的显著差异，常出现视场狭窄、运动杂乱、首尾断裂等问题。
- **高质量带标注全景视频数据稀缺**：现有 360° 视频数据集（如 ODV360）缺乏文本描述，且直接对 ERP 图像使用 BLIP 等标注工具会因极区畸变产生误标，Caption 粒度不足。

## 核心贡献（创新点）
1. **提出 360-Adapter 插件模块**：以极低参数量接收运动条件（如光流）或零输入，将多尺度特征残差注入冻结 U-Net 编码器；与已有工作通过全量微调或 LoRA 适配特定模态的本质区别在于，该方法完全保留基础 T2V 模型的图像与时间先验，实现即插即用的域迁移。
2. **设计 360 Enhancement Techniques**：包含纬度感知损失、潜在旋转机制与循环填充机制；与以往仅关注空间连续性或依赖重训练的工作不同，本文从宏观语义（旋转）与微观像素（循环填充）两端协同保障 ERP 首尾闭环，并以余弦权重显式引导去噪过程关注低纬度失真较小区域。
3. **构建 WEB360 全景视频-文本数据集并引入 360 Text Fusion**：采集约 2,000 段高清 ERP 视频片段；与直接对整张 ERP 打标的既有管线不同，本文通过将 ERP 投影为四张 90° FoV 无畸变子图分别标注后交由 ChatGPT 融合摘要，显著提升标注准确性与场景粒度。

## 方法详解
- **基础架构**：以 Stable Diffusion v1.5 空间分支与 AnimateDiff Motion Module v14 时间分支为冻结主干，仅在 U-Net 各分辨率层级的编码器侧引入 360-Adapter。
- **360-Adapter 结构**：输入条件 $c$（尺寸 $batch \times channels \times frames \times H \times W$）先经 pixel unshuffle 下采样至 $H/8 \times W/8$，再经 2D 卷积提取初始特征；随后通过四个 Adapter 块，每块含一个下采样卷积与一个含伪 3D 残差块（pseudo-3D conv：$1 \times 3 \times 3$ 负责空间建模，$3 \times 1 \times 1$ 负责时序建模）的特征提取；最终输出多尺度特征 $\mathbf{f}_{360}^i$ 与 U-Net 中间特征相加：$\hat{\mathbf{f}}_{enc}^i = \mathbf{f}_{enc}^i + \mathbf{f}_{360}^i$。训练时以概率 $P=0.2$ 将输入置零，防止模型过度依赖运动条件。
- **训练目标**：标准潜在扩散噪声预测损失，扩展为 $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t^{1:N}, t, \tau_\theta(\mathbf{y}), \mathbf{f}_{360})\|_2^2]$。
- **360 Enhancement Techniques**：
  - **Latitude-aware Loss**：引入权重矩阵 $\mathbf{W}_{i,j} = \cos\left(\frac{2i - H/8 + 1}{H/4}\pi\right)$ 对 latent 空间像素做逐元素加权，使低/中纬度区域在去噪中获得更高梯度贡献。
  - **Latent Rotation Mechanism**：推理时每步去噪时对 $\mathbf{z}_t^{1:N}$ 与运动条件施加水平旋转角 $\theta$（实验取 $\pi/2$），训练时则以随机角度做数据增强，缓解宏观语义的首尾断裂。
  - **Circular Padding Mechanism**：仅在最后 $\lfloor T/2 \rfloor$ 步将 U-Net 卷积的 padding 替换为循环填充，确保像素级左右边界一致，同时避免早期布局阶段引入不稳定伪影。
- **数据集构建（WEB360）**：从 ODV360 提取 210 段、从 YouTube 筛选 1,904 段单场景视频，共 2,114 对。原始 ERP 标注经 BLIP+360 Text Fusion 流程重写，输出细粒度全景描述。

## 实验与结果
- **实验设置**：分辨率 $512 \times 1024$，帧数 16，batch size 1，学习率 $1 \times 10^{-5}$，训练 100k 步；推理使用 DDIM 25 步，text guidance scale 7.5，$\theta=\pi/2$。
- **对比基线**：A. AnimateDiff；B. A + 全景 LoRA (LatentLabs360)；C. B + 360ET（不含 Adapter）；D. 完整 360DVD。
- **主要结果**：31 人用户偏好研究表明，D 在五项指标上全面领先：Graphics Quality 51.6%、Frame Consistency 64.5%、End Continuity 71.8%、Content Distribution 67.0%、Motion Pattern 74.6%，较最佳基线 C 提升约 50%~60% 的偏好率。定性结果显示 360DVD 能稳定生成符合 360° 内容分布与曲线运动规律的视频，且与多种个性化 T2I 模型兼容。
- **消融结论**：移除 360TF 导致主体缺失（标注质量下降）；移除伪 3D 层引发帧间闪烁；移除纬度损失使画面清晰度与视场感略有退化。

## 相关工作脉络
- **Text-to-Video Diffusion**：AnimateDiff、Make-a-Video、MagicVideo 等依赖大空间-时间联合训练；360DVD 定位为“冻结基座+轻量适配器”的低成本域适配方案，避免重新训练海量参数。
- **Panorama Image Generation**：OmniDreamer、StitchDiffusion、PanoDiff、IPO-LDM 等聚焦静态 ERP/NFoV 图像合成；本文首次将该范式扩展至视频生成，引入时序建模与运动条件控制。
- **Controllable Video Synthesis**：MCDiff、LaMD、Video-Composer 等需针对特定条件重新训练或微调；360DVD 的 Adapter 支持零样本条件注入，保持插件特性。
- **Panoramic Captioning / Vision-Language**：传统 BLIP/CLIP 直接标注 ERP 易受极区畸变干扰；本文的 360 Text Fusion 借助多视角投影+LLM 融合的策略，与现有单图直标管线形成方法论差异。

## 局限性与未来方向
- **分辨率受限于基础 SD 架构**：实验仅展示 $512 \times 1024$，更高画质依赖外部超分模型介入，未在本框架内原生扩展。
- **数据集规模有限**：WEB360 约 2,000 段视频，与百亿参数 T2V 模型所需数据量存在数量级差距，可能限制生成多样性上限。
- **运动条件依赖外部估计器**：当前使用 PanoFlow 生成光流，未探索端到端运动生成或隐式运动先验学习。
- **未来方向**：结合超分模块实现高分辨率生成、扩展至更长时序建模、探索无需外部光流的全自动运动条件合成。

## 研究启发与可借鉴点
- **冻结基座+残差注入的插件化适配范式**：360-Adapter 的轻量设计与零概率训练策略可有效迁移至其他非标准空间/拓扑域的生成任务，避免灾难性遗忘。
- **空间域加权损失的设计思路**：纬度感知损失将几何畸变先验转化为像素级训练权重，该思路可直接复用于鱼眼、球面或任意非均匀投影的图像/视频生成。
- **循环边界处理的两阶段策略**：仅在后半程去噪阶段启用循环填充，兼顾早期布局稳定性与晚期细节一致性，可推广至任何具有环面拓扑假设的生成任务。
- **多视角投影+LLM 融合的标注管线**：360 Text Fusion 为畸变大、视野极端的数据模态提供了可复用的自动化细粒度标注范式。

## 关键术语表
- **ERP (Equirectangular Projection)**：等距柱状投影，将球面坐标线性映射为 $1:2$ 宽高比矩形的全景图像标准格式。
- **360-Adapter**：论文提出的轻量可训练模块，通过伪 3D 卷积提取运动条件特征并以残差方式注入冻结 U-Net 编码器。
- **360 Enhancement Techniques**：包含纬度感知损失、潜在旋转机制与循环填充机制的组合技术，用于弥补 ERP 投影畸变与边界不连续缺陷。
- **WEB360**：作者构建的约 2,114 段高清 ERP 视频-文本对数据集，专为全景扩散模型训练设计。
- **360 Text Fusion**：将 ERP 投影为四张 90° FoV 子图分别标注后交由 ChatGPT 融合摘要，以提升全景描述粒度与准确性。
- **Pseudo-3D Convolution**：时空分离卷积操作（如 $1\times3\times3$ 与 $3\times1\times1$），在 Adapter 中分别建模空间细节与帧间时序依赖。
- **DDIM**：Denoising Diffusion Implicit Models，本文用于加速推理的确定性采样器。

## 可复现要素
- **数据集**：WEB360（来源 ODV360 与 YouTube，论文未声明是否完全开源，项目页链接为 https://akaneqwq.github.io/360DVD/）。
- **代码/权重**：基础模型 SD v1.5 与 Motion Module v14 开源；Adapter 权重与训练脚本建议查阅项目页面获取。
- **关键超参**：分辨率 $512 \times 1024$，帧数 16，batch size 1，学习率 $1 \times 10^{-5}$，训练步数 100k，零输入概率 $P=0.2$；推理使用 DDIM 25 步、text scale 7.5、旋转角 $\theta=\pi/2$；beta schedule：$\beta_{start}=0.00085$，$\beta_{end}=0.012$。
- **运动估计器**：PanoFlow（全景光流估算）。
