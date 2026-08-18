---
title: "CosmicMan-A-Text-to-Image-Foundation-Model-for-Humans"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_CosmicMan_A_Text-to-Image_Foundation_Model_for_Humans_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:15:33"
---

# 论文速读：CosmicMan-A-Text-to-Image-Foundation-Model-for-Humans

## 一句话总结
本文提出面向人类图像生成的专用基础模型 CosmicMan，通过人机协同数据飞轮 Annotate Anyone 构建 600 万张高分辨率细粒度标注数据集 CosmicMan-HQ 1.0，并在此基础上设计仅对 Stable Diffusion 轻微改动的 Daring 训练框架与 HOLA 对齐损失，实现了高保真、密集文本-图像精准对齐的人类图像生成，全面超越现有通用 T2I 基础模型。

## 研究问题与动机
- 通用文本生成图像模型（SD、DALLE、MidJourney 等）在生成人类图像时普遍存在解剖结构失真、服装细节错位、密集属性描述与图像像素对齐困难等固有缺陷。
- 传统人形生成/编辑/3D 重建任务多依赖孤立、小尺度、分布偏倚的领域数据，难以泛化至真实世界中多样化的身份、外观与几何形态。
- 现有大规模图文对数据集（如 LAION-5B）虽推动通用模型发展，但图像质量参差不齐、标注噪声高，且缺乏针对人类细粒度属性的精确标签。
- 缺乏一个架构轻量、易于集成到下游任务、且无需推理期额外空间条件即可直接生成高质量真人类内容的专用基础模型。

## 核心贡献（创新点）
1. **提出 Annotate Anyone 人机协同数据生产范式**。通过流动数据池与人在回路的迭代标注构建持续更新的数据飞轮，与传统静态数据集或纯人工/纯自动标注范式相比，以仅 1% 的人工标注成本实现 VLM 准确率 30% 以上的持续提升。
2. **构建 CosmicMan-HQ 1.0 大规模高质量人体数据集**。收录 600 万张平均分辨率 1488×1255 的真人类图像，提供 70 类细粒度标签与 1.15 亿属性标注（含边界框、关键点、人体解析、美学评分等），填补了公开数据集中高分辨率、多模态精确标注真人类数据的空白。
3. **设计 Daring 训练框架与 HOLA 跨注意力对齐损失**。在几乎不修改 SD 架构的前提下，将密集文本显式离散化为与人体结构对应的固定组别，并通过监督跨注意力图的空间响应解决密集概念对齐难题，区别于依赖推理期梯度修改或训练无关后处理的现有方法。

## 方法详解
- **Annotate Anyone 数据飞轮**：数据源由两部分构成——回收学术数据集（LAION-5B、SHHQ、DeepFashion）与实时抓取互联网 API（Flickr、Unsplash、Pixabay）；过滤环节剔除假人、低质量图像。标注采用人在回路迭代：以 InstructBLIP 为 AI 基座，用评估集 $I_e$ 识别长尾难例类别（准确率<85% 的标签），人工仅标注此类，微调后 AI 自动标注全量池数据，形成持续迭代的闭环。
- **文本-图像数据离散化（Data Discretization）**：基于人体自然结构先验，将密集文本 $C$ 划分为 $C=\{C_{body}, C_{outfit}\}$。利用人体解析掩码 $\bar{H}=\{h_i\}_{i=1}^N$（$h_1$ 为前景聚合，其余为服装/部位细粒度区域），将文本按起止索引切片为子 caption $c_{(s_i, e_i)}$，实现文本概念与图像语义区域的硬绑定。
- **Daring 框架与 HOLA 损失**：在标准 SD 去噪损失 $\mathcal{L}_{noise} = \mathbb{E}_{z,c,\epsilon,t}[||\epsilon - \epsilon_\theta(z_t, t, c)||_2^2]$ 基础上，将跨注意力图 $M$ 按文本组别分解为 $(m_{(s_1,e_1)}, ..., m_{other})$，设计：
  $$\mathcal{L}_{\text{HOLA}} = \frac{1}{N}\sum_{i=1}^{N}\left(\sum_{j=s_i}^{e_i}||m_j - h_i||_2^2 + \left\|\frac{1}{e_i-s_i}\sum_{j=s_i}^{e_i}(m_j) - h_i\right\|_2^2\right)$$
  第一项以人体结构为先验，迫使各概念特征的高响应区贴近对应语义区域；第二项约束同组内（如服装属性）平均注意力图与语义图对齐，减少歧义。总损失 $\mathcal{L} = \alpha\mathcal{L}_{noise} + \beta\mathcal{L}_{\text{HOLA}}$，无需增加额外网络模块即可端到端训练。

## 实验与结果
- **数据集与基线**：训练使用 CosmicMan-HQ 1.0；测试集为 2048 张带细粒度人工标注提示的人像图像。基线涵盖 SD-1.5/2.0、SDXL、DeepFloyd-IF、DALLE2/3、MidJourney 及 HumanSD、Text2Human 等。
- **评估指标**：FID、HPSv2、CLIP-Score、细粒度对齐指标 $Acc_{obj}/Acc_{tex}/Acc_{shape}/Acc_{all}$（受 DSG 启发）。
- **核心结果**：CosmicMan-SDXL 取得 FID 35.42（较 SDXL 相对提升 27.13%），$Acc_{all}$ 83.6，全面优于所有对比模型。相比 DALLE-3，物体、纹理、形状及整体对齐准确率分别提升 7.54%、1.38%、15.97%、7.46%。CosmicMan-SD 的 $Acc_{all}$ 为 81.2，FID 为 36.78。
- **消融结论**：仅换用 CosmicMan-HQ 微调 SD 即可使 FID 降低 10.52、$Acc_{all}$ 提升 6.3；叠加 HOLA 损失后进一步改善 FID 0.79、$Acc_{all}$ 1.5。数据规模从 1M 扩至 6M 带来 FID 提升 2.51；标注质量从 AltText 升级至 Annotate Anyone 产出带来 $Acc_{all}$ 提升 3.2~3.6。
- **下游应用**：2D 人体编辑（T2I-Adapter）中 FID 降至 37.62，$Acc_{all}$ 达 82.9；3D 人体重建（Magic123）中 CLIP-Sim 提升至 0.88，用户偏好率达 73.64%（对比 SD 基线的 26.36%）。
- **最强结果**：CosmicMan-SDXL 在生成质量与细粒度对齐上均获最优，FID 35.42，$Acc_{all}$ 83.6，用户偏好度 93.06%（质量）/ 85.38%（对齐）。

## 相关工作脉络
- **Stable Diffusion / SDXL**：通用潜扩散底座，依赖 noisy 大规模图文对；本文保留其架构与下游生态兼容性，仅通过数据质量与注意力监督实现垂直专用化跃升。
- **HumanSD / Text2Human**：依赖骨架或解析图作为空间条件且数据规模有限；本文证明仅凭文本即可在推理期无需额外条件生成多样化高质量人像，拓展了通用基础模型的适用边界。
- **Prompt-to-Prompt / Attend-and-excite 等免训练对齐方法**：通过推理期梯度修改潜变量改善对齐，计算开销大；本文将其思路内化为训练期 HOLA 监督，实现端到端高效学习且不影响推理速度。
- **LAION-5B / COYO-700
