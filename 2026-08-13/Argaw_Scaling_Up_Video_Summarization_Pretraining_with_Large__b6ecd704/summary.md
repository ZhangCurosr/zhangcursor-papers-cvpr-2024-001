---
title: "Scaling Up Video Summarization Pretraining with Large Language Models"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:02:25"
field: "视频理解与生成"
keywords: ["Video Summarization", "Large Language Models", "Self-supervised Pretraining", "Multimodal Learning", "Dataset Curation"]
innovations: ["利用LLMs作为Oracle生成大规模视频摘要伪标签数据集", "提出自回归视频摘要模型解决类别不平衡和时序依赖问题", "构建新的长视频摘要基准LfVS-T"]
benchmarks: ["LfVS-T", "SumMe", "TVSum"]
---

# 论文速读：Scaling Up Video Summarization Pretraining with Large Language Models

## 一句话总结
本文提出一种利用大型语言模型（LLMs）作为Oracle摘要器的自动可扩展管道，从长视频生成大规模视频摘要训练数据（LfVS-P，250K对），并据此提出自回归视频摘要模型，通过解码连续特征表示而非并行分类，有效解决了类别不平衡与重复片段问题，在多个基准上达到SOTA。

## 研究问题与动机
- **核心问题**：现有视频摘要数据集规模极小（如TVSum仅50对、SumMe仅25对），严重制约模型的泛化能力。
- **现有方法不足**：
  1. 大多数方法将视频摘要建模为二分类（每帧是否属于摘要）或重要性分数预测，存在严重的长尾分布问题（摘要片段远少于非摘要片段）。
  2. 这些方法并行预测每个时间步的摘要决策，忽略已选摘要片段之间的时序依赖，导致重复片段被选中。
- **动机**：借助LLMs在长文本摘要上的卓越能力，以及长视频丰富且密集的语音-视频对齐信息，构建大规模视频摘要预训练数据集，并设计能捕捉时序上下文依赖的新模型。

## 核心贡献（创新点）
1. **自动可扩展的数据集构建管道**：利用LLMs（如GPT-4）作为Oracle摘要器，从HowTo100M等长视频中生成高质量伪标注视频-摘要对，创建250K规模的LfVS-P预训练数据集。
   - **本质区别**：现有数据集（如TL:DW?仅12.1K）规模有限且自动标注质量较低；本文利用LLMs抽取式摘要能力，结合CLIP对齐过滤，生成规模更大、多样性更高的训练数据。
2. **自回归视频摘要模型**：将摘要生成转化为连续特征表示的自回归解码问题，而非离散分类或重要性分数预测，并通过交叉注意力融合视觉与转录文本。
   - **本质区别**：与CLIP-It、TL:DW?、A2Summ等并行预测每帧重要性的方法不同，本文模型按顺序生成摘要片段，显式建模摘要片段间的上下文依赖，缓解重复选取问题。
3. **新基准LfVS-T**：引入包含1200个长视频（8–33分钟）及人工高质量摘要的测试基准，覆盖392个类别，评估模型的跨域泛化能力。
   - **本质区别**：传统基准（TVSum、SumMe）视频短、规模小；LfVS-T专注长视频，更贴近实际应用场景。
4. **大规模预训练显著提升泛化性能**：实验证明，在LfVS-P上预训练的模型在零样本和微调设置下，均在SumMe、TVSum和LfVS-T上取得SOTA性能。
   - **本质区别**：现有工作缺乏大规模预训练数据，跨数据集泛化能力有限；本文验证了数据规模对鲁棒性的关键作用。

## 方法详解
- **数据集构建管道（Fig. 1）**：
  1. **源数据筛选**：从HowTo100M（1.2M+ narrated videos）中选择时长≥8分钟的长视频。
  2. **语音转文本**：使用Whisper模型转录视频音频，得到带起始时间戳的句子文本。
  3. **LLM提示摘要**：将带时间戳的文本输入LLM（主要用GPT-3.5-16K，也测试GPT-4、Llama-2-13B），提示其选择最关键的句子并保持原文和时间戳，实现抽取式文本摘要。
  4. **伪真值生成**：根据文本摘要句子的起止时间戳切取对应视频片段，利用CLIP嵌入的最近邻搜索微调对齐，聚合得到伪真值摘要视频。
- **视频摘要模型（Fig. 3）**：
  1. **多模态输入**：视频帧序列 \(V = \{X_1, X_2, \ldots, X_n\}\)（每秒采样1帧）和转录文本序列 \(T = \{S_1, S_2, \ldots, S_k\}\)。
  2. **视觉编码**：用预训练CLIP-ViT-L/14提取每帧视觉token：\(\{x_1, x_2, \ldots, x_n\} = \text{CLIP}(\{X_1, X_2, \ldots, X_n\})\)，添加SOS/EOS token后经位置编码输入V-Encoder（6层Transformer编码器），输出上下文视觉特征 \(\{\hat{x}_0, \hat{x}_1, \ldots, \hat{x}_{n+1}\}\)。
  3. **文本编码**：用SRoBERTa-NLI-large提取句子嵌入：\(\{s_1, s_2, \ldots, s_k\} = \text{SRoBERTa}(\{S_1, S_2, \ldots, S_k\})\)，经T-Encoder（3层Transformer编码器）输出上下文文本特征 \(\{\hat{s}_1, \hat{s}_2, \ldots, \hat{s}_k\}\)。训练时随机用MASK token掩盖文本（掩码率0–100%），使模型能处理无语音视频。
  4. **跨模态注意力**：以视觉特征为查询，文本特征为键值，计算多头交叉注意力：\(\text{head} = \text{Att.}(\hat{x}W^Q, \hat{s}W^K, \hat{s}W^V)\)，得到文本条件化视觉特征 \(\{\hat{x}_i^{\hat{s}}\}\)。
  5. **自回归摘要解码**：Summary Decoder（6层Transformer解码器）自回归生成摘要片段视觉特征。在时间步t，解码器以跨模态特征和已生成的摘要特征 \(\{\text{SOS}, y_1, \ldots, y_{t-1}\}\) 为输入，预测下一特征 \(\hat{y}_t = \text{S-Decoder}(\{\hat{x}_i^{\hat{s}}\}, \{\text{SOS}, y_1, \ldots, y_{t-1}\})\)。摘要特征由CLIP提取：\(\{y_1, \ldots, y_m\} = \text{CLIP}(\{Y_1, \ldots, Y_m\})\)。
  6. **训练损失**：特征重建损失（L2 loss）：\(\mathcal{L} = \sum_{i=1}^{m+1} |\hat{y}_i - y_i|^2\)，其中 \(y_{m+1}\) 为EOS token。
  7. **推理**：解码器生成摘要特征后，通过CLIP嵌入最近邻检索匹配输入视频的帧，时间聚合得到最终摘要。

## 实验与结果
- **数据集**：LfVS-P（250K训练对）、LfVS-T（1200测试视频，人工标注）；经典基准SumMe、TVSum。
- **评估指标**：F1 Score、Kendall's τ、Spearman's ρ。
- **基线模型**：CLIP-It、TL:DW?、iPTNet、A2Summ。
- **主要结果（Table 2，LfVS-T）**：
  - 本文模型：F1=68.11，τ=0.158，ρ=0.277。
  - 相比TL:DW?（F1=66.25）提升2.8%，相比A2Summ（F1=66.04）提升3.1%，全面超越所有基线。
- **SumMe/TVSum结果（Table 3）**：
  - 零样本（在LfVS-P预训练，直接测试）：SumMe F1=56.72，TVSum F1=65.76。
  - 微调后：SumMe F1=60.42，TVSum F1=72.38，超越SOTA（A2Summ为57.09/66.10）。
- **最强结果与提升幅度**：LfVS-T上F1=68.11为当前最高；TVSum微调后F1=72.38，较A2Summ提升9.1%。

## 相关工作脉络
1. **文本摘要**：LLMs（如GPT-4）在长文本抽取式摘要上表现优异，本文将其能力迁移至视频域。
2. **视频摘要数据集**：TVSum（50对）、SumMe（25对）规模小；TL:DW?（12.1K对）规模较大但自动标注质量有限；本文LfVS-P（250K对）规模、多样性均显著领先。
3. **视频摘要模型**：CLIP-It（查询引导的基于CLIP的摘要）；TL:DW?（交叉模态显著性的多模态摘要）；A2Summ（双对比损失的视频-文本对齐）；本文自回归解码方法在问题设定上根本不同。
4. **多模态视频理解**：利用语音转录文本辅助视频理解，本文通过随机掩码文本使模型能灵活处理有无语音视频。

## 局限性与未来方向
- **LLM依赖**：管道质量高度依赖LLM的指令遵循能力；开源模型（如Llama-2-13B）效果较差，可能因无法精确遵循提示。
- **数据源限制**：依赖语音-视频对齐良好的长视频，可能遗漏无语音或对齐差的视频。
- **计算成本**：自回归解码逐帧生成，推理速度可能较慢。
- **未来方向**：探索更高效的解码策略；将方法扩展至其他视频理解任务（如动作识别、事件检测）；改进无语音视频的摘要生成能力。

## 研究启发与可借鉴点
1. **LLM伪标注数据生成**：利用LLMs作为Oracle生成大规模伪标签的方法，可迁移到其他视频理解任务，缓解标注稀缺问题。
2. **自回归连续特征解码**：将序列生成任务转化为连续特征表示的自回归解码，有助于缓解类别不平衡和重复生成问题。
3. **多模态输入灵活性**：训练时随机掩码一种模态，使模型能适应有无该模态的输入，值得在其他多模态任务中借鉴。
4. **大规模预训练优先**：实验证明数据规模扩张对泛化性能至关重要，提示在资源受限时优先考虑数据规模而非仅优化模型结构。

## 关键术语表
- **Oracle summarizer**：指作为理想摘要生成器的LLM，用于生成高质量伪标签数据。
- **Extractive summarization**：抽取式摘要，从原文中直接选取重要句子组成摘要，而非重新生成。
- **Cross-modal attention**：跨模态注意力，允许不同模态（如视觉和文本）的特征相互查询信息。
- **Autoregressive decoding**：自回归解码，按顺序逐个生成输出元素，每个元素依赖于之前生成的元素。
- **Pseudo-ground truth**：伪真值，指通过自动化方法（如LLM）生成的近似真实标签。
- **Long-form video summarization**：长视频摘要，针对时长较长的视频（通常>8分钟）生成概要。

## 可复现要素
- **数据集**：LfVS-P（250K训练对）、LfVS-T（1200测试视频）；论文未明确声明是否开源，但提到“publicly available YouTube content”。
- **代码/权重**：论文未提及开源，需联系作者获取。
- **关键超参**：采样率1 fps；CLIP-ViT-L/14；SRoBERTa-NLI-large；V-Encoder 6层，T-Encoder 3层，CM-Att 1层，Decoder 6层；隐藏维度1024，注意力头数8，FFN维度2048；AdamW优化器，初始学习率3e-4，余弦退火，batch size 64，100 epochs，4×NVIDIA A6000 GPUs。
