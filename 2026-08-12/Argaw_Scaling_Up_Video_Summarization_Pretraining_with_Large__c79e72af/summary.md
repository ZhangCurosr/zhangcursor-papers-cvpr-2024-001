---
title: "Scaling Up Video Summarization Pretraining with Large Language Models"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:14:42"
field: "视频理解与摘要"
keywords: ["视频摘要", "大规模预训练", "大语言模型", "多模态融合", "自回归解码", "伪真值构建"]
innovations: ["LLM作为Oracle抽取式摘要器自动构建25万条长视频-摘要对", "自回归回归式摘要解码替代二分类/重要性评分", "随机MASK文本输入实现多模态容错训练"]
benchmarks: ["LfVS-T", "SumMe", "TVSum"]
---

# 论文速读：Scaling Up Video Summarization Pretraining with Large Language Models

## 一句话总结
本文利用大语言模型（LLM）作为"Oracle摘要器"，从25万条带语音的长视频自动构建视频摘要训练数据（LfVS-P），并提出了一种基于自回归解码的Transformer编码器-解码器视频摘要模型，在多个基准上刷新了SOTA。同时发布了1200条专业人工标注的长视频测试集（LfVS-T）。

## 研究问题与动机
- **数据规模瓶颈**：现有视频摘要数据集极小（TVSum仅50条、SumMe仅25条），导致SOTA方法严重过拟合特定领域，泛化能力差。
- **现有方法的两类缺陷**：（1）将视频摘要建模为二分类/帧重要性预测，存在严重的长尾类别不平衡（摘要片段远少于非摘要片段）；（2）各时间步决策相互独立、并行预测，导致生成的摘要中大量重复片段。
- **LLM能力的空白迁移**：LLM已在长文本理解与摘要上表现卓越，但如何将其"作为Oracle"迁移到视觉摘要领域尚未被系统探索。

## 核心贡献（创新点）
1. **LLM驱动的自动化大规模数据构建流程**：利用HowTo100M中的1.2M+旁白视频，经Whisper转写、CLIP对齐过滤、LLM抽取式摘要后映射回视频片段，产出25万条视频-摘要对（LfVS-P），相比既有数据集规模提升两个数量级。
2. **自回归回归式视频摘要模型**：放弃离散二分类/重要性评分，改为逐帧自回归解码连续CLIP特征表示，并通过SOS/EOS令牌控制生成长度，从根本上缓解类别不平衡与重复片段问题。
3. **多模态交叉注意力机制**：分别用V-Encoder（视频时序上下文）和T-Encoder（文本时序上下文）进行编码，再以视频特征为Query、文本特征为Key/Value做交叉注意力，最后送入Summary Decoder。
4. **LfVS-T新基准**：1200条YouTube长视频（8–33分钟，392个类别），含专业人员手工标注的高质量摘要，覆盖旁白与非旁白场景。
5. **规模化预训练的实证收益**：训练数据从10%增至100%带来单调提升（F1从53.44→68.11），且仅在LfVS-P上预训练后，在SumMe/TVSum上微调即分别超越从头训练6.1%和9.1%。

## 方法详解
**1. 数据集构建流程**
- 从HowTo100M筛选时长≥8分钟的旁白视频，用Whisper转录为带时间戳的句子序列。
- CLIP帧-文相似度过滤低对齐视频；将每个句子前缀起始时间戳后输入LLM（GPT-3.5-16K主实验，辅以GPT-4/Llama-2-13B对比）。
- Prompt要求LLM**保留原句措辞和时间戳**做抽取式摘要，再将选取句子通过起止时间戳映射回视频片段，用CLIP最近邻搜索修正时间戳偏移，聚合为伪真值（pGT）摘要。

**2. 模型架构（Transformer Encoder-Decoder）**
- **长视频编码**：CLIP-ViT-L/14抽取帧特征 → 加入SOS/EOS → 位置编码 → 6层V-Encoder（1024维、8头）输出带时序上下文的视觉token {x̂₀,…,x̂ₙ₊₁}。
- **长文本编码**：SRoBERTa-NLI-large抽取句嵌入 → 3层T-Encoder输出{textual tokens} {ŝ₁,…,ŝₖ}；训练时以0–100%概率随机MASK替换文本为特殊MASK token，使模型在无文本时仍可工作。
- **跨模态注意力**：head = Softmax(QKᵀ/√dₖ)V，其中Q=x̂·W^Q，K=ŝ·W^K，V=ŝ·W^V，输出{text-conditioned视觉特征} {x̂ᵢ^ŝ}。
- **自回归摘要解码**：ŷₜ = S-Decoder({x̂ᵢ^ŝ}, {SOS, y₁,…,yₜ₋₁})，6层Decoder，下一步摘要片段依赖已生成片段与输入上下文。损失为L₂重构：ℒ = Σᵢ₌₁^{m+1} ‖ŷᵢ − yᵢ‖²（yₘ₊₁为EOS）。推理时用最近邻检索将解码特征映射回输入视频帧并聚合。

## 实验与结果
**数据集与基线**
- 测试集：LfVS-T（1200条，人工标注）、SumMe、TVSum。
- 基线：CLIP-It、TL:DW?、iPTNet、A2Summ。
- 评估指标：F1-score、Kendall's τ、Spearman's ρ。

**主要结果（LfVS-T，表2）**
| 方法 | F1 | τ | ρ |
|---|---|---|---|
| CLIP-It | 62.87 | 0.129 | 0.225 |
| TL:DW? | 66.25 | 0.138 | 0.233 |
| iPTNet | 65.80 | 0.140 | 0.237 |
| A2Summ | 66.04 | 0.143 | 0.246 |
| **Ours** | **68.11** | **0.158** | **0.277** |

相对第二名TL:DW?提升F1 +1.86pp；相对A2Summ +2.07pp。

**跨数据集泛化（表3）**
- Zero-shot（LfVS-P→SumMe/TVSum）：F1=56.72/65.76，与从头训练SOTA接近。
- Fine-tuned：F1=60.42（SumMe，↑6.1pp vs 从头）、72.38（TVSum，↑9.1pp vs 从头），刷新两榜SOTA。

**消融（表4）**
完整模型最优：去掉文本输入 -1.52pp、去掉V-Encoder -5.34pp、去掉T-Encoder -0.62pp、去掉跨模态注意力 -0.39pp。

**超参关键值**
- 采样：1 fps；优化器AdamW，初始lr=3e-4，余弦退火；batch=64；100 epochs；4×A6000。
- 隐藏维度1024、8头、FFN 2048；V-Encoder 6层、T-Encoder 3层、S-Decoder 6层。

## 相关工作脉络
1. **文本摘要**：TextRank / BART / Pegasus → LLM时代（GPT-4、Llama-2）的长文本理解与抽取式摘要能力是本文数据构建的源头。
2. **早期视频摘要**：TVSum[36]、SumMe[7]建立监督范式，但规模极小，导致后续工作泛化受限。
3. **多模态视频摘要**：TL:DW?[26]、A2Summ[8]引入旁白文本辅助，证明多模态有益；本文延续并扩展至大规模预训练场景。
4. **查询式摘要**：Sharghi等人[34][35]引入自然语言查询定制化摘要；本文未涉及但可天然扩展。
5. **CLIP-based视频理解**：CLIP-It[25]用CLIP做零样本视频定位/摘要；本文借鉴CLIP作为视觉tokenizer与对齐度量。
6. **无监督/自监督视频摘要**：Mahasseni et al.[21]（Adv-LSTM）、Jung et al.[11]等；本文完全转向大规模有监督预训练路线。

## 局限性与未来方向
- **伪真值噪声**：pGT依赖LLM抽取质量与Whisper时间戳精度，仍存在漂移和摘要不完整风险；虽经CLIP近邻修正，但非人工精标。
- **LLM成本与可扩展性**：大尺度数据构建依赖GPT-4/3.5调用，成本和延迟较高；不同LLM间质量差异显著（Llama-2-13B仅44.89 F1 vs GPT-4的55.96 F1）。
- **评测侧重长视频**：LfVS-T平均12.2分钟，与SumMe/TVSum的2–4分钟差距大；短视频场景的能力仍需验证。
- **纯抽取式限制**：当前仅做片段选取（extractive），无法生成全新内容摘要；对复杂叙事或需要重组的场景表现存疑。
- **文本依赖**：无旁白视频虽可用MASK token处理，但性能明显下降（表4：无文本-1.52pp）。

## 研究启发与可借鉴点
1. **"LLM as Oracle"数据构建范式可迁移**：凡是需要大规模弱监督标注但缺乏数据的任务（如视频定位、事件检测、3D描述生成），均可借鉴"强LLM→结构化输出→投影回模态空间"的流水线。
2. **自回归回归式输出替代离散分类**：对类别极度不平衡的"片段选择"类任务，用连续特征+自回归解码比importance score更稳健；这一思路可推广到动作检测、字幕生成等。
3. **随机MASK多模态输入的对抗式正则**：训练期0–100%随机用MASK token替代文本，可使模型在任意模态缺失下鲁棒，是一种简单有效的多模态容错训练技巧。
4. **CLIP对齐用于时间戳校正**：用CLIP帧-文相似度做最近邻搜索修正ASR漂移，比纯阈值过滤更精确；可复用于任何语音-视频对齐任务。
5. **规模-性能单调曲线作为方法有效性的佐证**：10%/50%/100%三档实验清晰展示预训练规模带来的收益，这种"缩放实证"比单一SOTA数字更有说服力。

## 关键术语表
- **LfVS-P**：Long-form Video Summarization Pretraining，本文自动构建的25万条长视频-摘要对预训练数据集。
- **LfVS-T**：Long-form Video Summarization Testing，1200条人工标注长视频测试集，用于评估跨域泛化。
- **Oracle Summarizer**：指将LLM视为理想"上帝视角"摘要器，以其输出作为伪真值训练下游视觉模型。
- **Extractive Summary**：从原文中直接选取关键句子/片段组成的摘要，区别于重新组织的abstractive摘要；本文选择此路径以保留时间戳映射关系。
- **pGT（pseudo-ground truth）**：由自动流水线生成的"近似真值"，作为训练信号替代昂贵的人工标注。
- **Autoregressive Decoding**：在当前时刻输出依赖于之前已生成输出的解码方式；本文用于依次生成摘要片段特征。
- **Cross-Modal Attention**：以视频特征为Query、文本特征为Key/Value的注意力机制，实现文本条件化的视频表示。

## 可复现要素
- **数据**：HowTo100M公开（需筛选时长≥8分钟）；LfVS-P、LfVS-T论文未声明开源链接（CVPR 2024常见于补充材料/项目页，建议查阅arXiv版）。
- **代码/权重**：论文未给出仓库URL；PyTorch实现、CLIP-ViT-L/14、SRoBERTa-NLI-large、Whisper均为公开组件，可复现。
- **关键超参**：lr=3e-4、batch=64、epochs=100、4×A6000、1 fps、隐藏1024、8头、FFN 2048、V-Encoder 6层/T-Encoder 3层/S-Decoder 6层。
- **LLM**：主实验GPT-3.5-16K，对比GPT-4/Llama-2-13B（50K样本实验）；prompt模板见图2（要求保留原句措辞与时间戳）。
