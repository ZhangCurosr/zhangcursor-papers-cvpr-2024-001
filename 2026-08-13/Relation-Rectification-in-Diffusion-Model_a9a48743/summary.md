---
title: "Relation-Rectification-in-Diffusion-Model"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Wu_Relation_Rectification_in_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:42:41"
field: "文生图关系生成"
keywords: ["Diffusion Model", "Relation Generation", "Heterogeneous GCN", "Vision-Language Model", "Text-to-Image"]
innovations: ["提出Relation Rectification任务，利用HGCN调整[EOT]嵌入以修正方向性关系生成", "设计正负损失组合策略分离OSP对嵌入，同时保持物体语义解耦", "构建包含21种关系的Relation Rectification Benchmark"]
benchmarks: ["Relation Rectification Benchmark", "COCO"]
---

# 论文速读：Relation-Rectification-in-Diffusion-Model

## 一句话总结
本文提出 Relation Rectification 任务，通过一个轻量级的异构图卷积网络（RRNet）对文本编码器生成的 [EOT] 嵌入进行定向调整，使扩散模型能够准确生成具有正确方向性视觉关系的图像，同时保持冻结的预训练模型参数和基础生成能力。

## 研究问题与动机
- **核心问题**：T2I 扩散模型（如 SD）在处理包含方向性关系（如 "on", "inside"）的文本提示时，经常混淆对象间的空间/动作关系（如将"书在碗上"生成成"碗在书上"）。
- **现有方法不足**：CLIP 等 VLM 使用对比学习训练，倾向于将句子视为"Bag-of-Words"，无法捕捉句子顺序和方向性语义；现有解决方法（如引入 canvas layout 或 ControlNet）仅绕过问题而未修复文本编码器对关系方向性的敏感性缺失。
- **关键发现**：原始 SD 中，对象交换提示（OSP，如"<A,R,B>"与"<B,R,A>"）产生的 [EOT] token 嵌入向量高度相似（余弦相似度接近1），导致模型无法区分关系方向。
- **任务定义**：提出 Relation Rectification 新任务，使 T2I 模型对 OSP 对产生不同的生成响应，而非将其简化为词袋表示。

## 核心贡献（创新点）
1. **提出 Relation Rectification 新任务**：首次系统性地定义并评估 T2I 模型对方向性关系的生成能力，填补了该研究空白。
2. **识别 [EOT] 嵌入是关系生成瓶颈**：通过消融实验发现 [EOT] token 嵌入控制着关系生成，且 OSP 对的 [EOT] 嵌入几乎不可区分，这是 vanilla SD 关系生成失败的根本原因。
3. **设计 RRNet 异构图卷积框架**：将 OSP 对建模为有向异构图（对象节点+关系节点），利用 HGCN 学习调整向量 $h_{\Delta EOT}$ 分离 [EOT] 嵌入，仅需训练轻量级附加模块，保持 SD 参数冻结。
4. **构建 Relation Rectification Benchmark**：发布包含21种关系（8种位置关系+13种动作关系）、4200张图像的评测数据集，为后续研究提供标准评测基准。
5. **提出正负损失组合的训练策略**：设计 denoising loss（正向引导关系语义）和 negative loss（负向分离 OSP 嵌入），配合对象节点解耦机制，实现关系的准确生成与物体语义的保留。

## 方法详解
- **Heterogeneous Graph Construction**：将 OSP 对 "<A,R,B>" 和 "<B,R,A>" 建模为有向异构图，包含三类节点：对象节点 $v_O$、关系节点 $v_R$、调整节点 $v_{\Delta EOT}$。边表示方向关系：$A \rightarrow R \rightarrow B$。
- **节点初始化**：对象节点使用 CLIP 词嵌入，关系节点同样使用 CLIP 词嵌入，调整节点 $v_{\Delta EOT}$ 随机初始化。
- **HGCN 聚合更新**：
$$h_{\Delta EOT}^{(l+1)} = \sum_{\varepsilon \in \hat{\mathcal{E}}} \sigma\left(b^{(l)} + \sum_{j:(e_{j,i}) \in \mathcal{E}} \alpha_{j,i} h_j^{(l)} W_{\varepsilon}^{(l)}\right)$$
不同节点类型使用不同的权重矩阵 $W_\varepsilon$，关系方向和对象信息沿边聚合至调整节点。
- **嵌入调整**：
$$V_{eot}' = V_{eot} + \lambda \cdot h_{\Delta EOT}^{(L)}$$
其中 $\lambda \in [0,1]$ 控制调整强度，平衡关系准确性与图像质量。
- **对象节点解耦**：为每个对象 A 设立独立调整节点 $v_{\Delta EOT}^A$，从模板句"This is a photo of {A}"提取 $V_{eot}^A$，添加高斯噪声强制模型学习有效调整向量，防止 trivial solution。
- **正负损失**：
  - Positive Loss（Denoising Loss）：$\mathcal{L}_{denoise} = \mathbb{E}[||\epsilon - \epsilon_\theta(x_t, t, \phi(c(y)))||_2^2]$，使用示例图像引导调整后的嵌入生成正确关系。
  - Negative Loss：$\mathcal{L}_{neg} = \mathbb{E}[-||\epsilon - \epsilon_\theta(x_t, t, \phi(c(\tilde{y})))||_2^2]$，压制 OSP 对的反向关系，促进 $V_{eot}$ 分离。
  - 总损失：$\mathcal{L} = \eta \cdot \mathcal{L}_{denoise} + \xi \cdot \mathcal{L}_{neg}$，实验取 $\eta=10, \xi=2$。

## 实验与结果
- **数据集**：Relation Rectification Benchmark，包含21种关系（8种位置+13种动作），每提示生成100张图像，共4200张。
- **基线**：Stable Diffusion 2-1（原始）、Personalized Diffusion（优化 CLIP 文本编码器）。
- **评估指标**：关系生成准确率（使用 Qwen-VL-Chat 和 LLaVA 评测）、物体生成准确率（OGA）、FID（图像质量）。
- **主要结果**（Table 1）：
  - RRNet ($\lambda=0.6$)：Position(Qwen)=0.697↑、Position(LLaVA)=0.684↑、Action(Qwen)=0.500↑、Action(LLaVA)=0.632↑、OGA=0.970↑、FID=100.78↓
  - 相对 SD：关系准确率提升约25%（Position Qwen: 0.467→0.697），OGA 从0.898提升至0.970
  - 相对 Personalized Diffusion：Position Qwen 提升显著（0.509→0.697），且避免了物体语义损失
- **用户研究**（Table 3）：63名评估者，RRNet 获得75.56%偏好率，较 SD（16.98%）和 Personalized Diffusion（7.46%）显著提升。
- **泛化能力**：零样本处理未见过的对象组合，如图6所示成功生成新对象对的正确关系。
- **调参**：$\lambda \in [0.4, 0.6]$ 时可在关系准确率和图像质量间取得最佳平衡。

## 相关工作脉络
1. **Compositional Image Generation**（Composable Diffusion、GLIGEN、ControlNet）：通过额外条件（如布局、边界框）控制对象位置和生成，本文聚焦于文本编码器的内在关系理解缺陷而非外部条件干预。
2. **Training-free Layout Control**（Attend-and-excite、BoxDiff）：修改 cross-attention 实现无训练控制，但仅适用于简单空间关系，本文处理更复杂的方向性和动作关系。
3. **Personalized Diffusion**（Textual Inversion、DreamBooth）：优化文本编码器嵌入实现个性化，但易导致物体语义损失（如图5所示），本文通过 HGCN 仅调整关系方向而保留物体语义。
4. **Bag-of-Words 问题**（Winoground、When and why VLMs behave like BoW models）：指出现有 VLM 因对比学习目标无法捕捉句子顺序，本文将此问题延伸至 T2I 扩散生成并给出解决方案。
5. **Heterogeneous GCN**（HetGNN、HGAT）：本文首次将 HGCN 引入文本嵌入调整以解决关系生成问题，区别于传统的图结构学习任务。
6. **Concept Algebra in VLMs**：本文嵌入调整灵感来源于已有概念代数工作，但应用于关系方向修正这一新场景。

## 局限性与未来方向
- **图像质量代价**：随着 $\lambda$ 增大（关系准确率提升），FID 得分恶化（图像保真度下降），需在准确性和质量间权衡。
- **少量参考图像依赖**：每种关系需要少量示例图像进行训练，限制了完全无监督场景的应用。
- **仅处理二元关系**：当前框架针对两个对象间的一对一关系，复杂多对象关系（如"A在B和C之间"）尚需扩展。
- **固定关系词**：需要预先定义关系类型（位置/动作），对动态或隐式关系的泛化能力待验证。
- **未来方向**：扩展到更复杂的关系结构、减少参考图像需求、探索无需微调的 zero-shot 关系修正方法。

## 研究启发与可借鉴点
1. **Embedding 空间的方向性建模**：利用图神经网络显式编码句子中的方向性关系，为文本嵌入修正提供了可迁移的思路，可应用于其他需要关系感知的 VLM 任务。
2. **正负损失分离策略**：同时优化正向引导（生成正确关系）和负向压制（分离 OSP 嵌入）的对比学习思路，对训练关系感知的 embeddings 具有通用参考价值。
3. **冻结大模型+轻量适配器**：保持 SD 参数冻结仅训练 HGCN 的设计，既保留了预训练模型的强大生成能力，又实现了特定能力的增强，符合 Parameter-Efficient Fine-tuning 趋势。
4. **对象解耦机制**：引入独立节点和噪声扰动防止 trivial solution 的策略，可推广至其他需要分离多个概念特征的 embedding 调整任务。
5. **评测基准构建**：系统性地构建包含 OSP 对和对应图像的 benchmark，结合 VL chatbot 自动化评测，为关系生成研究提供了可复用的评估范式。

## 关键术语表
**Relation Rectification**：新定义的任务，旨在修正扩散模型对方向性关系的错误生成，使其能根据对象顺序差异产生不同响应。
**Object-Swapped Prompts (OSP)**：描述相同关系但对象位置互换的提示对，如"<A,R,B>"与"<B,R,A>"，用于测试模型的关系感知能力。
**[EOT] Token Embedding**：文本编码器中特殊结束标记的嵌入向量，汇聚了整个句子的语义信息，对关系生成起关键控制作用。
**Heterogeneous Graph Convolutional Network (HGCN)**：处理多种节点类型的图卷积网络，本文用于建模对象和关系节点间的方向性连接。
**Adjustment Vector ($h_{\Delta EOT}$)**：HGCN 输出的向量调整量，被加到原始 [EOT] 嵌入上以分离 OSP 对的语义表示。
**Positive/Negative Loss**：正向 loss 引导生成正确关系，负向 loss 压制反向关系，两者结合实现关系的准确定向。
**Relation Rectification Benchmark**：本文构建的包含21种关系、4200张图像的评测数据集，含 OSP 对和对应图像。
**Object Node Disentanglement**：将不同对象的语义特征从关系信息中分离的机制，防止调整过程中物体特征被污染或合并。

## 可复现要素
- **数据集**：Relation Rectification Benchmark，论文提供了链接（https://wuyinweihah.github.io/rrnet.github.io/），论文未明确声明是否开源，但提供了 project page。
- **代码/权重**：论文提供了 project page 链接，未明确说明代码开源状态；基于 Stable Diffusion 2-1，预训练模型可从 HuggingFace 获取。
- **关键超参**：$\lambda=0.6$（推荐平衡点）、$\eta=10$（denoising loss 权重）、$\xi=2$（negative loss 权重）、训练100 epochs、PNDM scheduler、30 steps、1 NVIDIA V100 GPU 训练约20分钟/关系。
