---
title: "Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speec"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_Faces_that_Speak_Jointly_Synthesising_Talking_Face_and_Speech_from_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:54:01"
field: "多模态人脸生成"
keywords: ["Talking Face Generation", "Text-to-Speech", "OT-CFM", "Multimodal Synthesis", "Face-stylised TTS", "Motion Disentanglement"]
innovations: ["首个端到端文本驱动统一框架TTSF，联合生成说话人脸与语音", "基于OT-CFM的运动采样器结合自动编码器噪声降低器，提升运动多样性与时间一致性", "运动解耦的身份条件化TTS机制，消除源图像运动对语音一致性的干扰"]
benchmarks: ["LRS2", "VoxCeleb2", "LRS3"]
---

# 论文速读：Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speech-from-Text

## 一句话总结
本文提出了首个端到端的**文本驱动多模态合成统一框架（TTSF）**，能够从单张静态人脸图片和文本输入，同步生成自然逼真的说话人脸视频与高质量语音。通过融合基于 OT-CFM 的运动采样器和运动解耦的 TTS 条件化策略，该方法在未知身份上实现了强泛化能力，同时提升了唇形同步精度与语音一致性。

---

## 研究问题与动机
1. **文本驱动 TFG 的研究空白**：相比音频驱动的 TFG，文本驱动的 TFG 研究较少；现有 cascade 式方案（TTS + TFG）存在误差累积与推理瓶颈，且缺乏统一的联合优化机制。
2. **多样头部姿态生成困难**：已有 TFG 模型（尤其 audio-driven）在自然场景下生成的头部运动方差较小，难以覆盖真实世界中的多样化姿态。
3. **同一身份的语音一致性问题**：当输入身份不变但面部运动（表情/姿态）变化时，传统 face-stylised TTS 方法因未对源图像中的运动特征进行解耦，导致生成语音的音色、语调不一致。
4. **未见身份的泛化需求**：面向元宇宙、虚拟助手等应用，模型需仅凭一张静态人像即可对未见过的身份生成高保真的视听内容。

---

## 核心贡献（创新点）
1. **首个端到端统一文本驱动多模态合成框架（TTSF）**：将 TFG 与 TTS 融合为单一联合训练模型，消除 cascade 方法中的误差传递与额外音频编码器需求。
2. **基于 OT-CFM 的运动采样器 + 自动编码器噪声降低器**：通过 Optimal-Transport Conditional Flow Matching 学习精确的运动分布，并引入 auto-encoder 压缩目标运动以消除直接回归导致的抖动伪影，显著提升时间一致性与运动多样性。
3. **运动解耦的身份条件化 TTS 机制**：首次在 TTS 系统中利用 TFG 模块提取的无运动身份特征 $f_{id}$ 作为条件，有效去除源图像中的运动因子，从而在不同面部运动下仍保持说话人音色、语调的一致性。
4. **无需反转网络的高效生成器设计**：采用 StyleGAN 风格的流式生成器（LIA baseline），避免计算成本高昂的 inversion network，增强未见身份的泛化能力。

---

## 方法详解
整体架构名为 **TTSF（Text-to-Speaking Face）**，包含 TFG 与 TTS 两条流水线，两者通过互补条件相互驱动（图 2）。

**TFG 流水线：**
- **Motion Extractor**：借鉴 LIA 架构，在正交约束下学习从视觉特征 $f = f_{id} + f_{m}$ 中独立提取身份特征 $f_{id}$ 与运动特征 $f_{m}$，区别于 LIA 的相对运动计算。
- **Audio Mapper**：使用 TTS 的中间表征（文本嵌入 $e_t$、上采样文本特征 $\tilde{f}_t$、能量均值）通过 MRF 模块融合，生成唇部运动特征 $f_{lip}$，替代传统音频编码器。
- **Motion Fusion & Generator**：在非唇动层融合 $f_{id} + f_{m}$，在唇动层融合 $f_{id} + f_{lip}$，经 StyleGAN 风格生成器 $G$ 输出视频 $\hat{I}_d$。
- **变分运动采样（OT-CFM Motion Sampler）**：训练时通过 Prior Network（4 层 Conformer）预测帧级均值 $\mu$，结合压缩后的运动特征 $f_c$（经 auto-encoder 降噪）作为 OT-CFM 目标；推理时从 $\mathcal{N}(\mu, I)$ 采样生成多样化且平滑的运动序列。
- **损失函数**：$\mathcal{L}_{GAN}$（对抗损失）、$\mathcal{L}_{rec}$（L1 + LPIPS）、$\mathcal{L}_{id}$（余弦相似度保身份）、$\mathcal{L}_{sync}$（改进 SyncNet 唇形同步损失）、$\mathcal{L}_{OT-CFM}$、$\mathcal{L}_{AE}$、$\mathcal{L}_{prior}$，总损失按权重加权求和（$\lambda_1 \sim \lambda_8$）。

**TTS 流水线：**
- 基于 **Matcha-TTS**（OT-CFM 架构），使用 TFG 输出的无运动身份特征 $f_{id}$ 同时条件化 encoder 与 decoder。
- 损失为 Matcha-TTS 原有的 prior loss、duration loss 与 OT-CFM loss 之和（$\mathcal{L}_{TTS}$）。
- 最终 mel-spectrogram 经预训练 vocoder（HiFi-GAN）转为音频。

---

## 实验与结果
**数据集：** 训练 — LRS3；测试 — LRS2、VoxCeleb2（更具挑战性，室外拍摄多）。

**评估指标：**
- 视频质量：FID ↓、ID-SIM ↑
- 唇形同步：LSE-C ↑
- 运动多样性：DIV ↑（基于 Hopenet 的头部运动特征标准差）
- 语音质量：WER ↓、MCD ↓、C-SIM ↑、RMSE ↓（F0）

**主要结果（LRS2，one-shot）：**
| 模型 | FID ↓ | ID-SIM ↑ | LSE-C ↑ | DIV ↑ |
|------|-------|----------|---------|-------|
| MakeItTalk (TTS) | 25.168 | 0.850 | 3.487 | 0.095 |
| Audio2Head (TTS) | 42.262 | 0.225 | 5.478 | 0.145 |
| SadTalker (TTS) | 20.729 | 0.859 | 6.256 | 0.111 |
| **TTSF (Ours)** | **18.348** | **0.864** | 5.686 | **0.143** |

- **视频质量最佳**：FID 和 ID-SIM 均超越所有基线。
- **唇形同步**：LSE-C 略低于 SadTalker，但用户感知评测（MOS）中 Lip Sync Accuracy 达 **4.09 ± 0.07**，显著优于基线（SadTalker 2.78）。
- **VoxCeleb2 跨域**：ID-SIM 达 **0.876**，再次验证未见身份泛化能力。
- **语音质量（Table 3）**：WER 14.56（优于 Face-TTS 的 18.02）、C-SIM 0.593（显著优于 Face-TTS 的 0.272），证明运动解耦对语音一致性的关键作用；使用含运动特征的条件（w/ motion）性能大幅下降（C-SIM 仅 0.272）。
- **Ablation（Audio Mapper 特征融合，Table 5）**：移除能量特征后 LSE-C 从 5.721 降至 5.555；移除能量+上采样特征后降至 3.935，验证各组件的必要性。

---

## 相关工作脉络
1. **Audio-driven TFG**（MakeItTalk, SadTalker, Audio2Head）：依赖真实音频驱动人脸生成；本文将其扩展为文本驱动，并通过联合训练消除 cascade 方法的误差累积。
2. **Cascade Text-driven TFG**（MakeItTalk + TTS 等）：先将文本转为音频再驱动 TFG；本文通过共享特征实现端到端联合优化，避免级联瓶颈。
3. **Text-to-Speech with Face Stylisation**（Face-TTS, 2023）：使用源图像embedding作为语音条件，但未考虑运动因素，导致同一身份不同表情下语音风格不一致；本文引入运动解耦模块解决了该问题。
4. **Flow Matching / OT-CFM for Generative Modeling**（Lipman et al., 2023；Matcha-TTS）：将 OT-CFM 引入面部运动生成领域，适配了运动特征的时间连续性约束，区别于原有语音合成应用。
5. **Uniflg**（Mitsui et al., 2023）：从 TTS latent 生成 facial landmarks，仍需额外阶段合成 RGB 视频；本文直接在像素空间联合生成语音与视频。

---

## 局限性与未来方向
1. **LSE-C 指标略低于 SadTalker**：尽管 MOS 用户感知更优，但客观指标存在差距，提示可在同步损失设计上进一步探索更精准的判别器。
2. **运动采样依赖 32 帧训练**：推理时处理全长度视频，长视频的时间一致性仍有优化空间。
3. **仅使用单张静态图片**：对于大角度侧脸或遮挡严重的人像，身份特征提取可能受限。
4. **未探索多说话人场景下的语言多样性**：当前主要在英文 LRS3/LRS2 上验证，多语言扩展待研究。
5. **计算资源需求较高**：训练需 8 块 48GB A6000 GPU，部署成本较大。

---

## 研究启发与可借鉴点
1. **运动-身份解耦的跨模态条件设计**：将 TFG 中提取的纯身份特征 $f_{id}$ 用于 TTS 条件化，解决"同身份不同表情导致语音风格漂移"的问题，该方法可迁移至其他多模态融合场景（如语音驱动手势生成）。
2. **OT-CFM 适配视频/时序生成**：将 OT-CFM 应用于面部运动序列采样，并通过 auto-encoder 压缩目标以缓解抖动，为其他时序生成任务（动作合成、视频插帧）提供了可复用的采样器设计范式。
3. **用文本中间表征替代音频编码器**：Audio Mapper 通过 MRF 融合文本嵌入、上采样文本特征与能量信息来驱动唇动，避免了昂贵的音频特征提取，降低了 cascade 方案的计算开销与误差传播。
4. **联合训练替代级联训练**：端到端联合训练（TFG + TTS）在唇形同步与语音质量上均优于 cascade 方案，提示在多模态生成任务中应优先考虑统一优化而非模块化堆叠。
5. **正交约束的运动特征学习**：Motion Extractor 采用正交约束下的可训练运动码（类似 LIA），以紧凑通道实现多样运动表示，该方法可与现有 TFG backbone 兼容复用。

---

## 关键术语表
**OT-CFM（Optimal-Transport Conditional Flow Matching）**：一种基于最优传输理论的流匹配生成建模方法，通过学习确定性 ODE 轨迹将简单先验分布映射到复杂数据分布，具有训练稳定、采样质量高的优点。

**TTSF（Text-to-Speaking Face）**：本文提出的端到端统一框架名称，将文本驱动的 TFG 与 face-stylised TTS 整合为一个联合优化系统。

**LSE-C（Lip Sync Error Confidence）**：基于预训练 SyncNet 计算的唇形同步误差指标，值越高表示同步越准确。

**DIV（Diversity）**：通过 Hopenet 提取生成帧头部运动特征的标准差，用于量化生成视频的头部姿态多样性。

**C-SIM（Cosine Similarity of X-vectors）**：目标语音与生成语音的 x-vector 余弦相似度，用于评估语音身份一致性。

**MRF（Multi-Receptive field Fusion）**：使用不同核大小与膨胀率的 1D 卷积残差块构成的特征融合模块，能够同时捕捉细粒度与粗粒度的时间上下文信息。

**Face-stylised TTS**：利用人脸图像作为说话人条件来生成个性化语音的 TTS 技术，区别于传统需要大量说话人录音的风格化方法。

**Motion Extractor**：从共享视觉特征中分离出身份特征 $f_{id}$ 和运动特征 $f_{m}$ 的模块，通过正交约束实现解耦表示。

---

## 可复现要素
- **数据集**：训练集 LRS3（公开）；测试集 LRS2、VoxCeleb2（均需申请使用）。
- **代码**：项目页面 https://mm.kaist.ac.kr/projects/faces-that-speak（论文未明确声明 GitHub 仓库，需关注后续发布）。
- **预训练权重**：未提及公开，需自行训练。
- **关键超参**：
  - 学习率：1e-4（Adam）
  - 训练轮数：Matcha-TTS 预训练 2000 epochs，联合训练 40 epochs
  - 视频帧数：运动采样器训练用 32 帧
  - 音频：16kHz，mel-spectrogram（window=640, hop=160, 80 bins）
  - 硬件：8 × 48GB A6000 GPU
  - 损失权重：λ₁~λ₈ 分别为 0.1, 1, 0.3, 0.1, 0.1, 1, 0.1, 1
