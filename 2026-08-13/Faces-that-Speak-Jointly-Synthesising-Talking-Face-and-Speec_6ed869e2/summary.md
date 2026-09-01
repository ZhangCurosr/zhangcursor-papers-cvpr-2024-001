---
title: "Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speec"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Jang_Faces_that_Speak_Jointly_Synthesising_Talking_Face_and_Speech_from_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:53:37"
field: "多模态生成与语音技术"
keywords: ["Talking Face Generation", "Text-to-Speech", "Flow Matching", "Multimodal Generation", "Identity Preservation", "OT-CFM"]
innovations: ["首个统一的文本驱动 TFG 与 TTS 联合生成框架，实现从文本到同步语音和说话人脸视频的端到端生成", "提出基于 OT-CFM 的运动采样器结合 AE 归一化器，实现高质量、时序稳定的面部运动序列生成", "通过在 TFG 模块内解耦身份与运动特征，并将纯净身份特征 $f_{id}$ 作为 TTS 条件，解决了变运动下语音风格一致性的难题"]
benchmarks: ["LRS2", "VoxCeleb2", "LRS3 (训练)"]
---

# 论文速读：Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speech

## 一句话总结
论文提出了一种名为 **Text-to-Speaking Face (TTSF)** 的统一框架，首次实现了从单张静态人脸照片和文本输入同时生成同步、自然的语音和说话人脸视频，并展现出对未见身份的强泛化能力。

## 研究问题与动机
*   **核心问题**：如何构建一个统一的端到端系统，仅凭文本和一张人脸照片，就能同时生成高保真、口型同步的语音和带有自然头部运动的说话人脸视频。
*   **现有方法不足**：
    1.  **音频驱动 TFG 的局限**：传统 TFG 方法高度依赖音频作为驱动条件，在纯文本到视频的生成场景中无法直接应用。
    2.  **级联方法的瓶颈**：现有文本驱动方法多采用“TTS + TFG”的级联方案，存在误差累积和推理效率低下的问题。
    3.  **语音一致性挑战**：在人脸风格化 TTS 中，当输入身份相同但面部运动不同时，现有方法难以生成语音风格一致（音色、韵律稳定）的语音，因为它们未有效解耦面部运动特征和身份特征。

## 核心贡献（创新点）
1.  **首个统一的文本驱动多模态生成系统**：将 TFG 与 TTS 整合于一个联合训练框架中，无需额外音频编码器即可替代音频条件，且避免了级联方法的误差累积。
2.  **基于 OT-CFM 的运动采样器**：提出利用最优传输条件流匹配（OT-CFM）来学习运动分布，能够高效地采样高质量、时序一致的面部运动代码，解决了直接回归导致 motion shaky 的问题。
3.  **运动解耦的身份条件 TTS**：在 TFG 模块中提取并剥离运动特征后，将纯净的身份特征 $f_{id}$ 作为 TTS 的条件，从而在相同身份但不同面部动作的输入下，仍能生成语音风格一致的语音。
4.  **基于 AE 的运动归一化器**：设计了一个自动编码器作为噪声减小器，对 OT-CFM 采样的运动特征进行压缩与重构，显著提升了生成运动的时序稳定性。

## 方法详解
框架命名为 **TTSF**，核心由两个互促的子系统构成：
*   **Text-to-Speech (TTS) 模块**：
    *   基于 **Matcha-TTS** 架构（本身也使用 OT-CFM）。
    *   输入文本经过文本编码器 $E_t$ 和持续时间预测器处理后，得到文本特征 $\tilde{f}_t$。
    *   TTS 解码器 $D_{TTS}$ 同时接受文本特征 $\tilde{f}_t$ 和来自 TFG 模块的身份特征 $f_{id}$ 作为条件，生成 mel-spectrogram。
    *   最终通过预训练的 HiFi-GAN vocoder 合成语音。
*   **Talking Face Generation (TFG) 模块**：
    *   **运动提取器**：基于 LIA 架构，使用一个正交约束的 MLP 从视觉特征 $f = E_v(I)$ 中分解出运动特征 $f_m$ 和身份特征 $f_{id}$（即 $f_m = f - f_{id}$）。
    *   **音频映射器 (Audio Mapper)**：这是连接文本和面部生成的关键。它将 TTS 的中间表示（文本嵌入 $e_t$、上采样文本特征 $\tilde{f}_t$ 以及能量特征）通过多 receptive field 融合模块（MRF），转换为唇部运动特征 $f_{lip}$，**替代了传统方法中的音频编码器**。
    *   **运动融合与生成器**：采用 LIA 的 StyleGAN 风格生成器。在非唇部层融合 $f_{id} + f_m$，在唇部层融合 $f_{id} + f_{lip}$，以此生成目标帧。
*   **运动采样器 (Motion Sampler)**：
    *   **核心机制**：采用 **Optimal-Transport Conditional Flow Matching (OT-CFM)**。该模块学习一个 ODE 向量场，将初始噪声分布流形变换到真实运动特征分布。
    *   **训练目标**：最小化预测向量场与基于 OT 理论推导出的目标向量场之间的损失 $\mathcal{L}_{OT-CFM}$。
    *   **归一化处理**：直接预测原始运动特征 $f_m$ 会导致推理时 motion shaky。因此引入一个**自动编码器运动归一化器**，将其压缩为更平滑的特征 $f_c$ 作为 OT-CFM 的目标，并辅以 MSE 重建损失 $\mathcal{L}_{AE}$ 和 prior loss $\mathcal{L}_{prior}$。
*   **损失函数**：
    *   TFG 部分：$\mathcal{L}_{GAN}$ (对抗损失), $\mathcal{L}_{rec}$ (L1+LPIPS), $\mathcal{L}_{id}$ (身份保持损失), $\mathcal{L}_{sync}$ (唇形同步损失，基于 SyncNet)。
    *   运动采样部分：$\mathcal{L}_{OT-CFM}, \mathcal{L}_{AE}, \mathcal{L}_{prior}$。
    *   TTS 部分：$\mathcal{L}_{TTS}$ (Matcha-TTS 的 prior, duration, OT-CFM 损失)。
    *   总损失为以上各项的加权和：$\mathcal{L}_{total} = \sum \lambda_i \mathcal{L}_i$。

## 实验与结果
*   **数据集**：在 **LRS3** 上进行训练，在 **LRS2** 和 **VoxCeleb2** 上进行评估（one-shot 设置）。
*   **评估指标**：
    *   视频质量：**FID** (↓), **ID-SIM** (↑)
    *   同步性：**LSE-C** (↑)
    *   运动多样性：**DIV** (↑)
    *   语音质量：**WER** (↓), **MCD** (↓), **C-SIM** (↑), **RMSE** (↓)
*   **主要结果**：
    *   **LRS2 上**：TTSF 在视频质量上显著优于所有对比方法，**FID 降至 18.348**（SadTalker 为 20.729，MakeItTalk 为 25.168），**ID-SIM 达到 0.864**（最高）。
    *   **VoxCeleb2 上**：TTSF 的 **ID-SIM 达到 0.876**，同样领先。
    *   **语音质量上**：与 SOTA 人脸风格化 TTS 模型 **Face-TTS** 相比，TTSF 在 **WER (14.56 vs 18.02)**、**C-SIM (0.593 vs 0.272)** 和 **RMSE (48.52 vs 52.33)** 上均有大幅提升，证明了运动解耦的有效性。消融实验表明，去掉运动特征会导致语音相似性严重下降。
    *   **用户研究**：在唇形同步、运动自然度和视频真实感三项 MOS 评分上均获得最高分（同步性 4.09，运动自然度 3.85，真实感 3.87，满分 5 分），显著优于 SadTalker、MakeItTalk 和 Audio2Head。

## 相关工作脉络
1.  **音频驱动 TFG (如 MakeItTalk, Audio2Head, SadTalker)**：这些方法需要音频信号驱动，无法直接从文本生成。本文将其作为级联基线（用自己的 TTS 模块替换其音频输入）进行比较，证明联合框架的优越性。
2.  **级联文本驱动 TFG (如 UniFLG)**：采用 TTS 生成音频后再驱动 TFG 的两阶段方法，存在误差累积和计算瓶颈。本文的联合架构避免了这一缺陷。
3.  **人脸风格化 TTS (如 Face-TTS, Residual-guided TTS)**：此类方法利用人脸图像生成语音，但未考虑源图像中的面部运动会对语音风格产生干扰。本文的核心创新正是通过 TFG 模块的运动解耦机制，为 TTS 提供更纯净的身份条件。
4.  **Motion Model / 面部运动生成 (如 LIA, First Order Motion Model)**：本文的运动提取器和融合模块借鉴了 LIA 的架构思想，但针对文本驱动和联合训练场景进行了改造（如正交约束、音频映射器的引入）。
5.  **Conditional Flow Matching for TTS (如 Matcha-TTS, Grad-TTS)**：本文的 TTS 模块基于 Matcha-TTS，并将流匹配思想创新性地应用于更复杂的 facial motion 序列生成任务。

## 局限性与未来方向
*   **数据偏差**：训练数据主要来自 TED/TEDx 视频（室内、正面、清晰），在极端姿势、遮挡或低分辨率等 wild 场景下的泛化能力有待进一步验证。
*   **系统复杂度**：模型包含多个复杂模块（OT-CFM 采样器、AE 归一化器、双向条件交互），推理速度和计算成本可能较高。
*   **身份范围**：虽然声称能泛化到 unseen identities，但实验仅验证了在同类人脸数据集上的表现，对于跨年龄、跨种族等更广泛的身份分布，效果未知。
*   **未来方向**：可扩展至多说话人、多语言支持；探索更轻量化的运动采样器以提高推理效率；与 3D 面部表示结合，提供更可控的头部姿态。

## 研究启发与可借鉴点
1.  **流匹配用于非连续动态序列生成**：将 OT-CFM 从传统的 mel-spectrogram 生成成功迁移到面部运动代码序列的生成，并配合归一化器解决时序抖动问题，这一思路可推广到其他需要生成平滑时序动态特征的任务（如手势生成、生物标志物序列预测）。
2.  **跨模态任务间的特征解耦与复用**：在统一框架内，巧妙利用一个子任务（TFG）内部的特征分解（$f_{id}$ 和 $f_m$）来解决另一个子任务（TTS）的核心痛点（语音一致性）。这种“内部资源反哺”的联合建模思想具有重要启发。
3.  **用高级语义特征替代低级声学特征**：使用 TTS 的中间文本特征（结合能量）通过 MRF 模块生成唇部驱动信号，成功替代了传统的音频编码器。这为在多模态生成中绕过声学依赖、直接利用文本/语义条件提供了新范式。
4.  **对抗训练与条件流匹配的联合优化**：框架同时集成了 GAN（用于图像生成）和 CFM（用于运动和声学特征序列生成），展示了如何在一个统一目标下协同优化两种不同的生成范式。
5.  **实验设计借鉴**：通过“w/ motion”和“w/o motion”的消融对比，直观证明了运动解耦对于保持语音一致性的必要性，这种对比实验设计非常有力，值得借鉴。

## 关键术语表
*   **TTSF (Text-to-Speaking Face)**：本文提出的统一框架名称，指代整个联合生成系统。
*   **OT-CFM (Optimal-Transport Conditional Flow Matching)**：一种生成建模方法，通过学习从简单先验分布到复杂数据分布的确定性常微分方程（ODE）流来生成数据，具有训练稳定、采样高效的优点。
*   **Matcha-TTS**：一种基于 OT-CFM 的高效文本到语音模型，本文将其作为 TTS 子系统的基座。
*   **LIA (Latent Image Animator)**：一种基于 StyleGAN 潜空间导航的图像动画化方法，本文借鉴其运动提取器和生成器架构。
*   **LSE-C (Lip Sync Error Confidence)**：一种评估生成视频与参考音频唇形同步精度的客观指标，值越高表示同步越好。
*   **ID-SIM (Identity Similarity)**：使用预训练人脸识别模型计算生成视频人脸与源图像人脸之间的余弦相似度，用于衡量身份保持能力。
*   **MRF (Multi-Receptive field Fusion)**：通过并行的 1D 卷积核（不同大小和膨胀率）融合多尺度时序信息的模块，用于构建音频映射器。
*   **C-SIM (Cosine Similarity of x-vectors)**：计算生成语音和参考语音的 x-vector 嵌入之间的余弦相似度，用于评估语音的身份相似度。

## 可复现要素
*   **数据集**：训练集 LRS3；测试集 LRS2 和 VoxCeleb2。**公开**。
*   **代码与权重**：论文项目页面 (https://mm.kaist.ac.kr/projects/faces-that-speak) 可能提供代码和预训练模型，需进一步确认。
*   **关键超参**：学习率 1e-4；优化器 Adam；训练设备 8x NVIDIA A6000 (48GB)；Mel-spectrogram 参数：window size 640, hop length 160, 80 mel bins；音频采样率 16kHz；OT-CFM 的 $\sigma_{min} = 10^{-4}$；损失权重 $\lambda_1$ 至 $\lambda_8$ 分别为 0.1, 1, 0.3, 0.1, 0.1, 1, 0.1, 1。
