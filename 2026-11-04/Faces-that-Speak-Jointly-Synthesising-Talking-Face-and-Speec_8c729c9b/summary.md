---
title: "Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speec"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Jang_Faces_that_Speak_Jointly_Synthesising_Talking_Face_and_Speech_from_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:51:20"
field: "多模态人脸生成与语音合成"
keywords: ["Talking Face Generation", "Text-to-Speech", "Flow Matching", "Multimodal Synthesis", "Motion Disentanglement", "OT-CFM"]
innovations: ["首个统一端到端文本驱动多模态合成框架 TTSF", "OT-CFM Motion Sampler 结合 AE Noise Reducer 生成时序一致的自然面部运动", "Motion-Removed Identity Feature 条件化 TTS 保障语音一致性"]
benchmarks: ["LRS2", "VoxCeleb2", "LRS3"]
---

# 论文速读：Faces-that-Speak-Jointly-Synthesising-Talking-Face-and-Speech

## 一句话总结
论文提出统一的文本驱动多模态合成框架 **TTSF**（Text-to-Speaking Face），首次实现仅从单张人脸肖像和文本输入同步生成自然语音与说话人脸视频，解决了 cascade 方法误差累积问题，并具备对未见身份（unseen identities）的强泛化能力。

## 研究问题与动机
1. **现有音频驱动 TFG 的局限**：多数 TFG 方法依赖音频作为条件，无法在仅有文本输入的场景（如 AI 聊天机器人、影视制作）中同时生成同步语音与人脸视频。
2. **Cascade 方法瓶颈**：已有文本驱动 TFG 工作（如 [68, 69, 72]）采用级联架构（TTS → TFG），存在误差累积与推理时间瓶颈；近期方法 [43] 仍需额外阶段生成 RGB 视频。
3. **头部姿态多样性不足**：已有方法难以生成反映真实场景的多样化头部姿态（head poses），且在 wild 场景（非中心构图、低 landmarks 精度）下性能退化严重。
4. **语音一致性挑战**：相同身份但不同面部运动时，已有 face-stylised TTS 方法（如 Face-TTS [33]）未能解耦运动因素，导致生成的语音风格不稳定。

## 核心贡献（创新点）
1. **首个端到端统一文本驱动多模态合成系统**：将 TFG 与 TTS 联合训练于单一框架，避免 cascade 架构的误差累积，无需额外音频编码器（文本编码器即可替代）。
2. **OT-CFM 驱动的 Motion Sampler + AE Noise Reducer**：基于 Optimal-Transport Conditional Flow Matching 构建运动采样器，结合自编码器压缩-重建机制消除噪声，生成具有时序一致性的自然头部姿态与面部运动。
3. **Motion-Removed Identity Feature Conditioning TTS**：从 TFG 视觉编码器中提取去运动（motion-removed）的 identity feature $f_{id}$ 作为 TTS 条件，有效保留说话人音色、韵律与口音一致性，区别于先前忽略运动解耦的 face-stylised TTS 方法。
4. **跨任务互补特征融合设计**：TFG 使用 TTS 中间表征（$f_{lip}$）驱动唇部运动，TTS 使用 TFG 提取的 identity feature 条件化语音生成，形成双向信息流，提升唇形同步与自然度。

## 方法详解
**整体架构（TTSF）**：共享视觉编码器 $E_v$ 同时处理源图像 $I_s$ 与驱动帧 $I_d$，输出视觉特征 $f_s, f_d$。

1. **Motion Extractor**：学习身份与运动特征的线性可分表示：$E_v(I) = f_{id} + f_m$，采用 LIA [67] 架构（5层 MLP + 正交约束的 trainable motion codes），独立提取身份与运动特征，而非计算相对运动。

2. **Motion Fusion & Generator**：TFG 生成器采用 StyleGAN-styled flow-based generator（LIA 变体），仅在 lip motion 层（第6、7层）注入音频特征，其余层融合 $f_{id} + f_m$：
$$f_{z,n} = \begin{cases} f_{id} + f_m & i \in \text{non-lip layers} \\ f_{id} + f_{lip} & i \in \text{lip layers} \end{cases}$$

3. **Audio Mapper**：替代传统音频编码器，使用 TTS 中间表征（文本嵌入 $e_t$ + upsampled text feature $\tilde{f}_t$ + energy）经 Multi-Receptive field Fusion (MRF) 模块聚合生成唇部运动特征 $f_{lip}$。

4. **OT-CFM Motion Sampler**：基于 Conditional Flow Matching 的变分运动采样，通过 ODE 从先验分布生成高质量运动码序列：
$$\mathcal{L}_{\text{OT-CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_0(x_0)} \| u_t^{\text{OT}}(\phi_t^{\text{OT}}(x_0)|x_1) - v_t(\phi_t^{\text{OT}}(x_0)|\mu;\theta) \|^2$$
其中先验 $\mu$ 由 4 层 Conformer 输出的帧级均值参数化。由于直接回归 $f_m$ 会导致抖动，引入 auto-encoder-based motion normaliser 将压缩特征 $f_c$ 作为 target $x_1$，训练损失包括：
- AE 重建损失：$\mathcal{L}_{AE} = \|\hat{f}_m - f_m\|_2$
- Prior loss（负对数似然）：$\mathcal{L}_{prior} = -\sum_{j=1}^{T} \log \varphi(f_{c,j}; \mu_j, I)$

5. **TTS 系统**：基于 Matcha-TTS [40]，将 $f_{id}$ 输入 encoder 与 decoder，使用 prior、duration 与 OT-CFM 损失（合称 $L_{TTS}$）。

6. **总损失函数**：
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{GAN} + \lambda_2 \mathcal{L}_{rec} + \lambda_3 \mathcal{L}_{id} + \lambda_4 \mathcal{L}_{sync} + \lambda_5 \mathcal{L}_{OT-CFM} + \lambda_6 \mathcal{L}_{ae} + \lambda_7 \mathcal{L}_{prior} + \lambda_8 \mathcal{L}_{tts}$$
超参数：$\lambda = [0.1, 1, 0.3, 0.1, 0.1, 1, 0.1, 1]$

## 实验与结果
**数据集**：训练集 LRS3（TED/TEDx 室内视频）；测试集 LRS2 与 VoxCeleb2（户外场景、更具挑战性）。

**评估基线**：MakeItTalk [76]、Audio2Head [64]、SadTalker [73]（与本文 TTS 级联）；Face-TTS [33]（face-stylised TTS 基线）。

**主要结果（LRS2，one-shot 设置）**：

| 方法 | FID↓ | ID-SIM↑ | LSE-C↑ | DIV↑ |
|------|-------|----------|---------|-------|
| MakeItTalk+TTS | 25.168 | 0.850 | 3.487 | 0.095 |
| Audio2Head+TTS | 42.262 | 0.225 | 5.478 | 0.145 |
| SadTalker+TTS | 20.729 | 0.859 | 6.256 | 0.111 |
| **TTSF (Ours)** | **18.348** | **0.864** | 5.686 | 0.143 |

- **视频质量**：FID 最低（18.348），ID-SIM 最高（0.864），显著优于所有基线。
- **同步性**：LSE-C 低于 SadTalker（6.256 vs 5.686），但用户研究 MOS 显示唇同步性更优（见下表）。
- **多样性**：DIV 达 0.143，接近 Audio2Head（0.145）且后者视频质量极差。

**VoxCeleb2 结果**：ID-SIM 达 0.876（最优），LSE-C 5.721，DIV 0.143。

**TTS 评估（LRS2）**：

| 方法 | WER↓ | MCD↓ | C-SIM↑ | RMSE↓ |
|------|------|------|---------|--------|
| Face-TTS | 18.02 | 6.85 | 0.272 | 52.33 |
| **Ours (w/o motion)** | **14.56** | 7.23 | **0.593** | **48.52** |
| Ours (w/ motion) | 15.68 | 7.43 | 0.451 | 50.67 |

- 去运动 conditioning 显著提升语音清晰度（WER 14.56）、音色相似度（C-SIM 0.593）与 F0 精度（RMSE 48.52）。
- 引入 motion feature 后性能退化，验证 motion disentanglement 必要性。

**用户研究（MOS，满分 5）**：

| 方法 | Lip Sync | Motion Naturalness | Video Realness |
|------|----------|-------------------|----------------|
| MakeItTalk | 2.44 | 2.79 | 2.79 |
| Audio2Head | 2.73 | 2.92 | 2.48 |
| SadTalker | 2.78 | 2.80 | 2.90 |
| **TTSF (Ours)** | **4.09** | **3.85** | **3.87** |

- 三项指标全面领先，差距显著（95% 置信区间不重叠）。

**最强提升**：相比 SadTalker 级联，FID 降低 2.38（20.729→18.348），ID-SIM 提升 0.005，MOS 三项均提升约 1 分。

## 相关工作脉络
1. **MakeItTalk [76]**：音频驱动 TFG，通过 speaker-aware 机制生成 lip 运动；本文将其改造为文本驱动基线（级联 TTS），但 cascade 方式存在误差传播。
2. **Audio2Head [64]**：强调头部姿态多样性，但 ID-SIM 极低（0.151），且对非中心构图输入完全失效，泛化能力弱。
3. **SadTalker [73]**：学习 3D motion coefficients，lip sync 指标（LSE-C）较高，但面部运动范围受限（DIV 仅 0.111）。
4. **Face-TTS [33]**：首个 face-stylised TTS，但忽略 facial motion disentanglement，导致相同身份不同运动时语音风格不稳定（t-SNE 聚类效果差）。
5. **LIA [67]**：潜空间运动编码基础，正交约束与 motion codes 设计被本文沿用；但 LIA 计算 source-target 相对运动，不适合本文的去耦场景。
6. **Matcha-TTS [40]**：OT-CFM 基础 TTS 架构，本文将其扩展为 face-conditioned 版本，并联合 TFG 训练。
7. **Uniflg [43]**：近期仅生成 facial keypoints 的文本驱动方法，仍需额外 RGB 生成阶段，本文实现端到端 unified 框架。

## 局限性与未来方向
1. **LSE-C 指标相对较低**：尽管用户研究（MOS）证明唇同步感知更优，但 LSE-C 客观分数不及 SadTalker，反映该指标对 pretrained model 的依赖可能误导评估。
2. **训练数据限制**：LRS3 为室内 TED 视频，训练分布与户外 wild 场景存在 gap；VoxCeleb2 无对应文本转录，需借用 LRS3 随机文本模拟测试。
3. **Motion Normaliser 引入额外复杂度**：自编码器压缩-重建增加推理开销，且 AE 训练与主任务联合优化可能产生权衡。
4. **未见身份的泛化验证不足**：虽声称 generalise to unseen identities，但缺乏大规模跨数据集身份泛化实验（如训练于 LRS3，测试于完全独立的数据集）。
5. **潜在安全风险**：生成逼真说话人脸与语音的技术可能被滥用进行 deepfake 伪造。

## 研究启发与可借鉴点
1. **跨任务特征复用范式**：TFG 的 identity feature 直接用于 TTS conditioning，TTS 的中间表征替代音频编码器用于 TFG——这种"交叉赋能"设计可迁移至其他多模态联合生成任务（如 lipreading + TTS、视频续写等）。
2. **OT-CFM + AE 平滑策略**：直接将流匹配 target 设为原始 motion feature 会导致时序抖动，通过自编码器压缩目标分布可有效平滑；该思路适用于其他需要高保真时序生成的场景（如动作生成、视频插帧）。
3. **Motion Disentanglement 的重要性**：本文证明在 face-stylised TTS 中忽略运动解耦会严重损害语音一致性；后续工作可探索更精细的身份-运动-表情三元分解。
4. **用户研究补充客观指标的必要性**：LSE-C 与 MOS 结果不一致提示单一自动指标不足以评估说话人脸生成质量，建议建立多维度综合评测体系。
5. **One-Shot 设置的现实意义**：仅凭单张静态肖像生成多样化说话视频，符合实际应用场景（如数字人客服、虚拟主播），值得在更多下游任务中验证。

## 关键术语表
**TTSF**（Text-to-Speaking Face）：本文提出的统一文本驱动多模态合成框架，同时生成说话人脸视频与自然语音。
**OT-CFM**（Optimal-Transport Conditional Flow Matching）：一种基于最优传输理论的 flow matching 生成方法，通过求解 ODE 从先验分布生成高质量样本。
**Motion Extractor**：从视觉特征中解耦身份（$f_{id}$）与运动（$f_m$）的模块，采用正交约束的 MLP 实现独立提取。
**Audio Mapper**：将 TTS 文本中间表征转换为唇部运动特征 $f_{lip}$ 的模块，替代传统音频编码器。
**MRF Module**（Multi-Receptive field Fusion）：通过不同核大小与膨胀率的 1D 卷积残差块聚合多尺度时序特征。
**LSE-C**（Lip Sync Error Confidence）：衡量生成视频唇形与音频同步程度的自动评估指标。
**DIV**（Diversity）：生成帧中头部姿态特征的标准差，衡量运动多样性。
**Face-Stylised TTS**：仅凭人脸图像条件化生成的文本到语音系统，目标是复现输入说话人的音色与口音特征。

## 可复现要素
- **训练数据集**：LRS3（公开）；测试集 LRS2（公开）、VoxCeleb2（公开）。
- **代码开源状态**：项目页面 https://mm.kaist.ac.kr/projects/faces-that-speak，但论文未明确声明 GitHub 仓库。
- **预训练权重**：未提及。
- **关键超参数**：Adam lr=1e-4；$[λ_1,...,λ_8] = [0.1, 1, 0.3, 0.1, 0.1, 1, 0.1, 1]$；σ_min=1e-4；32-frame videos 训练 motion sampler；Mel-spectrogram window=640, hop=160, bins=80；16kHz 音频采样率；8×48GB A6000 GPU。
- **预训练组件**：Matcha-TTS 在 LRS3 上预训练 2000 epochs；VGG19、SyncNet、Face Recognition Network 均为预训练权重。
