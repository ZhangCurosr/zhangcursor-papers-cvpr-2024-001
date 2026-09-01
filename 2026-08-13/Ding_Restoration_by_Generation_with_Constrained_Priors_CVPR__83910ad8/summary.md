---
title: "Restoration by Generation with Constrained Priors"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Ding_Restoration_by_Generation_with_Constrained_Priors_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:42:01"
field: "低层视觉与图像复原"
keywords: ["图像复原", "扩散模型", "盲人脸复原", "生成约束先验", "个性化生成", "skip guidance"]
innovations: ["通过加噪将退化图像投影到扩散模型生成空间并直接去噪采样实现盲复原", "利用生成相册/个性化相册微调约束扩散模型生成空间，无需假设退化类型", "提出skip guidance策略周期性强弱引导，平衡生成质量与输入忠实度"]
benchmarks: ["Wider-Test", "LFW-Test", "WebPhoto-Test", "Deblur-Test", "AFHQ Dog/Cat"]
---

# 论文速读：Restoration by Generation with Constrained Priors

## 一句话总结
本文提出利用预训练扩散模型的生成能力直接进行图像复原：只需向退化输入添加足够噪声使其分布与干净图像接近，再用微调后的扩散模型去噪采样即可。核心创新是通过"锚图像"（personal album 或 generative album）约束生成空间，在不假设退化类型的前提下，同时保证生成质量与输入特征的忠实保留。

## 研究问题与动机
1. **现有判别式方法依赖合成配对数据**：GFPGAN、CodeFormer、VQFR 等方法在合成退化数据上训练，难以泛化到真实世界具有未知复合退化的图像。
2. **基于模型的方法要求已知退化算子**：MAP 后验采样类方法（如 DGP、Diffusion Posterior Sampling）在推理时需已知退化过程 H，限制了实际应用。
3. **直接利用扩散模型去噪会丢失输入信息**：预训练扩散模型的生成空间过于发散，从含噪输入出发的去噪路径可能偏离输入的真实内容（如人脸身份）。
4. **单张图像复原本质病态**：仅凭一张低质量图像无法唯一确定高质量原图，需要额外约束——本文通过锚图像将生成空间收缩到贴近输入的流形子空间。

## 核心贡献（创新点）
1. **扩散模型直接用于图像复原的新范式**：无需假设退化类型，仅需加噪后使用预训练扩散模型采样；与 CodeFormer 等判别式方法本质区别在于完全绕过退化建模，依靠生成模型的先验。
2. **生成空间约束（Constrained Priors）机制**：通过 personal album 或 generative album 微调扩散模型，将生成空间限制在锚图像邻近的子空间；与 DGP/Diffusion Posterior Sampling 的区别是不依赖已知退化算子进行 MAP 优化。
3. **Skip Guidance 策略构建生成相册**：在每 n 步施加一次弱 $L_1$ 引导而非每步强约束，平衡生成质量与输入忠实度；区别于 IDM/DR2 等多层注入或条件扩散方案，本文采用解耦的"先构建相册、再微调"两步流程。
4. **个性化复原的无缝扩展**：直接将 personal album 作为锚图像微调，无需额外编码器或特征注入机制，相比 InstantBooth/Dreambooth 等个性化方法，省去测试时优化开销。

## 方法详解
**基本观察与原理**：对退化图像 $y_0$ 添加 Gaussian 噪声至时刻 $K$，当 $K$ 足够大时，$y_K \approx x_K$（$x_0$ 为对应干净图像），即 $p(x_0|y_K) \approx p(x_0|x_K)$，因此可直接用预训练扩散模型从 $y_K$ 开始去噪采样，得到高质量图像。

**生成空间约束（Two-stage anchoring）**：
- **Personal Album**：给定同一主体的多张干净图像（约 20 张），直接用于微调扩散模型（5,000 iterations，lr=1e-5），使生成空间贴合该主体的高频细节与身份特征。
- **Generative Album（单图场景）**：对输入 $y_0$ 加噪至 $y_K$，使用 Skip Guidance 生成一组高质量图像。Skip Guidance 公式：
  $$x_t' = x_t - \lambda \nabla_{x_t} \| y_0 - \hat{x}_{0,t} \|_2^2$$
  每隔 $n$ 步施加一次弱引导（而非每步），重复多次得到 ~16 张生成相册图像，再以此微调模型（3,000 iterations，batch=4，lr=1e-5）。
- **推理阶段**：对测试退化图像加噪至 $y_K$（K=200），使用微调后模型直接去噪 K 步，无需额外引导。

**关键超参**：$K=200$（噪声步数），生成相册大小 16，微调迭代 3,000，学习率 1e-5。

## 实验与结果
**数据集**：Wider-Test（970 张）、LFW-Test（1771 张）、WebPhoto-Test（407 张）为真实人脸盲复原基准；Deblur-Test（67 张）为含真实运动模糊的新建基准。

**基线对比**：GFPGAN、CodeFormer、VQFR、DR2(+VQFR)、ASFFNet（个性化场景）。

**定量结果（Table 1）**：
| 数据集 | 指标 | 最优方法 | Ours | 提升 |
|---|---|---|---|---|
| Wider-Test | FID↓ | CodeFormer 48.57 | **46.38** | −2.19 |
| Wider-Test | MUSIQ↑ | GFPGAN 56.48 | **58.73** | +2.25 |
| LFW-Test | FID↓ | CodeFormer 66.31 | **56.32** | −9.99 |
| LFW-Test | MUSIQ↑ | GFPGAN 60.46 | **60.68** | +0.22 |
| Deblur-Test | FID↓ | CodeFormer 163.47 | **135.33** | −28.14 |

**最强结果**：LFW-Test FID 56.32（大幅领先第二名），Deblur-Test FID 135.33（显著优于所有基线，证明对未知退化的强泛化性）。

**个性化复原（Table 2，ArcFace 余弦相似度 IDS）**：在 Subject A（老年女性）、Obama、Hermione 三个主体上，Ours 得分分别为 0.731 / 0.716 / 0.664，全面优于 CodeFormer、DR2 及参考图像方法 ASFFNet。

**泛化能力**：在 AFHQ Dog/Cat 数据集上微调后，同样可在狗的毛发、猫的面部细节上恢复高频信息并保持身份。

## 相关工作脉络
1. **GFPGAN / GPEN / CodeFormer / VQFR / RestoreFormer**：基于 GAN/codebook/transformer 的监督学习方法，依赖合成退化配对数据训练；本文不依赖任何配对数据，泛化至真实退化。
2. **DR2**：结合预训练扩散模型与监督人脸复原网络；本文完全无监督，不借助任何预训练复原网络。
3. **Diffusion Posterior Sampling (DPS, Kawar et al.) / Zero-shot DNRM**：利用预训练扩散模型作为图像先验进行后验采样，但要求推理时已知退化算子 H；本文完全不假设退化类型。
4. **DGP（GAN Inversion based）**：通过搜索 latent code 恢复图像；本文采用直接加噪去噪范式，无需优化 latent space。
5. **ASFFNet / Li et al.**：利用多张参考图像辅助复原；本文同样支持相册参考，但通过微调而非特征注入实现，且单图场景下可自动生成相册。
6. **Dreambooth / InstantBooth / Textual Inversion**：个性化生成方法；本文将其思想迁移至图像复原，证明微调后的扩散模型本身就是约束生成空间的有效载体。

## 局限性与未来方向
1. **每幅单图需单独微调**：generative album 场景下每输入一张退化图像都要进行 3,000 步微调，推理速度较慢，不适合大规模实时应用。
2. **验证局限于类别特定任务**：当前主要验证于人脸、猫狗等类别特定扩散模型，通用自然图像盲复原尚待探索（缺乏高质量通用预训练扩散模型）。
3. **生成相册质量依赖 skip guidance 超参**：$\lambda$、$n$ 等参数需合理设置，否则可能导致锚图像偏离输入或质量不足。
4. **未来方向**：研究免微调的空间约束方法（如编码器注入、prompt-based约束），扩展到更一般的自然图像复原。

## 研究启发与可借鉴点
1. **"加噪对齐分布→直接去噪采样"的思路可迁移**：任何预训练生成模型（如 GAN、VAE）均可借鉴此范式，将低质量输入映射到生成空间后再采样，适用于视频复原、医学图像增强等任务。
2. **生成相册 + 微调的两阶段设计具有通用价值**：对于任何缺少参考图像的复原/超分任务，可通过 skip guidance 生成锚集再微调，避免引入额外的结构假设。
3. **Skip Guidance 解耦引导强度的策略**：周期性弱引导优于每步强引导，可推广至其他需要平衡生成质量与输入忠实度的扩散生成任务（如图像编辑、风格迁移）。
4. **与团队方向结合的机会**：可探索将本方法与低资源机器翻译中的领域适配思想结合——用少量目标域样本微调语言模型生成器；或将 constrained prior 思想用于跨模态检索中的质量提升。

## 关键术语表
**Diffusion Model（扩散模型）**：通过逐步添加高斯噪声再学习逆向去噪过程来建模数据分布的生成模型，具有强大的生成质量与多样性。

**Generative Album（生成相册）**：由 skip guidance 从单张退化图像生成的一组高质量锚图像，用于微调扩散模型以约束生成空间。

**Personal Album（个性化相册）**：同一主体已有的多张干净图像集合，直接作为锚图像微调扩散模型，用于个性化复原任务。

**Skip Guidance（跳过引导）**：在扩散采样过程中每隔 n 步施加一次弱 $L_1$ 引导，平衡生成质量与输入忠实度的策略。

**Blind Face Restoration（盲人脸复原）**：在退化类型未知的情况下，从单张低质量人脸图像恢复高质量自然人脸的任务。

**Constrained Prior（约束先验）**：通过锚图像微调将扩散模型的生成空间限制在输入相近子空间的技术，是本文的核心思想。

**FID（Fréchet Inception Distance）**：衡量生成图像与真实图像分布差异的指标，越低越好；常用于评估图像生成/复原质量。

**ArcFace Identity Score（身份保持分数）**：基于 ArcFace 特征余弦相似度计算的身份保持量化指标，越高表示还原后身份越忠实。

## 可复现要素
- **数据集**：Wider-Test、LFW-Test、WebPhoto-Test、Deblur-Test（部分为论文自建）、AFHQ Dog/Cat；FFHQ 用于预训练。论文未明确说明各数据集的公开链接状态，但项目网页 https://gen2res.github.io 应有补充材料。
- **代码/权重**：论文提供项目网页，代码通常开源（未在当前文本中明确声明 GitHub 链接，需查阅网页）。
- **关键超参**：K=200（噪声步数），生成相册大小 16，微调迭代 3,000（单图）/ 5,000（个性化），batch size=4，学习率 1e-5，Skip Guidance 间隔 n 步（见附录）。
