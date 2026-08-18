---
title: "CommonCanvas-Open-Diffusion-Models-Trained-on-Creative-Commo"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Gokaslan_CommonCanvas_Open_Diffusion_Models_Trained_on_Creative-Commons_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:13:57"
field: "版权友好的文生图模型"
keywords: ["扩散模型", "文生图", "Creative Commons", "数据效率", "合成字幕", "版权", "Stable Diffusion", "Telephoning"]
innovations: ["提出telephoning范式，用BLIP-2为CC图像合成字幕", "实证证明SD2模型在~90M数据量即性能饱和", "开源基于CC图像的CommonCatalog数据集和CommonCanvas系列模型"]
benchmarks: ["MS COCO", "PartiPrompts"]
---

# 论文速读：CommonCanvas: Open Diffusion Models Trained on Creative-Commons Images

## 一句话总结
本研究构建了一个完全基于**创意共享（Creative-Commons, CC）**许可图像的新训练数据集（CommonCatalog），并通过**“telephoning”**（用预训练BLIP-2模型生成合成字幕）解决CC图像缺乏高质量文本标注的问题，证明了仅用约70M（LAION-2B的不到3%）这样的数据量，配合系统级训练优化，即可训练出与公开版Stable Diffusion 2 (SD2)在人类评估上性能相当的开源文生图扩散模型。

## 研究问题与动机
*   **CC图像缺乏高质量字幕：** 现存的CC图像大多只有简短的图片标题或URL，缺乏训练高质量文生图模型所需的丰富、结构化自然语言描述（caption）。
*   **高质量CC图像数据相对稀缺：** 可用的、分辨率适合训练SD2的高分辨率CC图像估计仅有约7000万张，远少于LAION-2B的约20亿张。
*   **LAION数据的版权与可复现性困境：** LAION-2B源自网页爬取，图像版权状态不明确，面临诸多法律诉讼；且仅提供URL，存在链接失效（link rot）风险，难以完全复现训练数据。
*   **SD2训练成本高昂：** 训练一个SD2模型需要约20万A100 GPU小时，进行数据量与质量的影响分析代价巨大，需要高效的训练流程。

## 核心贡献（创新点）
1.  **提出“telephoning”作为解决多模态生成模型数据稀缺的转移学习范式：** 使用预训练的BLIP-2视觉-语言模型，将高维图像“有损压缩”为低维文本描述，再用于训练目标文生图扩散模型，实现了在无真实标签的CC图像上生成高质量合成字幕。
2.  **系统性地验证了小数据量训练高质量扩散模型的可行性：** 通过在不同规模的LAION子集上训练SD2，证明SD2在约90M样本时即趋于饱和，性能不再随数据量增加而显著提升，这与CommonCatalog的数据规模相当。
3.  **实现并开源了一套高效的训练优化方案：** 综合应用Flash Attention、预计算VAE/文本编码器latent、将GroupNorm/LayerNorm降至float16、使用FSDP数据并行等技术，将SD2基线训练速度提升了2.71倍，使得大规模数据效率分析成为可能。
4.  **构建并开源了完整的CC图像文生图模型研究生态：** 发布了规模约70M、包含商业和非商业许可类别的CommonCatalog数据集（含BLIP-2合成字幕）、训练得到的CommonCanvas系列模型，以及全部训练代码。

## 方法详解
*   **Telephoning Caption Synthesis：**
    *   **第一步（有损压缩）：** 使用预训练并冻结的**BLIP-2-OPT-2.7B**模型（视觉编码器+跨模态Transformer+冻结的LLM）处理输入的CC图像。将图像编码为视觉特征，生成文本提示，再由LLM生成一段描述性文本作为合成字幕（caption）。此过程被形象地比喻为“传声筒”游戏。
    *   **第二步（模型训练）：** 使用生成的(CC图像, 合成字幕)对作为训练数据，去训练目标文生图扩散模型。作者强调，在此过程中只使用了BLIP-2的“映射”能力（输入图像，输出文本），并未直接访问BLIP-2预训练所依赖的LAION-400M等原始图像数据。
*   **CommonCatalog数据集构建：**
    *   **数据源：** 从**YFCC100M**数据集出发，根据其提供的Flickr ID重新下载高分辨率（超过4K）图像。
    *   **过滤：** 排除带有“禁止演绎”（ND）许可证的图像。
    *   **划分：** 根据是否允许商业用途，构建**CommonCatalog-C**（约2620万张，允许商用）和**CommonCatalog-NC**（约6700万张，包含C及非商用图像）。
    *   **预处理：** 为节省BLIP-2推理成本，图像被中心裁剪并缩放至最大边512x512像素。但在最终训练CommonCanvas模型时，使用原始高分辨率图像。
*   **CommonCanvas模型训练与优化：**
    *   **模型架构：** 主要基于SD2架构（约8.65亿参数的UNet），并额外尝试了SDXL的更大UNet架构（CommonCanvas-L系列）。
    *   **训练加速技术：**
        1.  **Flash Attention** with xFormers库。
        2.  **预计算Latent：** 在整个训练集上预先计算好VAE和文本编码器（OpenCLIP）的latents，避免在训练循环中重复计算。
        3.  **降低精度：** 将GroupNorm和LayerNorm的计算从float32降至float16。
        4.  **Fully Sharded Data Parallelism (FSDP)** 用于分布式训练。
        5.  **EMA权重保存：** 仅在最后3.5%的训练步骤中保存指数移动平均（EMA）权重。
    *   **数据效率分析：** 通过在LAION-2B的不同随机子集（1.1B, 90M, 10M, 1M）上训练SD2，评估数据量对模型性能的影响，得出结论：约90M数据足以训练出高质量的SD2。

## 实验与结果
*   **数据集：** CommonCatalog-C (26.2M), CommonCatalog-NC (67.0M)。基线对比使用LAION子集训练的SD2变体。
*   **评估基线与指标：**
    *   **自动化指标：** MS COCO验证集上的**FID** (Frechet Inception Distance), **KID** (Kernel Inception Distance), **CLIP-FID**, 以及文本-图像对齐度**CLIP-Score**。越低/越高越好。
    *   **人类评估：** 使用**PartiPrompts**数据集，进行成对偏好比较，计算用户更偏好CommonCanvas模型而非SD2的比例。
*   **主要结果：**
    *   **数据效率：** SD2在90M样本（使用原始LAION字幕或BLIP-2合成字幕）上的FID/KID性能与在1.1B全量数据上训练的模型**相当**。
    *   **合成字幕效果：** 使用BLIP-2合成字幕训练，在MS COCO（人类字幕）上的CLIP-FID优于在web抓取字幕数据集Conceptual Captions上的表现，表明合成字幕与高质量人类字幕的对齐更好。
    *   **CommonCanvas vs SD2：**
        *   **小型模型 (S):** CommonCanvas-S-C和S-NC在人类评估中分别以**37%**和**38%**的偏好率略低于SD2，但考虑到数据量和字幕性质，表现已属不错。
        *   **大型模型 (L):** **CommonCanvas-L-NC**（使用SDXL UNet）在人类评估中与SD2相比**无统计显著差异**，达到了可比的性能水平。
    *   **弱点：** 在人脸、通用摄影和绘画等类别上，CommonCanvas表现弱于SD2。对于包含著名人物/角色的提示词，生成结果偏离原角色程度更高。
    *   **速度提升：** 优化的训练流水线实现了**2.71倍**的训练速度提升。

## 相关工作脉络
*   **Stable Diffusion / SD2 / SDXL：** 本文研究的直接对标对象。SD2基于LAION-2B训练；SDXL是更大架构的演进版本，CommonCanvas-L采用了其UNet。
*   **BLIP-2：** 用于本文“telephoning”步骤的核心预训练视觉-语言模型。其预训练数据包含LAION-400M，但本文只利用其生成字幕的能力，不直接使用原始数据。
*   **DataComp / OBELICS：** 其他大规模多模态数据集构建工作。DataComp探索了不同规模和类型数据的影响；本文与其不同之处在于聚焦于**版权明确**的CC图像，并通过**合成字幕**解决标注问题。
*   **SynthCap (Concurrent Work)：** 同步发表的工作，使用扩散模型根据提示词生成图像再用captioning模型生成描述，过程与本文相反。两者都探索了合成数据，但起点和路径不同。
*   **Laion相关研究及法律讨论：** 本文动机来源于LAION数据集面临的版权诉讼和可复现性问题。引用了关于“合理使用”的法律专家观点，以及针对生成模型版权问题的学术讨论，将本文工作与法学界的研究联系起来。

## 局限性与未来方向
*   **数据来源时效性：** YFCC100M数据源较旧（约十年前），其CC图像不如LAION-2B中的网页爬取数据新颖和多样。
*   **特定领域性能不足：** 在人脸、专业摄影、绘画等类别上，模型生成质量不及SD2，可能与CC图像中此类内容占比或特征有关。
*   **潜在的人物/IP生成风险：** 尽管模型倾向于偏离知名角色，但通过诱导性提示词仍有可能生成与真实人物或受版权保护形象相似的图像，并未从根本上解决所有IP相关问题。
*   **未充分探索更大规模数据/模型：** 本文主要验证了在小数据量下训练SD2级别模型的可行性，未来需要探索将数据量扩大到数百万甚至十亿级CC图像，并测试更大架构（如DiT）的效果。
*   **未来方向：** 作者计划扩充CommonCatalog，纳入更多来源的CC图像；测试更大模型架构；使用更先进的captioning模型（如LLaVA）改进字幕质量；以及进一步探索数据效率与模型能力的关系。

## 研究启发与可借鉴点
1.  **“Telephoning”范式的可迁移性：** 本文提出的“用强大预训练模型为无标注数据生成软标签，再训练目标模型”的思路，不仅限于图像-文本，也可借鉴于其他模态组合（如视频-文本、3D-文本）或解决其他标注稀缺问题。
2.  **对“数据饱和点”的实证研究价值：** 系统性地评估不同数据量对主流基础模型性能的影响，其方法论和对“SD2可能存在欠拟合”的发现，对规划其他大模型训练的数据预算具有重要参考价值。
3.  **训练效率优化组合拳：** 将Flash Attention、latent预计算、混合精度归一化、FSDP等技术系统化应用并开源，为社区复现和高效训练扩散模型提供了实用指南。
4.  **开放版权数据路线的可行性验证：** 证明了在不依赖LAION等灰色地带数据集的情况下，通过仔细筛选开源许可数据和智能标注，可以训练出具有竞争力的开源生成模型，为关注版权合规的研究者和机构提供了一条可行路径。
5.  **合成字幕的质量与分布偏移：** 实验揭示了合成字幕在分布偏移（web caption vs. human caption）下的不同表现，提示在构建训练数据时，需要考虑字幕的“风格”与下游任务评估标准之间的匹配。

## 关键术语表
*   **CommonCatalog:** 本文构建的开源训练数据集，包含约7000万张带有BLIP-2合成字幕的Creative-Commons许可高分辨率图像，分为允许商用(C)和非商用(NC)两个子集。
*   **Telephoning:** 本文提出的术语，指代一种将预训练的“编码器”模型（如BLIP-2）用于为无标签数据生成合成标签（如图像字幕），再用这些合成标签去训练另一个“解码器”模型（如扩散模型）的转移学习模式。
*   **SD2-90M:** 作者在LAION-2B的9000万样本子集上训练的SD2基准模型，用于证明SD2模型在远低于原始数据量（1.1B可用样本）下即可达到性能饱和。
*   **PartiPrompts:** 一个包含多种复杂场景和概念的提示词数据集，被用作本文人类偏好评估的实验基准。
*   **MS COCO:** Microsoft COCO图像数据集，因其字幕由人工撰写且质量较高，被用作评估模型与高质量人类语言对齐程度的测试集。
*   **FSDP (Fully Sharded Data Parallelism):** PyTorch中的一种数据并行训练策略，通过将模型参数、梯度和优化器状态分片到多个GPU上，以降低显存占用。
*   **CLIP-Score / CLIP-FID:** 基于预训练的CLIP模型计算的评估指标。CLIP-Score衡量生成图像与文本提示的语义对齐度；CLIP-FID则衡量生成图像在CLIP特征空间中的分布与真实图像分布的距离。

## 可复现要素
*   **数据集：** CommonCatalog-C 和 CommonCatalog-NC 数据集及数据卡片已在GitHub开源。
*   **代码：** 训练CommonCanvas模型和优化后的SD2训练流水线的代码已在GitHub开源。
*   **模型权重：** CommonCanvas系列模型权重已计划在GitHub发布（链接指向MosaicML的diffusion仓库中的common-canvas.md）。
*   **关键超参/细节：**
    *   **训练规模：** 主要实验基于约70M（90M以下）样本。
    *   **分辨率：** 图像预处理最大边512x512用于captioning；训练时使用高分辨率原始图像。
    *   **骨干模型：** UNet架构基于SD2（~865M参数）和SDXL；文本编码器使用OpenCLIP；字幕生成使用BLIP-2-OPT-2.7B。
    *   **优化器：** 论文未明确提及，需查阅开源代码。
    *   **硬件：** 主要测试在128块NVIDIA A100 GPU上进行。
