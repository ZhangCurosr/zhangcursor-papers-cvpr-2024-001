---
title: "CommonCanvas-Open-Diffusion-Models-Trained-on-Creative-Commo"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Gokaslan_CommonCanvas_Open_Diffusion_Models_Trained_on_Creative-Commons_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:13:36"
field: "开源/合规文本到图像生成"
keywords: ["text-to-image diffusion", "Creative Commons", "synthetic captioning", "data efficiency", "copyright-safe training", "latent diffusion model"]
innovations: ["提出telephoning两阶段合成标注方法，用BLIP-2为CC图像生成caption", "证明SD2仅需LAION-2B的<3%（约70M样本）即可训练出同等质量模型", "实现2.71×训练加速并开源CommonCatalog数据集与CommonCanvas模型系列"]
benchmarks: ["MS COCO (FID, KID, CLIP-FID, CLIP-Score)", "Parti-Prompts human preference evaluation"]
---

# 论文速读：CommonCanvas: Open Diffusion Models Trained on Creative-Commons Images

## 一句话总结
论文提出了一种基于Creative-Commons（CC）许可图像的开源文本到图像扩散模型训练方案：通过预训练BLIP-2生成合成标注（"telephoning"），结合仅用LAION-2B约3%（约70M样本）的高效训练配方，训练出的CommonCanvas-L-NC模型在人类评估中与Stable Diffusion 2（SD2）性能相当。

## 研究问题与动机
1. **版权不确定性**：LAION-2B等Web爬取数据集的法律地位尚存争议，美国法院尚未明确判定其是否构成版权法下的"合理使用"，已有多起诉讼（Getty Images、LAION相关案件）。
2. **可复现性与数据安全性**：LAION仅提供URL而非图像本体，存在严重的"链接腐烂"问题（link rot），无法完全复现训练数据，且存在数据投毒风险；此外LAION已因包含CSAM内容不再公开。
3. **CC图像标注缺失**：CC图像普遍缺乏训练T2I模型所需的高质量alt-text caption，多数仅有图片标题或URL。
4. **CC图像数据量稀缺**：可用的高分辨率CC图像仅约7000万张，远低于LAION-2B的约20亿规模，需验证小规模数据能否训练出高质量模型。

## 核心贡献（创新点）
1. **提出"telephoning"合成标注方法**：利用预训练BLIP-2对无标注CC图像生成高质量合成caption，本质是将高维图像"有损压缩"为低维文本再用于训练T2I模型，与既往仅依赖Web爬取alt-text的方法有本质区别。
2. **发现SD2可用<3% LAION数据训练出同等质量模型**：通过系统的LAION子集实验（1.1B→90M→10M→1M），证明SD2可能存在欠参数化，仅需约70M高质量样本即可饱和训练，挑战了"需要数十亿数据"的固有认知。
3. **实现2.71×训练加速的系统级优化**：综合运用Flash Attention、VAE/text encoder latent预计算、GroupNorm/LayerNorm转float16、FSDP及EMA权重快照等技术，使大规模扩散模型训练时间大幅缩短。
4. **构建并开源CommonCatalog数据集与CommonCanvas模型系列**：收录约7000万CC图像（2600万商用+6700万非商用）及对应合成标注，同时开源两个规模模型（Small/865M参数与Large/SDXL-UNet），填补了纯开放许可T2I训练的空白。

## 方法详解
**Telephoning流程**：
- 输入：经筛选的高分辨率CC图像（中心裁剪+resize至512×512用于标注，训练时使用高分辨率原始图像）
- 标注生成：使用预训练的BLIP-2 OPT 2.5B模型（冻结视觉编码器+冻结LLM，仅训练中间Transformer桥接层），将图像编码为短文本caption；该过程约消耗1,120 GPU A100小时
- 模型训练：将合成caption与CC图像配对形成CommonCatalog，用于训练Latent Diffusion Model（LDM）

**训练架构**：
- CommonCanvas-S：采用SD2 UNet架构（约865M可训练参数）
- CommonCanvas-L：将UNet替换为SDXL的大规模网络（约2.6B参数），用于验证大数据量下大模型是否过拟合

**训练加速优化组合**：
- Flash Attention + xFormers库
- VAE和text encoder latent预计算（全训练集）
- GroupNorm/LayerNorm转换为float16精度
- Fully Sharded Data Parallelism（FSDP）
- 仅在训练最后3.5%阶段保留EMA权重

**数据规模验证**：在LAION-1.1B上随机采样不同比例子集训练SD2，评估FID/KID/CLIP-Score，发现10M和90M子集与1.1B全量数据表现相当，1M以下才开始下降。

## 实验与结果
**评测数据集与指标**：MS COCO（30K样本），FID、KID、CLIP-FID、CLIP-Score；Parti-Prompts人类偏好评估（ pairwise preference）

**主要结果**：
- **数据量分析**：SD2在10M和90M LAION子集上的FID/KID与1.1B全量数据无显著差异，印证"数据需求可大幅压缩"的结论
- **合成标注效果**：BLIP-2合成caption训练的模型CLIP-Score高于原始LAION caption；CLIP-FID在MS COCO（人工标注）上优于Conceptual Captions（机器标注）
- **人类评估**：CommonCanvas-S-C偏好率37%，CommonCanvas-S-NC偏好率38%（相对于SD2）；**CommonCanvas-L-NC与SD2无统计显著差异**，达到同等质量水平
- **与SD2对比**：CommonCanvas在人脸、通用摄影和绘画类别上表现略逊；但在生成知名人物/角色时有意偏离（Figure 10/11），隐私友好性更强

**最优模型**：CommonCanvas-L-NC（SDXL UNet + 67M CC图像 + 合成caption），在Parti-Prompts人类偏好评估中达到与SD2等效水平。

## 相关工作脉络
1. **SynthCap（Caffagni et al., 2023）**：用扩散模型根据caption生成图像后再反向caption，与本文"图像→caption"方向相反；本文是首个完全从零构建CC数据集并开源模型的工作。
2. **LLaVA（Liu et al., 2023）**：使用VQA模型增强caption数据集；本文指出此类caption upsample方法可应用于CommonCatalog以提升标注质量，但未在本文中实施。
3. **DataComp/OBELICS等Web爬取数据集**：直接从Common Crawl提取caption；本文强调LAION caption质量参差不齐（产品名、语法错误），BLIP-2合成caption在CLIP-Score上反而更优。
4. **Ablating Concepts in T2I（Kumari et al., 2023）**：针对版权问题的消融实验；本文从数据源层面彻底规避版权争议，而非事后修改模型。
5. **SILO Language Models（Min et al., 2023）**：针对text-to-text的版权隔离方案；本文将其思路扩展到T2I领域，但采用了完全不同的数据策略（CC图像替代Web爬取）。
6. **LAION-5B相关版权研究（Lee et al., 2023）**：从法律角度分析"non-expressive/non-consumptive"合理使用论点；本文引用其论证BLIP-2生成短句caption的版权安全性。

## 局限性与未来方向
1. **数据时效性**：CommonCatalog基于YFCC100M（约10年前的CC图像），不如LAION-2B内容新；计划从其他来源扩充CC图像。
2. **模型表现不均衡**：在人脸、摄影、绘画等依赖Conceptual Captions分布的类别上表现较弱；人类偏好略低于SD2（小幅）。
3. **caption多样性受限**：BLIP-2生成的标注CLIP-Score更高但caption diversity较低（引用Nguyen et al., 2023），可能限制模型创造性。
4. **未探索更大模型与先进captioner**：仅用了SDXL UNet作为大模型验证，未测试LLaVA等更新caption模型的融合效果。
5. **潜在风险**： adversarial prompt仍可能生成近似知名角色的图像（尽管偏离度高于SD2）。

## 研究启发与可借鉴点
1. **"telephoning"范式可迁移**：对于任何缺乏标注的高价值图像数据（如专业领域图像、历史档案），可复用"预训练I2T模型生成合成标注→训练目标模型"的两阶段transfer learning方案。
2. **数据效率分析的实验设计**：通过系统性的LAION子集缩放实验（1.1B→90M→10M→1M）验证模型饱和点，这种方法论可直接迁移到其他架构（如DiT、SD3）的数据需求评估。
3. **训练加速优化组合的复用价值**：2.71×提速所涉及的Flash Attention + latent预计算 + FSDP + float16 norm等技术栈，可作为后续扩散模型高效训练的标准基线配置。
4. **大模型在少数据下的过拟合防御**：验证SDXL规模UNet在70M数据上不过拟合，提示未来研究可在小数据+大模型组合上做进一步探索（当前主流认知是小数据易过拟合大模型）。
5. **开源合规替代方案的构建思路**：为希望规避LAION版权风险的研究团队提供了完整的参考实现（从数据筛选→标注生成→模型训练→开源发布的全链路），可直接作为合规T2I研究的起点。

## 关键术语表
**Telephoning**：本文提出的术语，指用预训练生成模型将高维模态（图像）"有损压缩"为低维模态（文本caption），再利用该合成标注训练另一生成模型的两阶段transfer learning方法，类比"电话游戏"的信息失真传递。
**CommonCatalog**：本文构建的开源多模态训练数据集，约7000万张CC许可图像及对应的BLIP-2合成caption，分为商用（C）和非商用（NC）两个子集。
**CommonCanvas**：基于CommonCatalog训练的家庭化LDM模型系列，包含Small（SD2 UNet）和Large（SDXL UNet）两个架构，分别训练于C和NC数据。
**SD2-90M**：在LAION-90M子集上训练的SD2变体，用于验证数据量下限的中间实验模型。
**CLIP-FID**：结合CLIP语义对齐与FID分布距离的混合评测指标，越低表示生成图像与caption的语义匹配度和视觉质量越好。
**Parti-Prompts**：Yu et al.提出的用于文本到图像模型人类评估的prompt集合，本文用于pairwise preference评级。
**FSDP（Fully Sharded Data Parallelism）**：一种分布式训练数据并行策略，将模型参数、梯度和优化器状态分片到多个GPU上以降低显存占用。
**YFCC100M**：包含1亿张CC许可图像的多媒体数据集，本文的CommonCatalog数据源，含Flickr ID和元数据。

## 可复现要素
- **数据集**：CommonCatalog约70M CC图像+合成caption，已开源（GitHub: mosaicml/diffusion）；LAION子集实验使用LAION-2B/1.1B公开数据
- **代码**：训练代码及优化实现已开源（GitHub链接见论文）
- **权重**：CommonCanvas-S-C、CommonCanvas-S-NC、CommonCanvas-L-NC模型权重已开源
- **关键超参**：图像resize至512×512用于caption生成；训练使用A100 GPU；约1,120 A100小时用于BLIP-2 caption生成；SD2基线训练估计约200,000 A100小时（Stability AI官方数据）
- **BLIP-2模型**：使用LAION-400M预训练的BLIP-2 OPT 2.5B版本
