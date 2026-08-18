---
title: "Amodal-Completion-via-Progressive-Mixed-Context-Diffusion"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Xu_Amodal_Completion_via_Progressive_Mixed_Context_Diffusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:01:22"
field: "图像补全与生成"
keywords: ["Amodal Completion", "Diffusion Models", "Image Inpainting", "Object Completion", "Co-occurrence Bias", "Training-free"]
innovations: ["提出Mixed Context Diffusion Sampling克服共现偏置", "渐进式遮挡感知补全流程无需训练", "反事实筛选系统自动化评估补全质量"]
benchmarks: ["COCO", "Open Images"]
---

# 论文速读：Amodal-Completion-via-Progessive-Mixed-Context-Diffusion

## 一句话总结
本文提出一种基于预训练扩散模型的无训练物体模态补全方法，通过混合上下文扩散采样（Mixed Context Diffusion Sampling）和渐进式遮挡感知流程，有效克服共现偏差，实现自然图像中被遮挡物体（含超出边界部分）的逼真补全。

## 研究问题与动机
1. **模态补全的定义与重要性**：模态补全（Amodal Completion）指补全被遮挡物体的完整外观（包括可见与隐藏部分），在机器人、自动驾驶、AR等领域具有重要应用价值。
2. **现有两步方法的局限**：传统方法先预测二值amodal mask再生成像素，但直接回归amodal mask是病态问题（完成结果多样性大），且合成数据与真实图像存在domain gap。
3. **扩散模型直接使用的失败模式**：直接使用预训练扩散inpainting模型去除遮挡物时，容易因上下文偏置（contextual bias）重新生成与原遮挡物相似的"共存对象"（co-occurrence bias），而非补全目标物体。
4. **缺乏自动化评估机制**：如何判断amodal completion是否成功缺乏无需人工标注的可靠标准。

## 核心贡献（创新点）
1. **渐进式遮挡感知补全流程（Progressive Occlusion-aware Completion Pipeline）**：直接利用预训练扩散inpainting模型补全遮挡区域，避免中间mask预测步骤；通过迭代逐步移除遮挡物直至目标物体完整。
2. **混合上下文扩散采样（Mixed Context Diffusion Sampling, MC）**：通过在扩散中途替换背景为纯色背景（产品摄影风格），打破目标物体与原始上下文的共现联系，从而抑制不期望的共存对象生成。
3. **反事实补全筛选系统（Counterfactual Completion Curation System）**：利用outpainting后的mask面积扩展比例作为训练无关的判断标准，自动筛选成功/失败的补全结果。
4. **首次实现仅用预训练模型无需微调的模态补全**：方法完全基于预训练的Stable Diffusion v2 inpainting模型，配合现成的grounded segmentation、depth和removal模块，无需额外训练或fine-tuning。

## 方法详解

### 3.4 渐进式遮挡感知补全流程
1. **Mask分析**：使用grounded segmentation（Grounded Dino + SAM）获取所有对象mask $\mathcal{M}_{obj}$，过滤出邻近query object的邻域mask $\mathcal{M}_{neighbor}$，通过深度排序分析识别遮挡物mask $\mathcal{M}_{occluder}$，聚合为单一二值遮挡mask $M_{occ} = \sum M_i$。
2. **条件Padding**：若query object mask $M_{modal}$触及图像边界，则在相应方向padding图像和mask，以支持边界外补全。
3. **扩散过程**：裁剪至bounding box后，运行MC扩散采样生成 $I_{amodal} = F_{0\to N}(I_{in}, M_{occ}, P)$。
4. **迭代检查**：若仍有遮挡物或触及边界，则以上一轮输出的 $I_{amodal}$ 为新输入 $I_{in}$、$M_{amodal}$ 为新 $M_{modal}$，继续迭代直至无遮挡。
5. **后处理**：裁剪额外背景后可叠加回原图。

### 3.5 混合上下文扩散采样（MC）
核心思想：在扩散中途临时替换背景，打破共现偏置。

1. **Swap Background（红色路径）**：将 $I_{in}$ 中 $M_{modal}$ 外区域替换为干净纯色背景（如灰色），得到合成图像 $I_{syn}$，运行扩散到第 $k$ 步：
   $$I_{syn\_amodal}^k = F_{0\to k}(I_{syn}, M_{occ}, P)$$

2. **创建去遮挡背景图像（蓝色路径）**：使用removal inpainter [47] 移除query object和occluders，得到干净背景后加噪至第 $k$ 步：
   $$I_{bg}^k = \text{AddNoise}(R(I_{in}, M_{modal} + M_{occ}), k)$$

3. **在噪声图中分割目标物体（绿色路径）**：从 $I_{syn\_amodal}^k$ 提取UNet decoder第 $l$ 层的特征，聚类后选择与 $M_{modal}$ 重叠度最高的cluster作为第 $k$ 步的amodal mask $M_{amodal}^k$。

4. **合成（紫色路径）**：
   $$I_{amodal}^k = I_{syn\_amodal}^k \odot M_{amodal}^k + I_{bg}^k \odot (1 - M_{amodal}^k)$$
   继续剩余 $N-k$ 步扩散得到最终结果。

### 3.6 反事实补全筛选系统
1. **生成步骤**：对 $I_{amodal}$ 进行outpainting（mask为除amodal mask和四个角外的区域），得到 $I_{amodal}'$，裁剪背景提取新amodal mask $M_{amodal}'$。
2. **决策步骤**：基于两个阈值判断——(1) 物体距图像边界的距离，(2) amodal mask面积扩展比例。实验确定阈值为20%：扩展<20%判定为complete，否则incomplete。

## 实验与结果

### 数据集
- 构建3000个pseudo-occluded样本：将COCO [26]（2500张）和Open Images [19,20]（500张）中的完整对象叠加到另一对象上模拟遮挡。
- 1500个easy case（20-50%遮挡）+ 1500个hard case（50-80%遮挡）。
- 覆盖至少55个object类别。

### 评估指标
- 高分辨率：CLIP cosine similarity（生成图像与文本category embedding）
- 中层次：DreamSim perceptual distance
- 低层次：LPIPS
- 用户偏好研究（MTurk）

### 主要结果（Table 1）

| Method | Easy CLIP↑ | Easy DreamSim↓ | Easy LPIPS↓ | Hard CLIP↑ | Hard DreamSim↓ | Hard LPIPS↓ | User Preference |
|--------|------------|----------------|-------------|------------|----------------|-------------|-----------------|
| SSSD [60] | 0.280/0.263 | 0.186/0.216 | 0.096/0.142 | 0.267/0.263 | 0.315/0.334 | 0.166/0.225 | 1.8% |
| LaMa [47] | 0.288/0.265 | 0.098/0.124 | 0.054/0.091 | 0.279/0.268 | 0.236/0.292 | 0.130/0.205 | 7.3% |
| Inst-Inpaint [56] | 0.264/0.257 | 0.325/0.304 | 0.185/0.195 | 0.252/0.254 | 0.451/0.446 | 0.263/0.283 | 0.0% |
| **Ours** | **0.290/0.266** | **0.096/0.106** | **0.054/0.078** | **0.290/0.267** | **0.184/0.185** | **0.110/0.141** | **90.9%** |

- **用户偏好**：90.9%的MTurk用户选择本文方法的补全结果（easy+hard合计）。
- 相比SOTA方法LaMa，在easy case上CLIP提升约+0.8%，在hard case上DreamSim显著降低（0.184 vs 0.236）。

### Ablation（Table 2）
- **MC的有效性**：在hard cases上，加入MC使成功率提升+18%（72%→90%）。
- **Naive Outpainting**：easy 66% success，hard 40% success，明显低于本文方法。

### MC组件分析（Figure 8-9）
- **背景类型**：灰色背景比森林/天空背景提升+20%/-16%/-9%。
- **UNet层**：第3层decoder（$D_3$）特征聚类效果最佳。
- **Timestep**：$k=20$（共50步）时复合效果最优。

### 反事实筛选系统（Table 3）
- 准确率0.70，精确率0.68，召回率0.68。
- 人类共识准确率0.83，说明该任务具有主观性。

## 相关工作脉络
1. **Amodal Completion传统方法**：SSSD [60]、OConet [3]等GAN方法缺乏高保真度；本工作利用预训练扩散模型的正则化先验实现photorealistic补全，且无需训练。
2. **直接Inpainting/Outpainting方法**：LaMa [47]、Inst-Inpaint [56]未针对amodal completion设计，常产生共现对象或不完整结果；本文通过MC和渐进策略解决。
3. **扩散模型控制方法**：ControlNet [62]、Prompt-to-Prompt [14]等需额外 conditioning（mask/edge map），且需资源密集型重训练；本文完全training-free。
4. **Concurrent Works**：pix2gestalt [36]、Tracking any object amodally [16]等同阶段工作，本文定位更侧重于无需训练的实用Pipeline。
5. **Amodal Segmentation**：KINS [39]、AMOS [3]等提供amodal mask标注，但appearance completion需额外生成模型；本文绕过mask预测直接生成像素。
6. **Diffusion Inpainting**：Stable Diffusion Inpainting [42]是本文基础，本文扩展其在amodal completion场景的应用方式。

## 局限性与未来方向
1. **小型query object被大型遮挡物遮挡时易过扩展**：物体本身较小但被大面积遮挡物覆盖时，可能过度延伸。
2. **阴影干扰**：query object上的细微阴影可能被扩散模型误认为上下文信号，生成兼容的遮挡物。
3. **强姿态/互动暗示**：如人骑马等姿态强烈暗示与其他物体互动，可能生成非预期的共存对象。
4. **反事实筛选系统精度有限**：当前0.70准确率仍有提升空间，可通过训练专用分类器改进。
5. **未来方向**：处理物体被完全包围的极端情况、结合3D先验（如novel view synthesis）、改善复杂姿态下的补全质量。

## 研究启发与可借鉴点
1. **训练无关的扩散模型应用范式**：证明预训练扩散模型无需fine-tuning即可通过巧妙prompt和流程设计解决复杂视觉任务，为其他任务提供思路。
2. **Mixed Context策略可迁移**：通过"中途打断-背景替换-重新合成"的思路解决共现偏置，可推广至其他inpainting场景（如去除不期望的共现对象）。
3. **UNet特征聚类用于语义分割**：利用扩散模型decoder特征进行无监督聚类，可替代传统分割模型在噪声图像上的检测任务。
4. **反事实推理作为评估工具**：通过outpainting后的mask变化判断完整性，为其他生成任务的自动化评估提供新思路。
5. **迭代渐进式补全框架**：将复杂任务分解为多步迭代，每步检查状态并更新输入，可应用于需要逐步 refinment 的生成任务。

## 关键术语表
- **Amodal Completion（模态补全）**：补全被遮挡物体的完整外观（包括视觉可见部分和隐藏部分）。
- **Modal Mask（模态mask）**：物体在图像中实际可见的二值掩码。
- **Amodal Mask（模态mask）**：物体完整（含隐藏部分）的二值掩码。
- **Co-occurrence Bias（共现偏置）**：扩散模型因训练数据的上下文关联，倾向于在原遮挡物位置生成与之共现的其他对象。
- **Mixed Context Diffusion Sampling（混合上下文扩散采样）**：在扩散中途替换背景为纯色，打破共现偏置的技术。
- **Counterfactual Completion Curation（反事实补全筛选）**：通过outpainting后mask扩展比例判断补全是否成功的训练无关规则。
- **Grounded Segmentation（接地分割）**：结合Grounded Dino和SAM的开放词汇对象分割方法。

## 可复现要素
- **数据集**：论文自构建3000个pseudo-occluded样本（COCO + Open Images），非公开。
- **代码**：论文未提供开源代码仓库链接。
- **权重**：使用公开Stable Diffusion v2 inpainting模型checkpoint [42]。
- **辅助模型**：Grounded Dino [29]、SAM [18]、Depth estimation [21]、Removal inpainter [47]——均为开源模型。
- **关键超参**：DDIM总步数N=50，MC混合timestep k=20，UNet decoder层l=3，mask扩展阈值20%，背景颜色为灰色。
- **硬件**：Nvidia Titan RTX 24GB GPU，无训练过程。
