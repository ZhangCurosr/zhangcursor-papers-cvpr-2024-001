---
title: "Sharingan-A-Transformer-Architecture-for-Multi-Person-Gaze-F"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:16:52"
---

# 论文速读：Sharingan-A-Transformer-Architecture-for-Multi-Person-Gaze-F

## 一句话总结
本文提出 Sharingan，一种基于 Transformer 的多目标注视跟随架构，通过将每个人物的头部特征与边界框编码为单个受控的 gaze token 并与图像 token 联合处理，首次在不破坏原始任务表述的前提下实现了高效、细粒度的多目标注视热图预测，在 GazeFollow、VideoAttentionTarget 和 ChildPlay 上均取得 SOTA。

## 研究问题与动机
1. **单目标前向效率瓶颈**：传统双塔 CNN 架构需对同一场景中的每个人物单独执行前向传播，推理耗时随人数线性增长，难以落地真实多人大规模场景。
2. **已有 Transformer 方法破坏任务表述**：现有基于 Transformer 的多目标方法（如 [38, 39]）采用固定数量的可学习 embedding 同时解码人头框与注视目标（SET PREDICTION 范式），依赖后验匹配步骤，无法直接对接标准 benchmark 的定量评估流程，也难以嵌入下游人类行为理解管线。
3. **解码机制薄弱且信息粒度粗糙**：既往工作多聚焦人物特征与场景显著图的融合设计，对最终热图解码关注不足（多使用极低维特征经转置卷积解码），限制了空间定位精度。
4. **冗余人物表征阻碍性能**：使用视觉注意力图或注视锥（GAZE CONE）编码中间人物信息在 Transformer 注意力框架下不仅增加 token 数量，还会引入不必要的归纳偏置，消融实验证实其反而损害精度。

## 核心贡献（创新点）
1. **受控单 Token 人物编码**：将头部 crop 与边界框坐标分别投影后相加生成唯一的位置感知 gaze token，本质区别在于摒弃 DETR 式的固定 learnable queries，严格保留“给定头部预测注视”的原始任务公式。
2. **Conditional DPT 多尺度条件解码器**：将密集预测架构 DPT 改造为条件控制版本，从编码器不同深度提取多分辨率特征，按人数复制后与 gaze token 点积注入 person-specific 信息，再经残差卷积逐级融合上采样；与以往直接使用 MLP 或单次点积重投影的方案形成架构级差异。
3. **训练人数固定化与推理人数解耦**：证明训练时固定采样 $N_p^{tr}$ 个 gaze token 并不会限制推理时的处理人数，模型凭借 token 置换不变性可泛化至任意数量目标，大幅简化多目标数据准备。
4. **统一多任务损失框架**：联合优化热图 MSE 损失、角注视向量余弦损失与 In-Out 二元分类损失，在单次前向中同步输出细粒度热图与注视方向先验。

## 方法详解
- **Image Tokens**：输入场景图 $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ 经 ViT 风格的 patch projection $\mathcal{P}_{img}$ 切块并附加位置编码，得到 $\mathbf{x}^{img} \in \mathbb{R}^{N \times D}$。
- **Gaze Tokens**：对第 $i$ 个人，头部 crop $\mathbf{h}_{crop}^i$ 输入预训练 ResNet-18 骨干 $\mathcal{G}$ 得到 embedding $\mathbf{g}^{emb}$；该 embedding 经 MLP $\mathcal{O}_{gv}$ 预测归一化 2D 注视向量 $\mathbf{g}_v$（受角损失监督）。同时 $\mathbf{g}^{emb}$ 通过线性投影 $\mathcal{P}_{gaze}$ 映射至 token 维度，头部边界框 $\mathbf{h}_{bbox}$ 通过 $\mathcal{P}_{bbox}$ 投影，两者相加得位置感知 gaze token：$\mathbf{x}_i^g = \mathcal{P}_{gaze}(\mathbf{g}^{emb}) + \mathcal{P}_{bbox}(\mathbf{h}_{bbox})$。
- **Transformer Encoder**：标准 ViT 编码器，输入 $\mathbf{x} = \mathbf{x}^{img} \oplus \mathbf{x}^g$，经 $L$ 层自注意力与 FFN 块输出 $\mathbf{x}^{out}$。模态差异由投影层 bias 隐式学习，无需显式 modality encoding。
- **Conditional DPT Decoder**：选取编码器第 4, 8, 16, 32 层的中间特征 $\mathbf{x}_{(l_k)}^{img}$ 与 $\mathbf{x}_{(l_k)}^{g}$。将图像特征重组为 $\left(\frac{H}{k}, \frac{W}{k}\right)$ 分辨率并投影至维度 $d_k$；将特征图沿批次维度复制 $N_p$ 份，与每个 gaze token 做逐元素点积，获得 person-specific 张量 $(B \times N_p, d_k, H/k, W/k)$。依次送入残差卷积融合模块并与上层输出相加，再经残差卷积、上采样（分辨率翻倍）与投影逐步融合。最终输出 $(B \times N_p, d_{out}, H/2, W/2)$，经卷积头压缩通道并 resize 至热图尺寸，分离 batch 与 person 维得到 $(B, N_p, 1, H_{hm}, W_{hm})$。
- **In-Out 预测头**：MLP 接收输入与输出 gaze token 拼接 $[\mathbf{x}_{(L)}^g, \mathbf{x}^g]$，预测画面内外二元标签。
- **损失函数**：$\mathcal{L} = \lambda_{reg} \mathcal{L}_{hm} + \lambda_{ang} \mathcal{L}_{ang} + \lambda_{io} \mathcal{L}_{io}$。其中 $\mathcal{L}_{hm} = \sum_{x,y} ||\mathcal{A}_{x,y}^{gt} - \mathcal{A}_{x,y}^{pred}||_2
