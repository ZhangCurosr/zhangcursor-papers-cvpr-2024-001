---
title: "Activity-Biometrics-Person-Identification-from-Daily-Activit"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Azad_Activity-Biometrics_Person_Identification_from_Daily_Activities_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:19:50"
---

# 论文速读：Activity-Biometrics-Person-Identification-from-Daily-Activit

## 一句话总结
本文首次系统研究基于RGB视频的日常活动人脸受限身份识别任务，提出ABNet框架，通过“无偏轮廓教师蒸馏”与“形态畸变对比学习”双路径解耦生物特征与外观偏置，并结合活动语义先验，在5个衍生基准上显著超越现有步态/图像重识别方法。

## 研究问题与动机
- 现有人员识别方法高度依赖行走步态或静态图像，难以覆盖个体执行多样化日常活动（如开药、上下楼、操作设备等）时的识别需求。
- RGB视频中存在强烈的外观偏置（衣物颜色、背景、配饰等），模型易过拟合浅层线索，导致跨场景、跨相机泛化失效。
- 真实安防与隐私保护场景常需隐藏或遮挡人脸，亟需发展仅依赖身体运动与活动上下文的鲁棒生物特征识别。
- 日常活动具有强上下文相关性，单一运动表征不足以区分个体，需挖掘活动先验与身份特征的协同判别能力。

## 核心贡献（创新点）
- **提出ABNet框架**：首次面向日常活动RGB视频设计人脸受限身份识别方案。与现有步态/图像基线不同，本文处理非结构化、高多样性的动态活动分布。
- **双路径特征解耦机制**：无偏教师蒸馏负责压制外观干扰，畸变负样本挖掘负责显式建模外观不变性。与仅依赖对抗损失或单一模态去偏的工作本质不同，两者正交互补。
- **活动先验融合策略**：在推理阶段将活动识别特征与生物特征拼接，利用高层语义上下文增强身份判别力。与纯时空特征聚合的视频识别方法相比，引入了任务级上下文约束。
- **构建5个专用评估基准**：从NTU RGB+D、PKU-MMD、Charades等主流活动识别数据集派生并标准化人脸模糊/色调偏移协议，填补该任务的评测空白。

## 方法详解
- **主干与双流头**：视频编码器 $S_\phi(\cdot)$ 采用 ResNet3D-50 提取时空特征 $F_{AB}$；经切分后分别输入活动头 $C^A$（输出 $F_{Ac}$）与演员头 $C^B$（含Transformer解码器 $D_\omega^B$，输出交织特征 $F_{BT}$，分解为生物特征 $f_{bb}$ 与外观特征 $f_{ba}$）。
- **无偏蒸馏（Bias-less Distillation）**：教师网络 $T_\theta(\cdot)$ 基于 GaitGL，仅接收由 Mask2Former 提取的二值轮廓视频 $b_s$，天然不含RGB外观信息。通过KL散度约束学生actor特征分布逼近教师：$\mathcal{L}_{KD} = \tau^2 KL(y_T \| y_S)$。
- **畸变偏置学习（Distortion-based Bias Learning）**：共享权重的畸变网络 $A_\varphi(\cdot)$ 对原始视频施加弹性形变（Elastic Transform）生成 $\hat{v}$，保持衣物/背景外观但扭曲人体形态（改变身份）。构造对比损失：$\mathcal{L}_{Dis} = \max(D(f_{ba}, f_{ba}^D) - D(f_{bb}, f_{bb}^D) + m, 0)$，推动外观特征对齐、生物特征分离。
- **联合学习目标**：活动头使用标准交叉熵 $\mathcal{L}_{Ac}$；生物特征头使用 $\mathcal{L}_{Bio} = \mathcal{L}_{ce} + \mathcal{L}_{tri}$（margin $m=0.3$）。总损失 $\mathcal{L} = \mathcal{L}_{Bio} + \lambda_1\mathcal{L}_{Ac} + \lambda_2\mathcal{L}_{KD} + \lambda_3\mathcal{L}_{Dis}$，所有 $\lambda_i=0.01$。推理时将 $F_{Ac}$ 与 $f_{bb}$ 拼接作为最终嵌入。

## 实验与结果
- **数据集与协议**：NTU RGB-AB（94类/88692样本）、PKU MMD-AB（41类/17k样本）、Charades-AB（157类/9848视频）、ACC-MM1-Activities（7类/1378视频）、BRIAR-BGC3（1055人/20K子集）。均做高斯人脸模糊与色调偏移增强。评估协议涵盖同活动/跨活动、同视向/跨视向，指标为R@1、mAP、TAR@0.1%FAR。
- **主要结果**：ABNet在全部核心数据集上均超越现有SOTA。NTU RGB-AB同活动View+ R@1=78.76 / mAP=40.31；PKU MMD-AB R@1=86.83 / mAP=57.31；ACC-MM1-Activities R@1=80.43 / mAP=52.71；Charades-AB R@1=45.84。相比最强对比方法
