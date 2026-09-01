---
title: "DuPL-Dual-Student-with-Trustworthy-Progressive-Learning-for"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Wu_DuPL_Dual_Student_with_Trustworthy_Progressive_Learning_for_Robust_Weakly_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:58:51"
---

# 论文速读：DuPL-Dual-Student-with-Trustworthy-Progressive-Learning-for

## 一句话总结
本文针对单阶段弱监督语义分割（One-stage WSSS）中CAM伪标签自我强化的确认偏误问题，提出双学生可信渐进学习框架（DuPL）。通过双网络交叉监督与表征差异约束生成多样化CAM，并结合动态阈值调整、基于GMM的自适应噪声过滤及丢弃区域的一致性正则化，在 PASCAL VOC 2012 和 MS COCO 上实现了超越现有单阶段方法并媲美多阶段方法的分割性能。

## 研究问题与动机
1. **CAM确认偏误（Confirmation Bias）**：单阶段方法同步优化CAM生成与分割头，共享backbone使错误的CAM伪标签不断反向强化自身的错误判断，导致分割性能随训练持续恶化。
2. **固定阈值过滤策略过于保守**：现有单阶段方法采用固定高阈值丢弃不可靠伪标签，虽能隐式缓解噪声，但会大量误删实际正确的像素，造成监督信号严重不足。
3. **丢弃区域缺乏训练覆盖**：被过滤的不可靠像素多位于语义模糊区、物体边界或背景，完全排除在监督外会导致分割头在这些关键区域无法有效学习。
4. **单阶段与多阶段性能鸿沟**：尽管单阶段流水线训练效率更高，但因偏误累积与数据利用率低，其最终mIoU仍显著落后于依赖复杂精炼模块的多阶段方法。

## 核心贡献（创新点）
1. **首次系统建模并缓解CAM确认偏误**：提出双学生架构，通过表征级差异损失强制两个子网络生成多样化CAM，利用交叉监督打破“错误预测自我强化”的恶性循环。
2. **可信渐进学习（Trustworthy Progressive Learning）**：设计动态阈值调整（DTA）与基于高斯混合模型（GMM）的自适应噪声过滤（ANF），在训练过程中渐进引入更多可信像素，替代传统固定阈值的一刀切策略。
3. **Every Pixel Matters 一致性正则化**：将ANF丢弃的不可靠区域视为无标签样本，施加强扰动分支并计算一致性损失，使模型在这些语义模糊区域也能获得隐式监督。
4. **SOTA性能与流程简化**：在ViT-B骨干下，DuPL在PASCAL VOC 2012上达到73.3%（val）/ 72.8%（test）mIoU，在MS COCO上达到44.6% mIoU，超越所有同期单阶段方法并可与多阶段方案竞争。

## 方法详解
- **双学生交叉监督**：构建两个参数独立的学生子网络 $\psi_1, \psi_2$（各自含Backbone、分类器、分割头）。为防同质化，在表征层施加差异损失 $\mathcal{L}_{dis} = \mathcal{D}(f_1, \Delta(f_2)) + \mathcal{D}(f_2, \Delta(f_1))$，其中 $\mathcal{D} = -\log(1-\text{cos\_sim})$，$\Delta$ 为 stop-gradient，迫使两网络提取多样化特征。两网络的CAM伪标签 $\mathbf{Y}_1, \mathbf{Y}_2$ 用于交叉监督对方分割头预测，$\mathcal{L}_{seg} = CE(\mathbf{P}_1, \mathbf{Y}_2) + CE(\mathbf{P}_2, \mathbf{Y}_1)$。
- **动态阈值调整（DTA）**：采用余弦退火逐步降低前景高阈值 $\tau_h$：$\tau_h(t) = \tau_h(0) - \frac{1}{2}(\tau_h(0)-\tau_h(T))(1-\cos(\frac{t\pi}{T}))$，使训练后期能纳入更多前景像素参与分割训练。
- **自适应噪声过滤（ANF）**：假设像素级CE损失 $l^x$ 服从双分量高斯混合模型 $\mathcal{P}(l^x) = w_c \mathcal{N}_c + w_n \mathcal{N}_n$，通过EM算法估计噪声后验概率 $\varrho_n(l^x)$。当 $\varrho_n(l^x) > \gamma$ 且均值间距 $\mu_n - \mu_c > \eta$ 时，判定该像素为噪声并从 $\mathcal{L}_{seg}$ 中剔除；两子网独立执行此策略。
- **一致性正则化（$\mathcal{L}_{reg}$）**：对每个被ANF标记为不可靠的像素掩码 $\mathcal{M}_i$，对输入施加强扰动 $\phi(\mathbf{X})$ 得到 $\tilde{\mathbf{X}}$，计算扰动预测 $\tilde{\mathbf{P}}_i$ 与变换后伪标签 $\phi'(\mathbf{Y}_i)$ 的CE损失，仅在 $\mathcal{M}_i=1$ 的区域累加，强制模型对扰动保持平滑预测。
- **总优化目标**：$\mathcal{L} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{dis} + \lambda_2 \mathcal{L}_{seg} + \lambda_3 \mathcal{L}_{reg}$，其中 $\mathcal{L}_{cls}$ 为多标签软margin分类损失。

## 实验与结果
- **数据集**：PASCAL VOC 2012（扩展SBD，train 10582 / val 1449 / test 1456）与 MS COCO 2014（train 82k / val 40k），报告 mIoU。
- **评估基线**：对比多阶段WSSS（EPS, L2G, PPC, OCR, ACR等
