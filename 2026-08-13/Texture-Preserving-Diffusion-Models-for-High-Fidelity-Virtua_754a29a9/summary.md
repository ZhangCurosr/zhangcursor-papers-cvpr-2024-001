---
title: "Texture-Preserving-Diffusion-Models-for-High-Fidelity-Virtua"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Yang_Texture-Preserving_Diffusion_Models_for_High-Fidelity_Virtual_Try-On_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 01:49:46"
---

# 论文速读：Texture-Preserving-Diffusion-Models-for-High-Fidelity-Virtua

## 一句话总结
本文提出了Texture-Preserving Diffusion (TPD)模型，通过空间拼接人物遮挡图与参考服装图，利用扩散模型原生自注意力块高效完成纹理迁移，并结合解耦掩码预测（DMP）自适应计算精确修复区域，在无需额外图像编码器与服装形变步骤的前提下，实现了高保真、高效率的虚拟试衣。

## 研究问题与动机
1. **额外编码器导致计算冗余与细节丢失**：现有扩散试衣方法（如DCI-VTON、LaDI-VTON）依赖CLIP、ViT或额外UNet提取服装特征，不仅增加显存开销，且CLIP类编码器预训练目标偏向粗粒度语义，难以保留细粒度纹理。
2. **固定掩码破坏非衣物身体细节**：传统方法仅基于原图估计一次性修复掩码，常误遮手臂、纹身、手指等区域，降低生成图像的身份一致性与真实感。
3. **形变（Warping）引入难以修正的伪影**：基于TPS/光流/关键点的方法在形变阶段产生的畸变会在后续合成中残留，影响整体质量。
4. **高效无形变的高保真生成方案仍待探索**：如何在不引入额外编解码结构的情况下，让扩散模型原生架构承担纹理迁移与掩码自适应任务，是当前研究的空白点。

## 核心贡献（创新点）
1. **提出SATT（基于自注意力的纹理迁移）机制**：将掩码人物图与参考服装图沿空间
