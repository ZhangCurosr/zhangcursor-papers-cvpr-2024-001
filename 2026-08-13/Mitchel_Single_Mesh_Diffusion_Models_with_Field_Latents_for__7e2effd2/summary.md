---
title: "Single Mesh Diffusion Models with Field Latents for Texture Generation"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Mitchel_Single_Mesh_Diffusion_Models_with_Field_Latents_for_Texture_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:40:33"
---

# 论文速读：Single Mesh Diffusion Models with Field Latents for Texture Generation

## 一句话总结
本文提出 Field Latents (FL) 与 Field Latent Diffusion Models (FLDMs)，将纹理编码为定义在三角网格顶点切空间的离散向量场，并在该潜空间上构建等距等变的扩散去噪模型，实现了仅凭单个纹理网格就能生成高保真、可编辑、可跨拓扑迁移的纹理。

## 研究问题与动机
- 高质量3D纹理资产构建成本高昂，但大规模带复杂纹理的3D数据集仍稀缺，仅部分类别具备非均匀纹理。
- 主流基于2D LDM的方法（如SDS优化、多视角渲染去噪）依赖图像先验，存在视图不一致、光照残留伪影，且难以合成高频细节。
- 基于三平面/体素栅格化的方法需将几何与纹理映射至3D网格，受显存限制必然降采样，导致高频纹理混叠与细节丢失。
- 直接在网格表面操作面临“表面点无全局标准朝向”的坐标歧义，传统2D CNN/ViT无法直接平移至流形。

## 核心贡献（创新点）
- **Field Latents 切向量场潜表示**：将纹理映射为顶点处的复数切向量，捕获局部方向信息，相比标量或纯重心插值方法能更精细地重建高频纹理细节。
- **等距等变 FL-VAE 架构**：编码器采用 VN-Transformer 提取局部1-ring切向量特征，解码器引入对数映射坐标函数 $c_{pq}^\psi = \log_p q \cdot \overline{z_p^\psi}$ 替代单纯双线性插值，显著提升隐式神经场的跨面片连续表达能力。
- **等变 Field Latent Diffusion Model**：在切丛上定义前向加噪与反向去噪过程，针对向量场无法直接使用加性时间步嵌入的问题，将 Field Convolutions (FC) 扩展为作用于 $\mathbb{C} \times \mathbb{R}^e$ 的滤波器，在保持等变性的前提下稳定注入时间步与条件信号。
- **生成式纹理迁移 (Generative Texture Transfer)**：等距等变性使模型无需显式点态对齐，即可将在网格 M 上学成的 FLDM 直接采样至拓扑不同但局部近似等距的网格 N，实现风格一致的新纹理生成。

## 方法详解
- **FL 潜表示与噪声定义**：纹理 $\psi \in L^2(M, \mathbb{R}^3)$ 在顶点 $p$ 的切空间 $T_pM$ 中表示为复数 $\mathbf{v}=re^{i\theta}$。定义切丛上的各向同性高斯分布 $\mathcal{TN}_M(0, I_d)$，满足 $\epsilon(p)=\epsilon_1+i\epsilon_2,\ \epsilon_{1,2}\sim\mathcal{N}(0,I_d)$。
- **FL-VAE Encoder**：对顶点 $p$ 的1-ring邻域均匀采样点 $\{q_i\}$，将纹理标量 $\psi(q_i)$ 与对数坐标 $\log_p q_i$ 交错拼接并附加 Token，经 8 层 VN-Transformer 后输出均值 $\mu_p^\psi \in \mathbb{C}^d$ 与标准差 $\sigma_p^\psi \in \mathbb{R}_{\ge0}^d$，潜代码 $z_p^\psi = \mu_p^\psi + \sigma_p^\psi \odot \epsilon(p)$。
- **FL-VAE Decoder**：基于对数映射诱导的局部参数化，构造旋转不变特征 $\mathrm{vec}_{j\ge i}(z_p^\psi [z_p^\psi]^*)$ 与位置感知特征 $c_{pq}^\psi$，拼接后输入 5 层 Real-valued MLP 预测 $\hat{\psi}_p(q)$。推理时利用三个相邻顶点的预测值经重心插值融合输出连续纹理。
- **FLDM 扩散过程**：前向加噪 $Z_t = \sqrt{\alpha_t} Z + \sqrt{1-\alpha_t}\epsilon$；反向去噪使用浅层两级 U-Net，损失 $\mathcal{L}_{FLDM} = \mathbb{E}_{\epsilon,t}\|\epsilon - \varepsilon(Z_t, t, \rho)\|_2^2$，其中 $\rho$ 为可选条件（如语义标签或掩码）。采样遵循标准 DDPM 递推公式。
- **等变性保障**：要求 $\mu$ 随等距推前旋转、$\sigma$ 保持不变；去噪网络通过 FC 卷积核内嵌条件实现等变，避免加性嵌入破坏切向量的旋转对称性。

## 实验与结果
- **数据集与设置**：FL-VAE 预训练于 Open-ImagesV4（叠加 1K 个 10K 顶点随机平面网格）；重建实验使用 Google Scanned Objects [13]（30K/5K 顶点）；生成实验使用 Objaverse [8] 与 Scanned Objects，每网格生成 500 份不同三角剖分副本防过拟合，T=1000 线性噪声调度，Adam 优化。
- **纹理压缩/重建 (Table 1)**：FL-VAE 在 30K/5K 顶点上 PSNR 达 22.38 / 20.59，DSSIM 0.51 / 0.83，LPIPS 1.02 / 1.81，全面优于 INFs [27]（21.33/18.86）与 FL-VAE (Bary.) 消融版。
- **无条件生成 (Table 2)**：FLDM SIFID 3.27、LPIPS 1.15，显著优于唯一单网格基线 Sin3DM [58]（6.58 / 2.20）。Sin3DM 因联合生成几何导致 LPIPS 虚高，实际纹理呈沿主轴重复/挤出伪影；FLDM 可在局部等距区域无缝复制细节。
- **可控编辑与迁移**：标签引导生成可分离鞋底/眼/嘴等语义区域纹理；Inpainting 在掩码边界自然融合；跨拓扑纹理迁移（如图 6 skull genus 0↔3）成功复现风格，且通过重网格粗细可连续控制纹理尺度（图 7）。

## 相关工作脉络
- **SDS/多视角 LDM 系（DreamFusion、TexFusion、Text2Tex 等）**：依赖 2D 先验与渲染环路，视图一致性与细节保真度受限；本文完全在内在流形上操作，规避外生渲染失真。
- **Sin3DM [58]**：首个单纹理网格 LDM，但采用外生三平面栅格化，纹理与 3D 嵌入强耦合；本文仅建模样本纹理分布，生成质量更高且支持纯纹理编辑。
- **INFs [27]**：基于 Laplacian 特征函数+重心插值的内在神经场；本文指出特征函数高频区与纹理细节常错位，且重心插值表达能力有限；FL 使用切向量场+对数坐标显著提升重建。
- **Manifold Diffusion Fields [14, 64]**：全局注意力内在 DM，复杂度限制其仅适用于粗网格；本文基于局部 Field Convolutions，可扩展至高分辨率三角网格。
- **GAN-based 纹理方法（Texturify、Mesh2Tex）**：面片属性+可微渲染对抗监督；本文采用扩散范式，生成多样性与 perceptual 指标更优，且天然支持条件编辑。

## 局限性与未来方向
- 单网格扩散模型训练时间较长，计算开销仍高于预训练 2D LDM 的 fine-tuning 路线。
- 当前模型无法在生成纹理中显式表达定向信息（如织物纹理走向、毛发方向）。
- 纹理迁移依赖网格间的局部近似等距性，曲率分布或拓扑差异极大的几何可能失效。
- 未来可将用户指定的切向量场作为条件，通过点积交互注入网络以控制生成纹理的方向性，并探索更轻量的单图/单网格扩散训练策略。

## 研究启发与可借鉴点
- **切向量场+复数表示的等变设计**可直接迁移至法线场生成、各向异性材质建模、曲线/矢量场扩散等流形生成任务，避免人工构造局部坐标系。
- **对数映射坐标函数**替代重心插值用于神经场特征延展，是一种提升表面连续表示容量与方向感知能力的通用技巧，适用于网格 NeRF 或隐式场工作。
- **浅层 U-Net + 单样本扩散**（SinDiffusion/Sin
