---
title: "DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Lou_DaReNeRF_Direction-aware_Representation_for_Dynamic_Scenes_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:22:51"
field: "动态场景神经辐射场"
keywords: ["NeRF", "动态场景重建", "小波变换", "DTCWT", "novel view synthesis", "场景分解", "方向感知表示"]
innovations: ["首次将双树复小波变换(DTCWT)引入NeRF优化，提出方向感知表示解决DWT的移位变分和方向模糊问题", "设计可学习掩码压缩方案，使DTCWT表示的存储效率与SOTA方法相当", "统一适用于动态4D和静态3D场景重建，在性能与模型大小间取得更优平衡"]
benchmarks: ["Plenoptic Video Dataset", "D-NeRF Dataset", "NeRF Synthetic", "NSVF", "LLFF"]
---

# 论文速读：DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes

## 一句话总结
本文提出了 DaReNeRF，一种基于**方向感知表示**的动态场景 NeRF 方法，首次将双树复小波变换（DTCWT）引入 NeRF 优化，解决了传统 DWT 在动态场景重建中的移位变分和方向模糊问题，在复杂动态场景的 novel view synthesis 中达到 SOTA 性能，同时训练时间减少 2×。

## 研究问题与动机
1. **动态场景重建的计算瓶颈**：真实世界场景具有复杂运动（多对象运动、相机运动、反射、光照变化），传统 NeRF 依赖大量 MLP 评估，训练时间长达数天到数周，且需要额外监督（深度图、光流等）。
2. **平面分解方法的纹理细节损失**：HexPlane、K-Planes 等分解方法通过 4D 体积分解到 2D 平面来加速训练，但无法保留高保真纹理细节。
3. **DWT 在动态场景中的两大缺陷**：① **移位变分性（shift variance）**：动态场景中的平移会显著破坏小波振荡模式；② **方向选择性差（poor direction selectivity）**：2D DWT 产生棋盘格模式，混合 ±45° 方向，无法有效捕捉线条和边缘，导致鬼影伪影。
4. **小波表示在动态场景的探索不足**：虽有 Wavelet-NeRF 等频率表示方法用于静态场景，但将其扩展到动态场景的能力尚未充分探索，HexPlane 也指出 DWT 系数作为基函数会导致显著性能下降。

## 核心贡献（创新点）
1. **首次将 DTCWT 引入 NeRF 优化**：提出方向感知表示（Direction-Aware Representation, DaRe），通过学习 6 个不同方向的复小波系数来捕捉动态场景特征，从根本上解决 DWT 的移位变分和方向模糊问题。*本质区别在于：DTCWT 通过 Hilbert 变换对构造实部/虚部滤波器组，实现方向选择性和近似移位不变性，而非 DWT 的棋盘格方向混合。*
2. **引入可学习掩码方法解决存储冗余**：针对 DTCWT 相比 DWT 产生的 2^d 冗余（d=2 时为 4 倍），设计可训练掩码（trainable mask）进行稀疏压缩，使模型存储效率与 SOTA 方法相当。*本质区别在于：将静态场景的压缩管线迁移到动态场景，结合 RLE + Huffman 编码实现高效压缩。*
3. **统一适用于动态与静态场景**：将方向感知表示同时扩展到静态 3D 场景重建，在 NeRF Synth.、NSVF、LLFF 三个数据集上均优于 DWT-based 方法，实现性能与模型大小的更优平衡。*本质区别在于：同一表示框架无需修改即可处理 4D 动态和 3D 静态场景，展现了通用表示潜力。*

## 方法详解
**整体框架**：基于平面分解的 4D 体积表示，将每个 2D 平面赋予方向感知能力，通过逆双树复小波变换（IDTCWT）恢复平面表示，再与向量融合后输入紧凑 MLP 回归颜色。

1. **动态场景分解（Baseline HexPlane 扩展）**：
   - 4D 体积 D 分解为 6 组平面-时间对：XY-ZT、XZ-YT、YZ-XT
   - 每个平面 $M_r^{AB} \in \mathbb{R}^{m \times n}$ 通过 IDTCWT 从 13 个小波系数（1 个近似系数 + 6 个实部 + 6 个虚部）恢复
   - 特征计算：$D = \sum_r M_r^{XY} \circ M_r^{ZT} \circ v_r^1 + \ldots$，然后拼接相乘结果并通过张量 $V^{RF}$ 融合

2. **方向感知表示（DTCWT 引入）**：
   - 对每个 2D 平面应用单层 DTCWT，得到 6 个实部方向系数 $\mathbf{R}\{\mathcal{W}_i^{AB}\}_{i=1}^6$ 和 6 个虚部方向系数 $\mathbf{I}\{\mathcal{W}_i^{AB}\}_{i=1}^6$，分别对应 6 个特定方向
   - 重建公式：$M_r^{AB} = \text{IDTCWT}([\mathcal{W}_{a,r}^{AB}, \mathbf{R}\{\mathcal{W}_{i,r}^{AB}\}, \mathbf{I}\{\mathcal{W}_{i,r}^{AB}\}])$
   - DTCWT 通过两组滤波器（实部 $h$、虚部 $g$，满足 Hilbert 变换关系）实现方向选择性和近似移位不变性

3. **稀疏表示与压缩**：
   - 为 13 组系数分别学习可训练掩码 $\mathcal{M}$，通过 gumbel-sigmoid 松弛实现端到端优化
   - 稀疏损失：$\mathcal{L}_m = \sum \mathcal{M}$，由 $\lambda_m$ 控制稀疏程度
   - 压缩：二进制掩码转为 8-bit → RLE 编码 → Huffman 编码

4. **优化与训练策略**：
   - 总损失：$\mathcal{L} = \frac{1}{|\mathcal{R}|}\sum_{r \in \mathcal{R}} ||\mathbf{C}(r) - \hat{\mathbf{C}}(r)||_2^2 + \lambda_{reg}\mathcal{L}_{reg} + \lambda_m\mathcal{L}_m$
   - 正则化：对方向感知表示施加 TV loss 促进时空连续性
   - 训练：coarse-to-fine 渐进分辨率策略 + 空体素跳过（emptiness voxel）加速

## 实验与结果
**动态场景（Plenoptic Video Dataset）**：
- DaReNeRF-S（稀疏版）：PSNR **30.224**，D-SSIM **0.015**，LPIPS **0.089**，训练时间 **5h**，模型大小 **244 MB**（100k steps）
- DaReNeRF（完整版）：PSNR **30.441**，D-SSIM **0.012**，LPIPS **0.084**，训练时间 **4.5h**，模型大小 **1,210 MB**
- 相比 HexPlane (650k)：PSNR 提升 **0.736**，训练时间缩短 **2.7×**（12h→4.5h）
- 相比 HexPlane (100k)：PSNR 提升 **0.872**，训练时间略增但质量显著提升

**动态场景（D-NeRF Dataset，单目视频）**：
- DaReNeRF：PSNR **31.95**，SSIM **0.97**，LPIPS **0.03**
- 超越所有非形变方法，且优于部分形变方法（如 D-NeRF 30.50、TiNeuVox-S 30.75）

**静态场景**：
- NeRF Synth.：PSNR **32.42**（8.91 MB），优于 DWT 方法（31.95，0.83 MB）+0.47 dB
- NSVF：PSNR **36.24**（8.98 MB），优于 DWT 方法（34.67，0.87 MB）+1.57 dB
- LLFF：PSNR **26.48**（13.67 MB），优于 DWT 方法（25.88，1.34 MB）+0.60 dB
- 稀疏度可达 **94%-99%**，模型大小约 1-14 MB

## 相关工作脉络
1. **HexPlane [7]**：平面分解代表作的基线方法，将 4D 体积分解为 6 个 2D 平面对，训练快但纹理细节丢失；本文在其基础上引入方向感知表示，解决其指出的 DWT 系数性能下降问题。
2. **D-NeRF [40] / Neural Volumes [28]**：早期动态 NeRF 方法，依赖大型 MLP 或额外监督，训练时间长；本文通过分解+频率表示避免形变场，实现无额外监督的高效重建。
3. **Masked Wavelet NeRF [42]**：静态场景的 DWT 压缩方案；本文将其压缩管线迁移到动态场景，但用 DTCWT 替代 DWT 以获得方向选择性和移位不变性。
4. **K-Planes [14] / Tensor4D [46]**：并发分解方法；本文在相同训练时间框架下提供更优的纹理重建质量。
5. **Wavelet-NeRF / Wire [44, 64]**：静态场景小波表示；本文填补了小波表示在动态场景应用的研究空白。
6. **4D-GS [57]**：动态高斯溅射；本文在 PSNR 指标上与之竞争，同时提供更小的模型体积（稀疏版 244 MB vs 145 MB）和可微渲染优势。

## 局限性与未来方向
1. **极端稀疏观测场景受限**：如 D-NeRF 类单目数据集，缺乏形变场导致信息共享机制不足，难以从极少量视角学习 3D 结构。
2. **模型紧凑性不如 DWT**：方向感知表示的冗余性限制了模型压缩到极小尺寸（如静态场景 <1MB），需要更紧凑的表示设计。
3. **未来方向**：探索更紧凑的方向感知表示构造方法；结合形变场处理稀疏观测；进一步优化压缩效率。

## 研究启发与可借鉴点
1. **DTCWT 在神经辐射场中的引入思路**：将信号处理中的双树复小波变换引入 NeRF 表示，为其他频域/几何表示方法提供借鉴——当传统 DWT 遇到移位敏感或方向模糊问题时，可考虑 DTCWT 替代。
2. **可学习掩码 + 传统编码的混合压缩策略**：gumbel-sigmoid 松弛实现端到端稀疏学习，再结合 RLE+Huffman 等传统编码，这种"学习+压缩"两阶段策略可在其他神经表示中复用。
3. **静态→动态的表示迁移设计**：将静态场景验证有效的方向感知表示直接扩展到 4D 动态场景，无需重大架构修改，这种跨维度泛化思路值得借鉴。
4. **无额外监督的动态场景重建**：相比依赖深度/光流监督的方法，本文仅用光度损失即可达到 SOTA，说明好的表示设计可减少对额外监督的依赖。
5. **与团队方向的结合机会**：若团队研究低资源/压缩 NeRF，可将 DTCWT 方向感知表示与团队已有的量化/剪枝方法结合；若研究医学影像（作者单位含联合影像智能），可探索该表示在 4D 医学 volume 重建中的应用。

## 关键术语表
**DaReNeRF**：Direction-aware Representation NeRF 的缩写，本文提出的动态场景神经辐射场方法。
**DTCWT（Dual-Tree Complex Wavelet Transform）**：双树复小波变换，通过两组滤波器（实部/虚部，满足 Hilbert 变换关系）实现方向选择性和近似移位不变性的复小波变换。
**Direction-Aware Representation (DaRe)**：方向感知表示，将 DTCWT 系数作为 2D 平面的隐式表示，捕捉 6 个特定方向的纹理特征。
**Trainable Mask**：可学习掩码，通过 gumbel-sigmoid 松弛实现的逐系数稀疏化机制，用于缓解 DTCWT 引入的存储冗余。
**Plenoptic Video Dataset**：由 21 个 GoPro 相机同步采集的多视图动态场景数据集，包含 specular、translucent、topology change 等复杂动态内容。
**IDTCWT（Inverse DTCWT）**：逆双树复小波变换，将从系数空间恢复为平面空间表示的变换操作。
**Coarse-to-Fine Training**：粗到细训练策略，渐进增加网格分辨率以加速训练并提供隐式正则化。
**Emptiness Voxel**：空体素，预计算的 3D 稀疏掩码，用于跳过场景中无内容的区域以加速渲染。

## 可复现要素
- **数据集**：Plenoptic Video Dataset（公开）、D-NeRF Dataset（公开）、NeRF Synth. / NSVF / LLFF（公开）
- **代码/权重**：论文未明确提及代码开源状态（需进一步确认）
- **关键超参**：空间网格 512、时间网格 300、batch size 4096、100k training steps、$\lambda_m = 2.5 \times 10^{-11}$（稀疏控制）、DTCWT level=1
- **硬件**：单卡 A100 GPU
