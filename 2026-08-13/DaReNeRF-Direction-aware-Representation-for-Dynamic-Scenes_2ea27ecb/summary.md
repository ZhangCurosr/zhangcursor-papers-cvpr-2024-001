---
title: "DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Lou_DaReNeRF_Direction-aware_Representation_for_Dynamic_Scenes_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:16:20"
field: "动态NeRF与3D场景表示"
keywords: ["动态场景重建", "NeRF", "对偶树复小波变换", "方向感知表示", "新视角合成", "场景分解"]
innovations: ["首次将DTCWT引入NeRF优化，提出方向感知表示解决DWT的位移敏感和方向模糊问题", "提出可训练掩码+RLE+Huffman级联压缩策略，在几乎不损失性能前提下实现存储效率与SOTA相当", "统一框架同时适用于动态4D和静态3D场景，在静态场景超越DWT基线"]
benchmarks: ["Plenoptic Video Dataset", "D-NeRF Dataset", "NeRF Synthetic", "NSVF", "LLFF"]
---

# 论文速读：DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes

## 一句话总结
本文提出了一种方向感知表示（Direction-Aware Representation, DaRe），基于对偶树复小波变换（DTCWT）将4D动态场景的平面分解与小波系数相结合，解决了传统2D离散小波变换（DWT）在动态场景中的位移敏感性和方向模糊问题，在复杂动态场景的新视角合成上取得了SOTA性能，同时训练时间减少了2倍。

## 研究问题与动机
1. **传统全MLP的动态NeRF方法训练极慢**：D-NeRF等需要数天至数周的训练时间，且依赖深度图等额外监督信号，难以实际应用。
2. **平面分解方法（如HexPlane、K-Planes）虽提速但细节丢失**：将4D场景分解为多个2D平面的方式显著降低了训练时间和内存，但在保留高频纹理细节方面存在不足。
3. **2D DWT在动态场景中存在两个固有缺陷**：一是**位移敏感性（shift variance）**，动态场景中运动、反射、光照变化导致微小位移会显著破坏小波振荡模式；二是**方向选择性差（poor direction selectivity）**，DWT产生棋盘格图案（±45°混叠），无法有效捕捉线条和边缘，导致运动物体周围出现鬼影伪影。
4. **现有基于小波的方法难以从静态扩展到动态场景**：Wavelet-based NeRF（如Masked Wavelet NeRF）主要在静态场景验证，直接套用于平面分解的动态方法会导致显著性能下降。

## 核心贡献（创新点）
1. **首次将DTCWT引入NeRF优化，提出方向感知表示（DaRe）**：通过学习6个方向的实部和虚部小波系数，利用复小波变换的移位不变性和方向选择性，解决了DWT在动态场景中的位移敏感和方向模糊问题，显著优于HexPlane等基线方法。
2. **提出可训练掩码（trainable mask）进行模型压缩**：针对DTCWT引入的2^d冗余（d=2时即4倍冗余），设计了逐方向学习的稀疏掩码，结合RLE+Huffman编码，在几乎不损失性能的前提下将模型尺寸压缩至与SOTA方法相当水平。
3. **通用表示框架同时适用于动态和静态场景**：不仅用于4D动态场景，还扩展到静态3D场景（基于TensorRF式分解），在NeRF Synthetic、NSVF、LLFF三个基准上均超越现有SOTA方法，实现了性能与模型尺寸的最优权衡。

## 方法详解
**整体框架**：以HexPlane的4D平面分解为基础，在每个2D平面上引入方向感知的小波表示，经逆对偶树复小波变换（IDTCWT）恢复平面特征后，通过配对平面相乘、拼接、乘以学习张量V^RF得到时空点的密度和外观特征，最后经小型MLP回归颜色值，通过体积渲染生成图像。

**4D动态场景分解（式1）**：将4D场景D分解为6组2D平面（XY-ZT、XZ-YT、YZ-XT）的低秩表示，内存复杂度从O(N³TF)降至O(RN²TF)。

**方向感知表示（式2）**：每个2D平面M_r^AB由逆DTCWT恢复，输入包含1个近似系数W_a^{AB}和12个小波系数（6个实部+6个虚部，对应6个方向），其中l为DTCWT变换层级（实验固定为1）。

**静态3D场景扩展（式3）**：采用TensorRF式分解，平面表示M_r^AB与向量表示v_r^C组合，方向感知表示同样适用于静态场景的平面部分。

**稀疏掩码与压缩（式4-5）**：引入可学习掩码M^{AB}，通过sg((H(M)-sigmoid(M))⊙W)实现梯度可控的稀疏化，配合L_m掩码损失（λ_m控制稀疏度），并对稀疏化后的二进制掩码进行RLE+Huffman编码压缩。

**损失函数（式6）**：L = (1/|R|) Σ||C(r)-Ĉ(r)||²₂ + λ_reg·L_reg + λ_m·L_m，其中L_reg为方向感知表示上的总变分（TV）损失，L_m为掩码稀疏损失。

**训练策略**：采用从粗到细的渐进分辨率训练、重要性采样、层级训练，以及空体素跳过加速。

## 实验与结果
**数据集**：Plenoptic Video Dataset（多视角动态视频，含高光、半透明、拓扑变化等复杂场景）、D-NeRF Dataset（单目合成视频）、NeRF Synthetic / NSVF / LLFF（静态场景）。

**Plenoptic Video（Table 1）**：
- **DaReNeRF-S**（100k步）：PSNR 30.224，D-SSIM 0.015，LPIPS 0.089，训练5h，模型244MB，相比HexPlane(650k步) PSNR提升0.754dB，训练时间减少10h。
- **DaReNeRF**（100k步）：PSNR 30.441，D-SSIM 0.012，LPIPS 0.084，训练4.5h，模型1,210MB，PSNR提升1.034dB，训练时间减少7.5h。

**D-NeRF（Table 2）**：
- **DaReNeRF**：PSNR 31.95，SSIM 0.97，LPIPS 0.03，无需变形场，超越D-NeRF(30.50)、TiNeuVox-S(30.75)，接近4D-GS(33.30)。

**静态场景**：
- NeRF Synthetic：8.91MB，PSNR 32.42，较DWT方法提升0.47dB。
- NSVF：8.98MB，PSNR 36.24，较DWT方法提升1.57dB，稀疏度94%。
- LLFF：13.67MB，PSNR 26.48，较DWT方法提升0.60dB。

**消融实验**：不同小波函数（Antonini/LeGall/Near Symmetric）对质量影响极小；λ_m=2.5×10⁻¹¹时稀疏度达94%（PSNR 36.24）；增加小波层级无显著性能提升但大幅增加训练时间和模型尺寸。

## 相关工作脉络
1. **HexPlane [7]**：平面分解式4D场景表示的代表作，将4D体素分解为6个2D平面组。DaReNeRF在其基础上将普通平面替换为方向感知小波表示，解决了HexPlane直接使用DWT时性能骤降的问题。
2. **D-NeRF [40]**：全MLP处理时空坐标的经典动态NeRF方法，需大量MLP推理、训练极慢且依赖变形场假设（无拓扑变化）。DaReNeRF无需变形场即可处理拓扑变化场景。
3. **K-Planes [14]**：同时在空间、时间和外观维度上显式分解的平面方法。DaReNeRF在PSNR和训练时间上均优于K-Planes-explicit，且无需变形场。
4. **Masked Wavelet NeRF [42]**：静态场景中小波压缩的代表方法，使用DWT。本文将其思想迁移至动态场景，但用DTCWT替代DWT，解决了方向模糊和位移敏感性。
5. **4D-GS [57]**：基于4D高斯溅射的动态场景实时渲染方法，在D-NeRF上PSNR达33.30。DaReNeRF以略低的PSNR（31.95）实现了无需变形场的纯监督学习。
6. **TensorF [10]**：静态3D场景的张量分解表示。本文在静态场景评估中基于TensoRF-192，加入方向感知表示后超越DWT版本。

## 局限性与未来方向
1. **极稀疏观测场景受限**：如D-NeRF式单目视频等极稀疏输入场景，因缺乏变形场和信息共享机制，鲁棒性不足。
2. **模型紧凑性不如DWT方法**：方向感知表示的冗余性导致难以达到<1MB的极致压缩（静态场景最优约8.91MB vs DWT的0.83MB），需在更小模型尺寸上进一步优化。
3. **未来方向**：探索更紧凑的方向感知表示构造方法；引入信息交互机制以适配极稀疏观测场景；进一步减小模型体积。

## 研究启发与可借鉴点
1. **DTCWT替代DWT解决动态场景方向/位移问题**：将复小波变换的方向选择性和近似移位不变性引入NeRF平面表示，是解决动态场景高频细节重建的有效思路，可迁移至其他频域表示工作。
2. **可训练掩码+RLE+Huffman的级联压缩策略**：该压缩pipeline直接从静态场景迁移到动态场景且效果良好，说明此类压缩策略具有良好的跨场景泛化能力。
3. **方向感知表示与平面分解的解耦设计**：将小波表示与平面分解正交解耦，意味着该表示可"插件式"接入任何基于平面分解的NeRF框架（如K-Planes、HexPlane），具有较高的方法通用性。
4. **静态-动态统一框架**：同一表示框架同时适用于4D动态和3D静态场景，避免了针对不同场景设计不同架构的繁琐，提示可探索更多跨场景的统一表示方法。
5. **消融验证小波函数选择的鲁棒性**：Table 6表明不同小波函数（Antonini/LeGall/Near Symmetric A/B）性能差异极小，这为实际部署时灵活选择实现提供了依据。

## 关键术语表
**Direction-Aware Representation (DaRe)**：基于对偶树复小波变换的方向感知表示，通过6个方向的小波系数捕获场景的方向特征，解决传统DWT的方向模糊和位移敏感问题。
**Dual-Tree Complex Wavelet Transform (DTCWT)**：对偶树复小波变换，通过两个交织的实小波树构建复小波，具有近似移位不变性和良好方向选择性，可同时输出6个方向的实部和虚部系数。
**Inverse DTCWT (IDTCWT)**：逆对偶树复小波变换，将方向感知的小波系数恢复为平面表示。
**Trainable Mask**：可学习的稀疏掩码，通过对小波系数逐元素调制实现有选择的特征保留，配合stop-gradient和Heaviside操作实现稀疏优化。
**Run-Length Encoding (RLE) + Huffman Encoding**：两种无损压缩算法的组合，先将二进制掩码做RLE编码，再对RLE结果做Huffman编码，大幅降低存储需求。
**Total Variation (TV) Loss**：总变分损失，施加于方向感知表示上以强制时空连续性。
**Emptiness Voxel**：空体素标记，通过聚合各时间步的最大不透明度生成的3D体素，用于跳过场景中空闲区域以加速渲染。
**Planar Decomposition (平面分解)**：将高维场景分解为多个低维平面表示的方法，如HexPlane将4D场景分解为3组2D平面。

## 可复现要素
- **数据集**：Plenoptic Video Dataset（公开）、D-NeRF Dataset（公开）、NeRF Synthetic / NSVF / LLFF（公开）
- **代码/权重是否开源**：论文未明确声明代码开源状态（论文未提及）
- **关键超参**：空间网格大小512、时间网格大小300、batch size 4096、wavelet level=1、λ_m取值范围1.0×10⁻¹⁰至2.5×10⁻¹¹、训练步数100k、训练设备单张A100 GPU
