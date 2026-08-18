---
title: "DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Lou_DaReNeRF_Direction-aware_Representation_for_Dynamic_Scenes_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:16:30"
field: "动态场景神经辐射场重建"
keywords: ["Dynamic Scene Reconstruction", "Neural Radiance Fields", "Dual-Tree Complex Wavelet Transform", "Novel View Synthesis", "Frequency-domain Representation", "Model Compression"]
innovations: ["首个将DTCWT引入NeRF的方向感知表示，解决DWT移位变分与方向模糊问题", "可训练掩码压缩机制实现12小波系数的高效稀疏化", "统一适用于动态与静态场景的高保真重建框架"]
benchmarks: ["Plenoptic Video Dataset", "D-NeRF Dataset", "NeRF Synthetic", "NSVF", "LLFF"]
---

# 论文速读：DaReNeRF-Direction-aware-Representation-for-Dynamic-Scenes

## 一句话总结
本文提出一种**方向感知表示（Direction-Aware Representation, DaRe）**，结合**双树复小波变换（DTCWT）**与平面分解框架，用于动态场景的高保真新视角合成；通过可训练掩码压缩冗余系数，在训练时间与模型体积之间实现更优权衡，并在动态/静态场景重建上均达到SOTA性能。

## 研究问题与动机
1. **动态场景建模的复杂性**：真实世界场景存在多目标运动、相机运动、光影变化等动态因素，传统隐式NeRF训练极慢且需额外监督（深度、光流等）。
2. **分解方法的局限性**：HexPlane等基于平面的分解方法虽加速训练，但难以保留高频纹理细节；直接引入2D离散小波变换（DWT）会导致性能显著下降。
3. **DWT的两个固有缺陷**：①**移位变分**：动态场景中微小偏移即破坏小波振荡模式，导致重影伪影；②**方向模糊**：DWT产生棋盘状图案，缺乏对特定方向边缘/线条的选择性。
4. **存储效率与性能的平衡**：方向感知表示引入$2^d$冗余（d=2时为6实+6虚），需高效压缩策略以匹配SOTA方法内存效率。

## 核心贡献（创新点）
1. **首个将DTCWT引入NeRF优化的方向感知表示**：通过6个方向的小波系数捕捉场景动态特征，从根本上解决DWT的移位变分和方向模糊问题，区别于HexPlane等纯分解方法。
2. **可训练掩码压缩机制**：为12个小波系数及近似系数引入独立稀疏掩码，配合RLE+Huffman编码，在几乎不损失性能前提下将模型体积压缩至与SOTA相当水平。
3. **统一适用于动态与静态场景**：在Plenoptic Video（动态）和NeRF Synth/NSVF/LLFF（静态）数据集上均超越现有方法，实现性能-效率更优权衡。
4. **无需额外监督信号**：仅依赖光度损失训练，不依赖深度图或光流，简化了动态场景重建流程。

## 方法详解
1. **动态场景分解基础**：将4D时空体积$D$分解为6组2D平面表示的张量积之和（公式1），每组平面$M_r^{AB} \in \mathbb{R}^{AB}$（如XY、ZT等），特征向量$v_r^i \in \mathbb{R}^F$，记忆复杂度从$\mathcal{O}(N^3TF)$降至$\mathcal{O}(RN^2TF)$。
2. **方向感知表示构建**：对每个2D平面$M_r^{AB}$应用**逆向双树复小波变换（IDTCWT）**，用1个近似系数+6实系数+6虚系数（共13个系数图）重建该平面（公式2）。DTCWT的双滤波器组满足Hilbert变换关系，使6个方向系数分别对齐特定角度，消除棋盘效应。
3. **静态场景适配**：采用TensorRF式分解（公式3），平面$M_r^{AB}$与向量$v_r^C$组合，同样可应用方向感知表示。
4. **稀疏压缩**：引入可学习掩码$\mathcal{M}^{AB}$，通过$\widehat{\mathcal{W}} = \text{sg}((H(\mathcal{M})-\text{sigmoid}(\mathcal{M}))\odot\mathcal{W})$（公式4）实现软阈值与梯度停滞，配合$\mathcal{L}_m=\sum\mathcal{M}$鼓励稀疏性。掩码值转8-bit后应用RLE+Huffman编码。
5. **优化目标**：总损失$\mathcal{L}=\frac{1}{|\mathcal{R}|}\sum\|\mathbf{C}(r)-\hat{\mathbf{C}}(r)\|_2^2+\lambda_{reg}\mathcal{L}_{reg}+\lambda_m\mathcal{L}_m$，其中正则化项为方向感知表示的全变差（TV）损失。采用coarse-to-fine训练策略与emptiness voxel加速。

## 实验与结果
- **动态场景数据集**：
  - **Plenoptic Video**（21相机同步视频，含复杂动态）：DaReNeRF（100k步）PSNR=32.258，D-SSIM=0.012，LPIPS=0.084，训练时间4.5h，模型体积1,210MB；稀疏版DaReNeRF-S PSNR=32.102，体积仅244MB。较HexPlane（100k步）PSNR提升0.69，训练时间减半。
  - **D-NeRF**（单目视频）：DaReNeRF PSNR=31.95，超越无形变场的HexPlane（31.04）与TiNeuVox-S（30.75），接近形变场方法4D-GS（33.30）。
- **静态场景数据集**：
  - **NeRF Synth**：VM-192(300)+DaRe PSNR=32.42 vs DWT版31.95（+0.47）。
  - **NSVF**：VM-192(300)+DaRe PSNR=36.24 vs DWT版34.67（+1.57）。
  - **LLFF**：VM-96(640)+DaRe PSNR=26.48 vs DWT版25.88（+0.60）。
- **消融实验**：不同DTCWT小波函数（Antonini/LeGall/Near Symmetric）性能相近且均优于DWT；$\lambda_m=2.5\times10^{-11}$时稀疏度达94.2%，模型8.98MB；小波层级增至1以上无显著收益。

## 相关工作脉络
1. **HexPlane [7]**：平面分解基准方法，直接应用DWT会导致性能退化，本文针对此缺陷提出方向感知表示。
2. **D-NeRF [40]**/ **DyNeRF [22]**：隐式形变场方法，需大量训练时间（DyNeRF 1344h）与额外监督，本文无需形变场即可处理拓扑变化场景。
3. **K-Planes [14]**/ **Tensor4D [46]**：并行分解方法，本文在纹理细节恢复上更优。
4. **Masked Wavelet NeRF [42]**：静态场景DWT压缩，本文将其思想迁移至动态场景并改用DTCWT解决方向/移位问题。
5. **4D-Gaussian Splatting [57]**：显式3D高斯方法，渲染速度快但训练数据要求高，本文在复杂动态场景重建质量上更具优势。

## 局限性与未来方向
1. **极端稀疏观测场景受限**：缺乏形变场机制，难以从极少量视角学习3D结构（如D-NeRF类单目设置）。
2. **模型紧凑性仍不及DWT方法**：即使压缩后静态场景模型体积仍大于1MB，难以达到极致压缩需求。
3. **未来方向**：探索更紧凑的方向感知表示构造方式；结合轻量化形变场以处理稀疏输入；拓展至实时渲染应用。

## 研究启发与可借鉴点
1. **频域表示与神经辐射场结合**：将信号处理中的DTCWT引入NeRF表示，为纹理细节恢复提供新思路，可迁移至静态场景压缩、Few-shot NeRF等方向。
2. **可训练掩码压缩管线**：软阈值+梯度停滞+熵编码的组合策略，适用于任何高维参数图的压缩任务。
3. **无监督动态场景重建**：仅凭光度损失训练即可处理复杂动态，避免了对深度/光流的依赖，简化了数据 pipeline。
4. **统一动态/静态表示**：同一框架通过调整分解形式适配两类场景，增强了方法通用性。

## 关键术语表
- **DaRe（Direction-aware Representation）**：基于DTCWT的方向感知表示，用13个小波系数图编码2D平面的方向性特征。
- **DTCWT（Dual-Tree Complex Wavelet Transform）**：双树复小波变换，通过两组滤波器产生复小波系数，具备近似移位不变性与良好方向选择性。
- **IDTCWT**：逆向双树复小波变换，用于从小波系数重建平面表示。
- **Plenoptic Video Dataset**：多相机同步采集的动态视频数据集，含复杂动态与透视效果，用于评估新视角合成。
- **D-NeRF Dataset**：单目视频数据集，场景为合成对象，用于测试稀疏观测下的重建能力。
- **可训练掩码（Trainable Mask）**：学习得到的稀疏掩码图，通过软阈值化抑制不重要的小波系数。
- **RLE+Huffman编码**：游程编码结合霍夫曼编码，用于压缩二值掩码以减小存储体积。
- **Emptiness Voxel**：记录场景中空白区域的3D体素，用于跳过无效光线采样以加速渲染。

## 可复现要素
- **数据集**：Plenoptic Video Dataset、D-NeRF Dataset、NeRF Synthetic、NSVF、LLFF（均为公开数据集）。
- **代码/权重**：论文未提及开源状态。
- **关键超参**：空间网格分辨率512，时间网格300，batch size 4096，小波层级1，$\lambda_m$取值$10^{-10}$~$2.5\times10^{-11}$，训练步数100k。
