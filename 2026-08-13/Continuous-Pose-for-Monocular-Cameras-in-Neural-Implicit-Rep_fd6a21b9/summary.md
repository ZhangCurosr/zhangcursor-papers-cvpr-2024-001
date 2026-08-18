---
title: "Continuous-Pose-for-Monocular-Cameras-in-Neural-Implicit-Rep"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Continuous_Pose_for_Monocular_Cameras_in_Neural_Implicit_Representation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:14:46"
---

# 论文速读：Continuous-Pose-for-Monocular-Cameras-in-Neural-Implicit-Rep

## 一句话总结
将单目相机位姿建模为时间的连续隐式神经函数（PoseNet），并与神经辐射场/SLAM联合优化；该设计规避了离散优化的累积误差与插值局限，在NeRF重建、异步事件相机及RGB-D vSLAM（含IMU融合）四个场景中均显著优于现有基线。

## 研究问题与动机
- 现有NeRF位姿优化（如BARF）采用离散SE(3)参数，面对高频IMU或异步事件时需人为累积或插值，引入测量误差并丢失细粒度运动细节。
- 离散优化在大初始位姿误差下极易陷入局部最优，导致图像拼接或场景重建失败。
- 传统连续轨迹表示（B-spline、Gaussian Process）难以直接与下游隐式神经表示（INR）端到端联合训练，且需额外设计基函数与超参。
- 实际相机运动常受机械约束，6-DOF全空间优化存在冗余，缺乏对低维运动流形的显式建模。

## 核心贡献（创新点）
1. **连续时间位姿神经表示**：用单个MLP将时间映射至SE(3)，可直接与NeRF/SLAM联合梯度下降，区别于传统离散Bundle Adjustment或需手工基函数的方法。
2. **内蕴运动（Intrinsic Motion）低维分解**：将相对位姿拆分为慢变参考系变换与低DOF内蕴运动，配合$\mathcal{L}_{dof}$稀疏正则显式诱导流形降维，优于全空间6-DOF优化。
3. **原生支持异步事件与IMU紧耦合**：无需事件累积或IMU预积分，利用自动微分直接计算姿态导数并施加约束，填补NeRF+SLAM中高频传感器融合的空白。
4. **解耦架构设计**：旋转与平移分别由独立MLP预测，结合正弦嵌入频段$F=5$，在多项实验中稳定超越单网络或B-spline基线。

## 方法详解
- **PoseNet基础结构**：8层MLP，隐藏维度256，ReLU激活；时间$t$先经正弦频域编码，末层tanh输出后归一化得到单位四元数$\mathbf{q}$与平移$\mathbf{v}$。
- **联合优化目标**：以NeRF为例，最小化 $\min_{\theta_s, \theta_p} \sum_i \| \mathcal{T}_i - g(\theta_s, T_i) \|^2$，其中预测位姿 $T_i = T_{init,i} \circ P(f(\theta_p, t_i))$，$P$为向量至刚体变换算子，场景隐式表示与位姿网络同步更新。
- **内蕴运动分解**：$T = T_o \circ T_I$，$f_o$预测参考系变换，$f_I$预测内蕴运动。$\mathcal{L}_{dof}$将旋转转为欧拉角并按视场角归一化，平移单位化后取$\ell_1$松弛近似$\ell_0$，强制内蕴运动分量稀疏；另加$\mathcal{L}_o = \|v_o\|_1$防止小旋转下平移发散。
- **IMU融合策略**：松散耦合对陀螺仪离散积分结果施加姿态$\ell_1$偏差损失；紧密耦合利用自动微分直接约束四元数导数 $\dot{\mathbf{q}} = \frac{1}{2}\Omega(\hat{\omega})$，绕过积分漂移，实现端到端IMU-视觉联合优化。

## 实验与结果
- **数据集与基线**：LLFF、Replica、ScanNet、TUM-RGBD、EUROC及事件相机合成/实物数据；基线含BARF、B-spline、EventNeRF、NICE-SLAM、Vox-Fusion、DI-Fusion、ORB-SLAM、VINS-MONO等。
- **NeRF从噪声位姿**：2D平面拼接成功率100%（BARF 30%/15%）；3D Real数据平均旋转误差从BARF的18.44°降至0.446°，平移误差从10.78降至0.32，PSNR提升约2 dB，且成功修复了BARF全部12处发散。
- **事件相机NeRF**：Translation Error显著下降（Chair 20帧从3.66 cm降至1.74 cm）；无需额外标定即可自动纠正偏心旋转带来的重建偏移。
- **vSLAM跟踪**：Replica平均ATE-RMSE从NICE-SLAM的1.06 cm降至0.49 cm（内蕴运动版）；ScanNet/TUM-RGBD全面持平或优于dense neural SLAM；EUROC紧密耦合IMU版本平均误差6.16，逼近稀疏特征方法（VINS-MONO 14.6）。
- **最强提升**：内蕴运动框架在Replica上普遍降低30%~50%轨迹误差，且DOF实测从5.22降至3.08（降幅41%），验证了低维流形先验的有效性。

## 相关工作脉络
- **BARF/DbARF系列**：离散参数联合优化位姿与NeRF；本文将其推广为连续时间函数，彻底消除关键帧插值误差。
- **B-spline/GP连续轨迹估计**：传统SLAM常用解析基函数建模连续位姿；PoseNet利用神经网络通用逼近能力与自动微分，更易嵌入可微渲染管线。
- **EventNeRF**：将异步事件累积为同步帧再重建；本文直接按事件触发时间查询连续位姿，保留事件相机微秒级时间戳优势。
- **NICE-SLAM/IMAP**：dense neural RGB-D SLAM代表；本文替换其tracking分支的离散关键帧优化为连续PoseNet，并引入低维流形约束。
- **VINS-MONO/ORBSLAM3**：稀疏特征+IMU紧耦合系统；本文证明隐式表示+连续位姿可在相似指标下逼近传统方法，拓展了隐式SLAM的传感器边界。

## 局限性与未来方向
- 实验集中于室内中速平稳场景，未覆盖高速剧烈运动或极端运动模糊/噪声分布。
- 内蕴运动参考系假设“缓慢变化”（每10个关键帧更新$f_o$），快速非线性机动下参考系滞后可能限制性能。
- 仅验证单目RGB、RGB-D与事件相机，多目立体或深度相机直推连续位姿的泛化性未充分探讨。
- 开源代码仅覆盖NeRF与部分模块，完整多场景工程实现未明确公开。

## 研究启发与可借鉴点
- **连续时间替代离散关键帧**：任何依赖多视图几何或动差优化的任务（SfM、视频稳像、光流场估计）均可尝试将位姿参数化为时间MLP，利用自动微分无缝对接高频传感器。
- **低DOF流形稀疏正则**：$\mathcal{L}_{dof}$的欧拉角归一化+$\ell_1$松弛设计简单高效，可直接迁移至刚体跟踪、动作分解等需提取主运动成分的任务。
- **四元数导数紧耦合范式**：绕过IMU预积分漂移、直接对网络输出求导匹配角速度，为神经状态估计与物理约束融合提供了干净且可微的实现路径。
- **旋转/平移解耦设计**：双MLP优于单MLP的结论提示后续高维流形学习任务可沿用解耦结构以提升优化稳定性。

## 关键术语表
- **PoseNet**：输入时间、输出相机SE(3)位姿的8层MLP，作为连续姿态表示的核心可微组件。
- **INR (Implicit Neural Representation)**：用神经网络隐式编码3D场景几何或辐射场的表示范式（如NeRF）。
- **Intrinsic Motion Frame**：将相机相对运动分解为慢变参考系变换与低维流形上的“内蕴运动”，后者自由度通常远低于6。
- **Loose/Tight IMU Coupling**：松散耦合指对IMU积分结果施加姿态偏差损失；紧密耦合指利用自动微分直接约束四元数导数与陀螺仪读数。
- **$\mathcal{L}_{dof}$**：作用于内蕴运动输出的稀疏正则项，通过归一化欧拉角与平移后的$\ell_1$范数近似鼓励低维流形。
- **EventNeRF**：利用事件相机异步流联合重建神经辐射场的代表性工作，本文在其基础上引入连续位姿优化以提升时间对齐精度。

## 可复现要素
- 数据集：LLFF、Replica、ScanNet、TUM-RGBD、EUROC 均为公开数据集；事件相机数据部分来自公开仿真模型与实物采集。
- 代码/权重：论文声明开源地址为 https://github.com/qimaqi/Continuous-Pose-in-NeRF，主要覆盖NeRF与部分SLAM模块，完整多场景工程代码未明确提供。
- 关键超参：PoseNet 为 8 层 MLP、隐藏维度 256、Sinusoidal 嵌入频段 $F=5$、旋转/平移解耦最优；IMU 紧密耦合使用四元数导数损失；内蕴运动参考系更新频率设为 10（每10个关键帧优化一次 $f_o$）。

<!--META
{"keywords": ["Continuous Pose", "Neural Implicit Representation", "NeRF", "vSLAM", "Event Camera", "IMU Fusion", "Intrinsic Motion", "Time-to-Pose Network"], "field": "神经辐射场与视觉SLAM", "innovations": ["将单目相机位姿建模为时间的连续神经函数以实现联合优化", "提出内蕴运动低维分解与DOF稀疏正则化", "利用自动微分原生支持事件相机与IMU紧耦合融合"], "benchmarks": ["LLFF", "
