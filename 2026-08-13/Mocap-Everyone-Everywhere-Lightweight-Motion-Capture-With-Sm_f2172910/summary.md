---
title: "Mocap-Everyone-Everywhere-Lightweight-Motion-Capture-With-Sm"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Lee_Mocap_Everyone_Everywhere_Lightweight_Motion_Capture_With_Smartwatches_and_a_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 00:14:34"
---

# 论文速读：Mocap-Everyone-Everywhere-Lightweight-Motion-Capture-With-Sm

## 一句话总结
本文提出了一种仅依靠两块智能手表腕部 IMU 与一台头戴式单目相机即可实现高质量 3D 全身动作捕捉的轻量化方法，通过 SLAM 提取的头部 6DoF 位姿、自适应楼层高度更新机制以及多阶段 Transformer 回归，有效克服了极端稀疏传感器输入与非平坦户外场景下的估计歧义与根漂移问题。

## 研究问题与动机
- 现有高性能动捕系统依赖 17~19 个专业 IMU 或 6 个以上专家级传感器，设备昂贵、穿戴繁琐，难以面向普通大众与长期户外使用。
- 基于 VR/AR 头显的方案虽能提供头部与手腕的 6DoF 信息，但严格依赖封闭室内环境与专用设备，无法扩展到广阔的真实世界场景。
- 仅使用 2 个腕部 IMU 时，输入信号极度稀疏且 IMU 固有歧义（方位/加速度积分误差）会导致严重的根节点漂移（root drift）与下半身姿态不确定。
- 在非平坦地形（楼梯、坡地）中，固定世界坐标系的 Z=0 平面无法代表人实际站立的地板高度，导致头部绝对高度信号失真。

## 核心贡献（创新点）
- **首创双智能手表 + 头戴相机的轻量级全身动捕框架**：证明消费级设备组合即可达到甚至超越 6 专业 IMU/VR 设备的动捕精度，大幅降低普及门槛。
- **自适应楼层高度更新算法**：基于脚部接触检测将足部投影至 3D 点云动态估计当前站立平面高度，使头部位姿在复杂户外地形中仍保持度量准确性。
- **多阶段 Transformer 运动估计与流形视觉优化模块**：通过末端位置中间表示解耦时序回归，并在预学得的运动流形空间中融合首人称/第三人称视觉线索进行后优化，显著缓解稀疏传感器的固有歧义。

## 方法详解
- **输入与预处理**：系统输入为头灯相机序列 $\mathbf{I}$ 与左右腕部 IMU 的角速度/加速度 $\mathbf{S}_{IMU}$。首先运行 DROID-SLAM 获取相机轨迹 $\mathbf{C}$ 与环境点云 $\mathbf{W}$，将 SLAM 任意坐标系对齐至真实世界度量坐标系（负 Z 轴对齐重力），并校准 IMU 朝向使其与重力方向一致。
- **头部位姿与楼层更新**：由相机位姿推导头部关节位姿 $\mathbf{H}_t \in SE(3)$。针对非平坦地面，设计楼层更新机制：利用前一时刻脚部接触概率 $\mathbf{c}_t$，找到最近一次高置信度触地点 $\mathbf{p}_{t_m}^f$，将其投影至点云 $\mathbf{W}$ 并取邻近点的平均 Z 值作为当前楼层高度 $f_t$，进而计算相对身高 $h_t = \mathbf{H}_z - f_t$。
- **多阶段 Transformer 估计模块 $\mathcal{F}_{est}$**：采用滑动窗口 seq2seq 架构。输入归一化至窗口首帧头部坐标系，并拼接绝对高度 $h_\tau$ 与头部上向量 $\theta_\tau^{up}$。网络分为两子模块：$\mathcal{F}^{end}$ 预测手/脚末端位置中间表示 $\mathbf{x}^{mid}$；$\mathcal{F}^{body}$ 融合原始特征与 $\mathbf{x}^{mid}$ 输出全身关节旋转、根平移及左右脚接触概率。损失函数涵盖位置误差 $\mathcal{L}_{pos}$、旋转误差 $\mathcal{L}_{rot}$、根误差 $\mathcal{L}_{root}$、末端一致性 $\mathcal{L}_{cons}$、接触 BCE 损失 $\mathcal{L}_{contact}$ 及足部滑移惩罚 $\mathcal{L}_{footvel}$，端到端联合训练。
- **流形空间视觉优化模块 $\mathcal{F}_{opt}$**：构建视觉线索 $\Phi = [\phi_E, \phi_T]$，其中 $\phi_E$ 为首人称图像中的 2D 手腕关键点，$\phi_T$ 为多用户场景中他人相机提供的 3D 姿态。利用卷积自编码器在学习数据上学得的运动流形进行后处理：从初始潜向量 $\mathbf{z}=E(\mathbf{M}_{reg})$ 出发，最小化 $\mathcal{L}=\mathcal{L}_{vis}+\mathcal{L}_{reg}+\mathcal{L}_{contact}$，最终经解码器 $E^{-1}(\mathbf{z}^*)$ 还原出满足视觉约束且保持时空连贯性的最终动作 $\mathbf{M}$。

## 实验与结果
- **数据集**：训练使用 AMASS；IMU 基线评测使用 TotalCapture（真实 IMU）；VR 基线评测使用 AMASS 子集划分与 HPS（大规模场景）。
- **对比基线**：6 IMU 全身体感方案 PIP [55]、TIP [23]；VR 上部肢体方案 AGRoL [12] 及其变体 AGRoL*（腕部位置替换为加速度）。
- **核心数字**：
  - 相比 PIP/TIP，r.MPJPE 以 3.77 cm 优于 4.40/4.88 cm；MPJPE 大幅降低 87.7%（vs PIP）与 89.8%（vs TIP），Root PE 降低 90.5% 与 92.1%，根漂移抑制效果显著。
  - 相比 AGRoL，在 AMASS 测试集上性能相当；在 HPS 大规模
