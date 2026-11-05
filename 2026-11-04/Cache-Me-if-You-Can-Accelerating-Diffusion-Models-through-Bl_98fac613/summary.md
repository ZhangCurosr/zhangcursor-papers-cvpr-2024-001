---
title: "Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Bl"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wimbauer_Cache_Me_if_You_Can_Accelerating_Diffusion_Models_through_Block_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:21:01"
field: "扩散模型高效推理"
keywords: ["Diffusion Models", "Inference Acceleration", "Block Caching", "Latent Diffusion", "Scale-Shift Adjustment", "DPM-Solver"]
innovations: ["提出Block Caching机制复用diffusion step间冗余的layer block输出，实现1.5x-1.8x加速", "设计基于L1_rel累积阈值的自动Caching Schedule推导方法", "引入Scale-Shift Adjustment蒸馏微调机制消除缓存伪影且几乎不增加推理开销"]
benchmarks: ["COCO subset (face-removed)", "FID", "Human Evaluation (OUI Prompts, PartiPrompts)"]
---

# 论文速读：Cache-Me-if-You-Can-Accelerating-Diffusion-Models-through-Block-Caching

## 一句话总结
本文通过分析扩散模型U-Net内部layer的输出变化规律，发现多数block在不同denoising step间的输出变化很小且平滑，由此提出Block Caching技术——复用之前step的block输出以避免冗余计算，并结合轻量级Scale-Shift Adjustment机制防止伪影，在不降低图像质量的前提下实现1.5×-1.8×的推理加速。

## 研究问题与动机
- **扩散模型推理成本高**：Text-to-image基础模型（如LDM、EMU）需要反复调用大型U-Net进行多次denoising step，导致高延迟，阻碍实际应用。
- **现有加速方法将网络视为黑盒**：改进solver（如DPM-Solver）和蒸馏技术主要通过减少步数或训练更小模型来加速，但未挖掘U-Net内部的计算冗余。
- **layer输出存在时序平滑性**：作者观察到attention block的输出在不同step间变化平滑，且不同位置block的变化模式具有区分性，这为利用计算冗余提供了可能。
- **Step-to-step变化通常很小**：大多数step中block输出的改变量微乎其微，意味着大量计算是冗余的，值得通过缓存机制加以规避。

## 核心贡献（创新点）
- **提出Block Caching机制**：基于对U-Net内部block输出变化的系统分析，复用之前step的block计算结果以跳过冗余前向传播，与仅减少step数的黑盒方法本质不同。
- **设计自动Caching Schedule推导方法**：利用相对绝对变化量$L1_{rel}$作为指标，通过累积变化阈值$\delta$自动决定每个block何时刷新缓存，无需人工调参。
- **引入Scale-Shift Adjustment机制**：针对激进caching导致的特征错位伪影，为每个接收缓存输入的block添加 timestep-dependent 的scale和shift参数，通过student-teacher蒸馏式微调优化，几乎不增加推理开销。
- **系统验证广泛适用性**：在LDM-512（900M参数）和EMU-768（2.7B参数）两个模型上，结合DPM和DDIM两种solver，均实现1.5×-1.8×加速且FID和人类评估均优于等耗时基线。

## 方法详解
- **核心观察与动机**：对U-Net中各block的输出变化进行分析，定义相对绝对变化量：
$$L1_{rel}(i, t) = \frac{||C_i(x_t, s_t) - C_i(x_{t-1}, s_{t-1})||_1}{||C_i(x_t, s_t)||_1}$$
发现：① block输出随时间平滑变化；② 不同位置block呈现不同变化模式（高分辨率early/late block在末尾20%变化大，深层低分辨率block在初期变化大）；③ 多数step间变化量很小。
- **Block Caching机制**：由于U-Net使用残差连接$B_i(x, s) = C_i(x, s) + concat(x, s)$，可在block级别缓存$C_i$的输出。若当前step的block输出与上次缓存输出差异小于阈值，则直接复用缓存结果，跳过该block的前向计算。
- **自动Caching Schedule**：从step $t_a$开始缓存，只要累积变化满足$\sum_{t=t_a}^{t_b-1} L1_{rel}(i, t) \leq \delta$则继续使用缓存，一旦超过$\delta$则在$t_b$重新计算。$\delta=0.5$时在LDM-512上获得最佳速度与质量权衡。
- **Scale-Shift Adjustment**：为解决缓存导致的特征错位伪影，对每个接收缓存输入的block添加 timestep embedding 输入的线性层，预测每channel的scale和shift参数（形状为$N \times C$）。优化时采用蒸馏策略：student（带缓存）与teacher（同权重无缓存）共享参数，teacher使用student轨迹的中间step作为输入生成目标输出，仅优化scale-shift参数，冻结原始模型权重。
- **仅缓存SpatialTransformer blocks**：实验表明ResBlock变化不够平滑且对局部细节至关重要，缓存ResBlock仅获5%加速但质量显著下降，因此仅对计算开销最大的SpatialTransformer block应用caching。

## 实验与结果
- **数据集与模型**：LDM-512（900M参数，512×512，Shutterstock数据replacement train）和EMU-768（2.7B参数，768×768）；评估数据集为COCO子集（去除人脸）用于FID计算。
- **基线对比**：与相同步数baseline、等延迟baseline（减少步数）对比，使用DPM和DDIM两种solver，classifier-free guidance强度5.0，bfloat16推理，单卡A100测速。
- **主要结果（LDM-512，Tab.1）**：
  - DPM 20步caching vs 14步baseline：FID 17.15 vs 18.67，速度1.43×，SS后FID降至15.95（速度1.65×），优于20步无caching基线（FID 17.44）。
  - DPM 50步caching vs 30步baseline：FID 15.18（速度1.82×），优于50步基线（FID 17.44）。
  - DDIM结果类似，最高加速1.82×。
- **人类评估（EMU-768，Tab.2）**：12位标注员、1320票，caching在所有配置下均获更高胜率（DPM 20步win率34.7%，DDIM 50步win率29.7%）。
- **最优超参**：$\delta=0.5$在LDM-512上获得1.5×加速与最佳视觉质量平衡；Scale-Shift优化需15k迭代、8卡A100、耗时12-48小时。

## 相关工作脉络
- **Improved Solvers（DPM-Solver、DDIM等）**：通过优化采样轨迹减少所需步数，但仍将U-Net视为黑盒；本文与solver互补，可在相同solver下进一步加速。
- **Distillation方法（Progressive Distillation、Guidance Distillation等）**：通过重新训练学生模型减少step，但需昂贵重训且可能限制负prompt支持；本文仅添加轻量scale-shift参数、冻结原权重，无需改变模型架构。
- **Consistency Models / Latent Consistency Models**：旨在一步或多步生成，属于模型架构层面的根本改变；本文保持原始模型不变，仅通过推理时缓存加速。
- **Efficient Diffusion Inference（SDXL、PixArt-α等）**：通过模型设计或训练加速；本文聚焦推理阶段计算冗余挖掘，可与应用到任何现有diffusion model。
- **Black-box加速vs interpretability-driven加速**：已有工作多从solver或蒸馏角度处理，本文首次系统分析U-Net内部layer的时序行为并据此设计缓存策略。

## 局限性与未来方向
- **仅缓存SpatialTransformer blocks**：ResBlock因变化不平滑且对细节关键而被排除，未来可探索更精细的block级缓存策略或混合缓存。
- **Caching阈值$\delta$需手动调优**：虽然自动schedule相比固定步数减少更，但$\delta$仍依赖经验选择，可探索自适应或per-prompt动态调整。
- **Scale-Shift优化成本较高**：需15k迭代和8卡A100（12-48小时），虽远小于全模型重训，但仍需一定资源，未来可研究更高效的优化策略。
- **仅在LDM和EMU上验证**：尚未在SDXL、Stable Diffusion 3等更新架构上测试，适用性有待扩展。
- **未探索与更激进solver的结合**：如与DPM-Solver++或一致性模型配合的潜力未充分挖掘。

## 研究启发与可借鉴点
- **从黑盒转向白盒加速思路**：分析模型内部各组件的时序行为以发现计算冗余，而非仅调整外部solver，这一思路可迁移至视频生成、3D生成等迭代式扩散应用中。
- **蒸馏式微调思想用于推理优化**：用teacher-student框架仅优化少量附加参数来适配推理时修改（如缓存），既保持原始能力又适应新行为，值得在其它推理优化场景中借鉴。
- **残差连接结构利于fine-grained缓存**：U-Net的残差设计天然支持block级缓存而不破坏信息流，未来可探索其他具有类似结构的架构（如DiT）中的缓存机会。
- **L1_rel指标用于自动化调度**：以相对变化量作为缓存决策依据，简洁有效且跨prompt稳定，可推广至其他需要动态计算调度的场景。
- **与solver正交可组合**：Block Caching与改进solver、蒸馏技术均不冲突，可叠加使用，为构建高效推理pipeline提供模块化思路。

## 关键术语表
- **Block Caching**：在扩散模型denoising过程中，复用之前step中计算过的layer block输出，避免冗余前向传播。
- **SpatialTransformer Block**：U-Net中执行self-attention和cross-attention的模块，计算开销最大，是本文缓存的主要对象。
- **Scale-Shift Adjustment**：为每个接收缓存输入的block添加timestep-dependent的scale和shift参数，通过蒸馏微调消除缓存引入的伪影。
- **L1_rel**：衡量相邻step间block输出的相对绝对变化量，用于自动推导caching schedule。
- **Caching Schedule**：根据阈值$\delta$和$L1_{rel}$自动确定的每个block何时刷新缓存、何时复用历史输出的策略。
- **DPM-Solver**：将denoising过程建模为ODE的高阶求解器，可用较少step（如10-20步）生成高质量图像。
- **Classifier-free Guidance**：扩散模型中无需单独classifier的条件引导技术，通过联合训练条件/无条件分支实现。
- **FID（Fréchet Inception Distance）**：评估生成图像质量的常用指标，值越低表示分布越接近真实图像。

## 可复现要素
- **数据集**：LDM-512使用内部Shutterstock图像进行replacement training；评估使用COCO子集（去人脸）。论文未提及公开数据集链接。
- **代码**：项目页面为 https://fwmb.github.io/blockcaching，论文未明确声明GitHub开源，需访问项目页面确认。
- **权重**：未提及公开预训练权重，使用内部训练的LDM-512和开源的EMU-768。
- **关键超参**：$\delta=0.5$（LDM-512最优）、guidance scale=5.0、bfloat16精度、Scale-Shift优化15k iter/8×A100。
