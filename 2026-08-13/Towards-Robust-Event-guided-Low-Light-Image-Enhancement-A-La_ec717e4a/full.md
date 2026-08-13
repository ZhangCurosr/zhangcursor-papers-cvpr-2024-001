# Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach

Guoqiang Liang<sup>1</sup> Kanghao Chen<sup>1</sup> Hangyu Li<sup>1</sup> Yunfan Lu<sup>1</sup> Lin Wang<sup>1,2\*</sup>

<sup>1</sup>AI Thrust, HKUST(GZ) <sup>2</sup>Dept. of Computer Science and Engineering, HKUST

{gliang041,kchen879,hli886,ylu066}@connect.hkust-gz.edu.cn, linwang@ust.hk

Project Page: https://vlislab22.github.io/eg-lowlight/

## Abstract

Event camera has recently received much attention for low-light image enhancement (LIE) thanks to their distinct advantages, such as high dynamic range. However, current research is prohibitively restricted by the lack oflargescale, real-world, and spatial-temporally aligned eventimage datasets. To this end, we propose a real-world (indoor and outdoor) dataset comprising over 30K pairs of images and events under both low and normal illumination conditions. To achieve this, we utilize a robotic arm that traces a consistent non-linear trajectory to curate the dataset with spatial alignment precision under 0.03mm. We then introduce a matching alignment strategy, rendering 90% of our dataset with errors less than 0.01s. Based on the dataset, we propose a novel event-guided LIE approach, called EvLight, towards robust performance in real-world low-light scenes. Specifically, we first design the multiscale holisticfusion branch to extract holistic structural and textural information from both events and images. To ensure robustness against variations in the regional illumination and noise, we then introduce a Signal-to-Noise-Ratio (SNR)-guided regional feature selection to selectively fuse features ofimagesfrom regions with high SNR and enhance those with low SNR by extracting regional structure information from events. Extensive experiments on our dataset and the synthetic SDSD dataset demonstrate our EvLight significantly surpasses the frame-based methods, e.g., [4] by 1.14 dB and 2.62 dB, respectively.

## 1. Introduction

Images captured under sub-optimal lighting conditions often exhibit various types of degradation such as poor visibility, noise, and inaccurate color [23]. For this reason, low-light image enhancement (LIE) serves as an essential task in ameliorating low-light image quality. LIE is crucial for downstream tasks, e.g., face detection [27, 50] and nighttime semantic segmentation [29]. Recently, with the emergence of deep learning, abundant frame-based methods have been proposed, ranging from enhancing contrast [54], removing noise [47] to correcting color [38]. Although the performance has been remarkably boosted, these methods often suffer from unbalanced exposure and color distortion when the visual details, e.g., edges, provided by framebased cameras are less distinctive, as shown in Fig. 1 (c).

![](images/bb7240a48059afe2b0b96a97f71edfc8e4ee94b600ad558658c9c77937e8dbce.jpg)  
(a) Input low-light image

![](images/a640b452b35879de8cf8c04bb99896c70d6b2f6c1c673d769280ee19002795c0.jpg)  
(b) Input event

![](images/9408713f5afe8c0391822ad471e295159e4a00562d3acf24921d7cebb73c460b.jpg)

![](images/3d2578335f4644f8e77dbbf36db27cb80d3ce32c7722dd09a1ae58c28bd99d49.jpg)  
(c) Retinexformer [4]  
(d) Ours  
Figure 1. A challenging example of our dataset containing an extremely low-light image (a) and sparse events (b). Compared with the result from a SOTA frame-based method Retinexformer [4] (c), our EvLight (d) not only recovers the structure details (e.g., the pipe on the ceiling) but also avoids over-enhancement and saturation in the bright regions (e.g., the lights).

Event cameras are bio-inspired sensors that generate event streams with high dynamic range (HDR), high temporal resolution, etc. [33, 55]. However, few research efforts have been made in combining both frame-based and event cameras to address the LIE task [18, 24, 25, 52] to date. A hurdle is the prohibitive lack of large-scale real-world datasets with spatial-temporally aligned images and events. For example, [52] proposes an unsupervised framework without the need for paired event-image data, and [24, 25] leverage the synthetic datasets for training. Nonetheless, these methods are less competent for applications in realworld low-light scenarios. LIE dataset [18] is a real-world event-image dataset with paired low-light/normal-light sequences, obtained by simply adjusting indoor lamplight (artificial light fluctuations) and outdoor exposure time while maintaining a fixed camera position. Thus, similar to the previous frame-based dataset SMID [6], this dataset is only limited to static scenes.

![](images/32c64ae6986fb0281f2b8237dbcf170e941b78e83a432b5ba52c84233bee161c.jpg)  
Figure 2. (a) An illustration of collecting spatially-aligned image-event dataset by mounting a DAVIS 346 event camera on the robotic arm and recording the sequences with the same trajectory receptively. (b) An overview of our matching alignment strategy. (c) An example of our dataset with images and paired events captured in low-light (with an ND8 filter) and normal-light conditions.

In this paper, we propose a large-scale real-world dataset, named SDE dataset – containing over 30K pairs of spatiotemporally aligned images and events (see examples in Fig. 2 (c)) – captured under both low-light and normal-light conditions (Sec. 3). To construct such a dataset, the inherent difficulty stems from the complexities involved in ensuring precise spatial and temporal alignment between paired lowlight and normal-light sequences, especially for dynamic scenes in nonlinear motion. To achieve this, we design a robotic alignment system to guarantee spatial alignment, where a DAVIS346 event camera [35] is mounted on a Universal UR5 robotic arm, see Fig. 2 (a). Our system shows a remarkable spatial accuracy with an error margin of merely 0.03mm, a significant improvement over the frame-based dataset, SDSD [39] with the error of 1mm. Moreover, unlike the setup of uniform linear motion in SDSD and the static scene in the LIE dataset [18], our system embraces non-linear motions with complex trajectories. This significantly enhances the diversity of our dataset for real-world scenarios. As for temporal alignment, a direct way to obtain aligned sequences is to clip them according to the specific motion start and end timestamps. However, even with the same camera and robot setting, the intervals (blue regions in Fig. 2 (b)) between motion start timestamps (left red line) and the timestamps of the initial frame (magenta line) in each clipped sequence are different, resulting in random temporal errors. To this end, we propose a novel matching alignment strategy to reduce the temporal discrepancies.

Buttressed by the dataset, we propose an event-guided LIE approach, called EvLight, towards the robust performance in real-world low-light scenes. The underlying premise is that – while low-light images deliver crucial color contents and events offer essential edge details – both modalities may be corrupted by different kinds of noise, yielding different noise distributions. Therefore, directly fusing the features of both modalities, as commonly done in [18], may also aggravate the noise in different regions of the two inputs, as shown in the blue box area in Fig. 5 (g).

To tackle these problems, our key idea is to fuse event and image features holistically, followed by a selective region-wise manner to extract the textural and structural information with the guidance of Signal-to-Noise-Ratio (SNR) prior information. To ensure robustness against variations in the regional illumination and noise, we further introduce an SNR-guided feature selection to extract features of images from regions with high SNR and those of events from regions with low SNR. This preserves the regional textural and structural information (Sec. 4.2). Then, we design an attention-based holistic fusion branch to coarsely extract holistic structural and textural information from both events and images (Sec. 4.3). Finally, a fusion block with channel attention is employed to fuse the holistic feature with the regional feature of images and events.

We conduct extensive experiments by comparing with the frame-based e.g., [4] and event-guided e.g., [25] methods on our real-world dataset and SDSD dataset (framebased dataset) [39] with events generated from the event simulator [15]. The experiments show that our EvLight works decently for enhancing diverse underexposed images under extremely low-light conditions, as depicted in Fig. 1.

## 2. Related Work

LIE Datasets. The performance of learning-based methods heavily relies on the quality of the training datasets [10] for either images [3, 5, 7] or videos [6, 10, 17, 22, 39, 40]. For example, SDSD [39] obtains a pair of videos under various light conditions from a scene by mounting the camera on a mechatronic system. In this paper, we mainly focus on the event-image datasets. A summary of existing image-event datasets for low-light enhancement is shown in Tab. 1. EvLowLight [24] only includes lowlight images/events without corresponding normal-light images/events as ground truth, while DVS-Dark [52] provides unpaired low-light/normal-light images/events. LIE [18] is a real-world image-event dataset, captured by adjusting the camera’s light intake in a static scene, wherein events are triggered by the light changes (indoor) and exposure times (outdoor). In contrast, we present a real-world dataset with over 30K spatially and temporally aligned image-event pairs (both indoor and outdoor), using a robotic alignment system, considering the non-linear motion.

<table><tr><td>Dataset</td><td>Release</td><td>Dynamic Scene</td><td>With Ground Truth</td><td>Numbers</td></tr><tr><td>DVS-Dark [52]</td><td>x</td><td>√</td><td>x</td><td>17,765</td></tr><tr><td>LIE [18]</td><td>x</td><td>x</td><td>√</td><td>2,231</td></tr><tr><td>EvLowLight [24]</td><td>x</td><td>√</td><td>x</td><td>一</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>31,477</td></tr></table>

Table 1. A summary of existing real-world image-event datasets. Note that images in DVS-Dark are gray-scale.

Frame-based LIE. Frame-based methods for low-light image enhancement can be divided into non-learning-based methods [1, 11, 12, 28, 46] and learning-based methods [4, 7, 9, 38, 41, 44, 45, 48, 49, 53, 54]. Nonlearning-based methods typically rely on handcrafted features, such as histogram equalization [1, 28] and the Retinex theory [11, 12, 46]. Nonetheless, these methods lead to the absence of adaptivity and efficiency [44]. With the development of deep learning, an increasing number of learning-based methods have emerged, which can be bifurcated as Retinex-based methods [4, 7, 9, 44, 53, 54] and non-Retinex-based methods [38, 41, 45, 48, 49]. Specially, SNR-Aware [48] collectively exploits Signal-to-Noise-Ratio-aware transformers and convolutional models to dynamically enhance pixels with spatial-varying operations. However, these frame-based approaches often result in blurry outcomes and low Structural Similarity (SSIM) due to the buried edge in low-light images.

Event-based LIE. Event cameras enjoy HDR and provide rich edge information even under low-light scenes [55]. Zhang et al. [52] focuses on reconstructing grayscale images from low-light events but faces challenges in preserving original details using only brightness changes from events. Recently, some researchers have utilized events as guidance for low-light image enhancement [18, 19], lowlight video enhancement [24, 25], and deblurring for lowlight images [56]. ELIE [18] utilizes a residual fusion module to blend event and image for low light enhancement. Liu et al. [25] address artifacts in prior low-light video enhancement methods by synthesizing events from adjacent images for intensity and motion information, and propose a fusion transform module to fuse these event features with image features. EvLowLight [24] establishes temporal coherence by jointly estimating motion from both events and frames while ensuring the spatial coherence between events and frames with different spatial resolutions. However, these methods directly fuse features extracted from events and images without considering the discrepancy of the noise at the different local regions in events and images.

## 3. Our SDE Dataset

Capturing paired dynamic sequences from real-world scenes presents a formidable challenge, primarily attributed to the complexity involved in ensuring spatial and temporal alignment under varying illumination conditions. The first line of approaches employs a stereo camera system to simultaneously record the identical scenes, using non-linear transformations and cropping like DPED [16]. However, it struggles with SIFT keypoint computation and matching [26] in the low light. This hinders the identification of overlapped video segments. The second line of approaches [17, 22] constructs an optical system incorporating a beam splitter, allowing two cameras to capture a unified view. Nonetheless, achieving impeccable alignment with such systems remains challenging, resulting in spatial misalignments, as mentioned in [22, 24, 31]. The third line of approaches, e.g., SDSD [39] proposes a mechatronic system mounting the camera on an electric slide rail to capture low-light/normal-light videos separately (two rounds). However, SDSD is constrained by the limited linear motion of the electric slide rail. Differently, we design a robotic alignment system, equipped with an event camera to capture paired RGB images and events, under both low-light and normal-light conditions. Our system features the nonlinear motions with complex trajectories.

1) Data Capture with Spatial Alignment. To ensure the spatial alignment of paired sequences, a robotic arm (Universal UR5), exhibiting a minimal repeated error of 0.03mm, is equipped to capture sequences following an identical trajectory. We set the robotic system with a predefined trajectory and a DAVIS 346 event camera with fixed parameters, e.g. exposure time. Firstly, paired image and event sequences are acquired under normal lighting conditions. Subsequently, an ND8 filter is applied to the camera lens, which facilitates the capture of low-light sequences while maintaining consistent camera parameters, such as exposure time and frame intervals.

2) Temporal Alignment of Low-light/Normal-light sequences. The alignment of SDSD [39] dataset involves a manual selection of the initial and final frames of each paired video, based on the motion states depicted in the videos, leading to inevitable bias. To mitigate this problem, initial temporal alignment is performed by trimming the sequences based on the start and end timestamps of a predefined trajectory. However, even with consistent settings for exposure time and frame intervals, there exists a variable time interval between the start timestamp of the trajectory and the first frame timestamp captured post-initiation of the trajectory in each sequence. The bias causes the misalignment between each low-light image and its normal-light image pair, particularly in complex motion paths.

![](images/d6767014c2b8edb58daaad3d3e64f609e85b20a5af0a6d034922415c93181545.jpg)  
Figure 3. An overview of our framework. Our method consists of three parts, (a) Preprocessing (Sec. 4.1), (b) SNR-guided Regiona Feature Selection (Sec. 4.2), and (c) Holistic-Regional Fusion Branch (Sec. 4.3). Specifically, SNR-guided Regional Feature Selection consists of two parts: Image-Regional Feature Selection (IRFS) and Event-Regional Feature Selection (ERFS). Additionally, Holistic-Regional Fusion Branch encompasses Holistic Feature Extraction (HFE) and Holistic-Regional Feature Fusion (HRF).

To achieve further alignment, we introduce a matching alignment strategy, wherein sequences from each scene are captured multiple times to minimize the alignment error to the largest extent, as shown in Fig. 2 (b). Practically, we capture 6 paired event-image sequences per scene —three in low-light and three in normal-light conditionals. These 6 sequences are trimmed to the predefined trajectory’s start and end timestamps, ensuring uniform content across all videos. Subsequently, the time intervals between the trajectory’s start timestamps and the initial frame timestamps of each trimmed sequence are calculated. As shown in Fig. 2 (b), the time intervals (blue regions) of 6 sequences are different, and we match the low-light sequence with the normal-light sequence, which has the minimal absolute errors of their time intervals; thus, we can reduce the misalignment caused by the random time interval. With the matching alignment strategy, we achieve a remarkable precision, with 90% of the datasets, exhibiting temporal alignment errors below 0.01s, and maximum errors of 0.013s and 0.027s for our indoor and outdoor datasets, respectively.

## 4. The Proposed EvLight Framework

Based on our SDE dataset, we further propose a novel event-guided LIE framework, called EvLight, as depicted in Fig. 3. Our goal is to selectively fuse the features of the image and events to achieve robust performance for eventguided LIE. EvLight takes the low-light image I and paired event stream $\{ \mathbf { e } _ { k } \} _ { k = 1 } ^ { N }$ with $N$ events as inputs and outputs an ehnanced image ${ \mathbf I } _ { e n }$ . Our pipeline consists of three components: 1) Preprocessing, 2) SNR-guided Regional Feature Selection, and 3) Holistic-Regional Fusion Branch.

Event Representation. Given an event stream $\left\{ \mathbf { e } _ { k } \right\} _ { k = 1 } ^ { N } ,$ we follow [30] to obtain the event voxel grid E by assigning the polarity of each event to the two closest voxels. The bin size is set to 32 in all the experiments.

## 4.1. Preprocessing

Initial Light-up. As demonstrated in recent frame-based LIE methods [4, 41, 49], coarsely enhancing the low-light image benefits the image restoration process and further boosts the performance. For simplicity, we follow Retinexformer [4] for the initial enhancement. As shown in the Fig. 3, we estimate the initial light-up image ${ \mathbf { I } } _ { l u }$ as:

$$
\mathbf { I } _ { l u } = \mathbf { I } \odot \mathbf { L } , \mathbf { L } = \mathcal { F } ( \mathbf { I } , \mathbf { I } _ { p r i o r } ) ,\tag{1}
$$

where $\mathbf { I } _ { p r i o r } \ = \ m a x _ { c } ( \mathbf { I } )$ denotes the illumination prior map, with $m a x _ { c }$ denoting the operation that computes the max values for each pixel across channels. $\mathcal { F }$ outputs the estimated illumination map $\mathbf { L } ,$ which is then applied to the input image I through a pixel-wise dot product.

The SNR Map. Following the previous approaches [2, 8, 48], we estimate the SNR map based on the initial lightup image ${ \mathbf { I } } _ { l u }$ and make it an effective prior for the SNRguided regional feature selection in Sec. 4.2. Given the initial light-up image ${ \mathbf { I } } _ { l u }$ , we first convert it into grayscale one $\mathbf { I } _ { g } , i . \bar { e } . , \mathbf { I } _ { g } \mathbf { \bar { \in } } \mathbb { R } ^ { \bar { H } \times W }$ , followed by computing the SNR map $\mathbf { M } _ { s n r } = \tilde { \mathbf { I } } _ { g } / a b s ( \mathbf { I } _ { g } - \tilde { \mathbf { I } } _ { g } )$ , where $\tilde { \mathbf { I } } _ { g }$ is the denoised counterpart of $\mathbf { I } _ { g } .$ In practice, similar to SNR-Net [48], the denoised counterpart is calculated with the mean filter. Feature Extraction. Image feature $\mathbf { F } _ { i m g }$ of light-up image ${ \mathbf { I } } _ { l u }$ and event feature ${ \bf F } _ { e v }$ of the event voxel grid E are initially extracted with conv3 × 3.

![](images/79d77e9b84a2e12707eb1034041781f0b9f68992207b44f5e5dae34fc4ba353f.jpg)  
Figure 4. Details of each block in SNR-guided Regional Feature Selection and Holistic-Regional Fusion Branch’s decoder.

## 4.2. SNR-guided Regional Feature Selection

In this section, we aim to selectively extract the regional featuresfrom either images or events. We design an imageregional feature selection (IRFS) block to select image feature with higher SNR values, thereby obtaining imageregional feature, less affected by noise. However, SNR map assigns low SNR values to not only high-noise regions but also edge-rich regions. Consequently, solely extracting features from regions with high SNR values can inadvertently suppress edge-rich regions. To address this, we introduce an event-regional feature selection (ERFS) block for enhancing edges in areas with poor visibility and high noise.

As shown in Fig. 3, inputs of this module include the image feature $\mathbf { F } _ { i m g } .$ , the event feature $\mathbf { F } _ { e v } ,$ , and the SNR map $\mathbf { M } _ { s n r }$ . Firstly, the image feature $\mathbf { F } _ { i m g }$ and event feature ${ \bf F } _ { e v }$ are down-sampled with conv4×4 layers with the stride of 2 and SNR map $\mathbf { M } _ { s n r }$ undergoes an averaging pooling with the kernel size of 2. These donwsampling operations are represented as ‘Down Sample’ in Fig. 3 and we attain different scale image feature $\mathbf { \Delta } \mathbf { \bar { F } } _ { i m a } ^ { i } \in \mathbb { R } ^ { \frac { \sim } { 2 ^ { 2 } - i } \times \frac { W } { 2 ^ { 2 } - i } \times 2 ^ { 2 - i } C }$ event feature $\mathbf { F } _ { e v } ^ { i } \ \in \ \mathbb { R } ^ { \frac { H } { 2 ^ { 2 } - i } \times \frac { W ^ { ' } } { 2 ^ { 2 } - i } \times 2 ^ { 2 - i } C }$ , and SNR map $\mathbf { M } _ { s n r } ^ { i } \in \mathbb { R } ^ { \frac { H } { 2 ^ { 2 } - i } \times \frac { W } { 2 ^ { 2 } - i } }$ where $i \ = \ 0 , 1 , 2$ . Then, the image feature $\mathbf { F } _ { i m g } ^ { i }$ and event feature $\mathbf { F } _ { e v } ^ { i }$ are selected with the guidance of SNR map $\mathbf { M } _ { s n r } ^ { i }$ in IRFS block, and ERFS block. These two blocks then output the selected image features $\mathbf { F } _ { s e l - i m g } ^ { i }$ and event features $\mathbf { F } _ { s e l - e v } ^ { i }$ , respectively. We now describe the details of these two blocks.

Image-Regional Feature Selection (IRFS) Block. As depicted in Fig. 4 (a), for an image feature $\mathbf { F } _ { i m g } ^ { i } ,$ we initially process it through two residual blocks [13] to extract regional information and yield the output $\hat { \mathbf { F } } _ { i m g } ^ { i } .$ . Each block comprises two conv3 × 3 layers and an efficient channel attention layer [37]. The SNR map $\mathbf { M } _ { s n r } ^ { i }$ is then expanded along the channel to align with the image feature’s channel dimensions. Then, we normalize it and make it within the range of [0, 1]. We then apply a predefined threshold on the SNR map to attain $\hat { \mathbf { M } } _ { s n r } ^ { i }$ . To emphasize regions with higher SNR values and attain the selected image feature $\mathbf { F } _ { s e l - i m g } ^ { i } ,$ we perform an element-wise multiplication ⊙ between the extended SNR map and the image feature $\hat { \mathbf { F } } _ { i m g } ^ { i } .$ , formulated as:

$$
\mathbf { F } _ { s e l - i m g } ^ { i } = \hat { \mathbf { M } } _ { s n r } ^ { i } \odot \hat { \mathbf { F } } _ { i m g } ^ { i } .\tag{2}
$$

Event-Regional Feature Selection (ERFS) Block. Edgerich regions in the initial light-up image, particularly those underexposed, exhibit low SNR values. Additionally, we observe that events in high SNR regions $( e . g .$ ., wellilluminated smooth planes) are predominantly leak noise and shot noise events. Consequently, we design the ERFS block that utilizes the inverse of the SNR map to selectively enhance edges in low-visibility, high-noise areas, and to suppress noise events in sufficiently illuminated regions. The initial processing in this block follows a similar architecture to that used for the IRFS block, with ${ \bf F } _ { e \it { \tau } } ^ { i }$ as the input and $\hat { \mathbf { F } } _ { e v } ^ { i }$ as the output. Given the SNR map $\hat { \mathbf { M } } _ { s n r } ^ { i } ,$ we obtain the reserve of SNR map $\bar { \mathbf { M } } _ { s n r } ^ { i }$ by $\hat { \mathbf { l } } \mathbf { \Pi } - \hat { \mathbf { M } } _ { s n r } ^ { i }$ . To obtain the selected event-regional feature $\mathbf { F } _ { s e l - e v } ^ { i } ,$ the element-wise multiplication product ⊙ between the reserve of SNR map and the event feature is carried out, which is formulated as:

$$
\mathbf { F } _ { s e l - e v } ^ { i } = \bar { \mathbf { M } } _ { s n r } ^ { i } \odot \hat { \mathbf { F } } _ { e v } ^ { i } .\tag{3}
$$

## 4.3. Holistic-Regional Fusion Branch

In this section, we aim to extract the holistic features from both the eventfeatures and imagefeatures, so as to build up long-range channel-wise dependencies between them. Besides, the holistic features are enhanced with the selected image-regional and event-regional features in the holisticregion feature fusion process.

Fig. 3 (c) depicts our holistic-regional fusion branch, which employs a UNet-like architecture [32] with the skip connections. This branch takes the concatenated feature of image $\mathbf { F } _ { i m g }$ and event ${ \bf F } _ { e v }$ from the preprocessing stage (Sec. 4.1) as the input and the enhanced image ${ \mathbf I } _ { e n }$ as the output. In the contracting path, there are 2 layers and the output of each layer is $\mathbf { F } _ { h o } ^ { i + 1 } \in$ $\begin{array} { r } { \mathbb { R } \frac { H } { 2 ^ { 2 - | i + 1 | } } \times \frac { W } { 2 ^ { 2 - | i + 1 | } } \times 2 ^ { 2 - | i + 1 | } C } \end{array}$ where $i = - 2 , - 1$ . In the ith layer, the holistic feature $\mathbf { F } _ { h o } ^ { i }$ first undergoes the holistic feature extraction (HFE) block. Then with a strided conv4 $\times ~ 4$ down-sampling operation, the holistic feature $\mathbf { F } _ { h o } ^ { i + 1 }$ is obtained. In the expansive path, the output of each layer is $\mathbf { F } _ { h o } ^ { i }$ where $i = 0 , 1 , 2$ . As shown in Fig. 4, the holistic feature $\mathbf { F } _ { h o } ^ { i - 1 }$ is processed with the HFE block and $\hat { \mathbf { F } } _ { h o } ^ { i - 1 }$ is produced. Then, the holistic feature $\hat { \mathbf { F } } _ { h o } ^ { i - 1 }$ is up-sampled with a strided deconv2 × 2 and it is fused with the selected regional image $\mathbf { F } _ { s e l - i m g } ^ { i }$ and event features $\mathbf { F } _ { s e l - e v } ^ { i }$ in the holistic-regional fusion (HRF) block.

<table><tr><td rowspan="2">Input</td><td rowspan="2">Method</td><td colspan="3">SDE-in</td><td colspan="3">SDE-out</td><td colspan="3">SDSD-in</td><td colspan="3">SDSD-out</td></tr><tr><td>PSNR↑</td><td>PSNR*↑</td><td>SSIM↑ PSNR↑</td><td></td><td>PSNR*↑</td><td>SSIM↑</td><td>PSNR↑</td><td>PSNR*↑</td><td>SSIM↑</td><td>PSNR↑</td><td>PSNR*↑</td><td>SSIM↑</td></tr><tr><td>Event Only</td><td>E2VID+ (ECCV’20) [34]</td><td>15.19</td><td>15.92</td><td>0.5891</td><td>15.01</td><td>16.02</td><td>0.5765</td><td>13.48</td><td>13.67</td><td>0.6494</td><td>16.58</td><td>17.27</td><td>0.6036</td></tr><tr><td rowspan="4">Image Only</td><td>SNR-Net (CVPR’22) [48]</td><td>20.05</td><td>21.89</td><td>0.6302</td><td>22.18</td><td>22.93</td><td>0.6611</td><td>24.74</td><td>25.30</td><td>0.8301</td><td>24.82</td><td>26.44</td><td>0.7401</td></tr><tr><td>Uformer (CVPR’22) [43]</td><td>21.09</td><td>22.75</td><td>0.7524</td><td>22.32</td><td>23.57</td><td>0.7469</td><td>24.03</td><td>25.59</td><td>0.8999</td><td>24.08</td><td>25.89</td><td>0.8184</td></tr><tr><td>LLFlow-L-SKF (CVPR’23) [45]</td><td>20.92</td><td>22.22</td><td>0.6610</td><td>21.68</td><td>23.41</td><td>0.6467</td><td>23.39</td><td>24.13</td><td>0.8180</td><td>20.39</td><td>24.73</td><td>0.6338</td></tr><tr><td>Retinexformer (ICCV’23) [4]</td><td>21.30</td><td>23.78</td><td>0.6920</td><td>22.92</td><td>23.71</td><td>0.6834</td><td>25.90</td><td>25.97</td><td>0.8515</td><td>26.08</td><td>28.48</td><td>0.8150</td></tr><tr><td rowspan="4">Image+Event</td><td>ELIE (TMM&#x27;23) [18]</td><td>19.98</td><td>21.44</td><td>0.6168</td><td>20.69</td><td>23.12</td><td>0.6533</td><td>27.46</td><td>28.30</td><td>0.8793</td><td>23.29</td><td>28.26</td><td>0.7423</td></tr><tr><td>eSL-Net (ECCV’20) [36]</td><td>21.25</td><td>23.19</td><td>0.7277</td><td>22.42</td><td>24.39</td><td>0.7187</td><td>24.99</td><td>25.72</td><td>0.8786</td><td>24.49</td><td>26.36</td><td>0.8031</td></tr><tr><td>Liu et al. (AAAI’23) [25]</td><td>21.79</td><td>23.88</td><td>0.7051</td><td>22.35</td><td>23.89</td><td>0.6895</td><td>27.58</td><td>28.43</td><td>0.8879</td><td>23.51</td><td>27.63</td><td>0.7263</td></tr><tr><td>Ours</td><td>22.44</td><td>24.81</td><td>0.7697</td><td>23.21</td><td>25.60</td><td>0.7505</td><td>28.52</td><td>29.73</td><td>0.9125</td><td>26.67</td><td>30.30</td><td>0.8356</td></tr></table>

Table 2. Comparisons on our SDE dataset and SDSD [39] dataset. The highest result is highlighted in bold while the second highest result is highlighted in underline. Since E2VID+ [34] can only reconstruct grayscale images, its metrics are calculated in grayscale.

Holistic Feature Extraction (HFE) Block. As shown in Fig. 4 (c), holistic feature extraction is mainly composed of a multi-head self-attention module and a feed-forward network. Given a holistic feature $\mathbf { F } _ { h o } ^ { i - 1 }$ , the feature can be processed as:

$$
\begin{array} { r l } & { \hat { \mathbf { F } } _ { m i d } ^ { i - 1 } = \mathrm { A t t e n t i o n } ( \mathbf { F } _ { h o } ^ { i - 1 } ) + \mathbf { F } _ { h o } ^ { i - 1 } , } \\ & { \hat { \mathbf { F } } _ { h o } ^ { i - 1 } = \mathrm { F F N } ( \mathbf { L N } ( \hat { \mathbf { F } } _ { m i d } ^ { i - 1 } ) ) + \hat { \mathbf { F } } _ { m i d } ^ { i - 1 } , } \end{array}\tag{4}
$$

where $\hat { \mathbf { F } } _ { m i d } ^ { i - 1 }$ is the middle output, LN is the layer normalization, FFN represents the feed-forward network, and Attention signifies the channel-wise self-attention, analogous to the multi-head attention mechanism employed in [51].

Holistic-Regional Fusion (HRF) Block. This block first concatenates the selected image features $\mathbf { F } _ { s e l - i m g } ^ { i } ,$ selected event features $\mathbf { F } _ { s e l - e v } ^ { i } .$ , and up-sampled holistic features $\hat { \mathbf { F } } _ { h o } ^ { i - 1 }$ . This concatenated feature $\mathbf { F } _ { c a t } ^ { i }$ is then passed through conv3×3 layers to generate a spatial attention map. Sequentially, the element-wise multiplication is performed between the attention map and the concatenated features, which can be denoted as:

$$
\mathbf { F } _ { h o } ^ { i } = \mathcal { F } _ { 3 } ( \sigma ( \mathcal { F } _ { 1 } ( \mathbf { F } _ { c a t } ^ { i } ) ) \odot \mathcal { F } _ { 2 } ( \mathbf { F } _ { c a t } ^ { i } ) + \mathbf { F } _ { c a t } ^ { i } ) ,\tag{5}
$$

where ${ \mathcal { F } } _ { i }$ is the convolution operation indicated in Fig. 4 (d). σ and ⊙ denote the Sigmoid function and the elementwise production, respectively.

Optimization. The loss function L utilized for training is articulated as: $\mathcal { L } = \sqrt { | | \mathbf { I } _ { e n } - \mathbf { I } _ { g t } | | ^ { 2 } + \epsilon ^ { 2 } } + \lambda | | \Phi ( \mathbf { I } _ { e n } \bar { ) } -$ $\Phi ( \mathbf { I } _ { g t } ) | | _ { 1 }$ , where λ is a hyper-parameter, ϵ is set to $1 0 ^ { - 4 } , \mathbf { I } _ { e n }$ and $\mathbf { I } _ { g t }$ denote the enhanced and ground truth images, and Φ represents feature extraction using the Alex network [21].

## 5. Experiments

Implementation Details: We employ the Adam optimizer [20] for all experiments, with learning rates of 1e − 4 and $2 e - 4$ for SDE and SDSD datasets, respectively. Our framework is trained for 80 epochs with a batch size of 8 using an NVIDIA A30 GPU. We apply random cropping, horizontal flipping, and rotation for data augmentation. The cropping size is $2 5 6 \times 2 5 6$ , and the rotation angles include 90, 180, and 270 degrees.

Evaluation Metrics: We use the peak-signal-to-noise ratio (PSNR) [14] and SSIM [42] for evaluation. Following the finetuning of the overall brightness of predicted results in previous methods [45, 53], we introduce the PSNR\* as the metric to assess image restoration effectiveness beyond light fitting. The calculation of PSNR\* is formulated as:

$$
\begin{array} { r l } & { \mathrm { P S N R } ^ { * } = \mathrm { P S N R } ( \mathbf { I } _ { e n } \times \mathbf { R } _ { g t - e n } , \mathbf { I } _ { g t } ) , } \\ & { \mathbf { R } _ { g t - e n } = \mathbf { M e a n } ( \mathrm { G r a y } ( \mathbf { I } _ { g t } ) ) / \mathbf { M e a n } ( \mathrm { G r a y } ( \mathbf { I } _ { e n } ) ) , } \end{array}\tag{6}
$$

where ${ \mathbf { I } _ { e n } } , \ { \mathbf { I } _ { g t } }$ , Gray, Mean, and PSNR represent the enhanced image, the ground-truth image, the operation of converting RGB images to grayscale ones, the operation of getting mean value, and the operation of calculating PSNR value, respectively.

Datasets: 1) SED dataset contains 91 image+event paired sequences (43 indoor and 48 outdoor sequences) captured with a DAVIS346 event camera [33] which outputs RGB images and events with the resolution of 346 × 260. For all collected sequences, 76 sequences are selected for training, and 15 sequences are for testing. 2) SDSD dataset [39] provides paired low-light/normal-light videos with $1 9 2 0 \times 1 0 8 0$ resolution containing static and dynamic versions. We choose the dynamic version for simulating events and employ the same dataset split scheme as in SDSD [39]: 125 paired sequences for training and 25 paired sequences for testing. We first downsample the original videos to the same resolution (346 × 260) of the DAVIS346 event camera. Then, we input the resized images to the event simulator v2e [15] to synthesize event streams with noise under the default noisy model.

![](images/054ccc046fc903ea62c4e493a7f6ab069c820b75bc1c5dc16d4058f07e93c909.jpg)  
(f) Retinexformer  
(g) ELIE  
(h) Liu et al.  
Figure 6. Qualitative results on our SDE-out dataset.  
(i) Ours  
(j) GT

## 5.1. Comparison and Evaluation

We compare our method with recent methods with three different settings: (I) the experiment with events as input, including E2VID+ [34]. (II) the experiment with a RGB image as input, including SNR-Net [48], Uformer [43], LLFlow-L-SKF [45], and Retinexformer [4]. (III) the experiment with a RGB image and paired events as inputs, including ELIE [18], eSL-Net [36], and Liu et al. [25]. We reproduced ELIE [18] and Liu et al. [25] according to the descriptions in the papers, while the others are retrained with the released code. We replace the event synthesis module in [25] by inputting events captured with the event camera or generated from the event simulator [15].

Comparison on our SDE Dataset: Quantitative results in Tab. 2 showcase our method’s superior performance on the SDE dataset, outperforming baselines with higher PSNR by 0.65 dB for SDE-in and 0.29 dB for SDE-out. To assess image restoration effectiveness beyond light fitting, we computed PSNR\* and our method also notably surpasses SOTA techniques, achieving higher PSNR\* by 0.93 dB for SDEin and 1.21 dB for SDE-out. This marks a significant validation of our approach for low-light image enhancement.

Qualitatively, as depicted in Fig. 5 and Fig. 6 for indoor and outdoor scenes respectively, our method effectively reconstructs clear edges in dark areas (e.g., the red box areas in Fig. 5 and Fig. 6), surpassing frame-based methods like Retinexformer [4] and event-guided approaches such as Liu et al. [25]. Moreover, our method demonstrates less color distortion and noise on challenging regions (e.g., the wall in Fig. 6) than LLFlow-L-SKF [45] and ELIE [18], and Retinexformer [4], underscoring our method’s robustness.

Comparison on the SDSD Dataset: To evaluate our method’s generalization, we conducted comparisons on the SDSD dataset [39], with quantitative outcomes detailed in Tab. 2. Our method outperforms baselines significantly in PSNR, PSNR\*, and SSIM, leading by more than 0.94 dB for SDSD-in and 0.59 dB for SDSD-out. Although ELIE and Liu et al. [25] surpass frame-based methods in SDSD-in dataset, they suffer from the overfitting in SDSD-out dataset which is demonstrated by the substantial disparity between PSNR and PSNR\*. Qualitatively, as shown in Fig. 7, our method effectively restores underexposed images to more detailed structures, as highlighted in the red box area. Moreover, ELIE [18] tends to produce color distortions, as visible in the blue box area of Fig. 7 (d).

![](images/5ae0bed33cd4a59a45590909431aedf17cc7b76e5d92fdb783400cec3210074e.jpg)  
(a) Input image

![](images/df117e8932b0e1659779627cb10e78772ec1de58999eff64a0bda70c27972208.jpg)  
(b) SNR-Net

![](images/45e09a12984a9b81466d0659fa4005bdac54ebc1a66af15b5174081c20af72bd.jpg)  
(c) Retinexformer

![](images/d5f9c61f54dc90b2b70f0e349d8a00d314c3310f1927db4007f5519948585d42.jpg)  
(d) ELIE  
Figure 7. Qualitative results on SDSD dataset [39].

![](images/39af0711012a1a6be9e8d5caefdea30b20d6e9ff41966aa74b69423e897fbae2.jpg)  
(a) Base Model

![](images/7b9dad372d776c3cd542c739350605369274c53544519280810e458d67d6d059.jpg)

![](images/2c465f9037b0df328d14302b0e7bbf6baf3f3a2bfaee8d333bad0589946a2353.jpg)

![](images/cac87d59ead30fd187e83162d47f5f2bccb5b1ed0a0c4bc913abcd59c24c4d34.jpg)

(d) W IRFS  
![](images/fd39c0fcf651f40df89f8d5a650d9f5ba3c29e733b03e562049bc0da250e4820.jpg)  
(e) Ours

(c) W ERFS  
![](images/dd4d32e77cf6a06dec61990b51679a4fe5479bc7932cd05b2e935f73b747254e.jpg)  
(f) GT  
Figure 8. Visualization of ablation results.

## 5.2. Ablation Studies and Analysis

We conduct ablation studies on SDE-in dataset to assess the effectiveness of each module of our method. The basic implementation, without SNR-guided regional feature selection as described in Sec. 4.2, is called the Base model.

Impact of Events: To reveal the impact of events, we conduct experiments on the Base model. The variant excluding events attains a PSNR of 21.35 dB and an SSIM of 0.6985, whereas adding events results in a 0.23 dB improvement in PSNR and a 0.002 improvement in SSIM. However, the Base model cannot fully explore the potential of events demonstrated by the limited improvement in SSIM.

Impact of SNR-guided regional feature selection: To verify it, we conduct an ablation study in Tab. 3. We replace the SNR map with an all-ones matrix and remove the whole selection module (the Base model). Compared with the Base model (1<sup>st</sup> row), regional feature selection with an all-ones matrix (2<sup>nd</sup> row) and SNR-guided regional feature selection (3<sup>rd</sup> row) yield 0.28 dB and 0.86 dB increase in PSNR, respectively, demonstrating the necessity of regional features and the SNR map. Although regional feature selection with an all-ones matrix and Base model both have color distortion (e.g., the red box in Fig. 8 (a), (b)), (b) has better structure details than (a).

Impact of IRFS and ERFS: To verify them, we conduct an ablation study in Tab. 4. Compared with the Base model (1<sup>st</sup> row), image-regional feature selection (IRFS, 2<sup>nd</sup> row), event-regional feature selection (ERFS, 3<sup>rd</sup> row), and the combination of them (4<sup>th</sup> row) yields the 0.34 dB, 0.60 dB, and 0.86 dB increase in PSNR, respectively, demonstrating the necessity of the IRFS and ERFS block. As shown in

![](images/ffdaf5284759229771a1a92391c3d48b4c8d870b89aaea301aab380bab11557d.jpg)  
(e) Liu et al.

![](images/79af8f2e28407de91166e7b23f0f7f00fc53fbb0a539e67a57127dcaa8eeb8d7.jpg)  
(f) Ours

(g) GT  
![](images/dd7277085c6b04546e2964821b1591a19355dc452fb7ffa1f5cb87cafd249ce8.jpg)

<table><tr><td></td><td>Regional Feature Selection</td><td>SNR-guided</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td>x</td><td>x</td><td>21.58</td><td>0.7001</td></tr><tr><td>2</td><td>√</td><td>x</td><td>21.86</td><td>0.7490</td></tr><tr><td>3</td><td>√</td><td>√</td><td>22.44</td><td>0.7697</td></tr></table>

Table 3. Ablation of SNR-guided regional feature selection.
<table><tr><td></td><td>IRFS</td><td>ERFS</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td>x</td><td>x</td><td>21.58</td><td>0.7001</td></tr><tr><td>2</td><td>√</td><td>x</td><td>21.92</td><td>0.7108</td></tr><tr><td>3</td><td>x</td><td>√</td><td>22.18</td><td>0.7525</td></tr><tr><td>4</td><td>√</td><td>√</td><td>22.44</td><td>0.7697</td></tr></table>

Table 4. Impact of each module of SNR-guided regional feature selection.

Fig. 8, IRFS (d) or ERFS (c) can reduce the color distortion that appears in the Base model (a). With both IRFS and ERFS blocks, our results deliver the best visual quality (e.g., red box and blue box in Fig. 8).

Generalization Ability: To assess the generalization capability of our EvLight, we carry out an experiment on the CED [33] and MVSEC [57] with the model trained on our SDE dataset. Moreover, we use the model, trained on the synthetic events from the SDSD dataset [39] to evaluate the generalization capacity on real events of our SDE dataset. Detailed visual results are available in Suppl. Mat.

## 6. Conclusion

This paper presented a large-scale real-world event-image dataset, called SDE, curated via a non-linear robotic path for high-fidelity spatial and temporal alignment, encompassing low and normal illumination conditions. Based on the real-world dataset, we designed a framework, EvLight, towards robust event-guided low-light image enhancement, which adaptively fuse the event and image features in a holistic and region-wised manner resulting in robust and superior performance. Limitations and Future Work: Due to inherent limitations of DAVIS346 event cameras, RGB images in our SDE dataset may exhibit partial chromatic aberrations and the moire pattern. In the future, we will´ improve our hardware system to enable synchronous triggering of robots and event cameras, thereby significantly reducing labor costs associated with repetitive collection.

Acknowledgment. This paper is supported by the National Natural Science Foundation of China (NSF) under Grant No. NSFC22FYT45 and the Guangzhou City, University and Enterprise Joint Fund under Grant No.SL2022A03J01278.

## References

[1] Tarik Arici, Salih Dikbas, and Yucel Altunbasak. A histogram modification framework and its application for image contrast enhancement. IEEE Transactions on image processing, 18(9):1921–1935, 2009. 3

[2] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local algorithm for image denoising. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), pages 60–65. Ieee, 2005. 4

[3] Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Fredo´ Durand. Learning photographic global tonal adjustment with a database of input/output image pairs. In CVPR 2011, pages 97–104. IEEE, 2011. 3

[4] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinexbased transformer for low-light image enhancement. arXiv preprint arXiv:2303.06705, 2023. 1, 2, 3, 4, 6, 7

[5] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3291–3300, 2018. 3

[6] Chen Chen, Qifeng Chen, Minh N Do, and Vladlen Koltun. Seeing motion in the dark. In Proceedings ofthe IEEE/CVF International conference on computer vision, pages 3185– 3194, 2019. 2, 3

[7] Wenhan Yang Jiaying Liu Chen Wei, Wenjing Wang. Deep retinex decomposition for low-light enhancement. In British Machine Vision Conference, 2018. 3

[8] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising with block-matching and 3d filtering. In Image processing: algorithms and systems, neural networks, and machine learning, pages 354– 365. SPIE, 2006. 4

[9] Huiyuan Fu, Wenkai Zheng, Xiangyu Meng, Xin Wang, Chuanming Wang, and Huadong Ma. You do not need additional priors or regularizers in retinex-based low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18125– 18134, 2023. 3

[10] Huiyuan Fu, Wenkai Zheng, Xicong Wang, Jiaxuan Wang, Heng Zhang, and Huadong Ma. Dancing in the dark: A benchmark towards general low-light video enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12877–12886, 2023. 2, 3

[11] Xueyang Fu, Delu Zeng, Yue Huang, Xiao-Ping Zhang, and Xinghao Ding. A weighted variational model for simultaneous reflectance and illumination estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2782–2790, 2016. 3

[12] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on image processing, 26(2):982–993, 2016. 3

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 5

[14] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 6

[15] Yuhuang Hu, Shih-Chii Liu, and Tobi Delbruck. v2e: From video frames to realistic dvs events. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1312–1321, 2021. 2, 6, 7

[16] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey, and Luc Van Gool. Dslr-quality photos on mobile devices with deep convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 3277–3285, 2017. 3

[17] Haiyang Jiang and Yinqiang Zheng. Learning to see moving objects in the dark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7324–7333, 2019. 3

[18] Yu Jiang, Yuehang Wang, Siqi Li, Yongji Zhang, Minghao Zhao, and Yue Gao. Event-based low-illumination image enhancement. IEEE Transactions on Multimedia, 2023. 2, 3, 6, 7

[19] Haiyan Jin, Qiaobin Wang, Haonan Su, and Zhaolin Xiao. Event-guided low light image enhancement via a dual branch gan. Journal ofVisual Communication and Image Representation, 95:103887, 2023. 3

[20] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6

[21] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012. 6

[22] Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim, Byungju Woo, Haechan Lee, Sunghyun Cho, and Suha Kwak. Human pose estimation in extremely low-light conditions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 704–714, 2023. 3

[23] Chongyi Li, Chunle Guo, Linghao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, and Chen Change Loy. Low-light image and video enhancement using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(12):9396–9416, 2021. 1

[24] Jinxiu Liang, Yixin Yang, Boyu Li, Peiqi Duan, Yong Xu, and Boxin Shi. Coherent event guided low-light video enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10615–10625, 2023. 2, 3

[25] Lin Liu, Junfeng An, Jianzhuang Liu, Shanxin Yuan, Xiangyu Chen, Wengang Zhou, Houqiang Li, Yan Feng Wang, and Qi Tian. Low-light video enhancement with synthetic event guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1692–1700, 2023. 2, 3, 6, 7

[26] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60:91–110, 2004. 3

[27] Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image

enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5637– 5646, 2022. 1

[28] Keita Nakai, Yoshikatsu Hoshi, and Akira Taguchi. Color image contrast enhacement method based on differential intensity/saturation gray-levels histograms. In 2013 International Symposium on Intelligent Signal Processing and Communication Systems, pages 445–449. IEEE, 2013. 3

[29] Jingyi Pan, Sihang Li, Yucheng Chen, Jinjing Zhu, and Lin Wang. Towards dynamic and small objects refinement for unsupervised domain adaptative nighttime semantic segmentation. arXiv preprint arXiv:2310.04747, 2023. 1

[30] Henri Rebecq, Rene Ranftl, Vladlen Koltun, and Davide´ Scaramuzza. Events-to-video: Bringing modern computer vision to event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3857–3866, 2019. 4

[31] Jaesung Rim, Haeyun Lee, Jucheol Won, and Sunghyun Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16, pages 184–201. Springer, 2020. 3

[32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 5

[33] Cedric Scheerlinck, Henri Rebecq, Timo Stoffregen, Nick Barnes, Robert Mahony, and Davide Scaramuzza. Ced: Color event camera dataset. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019. 1, 6, 8

[34] Timo Stoffregen, Cedric Scheerlinck, Davide Scaramuzza, Tom Drummond, Nick Barnes, Lindsay Kleeman, and Robert Mahony. Reducing the sim-to-real gap for event cameras. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVII 16, pages 534–549. Springer, 2020. 6, 7

[35] Gemma Taverni, Diederik Paul Moeys, Chenghan Li, Celso Cavaco, Vasyl Motsnyi, David San Segundo Bello, and Tobi Delbruck. Front and back illuminated dynamic and active pixel vision sensors comparison. IEEE Transactions on Circuits and Systems II: Express Briefs, 65(5):677–681, 2018. 2

[36] Bishan Wang, Jingwei He, Lei Yu, Gui-Song Xia, and Wen Yang. Event enhanced high-quality image recovery. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 155–171. Springer, 2020. 6, 7

[37] Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, and Qinghua Hu. Eca-net: Efficient channel attention for deep convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11534–11542, 2020. 5

[38] Ruixing Wang, Qing Zhang, Chi-Wing Fu, Xiaoyong Shen, Wei-Shi Zheng, and Jiaya Jia. Underexposed photo enhance-

ment using deep illumination estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6849–6857, 2019. 1, 3

[39] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. Seeing dynamic scene in the dark: A high-quality video dataset with mechatronic alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9700–9709, 2021. 2, 3, 6, 7, 8

[40] Wei Wang, Xin Chen, Cheng Yang, Xiang Li, Xuemei Hu, and Tao Yue. Enhancing low light videos by exploring high sensitivity camera noise. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4111– 4119, 2019. 3

[41] Yinglong Wang, Zhen Liu, Jianzhuang Liu, Songcen Xu, and Shuaicheng Liu. Low-light image enhancement with illumination-aware gamma correction and complete image modelling network. arXiv preprint arXiv:2308.08220, 2023. 3, 4

[42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6

[43] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restoration. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 17683–17693, 2022. 6, 7

[44] Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2022. 3

[45] Yuhui Wu, Chen Pan, Guoqing Wang, Yang Yang, Jiwei Wei, Chongyi Li, and Heng Tao Shen. Learning semantic-aware knowledge guidance for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1662–1671, 2023. 3, 6, 7

[46] Jun Xu, Yingkun Hou, Dongwei Ren, Li Liu, Fan Zhu, Mengyang Yu, Haoqian Wang, and Ling Shao. Star: A structure and texture aware retinex model. IEEE Transactions on Image Processing, 29:5022–5037, 2020. 3

[47] Ke Xu, Xin Yang, Baocai Yin, and Rynson WH Lau. Learning to restore low-light images via decomposition-andenhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2281– 2290, 2020. 1

[48] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17714–17724, 2022. 3, 4, 5, 6, 7

[49] Xiaogang Xu, Ruixing Wang, and Jiangbo Lu. Low-light image enhancement via structure modeling and guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9893–9903, 2023. 3, 4

[50] Jun Yu, Xinlong Hao, and Peng He. Single-stage face detection under extremely low-light conditions. In Proceedings

of the IEEE/CVF International Conference on Computer Vision, pages 3523–3532, 2021. 1

[51] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022. 6

[52] Song Zhang, Yu Zhang, Zhe Jiang, Dongqing Zou, Jimmy Ren, and Bin Zhou. Learning to see in the dark with events. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16, pages 666–682. Springer, 2020. 2, 3

[53] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In Proceedings of the 27th ACM international conference on multimedia, pages 1632–1640, 2019. 3, 6

[54] Yonghua Zhang, Xiaojie Guo, Jiayi Ma, Wei Liu, and Jiawan Zhang. Beyond brightening low-light images. International Journal ofComputer Vision, 129:1013–1037, 2021. 1, 3

[55] Xu Zheng, Yexin Liu, Yunfan Lu, Tongyan Hua, Tianbo Pan, Weiming Zhang, Dacheng Tao, and Lin Wang. Deep learning for event-based vision: A comprehensive survey and benchmarks. arXiv preprint arXiv:2302.08890, 2023. 1, 3

[56] Chu Zhou, Minggui Teng, Jin Han, Chao Xu, and Boxin Shi. Delieve-net: Deblurring low-light images with light streaks and local events. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1155–1164, 2021. 3

[57] Alex Zihao Zhu, Dinesh Thakur, Tolga Ozaslan, Bernd<sup>¨</sup> Pfrommer, Vijay Kumar, and Kostas Daniilidis. The multivehicle stereo event camera dataset: An event camera dataset for 3d perception. IEEE Robotics and Automation Letters, 3 (3):2032–2039, 2018. 8