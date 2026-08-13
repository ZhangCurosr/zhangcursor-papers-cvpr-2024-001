# COSALPURE: Learning Concept from Group Images for Robust Co-Saliency Detection

Jiayi Zhu<sup>1</sup>, Qing Guo<sup>3†</sup>, Felix Juefei-Xu<sup>2</sup>, Yihao Huang<sup>4</sup>, Yang Liu<sup>4</sup>, Geguang Pu<sup>1,5†</sup>

<sup>1</sup> East China Normal University, China <sup>2</sup> New York University, USA IHPC & CFAR, Agency for Science, Technology and Research, Singapore <sup>4</sup> Nanyang Technological University, Singapore <sup>5</sup> Shanghai Industrial Control Safety Innovation Tech. Co., Ltd, China

![](images/87c58274dc84c3d1ec796fbefdcce317e3c00032df8583e24f7bf41089ca6e97.jpg)  
Figure 1. Examples of our method COSALPURE and comparative results before and after purification. COSALPURE comprises two modules: group-image concept learning and concept-guided purification. Firstly, the concept learning module inputs a group of images that contain some adversarial cases and obtain their shared co-salient semantic information (i.e., the learned concept), denoted as c. We can validate the effectiveness of the learned c through the visualization via a text-to-image (T2I) diffusion model. Secondly, steered by the previously learned concept, we employ certain diffusion generation techniques to purify the entire group of images. Before our purification, the co-salient object detection results are poor, but after purification, the detection results are satisfactory. Please enlarge to see more details.

## Abstract

Co-salient object detection (CoSOD) aims to identify the common and salient (usually in theforeground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) ofthe co-salient objects. In this paper, we propose a novel robustness enhancement framework byfirst learning the concept ofthe co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose COSALPURE containing two modules, i.e., group-image concept learning and conceptguided diffusion purification. For thefirst module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise predictionfunction as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness ofCoSODs significantly. The project is available at https://v1len.github.io/CosalPure/.

## 1. Introduction

Co-salient object detection (CoSOD) plays a pivotal role in visual information analysis, aiming to identify and accentuate common and salient objects across a set of images [33]. This study area, crucial for applications like image segmentation and object recognition, has witnessed considerable advancement with the advent of neural networkbased methodologies. These methods excel in discerning shared saliency cues among images, offering a significant leap over traditional saliency detections [20]. However, their robustness is severely tested under adverse conditions, such as adversarial attacks and various image common corruption, including but not limited to motion blur [13].

The susceptibility of CoSOD methods to adversarial perturbations, such as those introduced by Jadena [12], poses a significant challenge. These perturbations, while not altering the high-level semantic information of images, can drastically reduce the accuracy of co-salient object detection. The disparity between the corrupted image’s saliency map and the ground truth, as a result of these attacks, highlights a critical vulnerability in current CoSOD approaches.

Currently, there are indeed methods aimed at defending against adversarial attacks, such as DiffPure, which employs a noise addition and denoising strategy to eliminate perturbations. However, when restoring the image, Diff-Pure does not take into account the identity of the object within the image (i.e., without object-specific information). As a result, the restored images produced by DiffPure may contain artifacts that are artificially generated. These artifacts will affect the detection results of the CoSOD method.

To fill this gap, this work introduces a novel robustness enhancement framework, COSALPURE. The intuitive idea is to first learn a concept from the group images and then use it to guide the data purification based on text-to-image (T2I) diffusion. The ‘concept’ means the high-level semantic information of co-salient objects in the group images and falls within the text’s latent space. Specifically, this innovative approach comprises two meticulously designed modules: group-image concept learning and concept-guided diffusion purification. The first module focuses on learning the concept of co-salient objects from group images, demonstrating robustness and resilience to adversarial examples. The second module strategically maps adversarial images into a latent space, following which diffusion generation techniques, steered by the previously learned concept, are employed to purify these images effectively.

As shown in the left panel of Fig. 1, when a group of images containing some adversarial examples is passed into co-salient object detectors, the detection results are poor. We first apply the concept learning module to obtain the shared co-salient semantic information c (i.e., the learned concept). The bottom right corner of Fig. 1 shows the visualization results of c via a T2I diffusion model. It is evident that the semantic information in the visualized images aligns with the original group images, demonstrating the effectiveness of the concept learning module. Secondly, we utilize the just-proven effective learned concept c to guide the purification. From the right panel of Fig. 1, we can observe that the purified group images via COSALPURE exhibit satisfactory performance in the CoSOD task.

Extensive experimental results substantiate the effectiveness of COSALPURE. COSALPURE stands as a robust, concept-driven solution, paving the way for more reliable and accurate co-salient object detection in an era where image manipulation and corruption are increasingly prevalent.

## 2. Related Work

Co-salient object detection. Different from single-image saliency detection [1, 6, 18, 20, 27, 28], the goal of cosaliency detection is to detect common salient objects in a group of images [9, 16, 17, 29, 34, 35], evolving from early feature-based approaches to sophisticated deep learning and semantic-driven methods. Deciphering correspondences among co-salient objects across multiple images is pivotal for co-saliency detection. This challenge can be effectively tackled through optimization-based methods [3, 19], machine learning-based models [5, 31], and deep neural networks [29, 34, 35]. GICD [35] employs a gradientinduced mechanism that pays more attention to discriminative convolutional kernels which helps to locate the cosalient regions. GCAGC [34] presents an adaptive graph convolutional network with attention graph clustering for co-saliency detection.

Adversarial attack for co-salient object detection. Jadena [12] is an adversarial attack that jointly tunes the exposure and additive perturbations, which can drastically reduce the accuracy of co-salient object detection.

Text-to-image diffusion generation model. The popularity of Text-to-Image (T2I) generation [30] is propelled by diffusion models [7, 15, 23], necessitating training on extensive text and image paired datasets like LAION-5B [25]. The adeptly trained model demonstrates proficiency in generating diverse and lifelike images based on user-specific input text prompts, realizing T2I generation. T2I personalization [10, 24] is geared towards steering a diffusion-based T2I model to generate innovative concepts.

Diffusion-based image purification methods. DiffPure [22] is a notable approach in the field of image processing, specifically designed to enhance the robustness of images against adversarial attacks. It employs a strategy of introducing controlled noise via the forward stochastic differential equation (SDE) [21] and subsequently denoising the image via the reverse SDE to counteract adversarial perturbations. While effective in reducing these perturbations, it is noteworthy that DiffPure does not explicitly consider object semantics during the image restoration process. Diffusion-

Driven Adaptation (DDA) [11] is a test-time adaptation method that improves model accuracy on shifted target data by updating inputs through a diffusion model [15], effectively avoiding domain-wise re-training.

## 3. Preliminaries and Motivation

## 3.1. Co-salient Object Detection (CoSOD)

We have a group of images $\mathcal { T } = \{ \mathbf { I } _ { i } \in \mathbb { R } ^ { H \times W \times 3 } \} _ { i = 1 } ^ { N }$ that contain N images, and these images have common salient objects. We denote a CoSOD method as COSOD(·), taking I as input and predicting N salient maps,

$$
\mathcal { S } = \{ \mathbf { S } _ { i } \} _ { i = 1 } ^ { N } = \mathbf { C } \boldsymbol { 0 } \mathbf { S } \boldsymbol { 0 } \mathbf { D } ( \mathcal { T } ) ,\tag{1}
$$

where $\mathbf { S } _ { i } \in \mathbb { R } ^ { H \times W }$ is a binary map $( i . e . ,$ , the saliency map) indicating the salient region of the i-th image I . We show an example of co-saliency detection results in Fig. 1.

## 3.2. Robust Issues of CoSOD

However, at times, part of the acquired images may be of low quality (i.e., corrupted by some degradation), which will affect the robustness of CoSOD methods. In particular, [12] proposes the joint adversarial noise and exposure attack that can reduce the detection accuracy of state-of-theart CoSODs significantly. To be specific, within the entire group, there are M images that have been added adversarial perturbations, denoted as $\{ \mathbf { I } _ { j } ^ { ' } \} _ { j = 1 } ^ { M }$ , while the remaining are considered as clean images $\left\{ { \bf I } _ { k } \right\} _ { k = 1 } ^ { N - M }$ . In this scenario, we can reformulate Eq. (1) as

$$
\begin{array} { r } { \boldsymbol { \mathcal { S } } ^ { \prime } = \{ \mathbf { S } _ { i } ^ { \prime } \} _ { i = 1 } ^ { N } = \operatorname { C o S O D } ( \{ \mathbf { I } _ { j } ^ { \prime } \} _ { j = 1 } ^ { M } \cup \{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { N - M } ) . } \end{array}\tag{2}
$$

The difference between S from Eq. (1) and $S ^ { \prime }$ indicates the robustness of the CoSOD method. Previous research findings indicate that existing CoSOD methods are susceptible to the influence of anomalous data (e.g., adversarial noise and exposure) [12]. Note that the degraded images (e.g., $\{ \mathbf { I } _ { j } ^ { ' } \} _ { j = 1 } ^ { M ^ { - } } )$ may not only affect the saliency maps of themselves but also impact the saliency maps of clean images.

Therefore, to enhance the robustness of CoSOD methods, defense methods should be developed. However, few works are focusing on this direction. A typical defense method is to purify the input images to remove the effects of degradations. In the following, we study the SOTA purification method, i.e., DiffPure [22], to enhance the robustness and show that enhancing the robustness of CoSOD is a nontrivial task and new technologies should be developed.

## 3.3. DiffPure and Challenges

A highly intuitive approach is to perform image reconstruction on input images, hoping to remove degradations. As an existing method for image reconstruction, DiffPure [22] can remove adversarial perturbations by applying forward diffusion followed by a reverse generative process. However, DiffPure is limited in its ability to address adversarial additive perturbations. It presents limited capabilities for handling other degradations like adversarial exposure. Fig. 2 illustrates two cases, with the upper case depicting a koala and the lower case representing a train. The input images for both cases are under the attack method [12]. The images processed through DiffPure visually eliminate perturbations. However, when the purified images underwent co-salient object detection together with the images within their respective groups, the detection results are inferior. Fig. 2 illustrates that the DiffPure cannot enhance the robustness of CoSOD under the attack method [12].

![](images/3392607fb8235589246c4d2022301605a213484a3241f90c96e7c14324f54feb.jpg)  
Figure 2. CoSOD results for DiffPure.The input images are under the attack method [12]. Processed by DiffPure [22], the purified images perform inferior in the CoSOD task together with their respective group images.

We tend to design a more effective purification method. DiffPure is specifically designed against adversarial attacks for image classification and neglects the specific properties of the CoSOD task: ❶ Only partial images within the group are attacked, and the clean images contain rich complementary information, which could help enhance the robustness. ❷ Although the adversarial patterns may affect the semantic features of images, the fact group images contain co-salient objects has not changed. How to utilize such a property should be carefully studied.

## 4. Methodology: COSALPURE

## 4.1. Overview

Beyond DiffPure [22], we propose to learn the concept of co-salient objects from the group images and leverage it to guide the purification. Specifically, given group images $\mathbf { \mathcal { T } ^ { \prime } } = \{ \mathbf { I } _ { j } ^ { \prime } \} _ { j = 1 } ^ { M ^ { \prime } } \cup \{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { N - M }$ that contains M degraded images and $\dot { N } - M$ clean images, we first learn the concept from $\mathcal { T } ^ { \prime }$ via the recent developed textual inversion method. The learned concept is a token and lies in the latent space of texts. We name it as ‘concept’ since we can use it to generate new images containing the ‘concept’. Note that the number of the degraded images (i.e., M) is unknown during application. We denote the concept of learning as

$$
\mathbf { c } = \mathbf { C o n c e p t L e a r n } ( \mathcal { T } ^ { \prime } ) ,\tag{3}
$$

and we detail the whole process in Sec. 4.2.

![](images/31ec53c5e3c21e98850e31442155aac172cff3d46fd71a742c5d1d68ece9c736.jpg)  
Figure 3. Overview of COSALPURE. The details of (a) are in Sec. 4.2, while the details of (b) are in Sec. 4.3.

![](images/08a08f0cf80e1993410cedeade1dd533efbcc723d1b01d3598fa4d50c753a565.jpg)  
Figure 4. Demonstration of the effectiveness of concept learning. (a) Five clean images are utilized for concept learning, and the learned concept can be reconstructed into an image through a pre-trained text-to-image model. (b) The first two images are attacked by Jadena [12] while the subsequent three images are clean, and the learned concept can also be reconstructed into a high-quality image. (a) and (b) use the same random seed.

After obtaining the concept, we aim to leverage it for purification by

$$
\hat { \mathbf { I } } = \mathbf { C o n c e p t P u r e ( c , I ) } , \mathbf { I } \in \mathcal { T } ^ { \prime } ,\tag{4}
$$

where the image $\hat { \bf I }$ is the purified image of I that may be a clean image or a perturbed image. We detail the conceptguided diffusion purification in Sec. 4.3. For each image in $\mathcal { T } ^ { \prime }$ , we can handle it via Eq. (4) and get a novel group denoted as $\hat { \mathcal { T } } .$ Then, we feed $\hat { \mathcal { T } }$ to CoSOD methods to see whether their robustness is enhanced or not.

The core idea is valid based on a critical assumption: the perturbed images in $\mathcal { T } ^ { \prime }$ do not affect the concept learning. We detail this in the Sec. 4.2.

## 4.2. Group-Image Concept Learning

In this section, we introduce the detail of groupimage concept learning $( i . e .$ , Eq (3)), which tends to utilize a group of input images for learning the textaligned embedding of common objects they have, as shown in Fig. 3 (a). We denote this process as $\begin{array} { r l } { \mathbf { c } } & { { } = } \end{array}$ ConceptLearn $\left( \{ \mathbf { I } _ { j } ^ { ' } \} _ { j = 1 } ^ { M } \cup \{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { N - M } \right)$ where c represents a token aligned with the texts’ latent space and represents the semantic information of common objects.

To this end, we formulate group-image concept learning as the personalizing text-to-image problem [10, 24] to enable text-to-image (T2I) diffusion models to rapidly swift new concept acquisition.

Text-to-Image Diffusion Model. We introduce the architecture and procedure of a classical T2I diffusion model [23]. It consists of three core modules: (1) image autoencoder, (2) text encoder, (3) and conditional diffusion model. The image autoencoder module has two submodules: an encoder E and a decoder D. It serves a dual purpose, where the encoder maps an input image X to a low-dimensional latent space with $\mathbf { z } = \mathcal { E } ( \mathbf { X } )$ , while the decoder transforms the latent representation back into the image space with $D ( \mathcal { E } ( \mathbf { X } ) ) \approx \mathbf { X }$ . The text encoder Γ firstly processes a text y by tokenizing it and secondly translates it into a latent space text embedding Γ(y). The conditional diffusion model $\epsilon _ { \theta }$ takes the time step $t ,$ the noisy latent $\mathbf { z } _ { t }$ at t-th time step and the text embedding Γ(y) as input to predict the noise added on $\mathbf { z } _ { t }$ , denoted as $\epsilon _ { \theta } ( \mathbf { z } _ { t } , t , \Gamma ( \mathbf { y } ) )$ .

Given a pre-trained T2I diffusion model and group images $\mathcal { T } ^ { \prime } = \dot { \{ \mathrm { I } _ { j } ^ { \prime } \} } _ { j = 1 } ^ { M } \cup \{ \mathrm { I } _ { k } \} _ { k = 1 } ^ { N - M }$ used for CoSOD task, we aim to learn a concept of the common object within $\mathcal { T } ^ { \prime }$ by

$$
\begin{array} { r l } & { \mathbf { c } = \arg \operatorname* { m i n } _ { \mathbf { c } ^ { * } } \mathbb { E } _ { \mathbf { X } \in \mathcal { T } ^ { \prime } , \mathbf { z } \in \mathcal { E } ( \mathbf { X } ) , \mathbf { y } , \epsilon \in \mathcal { N } ( 0 , 1 ) , t } \big ( } \\ & { \qquad \mathbf { c } ^ { * } } \\ & { \qquad \quad \big \| \epsilon _ { \theta } \big ( \mathbf { z } _ { t } , t , \Upsilon \big ( \Gamma ( \mathbf { y } ) , \mathbf { c } ^ { * } \big ) \big ) - \epsilon \big \| _ { 2 } ^ { 2 } \big ) , } \end{array}\tag{5}
$$

where y is a fixed text $( i . e .$ , ‘a photo of $S ^ { \ast \dagger } )$ and the function $\Upsilon ( \Gamma ( \mathbf { y } ) , \mathbf { c } ^ { * } )$ is to replace the token of $^ { \bullet } S ^ { \ast \bullet }$ within $\Gamma ( \mathbf { y } )$ with $\mathbf { c } ^ { * }$ . Intuitively, Eq. (5) forces the concept $\mathbf { c } ^ { * }$ to represent the co-salient objects within group images and also lies in the text latent space corresponding to the text $^ { \bullet } S ^ { \ast \bullet }$ . After obtaining c, we can embed $^ { \bullet } S ^ { \ast \bullet }$ into other texts to generate new images via the T2I diffusion model. For example, in Fig. 4 (b), we learn a concept of the co-salient objects $( i . e .$ piano), which corresponds to the text $^ { \ast , } S ^ { \ast \mathrm { \bullet } }$ . Then, we feed a text $( e . g .$ , ‘a photo of $S ^ { \ast \ast \ast }$ ) to the T2I model that generates an image containing the object, which means that the learned concept represents the salient objects in $\mathcal { T } ^ { \prime }$ very well.

Robustness of concept learning. It is obvious that the above method naturally aligns with our objective since we can exploit it to obtain the common semantic content among the group images for the CoSOD task. The key problem is whether the concept learning would be affected by degradations like adversarial perturbation in the group images. We conduct an empirical study to validate this. Specifically, given a group of clean images $( i . e . , \mathcal { T } )$ , we use it to learn a concept via Eq. (5). Meanwhile, we conduct the adversarial CoSOD attack [12] on two images within I and form a new group $\mathcal { T } ^ { \prime } .$ . With I<sup>′</sup>, we learn another concept via Eq. (5). Then, we can leverage the two learned concepts to generate images based on the same text prompt. As shown in Fig. 4, we find that the two generated images based on two concepts are similar, demonstrating that the adversarial examples have limited influence on concept learning. This inspires us to leverage the learned concept to purify the adversarial examples.

## 4.3. Concept-guided Diffusion Purification

We propose reconstructing the group images based on the learned concept c to eliminate the potential adversarial patterns as shown in Fig. 3 (b). Considering the advantage of the continuous representation [4, 14] in the smooth image reconstruction and its ability to remove perturbations, we employ a continuous representation module for the initial processing of the input image X, denoted as $\tilde { \mathbf { X } } = \mathbf { C } \mathbf { R } ( \mathbf { X } )$ This module not only somewhat denoises X but also addresses the issue of pre-trained CoSOD models designed for specific resolutions that do not match the input resolution of our employed diffusion model. Subsequently, the encoder of the image autoencoder module maps X<sup>˜</sup> to the latent space ${ \bf z } _ { 0 } = \mathcal { E } ( \tilde { \bf X } )$ . The following procedure is based on the diffusion pipeline and needs two procedures: forward process and reverse process. The forward diffusion process is a fixed Markov chain that iteratively adds a Gaussian noise to the latent ${ \bf z } _ { 0 }$ over $T$ timesteps, obtaining a sequence of noised images ${ \bf z } _ { 1 } , { \bf z } _ { 2 } , \cdots , { \bf z } _ { T }$ . In each step of the forward progress, the latent at time step $t \in [ 1 , T ]$ is updated by

$$
\begin{array} { r } { \mathbf { z } _ { t } = a _ { t } \mathbf { z } _ { t - 1 } + b _ { t } \epsilon _ { t } , \epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , } \end{array}\tag{6}
$$

where $a _ { t }$ and $b _ { t }$ are coefficients and $\mathcal { N } ( 0 , \bf { I } )$ represents the standard Gaussian distribution. By superimposing time steps from t = 1 to T, Eq. (6) can be simplified to

$$
\mathbf { z } _ { t } = \sqrt { \overline { { \alpha } } _ { t } } \mathbf { z } _ { 0 } + \sqrt { 1 - \overline { { \alpha } } _ { t } } \epsilon _ { t } , \mathbf { \epsilon } _ { t } \sim \mathcal { N } ( 0 , \mathbf { I } ) ,\tag{7}
$$

where we have $a _ { t } ^ { 2 } + b _ { t } ^ { 2 } = 1$ , and $\begin{array} { r } { \alpha _ { t } = a _ { t } ^ { 2 } , \overline { { \alpha } } _ { t } = \prod _ { \tau = 1 } ^ { t } \alpha _ { \tau } } \end{array}$ As we set the time step as T, the complete forward process can be expressed as

$$
\mathbf { z } _ { T } \sim \mathbf { q } ( \mathbf { z } _ { 1 : T } | \mathbf { z } _ { 0 } ) = \prod _ { t = 1 } ^ { T } \mathbf { q } ( \mathbf { z } _ { t } | \mathbf { z } _ { t - 1 } ) .\tag{8}
$$

![](images/61a1bfcb017c253c570f570467b921e473e3b942e45bc285fe2c3ea4dd38fb77.jpg)  
Figure 5. Attention maps for learned concepts on processed images.

For the reverse process, it iteratively removes the noise to generate an image in T timesteps. Unlike doing the reverse process directly, our method incorporates the obtained semantic embedding c as additional object information into the pipeline. Then, we can start from $\mathbf { z } _ { T }$ (alternatively called $\hat { \mathbf { z } } _ { T } )$ and progressively predict

$$
\epsilon _ { t - 1 } = \epsilon _ { \theta } ( \hat { \mathbf { z } } _ { t } , t , \mathbf { c } ) ,\tag{9}
$$

and obtain the latent at time step t − 1 via

$$
\begin{array} { r l r } {  { \hat { \mathbf { z } } _ { t - 1 } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } ( 1 - \alpha _ { t } ) } { 1 - \bar { \alpha } _ { t } } \tilde { \mathbf { z } } _ { 0 } } } \\ & { } & { + \frac { \sqrt { \alpha } _ { t } ( 1 - \bar { \alpha } _ { t - 1 } ) } { 1 - \bar { \alpha } _ { t } } \hat { \mathbf { z } } _ { t } + \sigma _ { t } \boldsymbol { \xi } , } \end{array}\tag{10}
$$

with

$$
\tilde { \mathbf { z } } _ { 0 } = \frac { \hat { \mathbf { z } } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t - 1 } } { \sqrt { \bar { \alpha } _ { t } } } ,\tag{11}
$$

where $\begin{array} { r } { \sigma _ { t } ^ { 2 } \ = \ \frac { ( 1 - \alpha _ { t } ) ( 1 - \overline { { \alpha } } _ { t - 1 } ) } { 1 - \overline { { \alpha } } _ { t } } } \end{array}$ and $\xi \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ according to the sample process of DDPM [15]. We can directly obtain the reconstructed image ˆx by using the decoder with formula $\hat { \mathbf { x } } = \mathcal { D } ( \hat { \mathbf { z } } _ { 0 } )$ .

To confirm that the concept learned by COSALPURE is applied accurately in the image reconstruction, we employ DAAM [26] to establish attention maps for the learned concepts on processed images as shown in Fig. 5. In each case, the attention map of the semantic embedding c (i.e., the learned concept) aligns well with the object itself in the image, indicating the effectiveness of COSALPURE.

## 5. Experiment

## 5.1. Experimental Setup

Datasets. We conduct experiments on Cosal2015 [32], iCoseg [2], CoSOD3k [8], and CoCA [36]. These four datasets contain 2,015, 643, 3,316, and 1,295 images of 50, 38, 160, and 80 groups respectively. We apply the SOTA adversarial attack for CoSOD (i.e., Jadena [12]) to the first 50% of images in each group, while the remaining 50% of images are kept in the clean state. We select the “augment” version of Jadena and follow the settings[12].

Table 1. Co-saliency detection performance. “Source-Only” means the group of images before processing, including 50% adversarial images and 50% clean images. We highlight the top results of each CoSOD method and each dataset in red.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="4">GICD</td><td colspan="4">GCAGC</td><td colspan="4">PoolNet</td></tr><tr><td>SR↑</td><td>AP↑</td><td> $F _ { \beta } \uparrow$ </td><td>MAE↓</td><td>SR↑</td><td>AP↑</td><td>Fβ↑</td><td>MAE↓</td><td>SR↑</td><td>AP↑</td><td> $F _ { \beta } \uparrow$ </td><td>MAE↓</td></tr><tr><td></td><td>Source-Only</td><td>0.3493</td><td>0.7306</td><td>0.4038</td><td>0.1676</td><td>0.5285</td><td>0.7853</td><td>0.6302</td><td>0.1570</td><td>0.5677</td><td>0.7425</td><td>0.6095</td><td>0.1276</td></tr><tr><td></td><td>DiffPure</td><td>0.4595</td><td>0.7478</td><td>0.5118</td><td>0.1444</td><td>0.4987</td><td>0.6998</td><td>0.5901</td><td>0.2162</td><td>0.6327</td><td>0.7779</td><td>0.6714</td><td>0.1181</td></tr><tr><td>Cos2015</td><td>DDA</td><td>0.4565</td><td>0.7579</td><td>0.5158</td><td>0.1469</td><td>0.5955</td><td>0.7928</td><td>0.6774</td><td>0.1542</td><td>0.6233</td><td>0.7863</td><td>0.6691</td><td>0.1181</td></tr><tr><td></td><td>COSALPURE</td><td>0.5602</td><td>0.7898</td><td>0.6177</td><td>0.1296</td><td>0.5975</td><td>0.7449</td><td>0.6521</td><td>0.2063</td><td>0.6908</td><td>0.8268</td><td>0.7258</td><td>0.1086</td></tr><tr><td></td><td>Source-Only</td><td>0.4012</td><td>0.7269</td><td>0.5063</td><td>0.1420</td><td>0.6469</td><td>0.8237</td><td>0.7173</td><td>0.1146</td><td>0.5847</td><td>0.8116</td><td>0.6472</td><td>0.1057</td></tr><tr><td>oseg</td><td>DiffPure</td><td>0.4447</td><td>0.7291</td><td>0.5503</td><td>0.1269</td><td>0.6609</td><td>0.8043</td><td>0.7051</td><td>0.1257</td><td>0.6796</td><td>0.8328</td><td>0.7144</td><td>0.0905</td></tr><tr><td>DDA</td><td></td><td>0.4665</td><td>0.7519</td><td>0.5948</td><td>0.1280</td><td>0.6982</td><td>0.8257</td><td>0.7390</td><td>0.1235</td><td>0.6578</td><td>0.8483</td><td>0.7179</td><td>0.0940</td></tr><tr><td></td><td>COSALPURE</td><td>0.5396</td><td>0.7611</td><td>0.6329</td><td>0.1208</td><td>0.7060</td><td>0.8052</td><td>0.7265</td><td>0.1413</td><td>0.7278</td><td>0.8730</td><td>0.7577</td><td>0.0850</td></tr><tr><td></td><td>Source-Only</td><td>0.3281</td><td>0.6988</td><td>0.4003</td><td>0.1439</td><td>0.4445</td><td>0.7325</td><td>0.5702</td><td>0.1376</td><td>0.4466</td><td>0.6606</td><td>0.5255</td><td>0.1386</td></tr><tr><td>COSOD3K</td><td>DiffPure</td><td>0.3887</td><td>0.6976</td><td>0.4683</td><td>0.1342</td><td>0.4996</td><td>0.7364</td><td>0.6279</td><td>0.1272</td><td>0.5247</td><td>0.7021</td><td>0.6064</td><td>0.1340</td></tr><tr><td></td><td>DDA</td><td>0.3838</td><td>0.7083</td><td>0.4776</td><td>0.1344</td><td>0.5337</td><td>0.7655</td><td>0.6544</td><td>0.1251</td><td>0.5105</td><td>0.7078</td><td>0.5875</td><td>0.1311</td></tr><tr><td></td><td>COSALPURE</td><td>0.4659</td><td>0.7327</td><td>0.5487</td><td>0.1221</td><td>0.5946</td><td>0.7999</td><td>0.6881</td><td>0.1144</td><td>0.5859</td><td>0.7432</td><td>0.6605</td><td>0.1215</td></tr><tr><td></td><td>Source-Only</td><td>0.1837</td><td>0.5490</td><td>0.3402</td><td>0.1168</td><td>0.2339</td><td>0.5177</td><td>0.4698</td><td>0.1227</td><td>0.2239</td><td>0.4296</td><td>0.4082</td><td>0.1500</td></tr><tr><td>COCCA</td><td>DiffPure</td><td>0.1706</td><td>0.5362</td><td>0.3492</td><td>0.1213</td><td>0.2231</td><td>0.5051</td><td>0.4995</td><td>0.1190</td><td>0.2185</td><td>0.4426</td><td>0.4286</td><td>0.1649</td></tr><tr><td></td><td>DDA</td><td>0.2054</td><td>0.5543</td><td>0.3668</td><td>0.1156</td><td>0.2671</td><td>0.5476</td><td>0.5165</td><td>0.1129</td><td>0.2416</td><td>0.4503</td><td>0.4470</td><td>0.1548</td></tr><tr><td></td><td>COSALPURE</td><td>0.2409</td><td>0.5753</td><td>0.3976</td><td>0.1119</td><td>0.3057</td><td>0.5884</td><td>0.5512</td><td>0.1040</td><td>0.2633</td><td>0.4681</td><td>0.4745</td><td>0.1604</td></tr></table>

Table 2. Co-saliency detection success rates (SR) of entire group of images, only adversarial images and only clean images. This table is to intuitively illustrate the impact of different methods on the adversarial and clean portions of group images.
<table><tr><td rowspan="2"></td><td colspan="3">GICD</td><td colspan="3">GCAGC</td><td colspan="3">PoolNet</td></tr><tr><td></td><td>avg ↑</td><td>adv ↑ clean ↑</td><td></td><td>avg ↑ adv ↑</td><td>clean ↑</td><td>avg ↑</td><td>adv ↑</td><td>clean ↑</td></tr><tr><td></td><td>Source-Only</td><td>0.3493</td><td>0.1053</td><td>0.5884</td><td>0.5285</td><td>0.3741</td><td>0.6797</td><td>0.5677</td><td>0.3671</td><td>0.7642</td></tr><tr><td></td><td>DiffPure</td><td>0.4595</td><td>0.3560</td><td>0.5609</td><td>0.4987</td><td>0.4533</td><td>0.5432</td><td>0.6327</td><td>0.5636</td><td>0.7003</td></tr><tr><td>Cos2015</td><td>DDA</td><td>0.4565</td><td>0.3079</td><td>0.6021</td><td>0.5955</td><td>0.4924</td><td>0.6964</td><td>0.6233</td><td>0.5185</td><td>0.7259</td></tr><tr><td></td><td>COSALPURE</td><td>0.5602</td><td>0.5416</td><td>0.5785</td><td>0.5975</td><td>0.5977</td><td>0.5972</td><td>0.6908</td><td>0.6569</td><td>0.7239</td></tr><tr><td rowspan="4">Coses</td><td>Source-Only</td><td>0.4012</td><td>0.1516</td><td>0.6336</td><td>0.6469</td><td>0.6161</td><td>0.6756</td><td>0.5847</td><td>0.3451</td><td>0.8078</td></tr><tr><td>DiffPure</td><td>0.4447</td><td>0.3161</td><td>0.5645</td><td>0.6609</td><td>0.6258</td><td>0.6936</td><td>0.6796</td><td>0.5741</td><td>0.7777</td></tr><tr><td>DDA</td><td>0.4665</td><td>0.3483</td><td>0.5765</td><td>0.6982</td><td>0.6903</td><td>0.7057</td><td>0.6578</td><td>0.5419</td><td>0.7657</td></tr><tr><td>COSALPURE</td><td>0.5396</td><td>0.5129</td><td>0.5645</td><td>0.7060</td><td>0.7064</td><td>0.7057</td><td>0.7278</td><td>0.6645</td><td>0.7867</td></tr><tr><td>COSOD3K</td><td>Source-Only</td><td>0.3281</td><td>0.1118</td><td>0.5364</td><td>0.4445</td><td>0.2901</td><td>0.5932</td><td>0.4466</td><td>0.2354</td><td>0.6500</td></tr><tr><td rowspan="3"></td><td>DiffPure</td><td>0.3887</td><td>0.2987</td><td>0.4754</td><td>0.4996</td><td>0.4333</td><td>0.5636</td><td>0.5247</td><td>0.4462</td><td>0.6003</td></tr><tr><td>DDA</td><td>0.3838</td><td>0.2606</td><td>0.5026</td><td>0.5337</td><td>0.4394</td><td>0.6246</td><td>0.5105</td><td>0.3945</td><td>0.6222</td></tr><tr><td>COSALPURE</td><td>0.4659</td><td>0.4597</td><td>0.4718</td><td>0.5946</td><td>0.5955</td><td>0.5938</td><td>0.5859</td><td>0.5703</td><td>0.6009</td></tr><tr><td rowspan="4">COCA</td><td>Source-Only</td><td>0.1837</td><td>0.0877</td><td>0.2739</td><td>0.2339</td><td>0.1818</td><td>0.2829</td><td>0.2239</td><td>0.1371</td><td>0.3053</td></tr><tr><td>DiffPure</td><td>0.1706</td><td>0.1212</td><td>0.2170</td><td>0.2231</td><td>0.1802</td><td>0.2634</td><td>0.2185</td><td>0.1754</td><td>0.2589</td></tr><tr><td>DDA</td><td>0.2054</td><td>0.1499</td><td>0.2574</td><td>0.2671</td><td>0.2264</td><td>0.3053</td><td>0.2416</td><td>0.1897</td><td>0.2904</td></tr><tr><td>COSALPURE</td><td>0.2409</td><td>0.2360</td><td>0.2455</td><td>0.3057</td><td>0.2998</td><td>0.3113</td><td>0.2633</td><td>0.2264</td><td>0.2979</td></tr></table>

Evaluation settings. We choose GICD [35] and GCAGC [34] to evaluate our method as they are commonly used state-of-the-art CoSOD methods. Additionally, we take PoolNet [20] into consideration, assessing the performance in salient object detection.

Baseline methods. Indeed, there is currently no specific image processing method designed for CoSOD attacks. Hence, we employ two alternative approaches as baselines. DiffPure [22] is a method that utilizes a diffusion model for purifying perturbation-based adversarial images. Diffusion-Driven Adaptation (DDA) [11] builds upon a diffusion-based model by introducing a novel selfensembling scheme, enhancing the adaptation process by dynamically determining the degree of adaptation. DiffPure and DDA employ the same sampling noise scale as our proposed COSALPURE.

Metrics. We employ four metrics to evaluate the cosalient object detection result, including detection success rate (SR), average precision (AP) [33], F-measure score $F _ { \beta }$ with $\beta ^ { 2 } = 0 . 3 [ 1 ]$ and mean absolute error (MAE) [33]. For the detection success rate, we calculate the intersection over union (IOU) between each co-salient object detection result of the reconstructed image and the corresponding groundtruth map. We divide the number of successful results (IOU > 0.5) by the total number of results to calculate SR. In addition, to intuitively illustrate the impact of different methods on CoSOD results, we not only compute SR for the entire group of images but also separately calculate SR for only adversarial images and only clean images.

Implementation details. In the group-image concept learning procedure, the sampled images are simply resized from 224 × 224 resolution to $7 6 8 \times 7 6 8$ resolution before being passed into the image encoder. For the continuous representation [4, 14] module employed in the concept-guided diffusion purification procedure, we constructed a dataset to train it. We select 50,000 samples with a resolution of 224 × 224 from ImageNet (1,000 categories, each with 50 samples) and apply noise with the intensity of 16/255 via the PGD attack to these samples to construct the inputs of the continuous representation module. For the ground truth images, we apply clean images with a resolution of 768 × 768 corresponding to the input images. We follow the experimental setup of [14] and trained for 10 epochs to obtain the required module. The group-image concept learning procedure and the concept-guided purification procedure utilize the same pre-trained image encoder and conditional diffusion model. For the concept-guided purification procedure, we set the number of timesteps T to 250, and the same configuration is applied to baseline methods.

![](images/24f0a32e5eb25909fcb2be95b24dfcf79804a33b13d320b78c901689488889db.jpg)  
Figure 6. Visualization of co-salient object detection results. We show four visualized cases in this figure, with the source-only/purified image, the groundtruth of the co-saliency map, and the results of GICD, GCAGC, and PoolNet in the columns. Our method, COSALPURE, is highlighted in green.

## 5.2. Comparison on Adversarial Attacks

We denote the images before reconstruction (containing 50% adversarial images and 50% clean images) as "Source-Only". The comparison between our proposed COSALPURE and baselines are shown in Table 1. We consider an image to be successfully detected in the co-salient object detection (CoSOD) task if the IOU of its CoSOD result and the ground-truth map exceed 0.5. Compared to DiffPure [22] and DDA [11], COSALPURE outperforms them in terms of co-salient object detection success rates (SR) across all four datasets. For the other three metrics (i.e., AP [33], F [1] and MAE [33]), COSALPURE remains the best at most of the time. We show four visualized cases in Fig. 6. For each case, we present the generated images of COSALPURE and two baselines, DiffPure and DDA. Obviously COSALPURE generates higher-quality images, as it leverages the intrinsic commonality of objects across the group of images. Additionally, we showcase the comparison of the detection results on GICD, GCAGC, and Pool-Net. The results from COSALPURE closely approximate the ground-truth map, while the baseline methods struggle to display the correct results.

To intuitively illustrate the impact of different methods on the adversarial and clean portions of group images, we measure the co-salient object detection success rate from three perspectives. In Table 2, "avg" represents the evaluation across the entire group of images, "adv" and "clean" correspond to evaluations on only the 50% images that are under the SOTA attack [12] and on only the 50% images that remain clean. COSALPURE at some times have a lower “clean” SR compared to DDA or source-only. However, DiffPure and DDA are unsatisfactory in “adv” SR, while COSALPURE exhibits a significant lead in “adv” SR, resulting in it consistently performing the best in “avg” SR.

## 5.3. Ablation Study

To validate the effect of the learned concepts on CoSOD results, we conduct ablation studies on Cosal2015 [32] and CoSOD3k [8]. In Table 3, “w/o concept inversion” represents only utilizing the continuous representation module and not applying the subsequent purification process. “w/ None concept” denotes passing a meaningless “None” as the concept during the purification. “w/ learned concept” denotes the complete pipeline, firstly learning the concept from the entire group of images and secondly passing the learned concept during the purification procedure to accomplish image reconstruction. From Table 3, it is evident that the learned concept contributes to significant improvements in various metrics. As shown in Fig. 7, when we do not apply the concept-guided purification or when we pass in a meaningless concept, the generated image performs poorly in the CoSOD task. This improves when we pass in the learned effective concept which includes object semantics. The example proves that the learned concept contributes to the reconstruction of images used for the CoSOD task.

Table 3. Ablation study. “w/o concept inversion” represents only utilize the continuous representation module and not apply the subsequent purification process. “w/ None concept” denotes passing a meaningless “None” as the concept during the purification procedure. “w/ learned concept” denotes the complete pipeline, firstly learning the concept from the entire group of images and secondly passing in the learned concept during the purification procedure.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="4">GICD</td><td colspan="4">GCAGC</td><td colspan="4">PoolNet</td></tr><tr><td>SR↑</td><td>AP↑</td><td> $F _ { \beta } \uparrow$ </td><td>MAE↓</td><td> ${ \mathrm { S R } } \uparrow$ </td><td> $\mathbf { A P \uparrow }$ </td><td> $F _ { \beta } \uparrow$ </td><td>MAE↓</td><td>SR↑</td><td>AP↑</td><td> $F _ { \beta } \uparrow$ </td><td>MAE↓</td></tr><tr><td></td><td>Source-Only</td><td>0.3493</td><td>0.7306</td><td>0.4038</td><td>0.1676</td><td>0.5285</td><td>0.7853</td><td>0.6302</td><td>0.1570</td><td>0.5677</td><td>0.7425</td><td>0.6095</td><td>0.1276</td></tr><tr><td>Cos2015</td><td>COSALPURE w/o concept inversion</td><td>0.5186</td><td>0.7791</td><td>0.5809</td><td>0.1350</td><td>0.5528</td><td>0.7322</td><td>0.6430</td><td>0.2145</td><td>0.6843</td><td>0.8241</td><td>0.7225</td><td>0.1098</td></tr><tr><td></td><td>COSALPURE w/ None concept</td><td>0.5225</td><td>0.7784</td><td>0.5886</td><td>0.1354</td><td>0.5334</td><td>0.7091</td><td>0.6132</td><td>0.2263</td><td>0.6774</td><td>0.8172</td><td>0.7160</td><td>0.1110</td></tr><tr><td></td><td>COSALPURE w/ learned concept</td><td>0.5602</td><td>0.7898</td><td>0.6177</td><td>0.1296</td><td>0.5975</td><td>0.7449</td><td>0.6521</td><td>0.2063</td><td>0.6908</td><td>0.8268</td><td>0.7258</td><td>0.1086</td></tr><tr><td></td><td>Source-Only</td><td>0.3281</td><td>0.6988</td><td>0.4003</td><td>0.1439</td><td>0.4445</td><td>0.7325</td><td>0.5702</td><td>0.1376</td><td>0.4466</td><td>0.6606</td><td>0.5255</td><td>0.1386</td></tr><tr><td></td><td>COSALPURE w/o concept inversion</td><td>0.4424</td><td>0.7314</td><td>0.5317</td><td>0.1243</td><td>0.5753</td><td>0.7923</td><td>0.6804</td><td>0.1170</td><td>0.5747</td><td>0.7424</td><td>0.6508</td><td>0.1223</td></tr><tr><td>COoSOOD3K</td><td>COSALPURE w/ None concept</td><td>0.4297</td><td>0.7216</td><td>0.5158</td><td>0.1273</td><td>0.5488</td><td>0.7715</td><td>0.6584</td><td>0.1288</td><td>0.5711</td><td>0.7373</td><td>0.6453</td><td>0.1231</td></tr><tr><td></td><td>COSALPURE w/ learned concept</td><td>0.4659</td><td>0.7327</td><td>0.5487</td><td>0.1221</td><td>0.5946</td><td>0.7999</td><td>0.6881</td><td>0.1144</td><td>0.5859</td><td>0.7432</td><td>0.6605</td><td>0.1215</td></tr></table>

![](images/09b8825157e759b75e0bdfde1c621791e4bd66c1d8f43a2a397dd3a80124e663.jpg)  
Figure 7. Visualization for ablation study.

Table 4. Extension to motion blur.
<table><tr><td></td><td>SR↑</td><td>AP↑</td><td> $\overline { { F _ { \beta } \mathrm { \Delta } \mathrm { \uparrow } } }$ </td><td>MAE↓</td></tr><tr><td>Source-Only</td><td>0.3915</td><td>0.7408</td><td>0.4373</td><td>0.1590</td></tr><tr><td>DiffPure</td><td>0.3146</td><td>0.6774</td><td>0.3763</td><td>0.1738</td></tr><tr><td>DDA</td><td>0.3900</td><td>0.7381</td><td>0.4425</td><td>0.1579</td></tr><tr><td>COSALPURE</td><td>0.4575</td><td>0.7419</td><td>0.5241</td><td>0.1505</td></tr></table>

## 5.4. Extention to Common Corruption

In addition to adversarial attacks on CoSOD, we also broaden our experiments to include a common corruption type: motion blur. We select the Cosal2015 dataset [32] and, similar to the adversarial experiments, apply motion blur [13] to the first 50% of images in each group while keeping the remaining 50% of images clean. Here we set the number of timesteps $T$ to 500 and do not employ the continuous representation module. As shown in Table 4, COSALPURE performs better than other diffusionbased image processing methods at all four metrics.

## 6. Conclusions

This paper presented COSALPURE, an innovative framework enhancing the robustness of co-salient object detection (CoSOD) against adversarial attacks and common image corruptions. Central to our approach are two key innovations: group-image concept learning and conceptguided diffusion purification. Our framework effectively captures and utilizes the high-level semantic concept of cosalient objects from group images, demonstrating notable resilience even in the presence of adversarial examples.

Empirical evaluations across datasets like Cosal2015, iCoseg, CoSOD3k, and CoCA showed that COSALPURE significantly outperforms existing methods such as DiffPure and DDA in CoSOD tasks. Not only did it achieve higher success rates, but it also excelled in performance metrics like AP, F-measure, and MAE. Additionally, its effectiveness against common image corruptions, like motion blur, underscores its versatility.

Our COSALPURE represents a substantial advancement in CoSOD, offering robust, concept-driven image purification. It opens avenues for more resilient co-salient object detection, vital in today’s landscape of sophisticated image manipulation and corruption. Future work might extend this framework to broader image analysis applications and explore its adaptability to real-world scenarios.

## 7. Acknowledgments

Geguang Pu is supported by National Key Research and Development Program (2020AAA0107800), and Shanghai Collaborative Innovation Center of Trusted Industry Internet Software. This work is also supported by the National Research Foundation, Singapore, and DSO National Laboratories under the AI Singapore Programme (AISG Award No: AISG2-GC-2023-008), and Career Development Fund (CDF) of the Agency for Science, Technology and Research (A\*STAR) (No.: C233312028).

## References

[1] Radhakrishna Achanta, Sheila Hemami, Francisco Estrada, and Sabine Susstrunk. Frequency-tuned salient region detection. In 2009 IEEE conference on computer vision and pattern recognition, pages 1597–1604. IEEE, 2009.

[2] Dhruv Batra, Adarsh Kowdle, Devi Parikh, Jiebo Luo, and Tsuhan Chen. icoseg: Interactive co-segmentation with intelligent scribble guidance. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3169–3176. IEEE, 2010.

[3] Xiaochun Cao, Zhiqiang Tao, Bao Zhang, Huazhu Fu, and Wei Feng. Self-adaptively weighted co-saliency detection via rank constraint. IEEE Transactions on Image Processing, 23(9):4175–4186, 2014.

[4] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8628–8638, 2021.

[5] Ming-Ming Cheng, Niloy J Mitra, Xiaolei Huang, and Shi-Min Hu. Salientshape: group saliency in image collections. The visual computer, 30:443–453, 2014.

[6] Ming-Ming Cheng, Niloy J Mitra, Xiaolei Huang, Philip HS Torr, and Shi-Min Hu. Global contrast based salient region detection. IEEE transactions on pattern analysis and machine intelligence, 37(3):569–582, 2014.

[7] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

[8] Deng-Ping Fan, Zheng Lin, Ge-Peng Ji, Dingwen Zhang, Huazhu Fu, and Ming-Ming Cheng. Taking a deeper look at co-salient object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 2919–2929, 2020.

[9] Deng-Ping Fan, Tengpeng Li, Zheng Lin, Ge-Peng Ji, Dingwen Zhang, Ming-Ming Cheng, Huazhu Fu, and Jianbing Shen. Re-thinking co-salient object detection. IEEE transactions on pattern analysis and machine intelligence, 44(8): 4339–4354, 2021.

[10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

[11] Jin Gao, Jialing Zhang, Xihui Liu, Trevor Darrell, Evan Shelhamer, and Dequan Wang. Back to the source: Diffusion-driven test-time adaptation. arXiv preprint arXiv:2207.03442, 2022.

[12] Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, and Song Wang. Can you spot the chameleon? adversarially camouflaging images from co-salient object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2150–2159, 2022.

[13] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturba-

tions. Proceedings of the International Conference on Learning Representations, 2019.

[14] Chih-Hui Ho and Nuno Vasconcelos. Disco: Adversarial defense with local implicit functions. Advances in Neural Information Processing Systems, 35:23818–23837, 2022.

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[16] Bo Jiang, Xingyue Jiang, Ajian Zhou, Jin Tang, and Bin Luo. A unified multiple graph learning and convolutional network model for co-saliency estimation. In Proceedings ofthe 27th ACM international conference on multimedia, pages 1375– 1382, 2019.

[17] Bo Li, Zhengxing Sun, Lv Tang, Yunhan Sun, and Jinlong Shi. Detecting robust co-saliency with recurrent co-attention neural network. In IJCAI, page 6, 2019.

[18] Guanbin Li and Yizhou Yu. Deep contrast learning for salient object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 478–487, 2016.

[19] Hongliang Li, Fanman Meng, Bing Luo, and Shuyuan Zhu. Repairing bad co-segmentation using its quality evaluation and segment propagation. IEEE Transactions on Image Processing, 23(8):3545–3559, 2014.

[20] Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Jiashi Feng, and Jianmin Jiang. A simple pooling-based design for real-time salient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3917–3926, 2019.

[21] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

[22] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. arXiv preprint arXiv:2205.07460, 2022.

[23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[24] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023.

[25] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.

[26] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the daam: Interpreting stable diffu-

sion using cross attention. arXiv preprint arXiv:2210.04885, 2022.

[27] Linzhao Wang, Lijun Wang, Huchuan Lu, Pingping Zhang, and Xiang Ruan. Saliency detection with recurrent fully convolutional networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 825–841. Springer, 2016.

[28] Wenguan Wang, Qiuxia Lai, Huazhu Fu, Jianbing Shen, Haibin Ling, and Ruigang Yang. Salient object detection in the deep learning era: An in-depth survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6): 3239–3259, 2021.

[29] Zheng-Jun Zha, Chong Wang, Dong Liu, Hongtao Xie, and Yongdong Zhang. Robust deep co-saliency detection with group semantic and pyramid attention. IEEE transactions on neural networks and learning systems, 31(7):2398–2408, 2020.

[30] Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. Text-to-image diffusion model in generative ai: A survey. arXiv preprint arXiv:2303.07909, 2023.

[31] Dingwen Zhang, Deyu Meng, Chao Li, Lu Jiang, Qian Zhao, and Junwei Han. A self-paced multiple-instance learning framework for co-saliency detection. In Proceedings of the IEEE international conference on computer vision, pages 594–602, 2015.

[32] Dingwen Zhang, Junwei Han, Chao Li, Jingdong Wang, and Xuelong Li. Detection of co-salient objects by looking deep and wide. International Journal of Computer Vision, 120: 215–232, 2016.

[33] Dingwen Zhang, Huazhu Fu, Junwei Han, Ali Borji, and Xuelong Li. A review of co-saliency detection algorithms: Fundamentals, applications, and challenges. ACM Transactions on Intelligent Systems and Technology (TIST), 9(4):1– 31, 2018.

[34] Kaihua Zhang, Tengpeng Li, Shiwen Shen, Bo Liu, Jin Chen, and Qingshan Liu. Adaptive graph convolutional network with attention graph clustering for co-saliency detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9050–9059, 2020.

[35] Zhao Zhang, Wenda Jin, Jun Xu, and Ming-Ming Cheng. Gradient-induced co-saliency detection. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pages 455–472. Springer, 2020.

[36] Jia-Xing Zhao, Jiang-Jiang Liu, Deng-Ping Fan, Yang Cao, Jufeng Yang, and Ming-Ming Cheng. Egnet: Edge guidance network for salient object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8779–8788, 2019.