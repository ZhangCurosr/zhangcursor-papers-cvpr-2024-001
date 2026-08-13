# Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction

Junuk Cha1Jihyeon Kim1,2† Jae Shin Yoon³\* Seungryul Baek1\* 1UNIST 2KETI 3Adobe Research

![](images/874e6fea41c8063f1117225191311f00f3cf0446288315fe87ee143bf2c889f4.jpg)

![](images/eb6b543d5fcf55b49fac1f855900c9fb9ae85c11b331d48843c97ae0e9e359ae.jpg)  
Figure 1. Given a text and a canonical object mesh as prompts, we generate 3D motion for hand-object interaction without requiring object trajectory and initial hand pose. We represent the right hand with a light skin color and the left hand with a dark skin color. The articulation of a box in the first row is controlled by estimating an angle for the pre-defined axis of the box.

## Abstract

This paper introduces the first text-guided work for generating the sequence of hand-object interaction in 3D. The main challenge arises from the lack of labeled data where existing ground-truth datasets are nowhere near generalizable in interaction type and object category, which inhibits the modeling of diverse 3D hand-object interaction with the correct physical implication (e.g., contacts and semantics) from text prompts. To address this challenge, we propose to decompose the interaction generation task into two subtasks: hand-object contact generation; and hand-object motion generation. For contact generation, a VAE-based network takes as input a text and an object mesh, and generates the probability of contacts between the surfaces of hands and the object during the interaction. The network learns a variety of local geometry structure of diverse objects that is independent of the objects’ category, and thus, it is applicable to general objects. For motion generation, a Transformer-based diffusion model utilizes this 3D contact map as a strong prior for generating physically

plausible hand-object motion as a function of text prompts by learning from the augmented labeled dataset; where we annotate text labels from many existing 3D hand and object motion data. Finally, we further introduce a hand refiner module that minimizes the distance between the object surface and hand joints to improve the temporal stability of the objecthand contacts and to suppress the penetration artifacts. In the experiments, we demonstrate that our method can generate more realistic and diverse interactions compared to other baseline methods. We also show that our method is applicable to unseen objects. We will release our model and newly labeled data as a strong foundation for future research. Codes and data are available in: https://github.com/JunukCha/Text2HOI.

## 1. Introduction

Imagine handing over an apple on a table to your friends: you might first grab it and convey this to them. During a social interaction, the hand pose and motion are often defined as a function of object's pose, shape, and category. While existing works [3, 8, 9, 15, 21, 27, 30, 31] have been successful in modeling diverse and realistic 3D human body motions from a text prompt (where there exists no text-guided hand motion generation works), the context of object interaction has been often missing, which significantly limits the expressiveness in the semantics of the generated motion sequence. In this paper, we propose a first work that can generate realistic and assorted hand-object motions in 3D from a text prompt as shown in Fig. 1. Our work can be used for various applications such as generating surgical simulations, interactive control of a character for gaming, and future path planning between a robot hand and objects for robotics.

Learning to generate a sequence of 3D hand-object interaction from a text prompt is extremely challenging due to the scarcity of the dataset: the diversity of existing datasets for a sequence of 3D meshes and associated text labels is far behind the one of real-world distribution which is determined by a number of parameters such as hand type (e.g., left or right), object's category and structure, scale, contact regions, and so on. A generative model learned from such limited data will fail in the diverse modeling of physically and semantically plausible 3D hand-object interaction.

To overcome this challenge, we propose to decompose the interaction generation task into two subtasks, "object contact map generation" and “hand-object motion generation", where the models dedicated to each task learn a general geometry representation from the augmented dataset, which leads to the significant improvement in the generalizability and physical plausibility of the combined pipeline.

For contact map generation, we newly develop a contact map prediction network that encodes a local geometry surface of a 3D object mesh along with a target motion text; and generates a 3D contact map—3D probability map at the object's surface that describes the potential regions contacted by hand meshes during the interaction—along with the general geometric features. Since the local geometry representation is category-agnostic, the network is applicable to general objects. By adding condition of the scale information, our contact map generation module is, in nature, able to decode scale-variant probability, e.g., if the object's scale is smaller, the region of the predicted contact probability becomes wider, reflecting the natural tendency to grasp smaller objects over a wider area.

For motion generation, a Transformer-based diffusion model utilizes the contact map and geometric features as strong guidance to generate the sequence of 3D hand and object movements from a text prompt. Unlike a conventional diffusion process [11], the model is designed to directly estimate the final sample at each step, which allows us to apply explicit geometric loss (e.g., relative distance or orientation) to improve the geometric correctness. Our diffusion model learns the augmented data where we perform extensive manual annotation of the text labels from external motion datasets [5, 16, 25].

Using these two modules, we introduce the first text-guided hand-object interaction generation framework that generates the 3D interaction in a compositional way. Given a text prompt, canonical 3D object mesh, and object's scale, our VAE-based contact predictor generates a 3D contact map, and geometry features. Our Transformer-based diffusion model encodes the contact, text, and geometry information with frame-wise and agent-wise (i.e., object, and left and right hand) positional embedding to decode realistic 3D hand-object interaction. Finally, our new Transformer-based refiner module pushes the physical correctness of the 3D interaction in a single feed-forward manner by refining the contacts and suppressing the penetration artifacts.

In the experiments, we validate our model on three datasets (H2O [16], GRAB [25], and ARCTIC [5]) where our method outperforms other baseline methods in terms of accuracy, diversity, and physical realism by large margins. We also demonstrate that our compositional framework enables the application of our method to new objects that are not seen during training.

Our contributions can be summarized as follows:

• To the best of our knowledge, we propose the first approach that can generate a sequence of 3D hand-object interaction in various styles and lengths from a text prompt.

• We propose a novel compositional framework that enables the modeling of high-quality hand-object interaction from limited data.

• We introduce a new fast and efficient hand refinement module that improves physical realism (e.g., penetration-free interaction) without any test-time optimization.

• We annotate text labels from existing hands and object motion datasets, which will be made public.

## 2. Related Work

Text to human motion generation. Thanks to the userfriendly nature of textual inputs, there has been substantial progress in the field of text-guided human motion generation [3, 8, 9, 12, 15, 17, 19, 21, 27, 30, 31, 33]. Guo et al. [8] proposed the text2length and text2motion modules to generate human motion in varying time length, while remaining realistic and faithful to the text. Tevet et al. [27] introduced a Motion Diffusion Model (MDM) for generating natural and expressive human motion, utilizing the geometric losses and Transformer-based approach that predicts the sample instead of noise in each diffusion step. Recently, Liang et al. [19] presented a method that can generate interactive motion between two people. But it cannot handle three or more multi-agents.

Hand and object motion generation. Existing approaches [1, 2, 4, 7, 10, 13, 14, 18, 20, 34] focus on grasping the stationary object. They are limited in their ability to manipulate the object and are therefore inadequate to generate a natural hand-object motion. Ghosh et al. [6] proposed a method for generating full-body motion in interaction with 3D objects, which is guided by action labels, while it requires an optimization stage for full performance. To generate hand and manipulated object motion, Zhang et al. [29] proposed a network that relies on the current hand pose, past and future trajectories of both hands and object, and diverse spatial representations. Zheng et al. [32] generate the hand-object motion covering both rigid and articulated objects, given an initial hand pose, object geometry, and sparse sequences of object poses. While plausible, these methods [29, 32] require the 3D object sequence as inputs, which is often not available from a user. In addition, they cannot utilize text modality.

![](images/c718a1c124fb082a5ecdf864e9333dd4ac144ea3a653f07cfd165877c530fadd.jpg)  
Figure 2. Schematic diagram of the overall framework. Given a text prompt and a canonical object mesh prompt, our aim is to generate the 3D motion for hand-object interaction. We first generate a contact map from the canonical object mesh conditioned by the text prompt and object's scale. The hand-object motion generation module removes the noise from the inputs for the denoised outputs to align with the predicted contact map and the text prompt. The denoised outputs exhibit artifacts, including the penetration. To address these artifacts, the hand refinement module adjusts the generated (denoised) hand pose parameters to restrain the penetration and to improve contact interactions.

## 3. Method

Our goal is to generate hand-object interacting motions given a text prompt T and a canonical object mesh $\mathbf { M } _ { \mathrm { o b j } }$ . To address them, we design our framework with three stages, as shown in Fig. 2. First, we use the canonical object mesh $\mathbf { M } _ { \mathrm { { o b j } } }$ combined with the text feature $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ via the CLIP text encoder $f ^ { \mathrm { C L I P } }$ [23] to estimate the contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } }$ that provides a strong prior for relative 3D locations of hands and an object. Then, we use the Transformer-based diffusion model to denoise the noised input data $\{ \mathbf { x } _ { t } ^ { l } \} _ { l = 1 } ^ { L }$ at the t-th diffusion timestep, where L is the overall sequence length. By conditioning the text features $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ , contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } ,$ object features $\mathbf { F } _ { \mathrm { o b j } }$ and scale $s _ { \mathrm { o b j } }$ on the diffusion model, we estimate the denoised sample $\hat { \mathbf { x } } _ { 0 }$ from the noised one $\mathbf { x } _ { t } .$ Lastly, hand refiner improves the initial generated hand-object motions considering penetration and contact between hands and an object.

## 3.1. Contact map prediction

To generate natural motions for hand-object interaction, it is crucial to understand contact points between hands and an object. For this, we design the contact prediction network $f ^ { \mathrm { c o n t a c t } }$ that encodes contact points on the surfaces of the object mesh $\mathbf { M } _ { \mathrm { { o b j } } }$ along with a text prompt T and object's scale $s _ { \mathrm { o b j } }$

We first compute $s _ { \mathrm { o b j } }$ which represents the maximum distance from center of object mesh to its vertices. We then sample N-point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ from the vertices of canonical object mesh using the farthest point sampling (FPS) algorithm [22]. Subsequently, we normalize P to $\mathbf { P } _ { \mathrm { n o r m } }$ by dividing it with $s _ { \mathrm { o b j } } .$ The contact prediction network $f ^ { \mathrm { c o n t a c t } }$ receives the normalized point cloud $\mathbf { P } _ { \mathrm { n o r m } } .$ text features $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ , object's scale $s _ { \mathrm { o b j } } .$ and Gaussian random noise vector $\mathbf { z } _ { \mathrm { c o n t a c t } } \in \mathbb { R } ^ { 6 4 }$ , and produces the contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } \in \mathbb { R } ^ { N \times 1 }$ . In the middle of $f ^ { \mathrm { c o n t a c t } }$ , we obtain the object features $\mathbf { F } _ { \mathrm { o b j } } \in \mathbb { R } ^ { 1 , 0 2 4 }$ . To train $f ^ { \mathrm { c o n t a c t } }$ , we use the combination of binary cross-entropy loss, dice loss and kullback-leibler (KL) divergence loss following [18].

## 3.2. Text-to-3D hand-object motion generation

Our text-to-3D hand-object interaction generator (Text2HOI) $f ^ { \mathrm { T H O I } }$ , whose architecture is the Transformer encoder [28], is trained via the diffusion-based approach [11].

## 3.2.1 Preliminaries.

The 3D hand-object motion is represented as $\begin{array} { r l } { \mathbf { x } _ { 0 } } & { { } = } \end{array}$ $\{ \mathbf { x } _ { 0 , \mathrm { l h a n d } } ^ { l } , \mathbf { x } _ { 0 , \mathrm { r h a n d } } ^ { l } , \mathbf { x } _ { 0 , \mathrm { o b j } } ^ { l } \} _ { l = 1 } ^ { L _ { \mathrm { m a x } } }$ , where l denotes the frame index. This motion comprises $3 { \cdot } L _ { \mathrm { m a x } }$ elements, which accounts for the maximum motion length $L _ { \mathrm { m a x } }$ of three agents $( i . e .$ , left and right hands and an object): For left and right hands, $\mathbf { x } _ { 0 , \mathrm { l h a n d } } ^ { l } \in \bar { \mathbb { R } } ^ { 9 9 }$ and $\mathbf { x } _ { 0 , \mathrm { r h a n d } } ^ { l } \in \mathbb { R } ^ { 9 9 }$ are composed of 99-dimensional vectors by flattening and concatenating the 3D hand translation parameters $\mathbf t _ { h } ^ { l } \in \mathbb { R } ^ { 3 }$ and MANO hand pose parameters $\theta ^ { l } \in \mathbb { R } ^ { 1 6 \times \bar { 6 } }$ in 6D representation [35]. For an object, $\mathbf { x } _ { \mathrm { 0 , o b j } } ^ { l } \in \mathbb { R } ^ { 1 0 }$ is 10-dimensional vector that concatenates the 3D object translation $\mathbf { t } _ { o } ^ { l } \in \mathbb { R } ^ { 3 }$ , object rotation $\mathbf { r } ^ { l } \in \mathbb { R } ^ { 6 }$ [35], and object articulation angle $\alpha ^ { l } \in \mathbb { R } ^ { 1 }$

The 3D hand-object interaction $\mathbf { x } _ { \mathrm { 0 } }$ is used to generate the mesh of hands, and to deform the mesh of objects: The left and right hand meshes are generated from ${ \bf x } _ { \mathrm { 0 , l h a n d } }$ and $\mathbf { x } _ { \mathrm { 0 , r h a n d } }$ by feeding them to the MANO layer [24] to output the hand vertices $\mathbf { \bar { V } } _ { \mathrm { l h a n d } } , \mathbf { V } _ { \mathrm { r h a n d } } \in \mathbb { R } ^ { L \times V \times \mathbf { \bar { 3 } } }$ , and hand joints Jlhand, $\mathbf { J } _ { \mathrm { { r h a n d } } } \in \mathbb { R } ^ { L \times J \times 3 }$ in 3D global space, where $V = 7 7 8$ and $J = 2 1$ . A deformed object's point cloud $\mathbf { P } _ { \mathrm { d e f } } \in \mathbb { R } ^ { L \times N \times 3 }$ is generated in 3D global space by transforming the object's point clouds $\mathbf { P }$ with the translation, rotation and articulation angles in $\mathbf { x } _ { 0 , \mathrm { o b j } } .$ The notation^ and \~ indicate that these values are derived from the estimated $\hat { \mathbf { x } } _ { 0 }$ and refined $\tilde { \mathbf { x } } _ { 0 } .$ respectively.

![](images/8975b2bee71dd1f2f7417c4bfa73db43c7eec1636f78feb0b0882298d483d192.jpg)  
Figure 3. The details of the text-to-3D hand-object motion generation in our framework. In the forward process, we generate the noised motion $\{ \mathbf { x } _ { t } ^ { l } \} _ { l = 1 } ^ { \hat { L } }$ by adding the noise to the original (ground-truth) motion $\{ \mathbf { x } _ { 0 } ^ { l } \} _ { l = 1 } ^ { \hat { L } }$ . In the backward process, the Transformer encoder denoises the noised motion $\{ \mathbf { x } _ { t } ^ { l } \} _ { l = 1 } ^ { \hat { L } } .$ , using various conditions c including text features $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ , contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } ,$ object features $\mathbf { F } _ { \mathrm { o b j } } .$ and object's scale $s _ { \mathrm { o b j } }$ The right panel illustrates a comparison between conventional positional encoding, which can only differentiate each patch, and our proposed encoding, which provides detailed differentiation of both frames and agents. A unique positional encoding value is assigned for each box, distinguished by different colors.

## 3.2.2 Forward process.

Our forward process is formulated as:

$$
\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t }\tag{1}
$$

following [11], where t is the diffusion time-step, $\mathbf { x } _ { \mathrm { 0 } }$ is the original 3D hand-object motion, $\mathbf { x } _ { t }$ is the noised 3D hand-object motion at the t-th diffusion time-step, and $\bar { \alpha } _ { t } \in ( 0 , 1 )$ is a set of constant hyper-parameters. The noise $\epsilon _ { t }$ is randomly sampled from the Gaussian distribution at each diffusion-time step t.

## 3.2.3 Backward process.

In the backward process, the text-to-3D hand-object interaction generator (Text2HOI) fTHoI denoises the noised motion $\mathbf { x } _ { t }$ to reconstruct the original (ground-truth) motion x0: $\hat { \mathbf { x } } _ { 0 } =$ $f ^ { \mathrm { T H O I } } ( \mathbf { x } _ { t } , t , c )$ , where c denotes the conditions, as described in [27]. Since we exploit the Transformer encoder as the architecture, the noised signal $\mathbf { x } _ { t }$ needs to be first converted to the proper input embedding $\mathbf { X } _ { t }$ . Similarly, the output of Transformer architecture $\hat { \mathbf { X } } _ { t }$ also needs to be converted to the denoised signal $\hat { \mathbf { x } } _ { 0 }$ Furthermore, the text features $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ , object features $\mathbf { F } _ { \mathrm { o b j } } .$ estimated contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } }$ and object's scale $s _ { \mathrm { o b j } }$ are merged together to constitute the conditional signals $\mathbf { X } _ { \mathrm { c o n d } }$ , which will be detailed in the remainder of the section:

Transformer input generation. The forwarded signal $\mathbf { x } _ { t } ^ { l } = \{ \mathbf { x } _ { t , \mathrm { l h a n d } } ^ { l } , \mathbf { x } _ { t , \mathrm { r h a n d } } ^ { l } , \mathbf { x } _ { t , \mathrm { o b j } } ^ { l } \}$ is passed through corresponding fully connected layers (i.e., fin,lhand, $f ^ { \mathrm { i n , r h a n d } }$ , and $f ^ { \mathrm { i n , o b j } } )$ respectively to obtain the input to the Transformer encoder, $\mathbf { X } _ { t } ^ { \hat { l } ^ { ' } } = \{ \mathbf { X } _ { t , \mathrm { l h a n d } } ^ { \hat { l } } \in \mathbb { R } ^ { 5 1 2 } , \mathbf { X } _ { t , \mathrm { r h a n d } } ^ { \hat { l } ^ { ' } } \in \mathbb { R } ^ { 5 1 2 } , \mathbf { X } _ { t , \mathrm { o b i } } ^ { l } \in \mathbb { R } ^ { 5 1 2 } \}$ respectively. Then, we apply two types of positional encoding: frame-wise and agent-wise. Frame-wise positional encoding adds an sinusoidal value to $\mathbf { X } _ { t } ^ { l }$ which varies according to the motion length index $l ;$ while irrespective to the type of agents.

Agent-wise positional encoding adds distinct encoding values for each agent (left hand, right hand, and object), which are consistent across different frames, to $\mathbf { X } _ { t , \mathrm { l h a n d } } , \mathbf { X } _ { t , \mathrm { r h a n d } }$ and $\mathbf { X } _ { t , \mathrm { o b j } }$ . These are designed to help the Transformer encoder to better understand the input data. The detail pipeline of these positional encodings is shown in the right bottom panel of Fig. 3.

The Transformer encoder has a maximum input capacity of 451. The first input is reserved for the conditioning, while the remaining inputs accommodate the maximum motion length $L _ { \mathrm { m a x } }$ of 150 frames, involving three distinct agents: left hand, right hand and object. We mask out all inputs except for the first $1 + 3 \hat { L }$ inputs where $\hat { L }$ is the estimated length of sequence and subsequently, we mask inputs which are not belonging to the estimated hand type H\* (see Sec. 4.1 for details about how Î and H\* are estimated).

Conditional input generation. To generate denoised handobject motions conditioned on the text prompt T and canonical object mesh $\mathbf { M } _ { \mathrm { o b j } } ,$ we need to generate the conditional input for the t-th diffusion time-step. Conditional input $\mathbf { X } _ { t , \mathrm { c o n d } }$ is generated by:

$$
\begin{array} { r c l } { { \bf X } _ { t , \mathrm { c o n d } } } & { = } & { { \bf X } _ { \mathrm { c o n d } } + t _ { \mathrm { e m b } } } \end{array}\tag{2}
$$

where the diffusion time-step embedding $t _ { \mathrm { e m b } } ~ = ~ f ^ { \mathrm { t s } } ( t )$ is obtained by applying the diffusion time-step t to the time-step embedding fully-connected layer $f ^ { \mathrm { t s } }$ and the condition embedding $\mathbf { X } _ { \mathrm { c o n d } }$ is generated as follows:

$$
\begin{array} { c c l } { \mathbf { X } _ { \mathrm { c o n d } } } & { = } & { \mathbf { X } _ { \mathrm { t e x t } } ^ { \mathrm { c o n d } } \mathbf { + } \mathbf { X } _ { \mathrm { o b j } } ^ { \mathrm { c o n d } } } \end{array}\tag{3}
$$

where the text condition $\mathbf { X } _ { \mathrm { t e x t } } ^ { \mathrm { c o n d } } { = } f ^ { \mathrm { t e x t } } ( f ^ { \mathrm { C L I P } } ( \mathbf { T } ) )$ is generated by applying the text feature $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ to the fc layer $f ^ { \mathrm { t e x 1 } }$ . The object condition $\mathbf { X } _ { \mathrm { o b j } } ^ { \mathrm { c o n d } } = f ^ { \mathrm { o b j } } ( \{ \mathbf { F } _ { \mathrm { o b j } } , \hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } , S _ { \mathrm { o b j } } \} )$ is obtained by concatenating object feature $\mathbf { F } _ { \mathrm { o b j } }$ contact map mcontact and object's scale $s _ { \mathrm { o b j } } ,$ and feeding them to the fc layer $f ^ { \mathrm { o b j } }$

Transformer output conversion. Masked inputs $\mathbf { X } _ { t } ~ = ~ \{ \mathbf { X } _ { t , \mathrm { c o n d } } , \mathbf { X } _ { t } ^ { 1 } , \mathbf { X } _ { t } ^ { 2 } , \dots , \mathbf { X } _ { t } ^ { \hat { L } } \}$ are fed to the Transformer encoder to estimate the outputs $\hat { \mathbf { X } } _ { 0 } = \{ \hat { \mathbf { X } } _ { 0 } ^ { l } \} _ { l = 1 } ^ { \hat { L } } ,$ where $\hat { \mathbf { X } } _ { 0 } ^ { l } = \{ \hat { \mathbf { X } } _ { 0 , \mathrm { l h a n d } } ^ { l } , \hat { \mathbf { X } } _ { 0 , \mathrm { r h a n d } } ^ { l } , \hat { \mathbf { X } } _ { 0 , \mathrm { o b j } } ^ { l } \}$ Each outputs—X0,hand, $\hat { \mathbf { X } } _ { \mathrm { 0 , r h a n d } } ^ { l } ,$ and $\hat { \mathbf { X } } _ { 0 , \mathrm { o b j } } ^ { l } -$ are passed through its own dedicated fully connected layer, denoted as $f ^ { \mathrm { o u t , l h a n d } } , \ f ^ { \mathrm { o u t , r h a n d } } ,$ and $f ^ { \mathrm { o u t , o b j } }$ , respectively, to obtain the denoised signal $\hat { \mathbf { x } } _ { 0 } ^ { l } = \{ \hat { \mathbf { x } } _ { 0 , \mathrm { l h a n d } } ^ { l } , \hat { \mathbf { x } } _ { 0 , \mathrm { r h a n d } } ^ { l } , \hat { \mathbf { x } } _ { 0 , \mathrm { o b j } } ^ { l } \}$

Training. Note that the losses related to left and right hands are activated by indicator functions $\mathbb { 1 } _ { \mathrm { l e f t } }$ and $\mathbb { 1 } _ { \mathrm { i g h t } } .$ respectively, which are derived from the hand type H\*. The $\breve { f } ^ { \mathrm { T H O I } }$ is trained with loss functions as follows:

$$
L _ { \mathrm { T H O I } } ( f ^ { \mathrm { T H O I } } ) { = } L _ { \mathrm { d i f f } } ( f ^ { \mathrm { T H O I } } ) { + } L _ { \mathrm { d m } } ( f ^ { \mathrm { T H O I } } ) { + } L _ { \mathrm { r o } } ( f ^ { \mathrm { T H O I } } )\tag{4}
$$

where

$$
L _ { \mathrm { d i f f } } ( f ^ { \mathrm { T H O I } } ) { = } E _ { { \mathbf { x } } _ { t } \sim q ( { \mathbf { x } } _ { 0 } | c ) , t \sim [ 1 , T ] } | | { \mathbf { x } } _ { 0 } - f ^ { \mathrm { T H O I } } ( { \mathbf { x } } _ { t } , t , c ) | | _ { 2 } ^ { 2 }\tag{5}
$$

is the loss which is used to reconstruct $\mathbf { x } _ { \mathrm { 0 } }$ from $\mathbf { x } _ { t }$ similar to [27]. We have two more losses $( i . e . , L _ { \mathrm { d m } } , L _ { \mathrm { r o } } )$ to make the $f ^ { \mathrm { T H O I } }$ to generate more accurate hand-object motions. The distance map loss $L _ { \mathrm { d m } } .$ proposed in [19], is employed in our hand-object motion generation to align the estimated distance map with ground-truth distance map as follows:

$$
\begin{array} { r l } { L _ { \mathrm { d m } } \big ( f ^ { \mathrm { T H O I } } \big ) = } & { \displaystyle \sum _ { i = 1 } ^ { \hat { L } \times J \times N } \bigg \{ \mathbb { 1 } _ { \mathrm { l e f t } } \cdot \bigg ( \big ( \hat { \mathbf { d } } _ { \mathrm { l e f t } } ^ { i } - \mathbf { d } _ { \mathrm { l e f t } } ^ { i } \big ) \cdot I \big ( \mathbf { d } _ { \mathrm { l e f t } } ^ { i } < \tau \big ) \bigg ) ^ { 2 } } \\ & { + \mathbb { 1 } _ { \mathrm { r i g h t } } \cdot \bigg ( \big ( \hat { \mathbf { d } } _ { \mathrm { r i g h t } } ^ { i } - \mathbf { d } _ { \mathrm { r i g h t } } ^ { i } \big ) \cdot I \big ( \mathbf { d } _ { \mathrm { r i g h t } } ^ { i } < \tau \big ) \bigg ) ^ { 2 } \bigg \} } \end{array}\tag{6}
$$

where $\hat { \mathbf { d } } _ { \mathrm { l e f t } } ^ { i }$ and $\hat { \mathbf { d } } _ { \mathrm { { r i g h t } } } ^ { i }$ denote the i-th element of $\hat { \mathbf { d } } _ { \mathrm { l e f t } }$ and $\hat { \mathbf { d } } _ { \mathrm { { r i g h t } } } \in \mathbb { R } ^ { \hat { L } \times J \times N }$ , respectively. These represent the estimated distance maps between the J hand joints (left $\hat { \mathbf { J } } _ { \mathrm { l h a n d } }$ and right $\hat { \mathbf { J } } _ { \mathrm { r h a n d } } )$ and the N object points $\hat { \mathbf { P } } _ { \mathrm { d e f } }$ across a sequence of $\hat { L }$ frames, derived from their 3D global positions. $\mathbf { d } _ { \mathrm { l e f t } } ^ { i }$ and $\mathbf { d } _ { \mathrm { { r i g h t } } } ^ { i }$ denote the i-th element of $\mathbf { d } _ { \mathrm { l e f t } }$ and $\mathbf { d } _ { \mathrm { r i g h t } } \in \mathbb { R } ^ { \hat { L } \times J \times N }$ which are the ground-truth distance maps obtained for left and right hands, respectively. The indicator function $I ( \cdot )$ outputs 1 when the statement is true and 0, otherwise. It activates the loss only when the hand-object distance is below the distance threshold τ, where it is empirically set as 2cm.

In the relative orientation loss $L _ { \mathrm { r o } } .$ we consider the 3D relative rotation as follows, as hands and objects exhibit severe rotation changes:

$$
\begin{array} { r l } & { L _ { \mathrm { r o } } ( f ^ { \mathrm { T H O I } } ) { = } \mathbb { 1 } _ { \mathrm { l e f t } } \cdot \| R ( \hat { \mathbf { x } } _ { 0 , \mathrm { l h a n d } } , \hat { \mathbf { x } } _ { 0 , \mathrm { o b j } } ) { - } R ( \mathbf { x } _ { 0 , \mathrm { l h a n d } } , \mathbf { x } _ { 0 , \mathrm { o b j } } ) \| _ { 2 } ^ { 2 } } \\ & { \quad \quad \quad + \mathbb { 1 } _ { \mathrm { i j g h t } } \cdot \| R ( \hat { \mathbf { x } } _ { 0 , \mathrm { f h a n d } } , \hat { \mathbf { x } } _ { 0 , \mathrm { o b j } } ) - R ( \mathbf { x } _ { 0 , \mathrm { f h a n d } } , \mathbf { x } _ { 0 , \mathrm { o b j } } ) \| _ { 2 } ^ { 2 } , } \end{array}\tag{7}
$$

where $R ( \cdot , \cdot )$ indicates the 3D relative orientation between hand and object.

Sampling. At each time-step t, the model $f ^ { \mathrm { T H O I } }$ predicts a clean motion, denoted as $\hat { \mathbf { x } } _ { 0 } { = } f ^ { \mathrm { T H O I } } ( \mathbf { x } _ { t } , t , c )$ , and then re-noise $\hat { \mathbf { x } } _ { 0 }$ to $\mathbf { x } _ { t - 1 }$ [27]. This procedure is conducted repeatedly, starting from $t = T$ to $t = 1$

## 3.3. Hand refinement network

We propose a hand refinement network $f ^ { \mathrm { r e f } }$ that considers the contact and penetration between hands and an object generated from Text2HOI $f ^ { \mathrm { T H O I } }$ in Sec.3.2. The architecture of $f ^ { \mathrm { r e f } }$ is similar to that of $f ^ { \mathrm { { \hat { T H O I } } } }$ : 1) it employs a Transformer encoder architecture, and 2) it utilizes frame-wise and agent-wise position encoding. The main differences between $f ^ { \mathrm { T H O I } }$ and $f ^ { \mathrm { r e f } }$ are that $f ^ { \mathrm { r e f } }$ does not involve the diffusion mechanism; it does not receive any conditions as input; and it refines only hand motions. Inputs and outputs. The hand refinement network receives several inputs: Text2HOI's hand output $\hat { \mathbf { X } } _ { \mathrm { 0 , h a n d } } .$ , hand joints $\hat { \mathbf { J } } _ { \mathrm { h a n d } }$ , predicted contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } ,$ deformed object's point cloud $\hat { \mathbf { P } } _ { \mathrm { d e f } } .$ and distance-based attention map $\mathbf { m } _ { \mathrm { a t t } }$ The attention map $\mathbf { m } _ { \mathrm { a t t } } = \exp ( - 5 0 \times \mathbf { D } )$ is defined as [26], where $\mathbf { D } \in \mathbb { R } ^ { J \times 3 }$ represents the 3D displacement between J hand joints $\hat { \mathbf { J } } _ { \mathrm { h a n d } }$ and the nearest object points in $\hat { \mathbf { P } } _ { \mathrm { d e f } } .$ These components, denoted as $\hat { \mathbf { x } } _ { 0 , \mathrm { h a n d } } , \hat { \mathbf { J } } _ { \mathrm { h a n d } } , \hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } , \hat { \mathbf { P } } _ { \mathrm { d e f } }$ , and $\mathbf { m } _ { \mathrm { a t t } } .$ are flattened and concatenated to form the hand refiner input. As indicated in Sec. 3.2, these inputs are masked using H\*. Then, $f ^ { \mathrm { r e f } }$ outputs the refined hand motions $\tilde { \mathbf { X } } _ { \mathrm { h a n d } }$ . They are masked using H\* for loss calculation and result visualization. Training. The hand refinement network is trained using the loss function $L _ { \mathrm { r e f i n e } }$ as follows:

$$
L _ { \mathrm { r e f i n e } } ( f ^ { \mathrm { r e f } } ) { = } L _ { \mathrm { s i m p l e } } ( f ^ { \mathrm { r e f } } ) { + } L _ { \mathrm { p e n e t } } ( f ^ { \mathrm { r e f } } ) { + } \lambda _ { 1 } L _ { \mathrm { c o n t a c t } } ( f ^ { \mathrm { r e f } } ) ,\tag{8}
$$

where $\lambda _ { 1 }$ is set as 5. The simple L2 loss is expressed as follows:

$$
L _ { \mathrm { s i m p l e } } ( f ^ { \mathrm { r e f } } ) { = } \| \tilde { \mathbf { x } } _ { \mathrm { h a n d } } { - } \mathbf { x } _ { \mathrm { h a n d } } \| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $\mathbf { X } _ { \mathrm { h a n d } }$ denotes the ground-truth hand motions. The penetration loss $L _ { \mathrm { p e n e t } } \left[ 1 3 \right]$ is applied only on hand vertices that penetrate the object surfaces as follows:

$$
L _ { \mathrm { p e n e t } } ( f ^ { \mathrm { r e f } } ) { = } \mathbb { 1 } _ { \mathrm { l e f t } } { \cdot } | | d ( \tilde { v } _ { \mathrm { l h a n d } } , \hat { p } _ { \mathrm { o b j } } ^ { \mathrm { l e f t } } ) | | ^ { 2 } { + } \mathbb { 1 } _ { \mathrm { r i g h t } } { \cdot } | | d ( \tilde { v } _ { \mathrm { r h a n d } } , \hat { p } _ { \mathrm { o b j } } ^ { \mathrm { r i g h t } } ) | | ^ { 2 } ,\tag{10}
$$

where $d ( \cdot , \cdot )$ denotes the Euclidean distance between two points, $\tilde { v } _ { \mathrm { l h a n d } } \in \tilde { \mathbf { V } } _ { \mathrm { l h a n d } }$ and $\tilde { v } _ { \mathrm { r h a n d } } \in \tilde { \mathbf { V } } _ { \mathrm { r h a n d } }$ are hand vertices that penetrate the object surface, and $\hat { p } _ { \mathrm { o b j } } ^ { \mathrm { l e f t } } \in \hat { P } _ { \mathrm { d e f } }$ and $\hat { p } _ { \mathrm { o b j } } ^ { \mathrm { r i g h t } } \in \hat { P } _ { \mathrm { d e f } }$ denote the object points closest to $\tilde { v } _ { \mathrm { l h a n d } }$ and $\tilde { v } _ { \mathrm { r h a n d } }$ , respectively. The contact loss $L _ { \mathrm { { c o n t a c t } } }$ [13] is applied to the joints that are sufficiently close to the object surface, as follows:

$$
L _ { \mathrm { c o n t a c t } } ( f ^ { \mathrm { r e f } } ) { = } \mathbb { 1 } _ { \mathrm { l e f t } } { \cdot } | | d ( \tilde { j } _ { \mathrm { l h a n d } } , \hat { c } _ { \mathrm { o b j } } ^ { \mathrm { l e f t } } ) | | ^ { 2 } + \mathbb { 1 } _ { \mathrm { r i g h t } } \cdot | | d ( \tilde { j } _ { \mathrm { r h a n d } } , \hat { c } _ { \mathrm { o b j } } ^ { \mathrm { r i g h t } } ) | | ^ { 2 } ,\tag{11}
$$

where $\tilde { j } _ { \mathrm { l h a n d } } \in \tilde { \bf J } _ { \mathrm { l h a n d } }$ and $\tilde { j } _ { \mathrm { r h a n d } } \in \tilde { \mathbf { J } } _ { \mathrm { r h a n d } }$ represent the hand joints that are within a distance threshold τ from the object surface, respectively. $\hat { c } _ { \mathrm { o b j } } ^ { \mathrm { l e f t } } \in \hat { P } _ { \mathrm { d e f } }$ and $\hat { c } _ { \mathrm { o b j } } ^ { \mathrm { r i g h t } } \in \hat { P } _ { \mathrm { d e f } }$ represent object points closest to $\tilde { j } _ { \mathrm { l h a n d } }$ and $\tilde { j } _ { \mathrm { r h a n d } } .$ respectively.

Table 1. Comparison on H2O, GRAB, and ARCTIC datasets. † denotes our produced results. → denotes that the higher value of the metric, the closer to the GT distribution. Best results are emphasized in bold.
<table><tr><td colspan="6">H2O</td></tr><tr><td>Method</td><td> $\mathrm { A c c u r a c y } \left( \mathrm { t o p } { - } 3 \right) \uparrow$ </td><td> $\mathrm { F I D \downarrow }$ </td><td> $\mathrm { D i v e r s i t y } $ </td><td> $\mathrm { \Delta M u l t i m o d a l i t y \uparrow }$ </td><td> $\mathrm { P h y s i c a l ~ r e a l i s m } \uparrow$ </td></tr><tr><td>GT</td><td> $\overline { { 0 . 9 9 2 0 \pm 0 . 0 0 0 3 } }$ </td><td>–</td><td> $\overline { { 0 . 6 0 5 7 \pm 0 . 0 0 5 0 } }$ </td><td> $0 . 2 0 6 7 \pm 0 . 0 0 2 4$ </td><td> $\overline { { 0 . 4 7 9 0 \pm 0 . 0 0 0 2 } }$ </td></tr><tr><td>T2M† [8]</td><td> $\overline { { 0 . 6 4 6 3 \pm 0 . 0 0 1 4 } }$ </td><td> $\overline { { 0 . 3 4 3 9 \pm 0 . 0 0 0 6 } }$ </td><td> $\overline { { 0 . 3 4 7 5 \pm 0 . 0 0 4 0 } }$ </td><td> $\overline { { 0 . 0 6 3 4 \pm 0 . 0 0 2 2 } }$ </td><td> $\overline { { 0 . 3 8 9 0 \pm 0 . 0 1 6 } }$ </td></tr><tr><td>MDM† [27]</td><td> $0 . 5 8 3 2 \pm 0 . 0 0 1 1$ </td><td> $0 . 3 0 1 5 \pm 0 . 0 0 1 1$ </td><td> $0 . 5 1 2 7 \pm 0 . 0 0 5 4$ </td><td> $0 . 1 7 3 8 \pm 0 . 0 0 4 9$ </td><td> $0 . 5 5 7 2 \pm 0 . 0 0 1 3$ </td></tr><tr><td> $\mathrm { I M O S } ^ { \dagger } \left[ 6 \right]$ </td><td> $0 . 5 5 1 8 \pm 0 . 0 0 2 6$ </td><td> $0 . 2 9 4 5 \pm 0 . 0 0 1 1$ </td><td> $0 . 4 0 7 6 \pm 0 . 0 0 5 6$ </td><td> $0 . 1 7 9 8 \pm 0 . 0 1 1 5$ </td><td> $0 . 3 5 3 2 \pm 0 . 0 0 2 6$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 8 2 9 5 \pm 0 . 0 0 1 5 }$ </td><td> $\mathbf { 0 . 1 7 4 4 \ : \pm 0 . 0 0 1 3 }$ </td><td> $\mathbf { 0 . 5 3 6 5 \pm 0 . 0 0 7 3 }$ </td><td> ${ \bf 0 . 2 4 6 9 \pm 0 . 0 0 8 1 }$ </td><td> $\mathbf { 0 . 7 5 7 4 } \pm \mathbf { 0 . 0 0 2 2 }$ </td></tr><tr><td colspan="6"> $\overline { { \mathrm { ~ G R A B ~ } } }$ </td></tr><tr><td>GT</td><td> $\overline { { 0 . 9 9 9 4 \pm 0 . 0 0 0 1 } }$ </td><td></td><td> $\overline { { 0 . 8 5 5 7 \pm 0 . 0 0 5 4 } }$ </td><td> $\overline { { 0 . 4 3 9 0 \pm 0 . 0 0 4 5 } }$ </td><td> $\overline { { 0 . 8 0 8 4 \pm 0 . 0 0 0 2 } }$ </td></tr><tr><td> $\overline { { { \mathrm { ~ \bf ~ T } 2 { \bf M } ^ { \dag } \left[ 8 \right] } } }$ </td><td> $\overline { { 0 . 1 8 9 7 \pm 0 . 0 0 0 7 } }$ </td><td> $0 . 7 8 8 6 \pm 0 . 0 0 0 5$ </td><td> $0 . 5 7 1 2 \pm 0 . 0 0 7 8$ </td><td> $0 . 0 9 6 4 \pm 0 . 0 0 2 7$ </td><td> $0 . 5 8 4 4 \pm 0 . 0 0 0 2$ </td></tr><tr><td>MDM† [27]</td><td> $0 . 5 1 2 7 \pm 0 . 0 0 0 9$ </td><td> $0 . 6 0 2 3 \pm 0 . 0 0 1 1$ </td><td> $0 . 8 0 1 2 \pm 0 . 0 0 5 4$ </td><td> $0 . 5 1 9 4 \pm 0 . 0 1 4 5$ </td><td> $0 . 7 3 8 2 \pm 0 . 0 0 0 4$ </td></tr><tr><td> $\mathrm { I M O S } ^ { \dagger } \left[ 6 \right]$ </td><td> $0 . 4 0 9 7 \pm 0 . 0 0 0 5$ </td><td> $0 . 6 1 4 7 \pm 0 . 0 0 0 3$ </td><td> $0 . 6 8 6 1 \pm 0 . 0 0 6 0$ </td><td> $0 . 2 8 4 5 \pm 0 . 0 0 3 6$ </td><td> $0 . 6 4 1 8 \pm 0 . 0 0 1 4$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 9 2 1 8 \pm 0 . 0 0 1 0 }$ </td><td> $\mathbf { 0 . 3 0 1 7 \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 8 3 5 1 \pm 0 . 0 0 6 1 }$ </td><td> $\mathbf { 0 . 5 2 1 6 \pm 0 . 0 1 3 1 }$ </td><td> $\mathbf { 0 . 8 8 3 9 \pm 0 . 0 0 0 5 }$ </td></tr><tr><td colspan="6"></td></tr><tr><td>GT</td><td> $\overline { { 0 . 9 9 9 7 \pm 0 . 0 0 0 1 } }$ </td><td></td><td> $\overline { { \mathrm { A R C T I C } } }$   $\overline { { 0 . 5 9 1 6 \pm 0 . 0 0 3 7 } }$ </td><td> $\overline { { 0 . 3 2 7 9 \pm 0 . 0 0 3 8 } }$ </td><td> $\overline { { 0 . 9 5 7 3 \pm 0 . 0 0 0 0 } }$ </td></tr><tr><td>T2M† [8]</td><td> $\overline { { 0 . 5 2 3 4 \pm 0 . 0 0 1 5 } }$ </td><td> $\overline { { 0 . 3 5 9 9 \pm 0 . 0 0 0 5 } }$ </td><td> $\overline { { 0 . 3 3 0 1 \pm 0 . 0 0 2 3 } }$ </td><td> $\overline { { 0 . 0 8 4 9 \pm 0 . 0 0 1 7 } }$ </td><td> $\overline { { 0 . 0 1 4 3 \pm 0 . 0 0 0 1 } }$ </td></tr><tr><td>MDM† [27]</td><td> $0 . 5 5 7 2 \pm 0 . 0 0 1 2$ </td><td> $0 . 3 0 2 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 4 9 8 4 \pm 0 . 0 0 3 9$ </td><td> $0 . 2 6 3 2 \pm 0 . 0 0 6 5$ </td><td> $0 . 7 0 4 3 \pm 0 . 0 0 0 9$ </td></tr><tr><td> $\mathrm { I M O S } ^ { \dagger } \left[ 6 \right]$ </td><td> $0 . 8 1 9 0 \pm 0 . 0 0 3 9$ </td><td> $0 . 1 8 2 6 \pm 0 . 0 0 0 5$ </td><td> $0 . 5 7 0 2 \pm 0 . 0 0 3 9$ </td><td> $0 . 2 7 4 1 \pm 0 . 0 0 4 9$ </td><td> $0 . 7 5 6 9 \pm 0 . 0 0 2 3$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 9 2 0 5 \pm 0 . 0 0 1 2 }$ </td><td> $\mathbf { 0 . 1 3 2 9 \mathop { \pm } 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 5 7 5 8 \pm 0 . 0 0 4 2 }$ </td><td> $\mathbf { 0 . 3 1 7 0 \pm 0 . 0 0 6 8 }$ </td><td> $\mathbf { 0 . 8 7 6 0 \mathop { \pm } 0 . 0 0 0 9 }$ </td></tr></table>

prompts for ARCTIC. The characteristics of three datasets and details of our annotation process are further illustrated in the supplemental material.

## 4. Experiments

## 4.1. Implementation details

We use $T = 1 { , } 0 0 0$ noising steps and a cosine noise schedule. We use sinusoidal positional encoding for frame-wise and agent-wise positional encodings. We set the maximum length of motion sequences, denoted as $L _ { \mathrm { m a x } } ,$ to 150 frames. Further details about network architecture can be found in the supplemental material.

Hand-type selection. We use the CLIP text encoder [23] $f ^ { \mathrm { C L I P } }$ to calculate cosine similarity between the input text prompt T and predefined prompt templates Γ(H)=“A photo of H", where H ∈ {left hand, right hand, both hands}. The hand type H\* with the highest cosine similarity to T is selected for masking in the Transformer's inputs, outputs, and losses (see Secs. 3.2, 3.3 and supplemental material.).

Motion length prediction. To obtain proper motion length $\begin{array} { r } { \hat { L } \le L _ { \operatorname* { m a x } } . } \end{array}$ we design a motion-length prediction network $f ^ { \mathrm { L e n g t h } }$ It receives the text feature vector $f ^ { \mathrm { C L I P } } ( \mathbf { T } )$ and Gaussian random noise $\mathbf { n } \in \mathbb { R } ^ { 6 4 }$ , to predict the appropriate motion length Î for the text prompt T. To train $\hat { f } ^ { \mathrm { L e n g i h } }$ , we use the loss function $L _ { \mathrm { l e n g t h } } = \lVert \hat { L } - L \rVert ^ { 2 }$ , where L is ground truth.

## 4.2. Dataset

We use H2O [16], GRAB [25], and ARCTIC [5] in our experiment, which collects hand-object mesh sequences. We automatically generate text prompts by exploiting action labels for H2O and GRAB datasets; while we manually label text

## 4.3. Evaluation metrics and baselines

Evaluation metrics. We use the metrics of accuracy, frechet inception distance (FID), diversity, and multi-modality, as used in IMOS [6]. The accuracy serves as an indicator of how well the model generates motions and is evaluated by the pre-trained action classifier. We train a standard RNN-based action classifier to extract motion features and classify the action from the motions, as in IMOS [6]. The FID quantifies feature-space distances between real and generated motions, capturing the dissimilarity. The diversity reflects the range of distinct motions and multi-modality measures the average variance of motions for an individual text prompt. To assess the physical realism of generated hand-object motions, we employ a physical model following the approach in ManipNet [29], assigning a realism score of 0 (unreal) or 1 (real) for measuring the realism of each frame. Experiments are conducted 20 times to establish the robustness, and we reported results within a 95% confidence interval.

Baselines. We compare our approach with three existing textto-human motion generation methods: T2M [8], MDM [27] and IMOS [6]. T2M [8] employs a temporal VAE-based architecture and MDM [27] utilizes a diffusion model. IMOS [6] is designed to first generate human body and arm motions conditioned on both action labels and past body motions. It then optimizes object rotation and translation based on their history to generate body and arm motion. Since they were approaches. The similarity of our distribution to the ground truth (GT) distribution in terms of Diversity, along with our highest scores in Multimodality and Accuracy, demonstrates our model's capability to generate motions that are both diverse and accurate, and are well-aligned with text prompts. We compare our qualitative results with other baselines in Fig. 4. It shows that our method, compared to others, outperforms in generating motions where hand and object interact realistically, and these motions align closely with text prompts "Use a hammer with the right hand.". The right hand well grabs the hammer and mimics the motion of driving something into a wall

Table 2. Ablation study on the positional encoding, losses, and conditions for Ours w/o $f ^ { \mathrm { r e f } } ,$ and ablation study on losses for $\mathrm {  ~ \Omega ~ } ^ { \circ } \mathrm { \mathrm { u r s } } ^ { \mathrm { \Theta } } .$
<table><tr><td rowspan="2">Method</td><td rowspan="2"> $f ^ { \mathrm { r e f } }$ </td><td colspan="5"> $\overline { { \mathrm { ~ G R A B ~ } } }$ </td></tr><tr><td> $\mathrm { A c c u r a c y } \left( \mathrm { t o p } { - } 3 \right) \uparrow$ </td><td>FID↓</td><td> $\mathrm { D i v e r s i t y } $ </td><td>Multimodality ↑ Physical realism ↑</td><td></td></tr><tr><td>GT</td><td></td><td> $\overline { { 0 . 9 9 9 4 \pm 0 . 0 0 0 1 } }$ </td><td></td><td> $\overline { { 0 . 8 5 5 7 \pm 0 . 0 0 5 4 } }$ </td><td> $\overline { { 0 . 4 3 9 0 \pm 0 . 0 0 4 5 } }$ </td><td> $\overline { { 0 . 8 0 8 4 \pm 0 . 0 0 0 2 } }$ </td></tr><tr><td>w/o frame-wise &amp; agent-wise PE</td><td>X</td><td> $\overline { { 0 . 8 2 9 4 \pm 0 . 0 0 1 6 } }$ </td><td> $\overline { { 0 . 3 4 6 1 \pm 0 . 0 0 1 8 } }$ </td><td> $\overline { { 0 . 7 8 1 4 \pm 0 . 0 6 3 } }$ </td><td> $\overline { { 0 . 4 7 7 6 \pm 0 . 0 1 9 4 } }$ </td><td> $\overline { { 0 . 8 0 2 4 \pm 0 . 0 0 0 7 } }$ </td></tr><tr><td>w/o agent-wise PE</td><td>x</td><td> $0 . 8 3 1 4 \pm 0 . 0 0 1 2$ </td><td> $0 . 3 4 1 2 \pm 0 . 0 0 0 6$ </td><td> $0 . 8 0 1 1 \pm 0 . 0 6 7$ </td><td> $0 . 4 7 5 5 \pm 0 . 0 1 2 2$ </td><td> $0 . 8 2 2 1 \pm 0 . 0 0 0 9$ </td></tr><tr><td> $\overline { { \mathrm { \Delta w } / \mathrm { o } \ L _ { \mathrm { d m } } \ \& \ L _ { \mathrm { r o } } } }$ </td><td>x</td><td> $\overline { { 0 . 8 2 8 9 \pm 0 . 0 0 3 8 } }$ </td><td> $\overline { { 0 . 3 4 1 6 \pm 0 . 0 0 2 0 } }$ </td><td> $\overline { { 0 . 7 8 8 7 \pm 0 . 0 6 4 0 } }$ </td><td> $\overline { { 0 . 4 6 5 4 \pm 0 . 0 1 5 0 } }$ </td><td> $\overline { { 0 . 7 4 9 0 \pm 0 . 0 0 0 6 } }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ L _ { \mathrm { r o } }$ </td><td>x</td><td> $0 . 8 2 7 2 \pm 0 . 0 0 2 0$ </td><td> $0 . 3 4 0 7 \pm 0 . 0 0 1 5$ </td><td> $0 . 7 9 9 7 \pm 0 . 0 0 7 9$ </td><td> $0 . 4 6 2 7 \pm 0 . 0 1 0 4$ </td><td> $0 . 8 2 4 7 \pm 0 . 0 0 1 1$ </td></tr><tr><td>w/o  $L _ { \mathrm { d m } }$ </td><td>x</td><td> $0 . 8 2 0 2 \pm 0 . 0 0 1 7$ </td><td> $0 . 3 4 4 4 \pm 0 . 0 0 0 7$ </td><td> $\mathbf { 0 . 8 1 5 6 \pm 0 . 0 0 7 0 }$ </td><td> $0 . 4 8 1 9 \pm 0 . 0 1 2 5$ </td><td> $0 . 6 4 1 0 \pm 0 . 0 0 1 0$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } \& s _ { \mathrm { o b j } }$ </td><td>X</td><td> $\overline { { 0 . 8 1 9 7 \pm 0 . 0 0 0 9 } }$ </td><td> $\overline { { 0 . 3 4 2 8 \pm 0 . 0 0 1 2 } }$ </td><td> $\overline { { 0 . 7 9 9 4 \pm 0 . 0 0 5 5 } }$ </td><td> $\overline { { 0 . 4 3 0 5 \pm 0 . 0 1 2 1 } }$ </td><td> $0 . 7 8 1 5 \pm 0 . 0 0 0 6$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ s _ { \mathrm { o b j } }$ </td><td>x</td><td> $0 . 8 2 7 4 \pm 0 . 0 0 1 3$ </td><td> $0 . 3 4 1 3 \pm 0 . 0 0 1 0$ </td><td> $0 . 7 9 6 3 \pm 0 . 0 0 5 4$ </td><td> $0 . 4 4 0 5 \pm 0 . 0 1 3 9$ </td><td> $0 . 8 0 1 8 \pm 0 . 0 0 0 5$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \hat { \mathbf { m } } _ { \mathrm { c o n t a c t } }$ </td><td>x</td><td> $0 . 8 2 7 7 \pm 0 . 0 0 2 7$ </td><td> $0 . 3 4 1 1 \pm 0 . 0 0 0 6$ </td><td> $0 . 8 0 1 2 \pm 0 . 0 0 6 7$ </td><td> $0 . 4 4 5 5 \pm 0 . 0 1 1 5$ </td><td> $0 . 7 8 9 2 \pm 0 . 0 0 0 9$ </td></tr><tr><td> $\overline { { \mathrm { \mathrm { ~ O u r s ~ w } } / \mathrm { 0 ~ } f ^ { \mathrm { r e f } } } }$ </td><td>x</td><td> $\mathbf { 0 . 8 4 1 1 \pm 0 . 0 0 0 9 }$ </td><td> $\mathbf { \overline { { 0 . 3 3 2 1 \pm 0 . 0 0 0 6 } } }$ </td><td> $\overline { { 0 . 8 1 4 3 \pm 0 . 0 0 5 0 } }$ </td><td> $\mathbf { \overline { { 0 . 4 9 8 9 \pm 0 . 0 1 5 4 } } }$ </td><td> $\mathbf { \overline { { 0 . 8 3 1 2 \pm 0 . 0 0 0 5 } } }$ </td></tr><tr><td>w/o  $\overline { { L _ { \mathrm { p e n e t } } \& L _ { \mathrm { c o n t a c t } } } }$ </td><td>√</td><td> $\overline { { 0 . 8 8 3 8 \pm 0 . 0 0 1 4 } }$ </td><td> $\overline { { 0 . 3 2 3 4 \pm 0 . 0 0 0 7 } }$ </td><td> $\overline { { 0 . 8 2 7 7 \pm 0 . 0 0 6 8 } }$ </td><td> $\overline { { 0 . 5 1 1 1 \pm 0 . 0 1 4 } }$ </td><td> $\overline { { 0 . 6 2 4 9 \pm 0 . 0 0 0 8 } }$ </td></tr><tr><td>w/o  $L _ { \mathrm { { c o n t a c t } } }$ </td><td>√</td><td> $0 . 8 8 2 7 \pm 0 . 0 0 0 8$ </td><td> $0 . 3 1 1 4 \pm 0 . 0 0 1 3$ </td><td> $0 . 8 3 0 1 \pm 0 . 0 0 4 8$ </td><td> $0 . 4 8 0 8 \pm 0 . 0 1 5 1$ </td><td> $0 . 1 4 6 7 \pm 0 . 0 0 0 5$ </td></tr><tr><td>w/o  $L _ { \mathrm { p e n e t } }$ </td><td>√</td><td> $0 . 8 9 4 1 \pm 0 . 0 0 0 9$ </td><td> $0 . 3 0 2 4 \pm 0 . 0 0 0 5$ </td><td> $0 . 8 2 6 7 \pm 0 . 0 0 6 1$ </td><td> $0 . 5 1 8 2 \pm 0 . 0 0 9 9$ </td><td> $0 . 8 7 8 2 \pm 0 . 0 0 0 6$ </td></tr><tr><td>Ours</td><td>√</td><td> $\mathbf { \overline { { 0 . 9 2 1 8 \pm 0 . 0 0 1 0 } } }$ </td><td> $\mathbf { 0 . 3 0 1 7 \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { \overline { { 0 . 8 3 5 1 } } } \pm \mathbf { 0 . 0 0 6 1 }$ </td><td> $\overline { { { \bf 0 . 5 2 1 6 \pm 0 . 0 1 3 1 } } }$ </td><td> $\mathbf { \overline { { 0 . 8 8 3 9 \pm 0 . 0 0 0 5 } } }$ </td></tr></table>

"Use a hammer with the right hand."

![](images/6063468d1003d04ce9e755f0c9a57da14a0a8d173d859ed749049cd23aa331a5.jpg)  
Figure 4. We compare our generated hand-object motions with other baselines' results. Each row show the results of Text2Motion [8], MDM [27], IMOS [6], and ours.

Qualitative results are shown in Fig. 5. Our method generates realistic hand-object motions that are closely aligned with the input text prompts, effectively handling even unseen objects. Please refer to the supplemental material for more visualizations including video results, and text-guided and scale-variant contact maps.

originally designed for generating individual human motions from text prompts, to ensure a fair comparison, we re-train the methods using hand-object motion data, allowing them to generate hand-object motions from text prompts.

## 4.5. Ablation study

We conduct several ablation studies on GRAB dataset, to validate the effectiveness of our modules. The results are demonstrated in Tab. 2.

## 4.4. Experimental results

Comparison to other methods. We compare our method with other state-of-the-art methods (i.e., T2M [8], MDM [27], and IMOS [6]), as shown in Tab. 1. For all datasets, our method outperforms other baselines in multiple measures. Particularly, our method demonstrates exceptional performance in generating physically realistic hand-object motions, as evidenced by achieving the highest score in Physical realism compared to other

Position encoding. We introduce two types of positional encodings: frame-wise and agent-wise, which assists the Transformer to interpret inputs in a more distinct way. Seeing the results ‘w/o frame-wise & agent-wise PE’ and w/o agent-wise PE', we can conclude that by leveraging the specialized positional encoding, fTHol is capable of generating more realistic hand-object motions.

Losses. We remove the distance map loss $L _ { \mathrm { d m } }$ and relative orientation loss $L _ { \mathrm { r o } }$ in our approach, and see how the performance changes: Seeing w/o $L _ { \mathrm { d m } } \& L _ { \mathrm { r o } } ^ { \phantom { \dagger } } ,$ , 'w/o $L _ { \mathrm { d m } } \ '$ andw/o ${ L _ { \mathrm { r o } } } ^ { \mathrm { , } }$ , we can conclude that these losses induce better results by facilitating model's understanding of 3D relationship between hands and an object.

![](images/5704726693a5822707b2f70fb9187ededfce60435f87c0f54cc7a86a79d69d8a.jpg)

"Fly an airplane with the right hand."

![](images/f414c382b5ed5ef9d5b082cc31fe716d62cb1646ecd0f742e0b3e448297d5677.jpg)  
"Close a microwave with both hands."  
"Pour milk in round bottle with the right hand." (Unseenobject)

![](images/e48e853b5b8408081255be0641e02c6c5390c574138c686507813bd67843aff7.jpg)  
"Grab a teddy bear with the left hand." (Unseenobject)

![](images/d88f09e77cb5804a6fc5c96ad067dfb5968908502ccfb8833b92daf40b60adcd.jpg)  
Figure 5. We demonstrate the generated hand-object motions and the predicted contact map results. The first and second rows show the results with objects seen during training. The third and fourth rows show the results with objects unseen during training.

Condition inputs. We remove the contact map $\hat { \mathbf { m } } _ { \mathrm { c o n t a c t } }$ and scale of the object $s _ { \mathrm { o b j } }$ conditions from the original pipeline, and see how the performance changes. Seeing $\mathbf { \hat { w } } / 0 \hat { \mathbf { m } } _ { \mathrm { c o n t a c t } } \& s _ { \mathrm { o b j } } ^ { * }$ 'w/o $s _ { \mathrm { o b j } } \rrangle$ and $\mathrm { { \bf \hat { w } } / o \hat { \ m } _ { c o n t a c t } } ;$ , we can observe that gradually including additional conditions aids in generating more appropriate hand poses to the object.

Refiner. Compared to the Ours w/o $f ^ { \mathrm { r e f i n e r } } ,$ that does not involve the refiner, Ours' provides far better performance, especially in the physical realism of hand and object motions. Also, we demonstrate the effect of losses $L _ { \mathrm { p e n e t } }$ and $L _ { \mathrm { { c o n t a c t } } }$ by removing them in ‘w/o $L _ { \mathrm { p e n e t } } \& L _ { \mathrm { c o n t a c t } } ,$ , 'w/o $L _ { \mathrm { { c o n t a c t } } } { \mathrm { , } }$ and 'w/o $\scriptstyle L _ { \mathrm { p e n e t } } ,$ , as shown in the same table. Involving more losses consistently improve the performance.

## 5. Conclusion

In this paper, we propose a novel method for generating the sequence of 3D hand-object interaction from a text prompt and a canonical object mesh. This is achieved through the three-staged framework that 1) estimates the text-guided and scale-variant contact maps; 2) generates hand-object motions based on a Transformer-based diffusion mechanism; and 3) refines the interaction by considering the penetration and contacts between hands and an object. In experiments, we validate our effectiveness of hand-object interaction generation by comparing it to three baselines where our method outperforms previous methods with strong physical plausibility and accuracy.

Acknowledgements. This work was supported by IITP grants (No. 2020-0-01336 Artificial intelligence graduate school program (UNIST) 10%; No. 2021-0-02068 AI innovation hub 10%; No. 2022-0-00264 Comprehensive video understanding and generation with knowledge-based deep logic neural network 20%) and the NRF grant (No. RS-2023-00252630 20%), all funded by the Korean government (MSIT). This work was also supported by Korea Institute of Marine Science & Technology Promotion (KIMST) funded by Ministry of Oceans and Fisheries (RS-2022-KS221674) 20% and received support from AI Center, CJ Corporation (20%).

## References

[1] Samarth Brahmbhatt, Cusuh Ham, Charles C Kemp, and James Hays. Contactdb: Analyzing and predicting grasp contact via thermal imaging. In CVPR, 2019. 2

[2] Samarth Brahmbhatt, Ankur Handa, James Hays, and Dieter Fox. Contactgrasp: Functional multi-finger grasp synthesis from contact. In IROS, 2019. 2

[3] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In CVPR, 2023. 1, 2

[4] Enric Corona, Albert Pumarola, Guillem Alenya, Francesc Moreno-Noguer, and Grégory Rogez. Ganhand: Predicting human grasp affordances in multi-object scenes. In CVPR, 2020. 2

[5] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges. Arctic: A dataset for dexterous bimanual hand-object manipulation. In CVPR, 2023. 2, 6

[6] Anindita Ghosh, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, and Philipp Slusallek. Imos: Intent-driven full-body motion synthesis for human-object interactions. In Computer Graphics Forum, 2023. 2, 6, 7

[7] Patrick Grady, Chengcheng Tang, Christopher D Twigg, Minh Vo, Samarth Brahmbhatt, and Charles C Kemp. Contactopt: Optimizing contact to improve grasps. In CVPR, 2021. 2

[8] Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In CVPR, 2022. 1, 2, 6, 7

[9] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In ECCV, 2022. 1, 2

[10] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In CVPR, 2020. 2

[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NIPS, 2020. 2, 3, 4

[12] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. arXiv preprint arXiv:2306.14795, 2023. 2

[13] Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang. Hand-object contact consistency reasoning for human grasps generation. In ICCV, 2021. 2, 5

[14] Korrawe Karunratanakul, Jinlong Yang, Yan Zhang, Michael J Black, Krikamol Muandet, and Siyu Tang. Grasping field: Learning implicit representations for human grasps. In 2020 International Conference on 3D Vision (3DV), 2020. 2

[15] Jihoon Kim, Jiseob Kim, and Sungjoon Choi. Flame: Free-form language-based motion synthesis & editing. In AAAI, 2023. 1, 2

[16] Taein Kwon, Bugra Tekin, Jan Stühmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In ICCV, 2021. 2, 6

[17] Taeryung Lee, Gyeongsik Moon, and Kyoung Mu Lee. Multiact: Long-term 3d human motion generation from multiple action labels. In AAAI, 2023. 2

[18] Haoming Li, Xinzhuo Lin, Yang Zhou, Xiang Li, Yuchi Huo, Jiming Chen, and Qi Ye. Contact2grasp: 3d grasp synthesis via hand-object contact constraint. arXiv preprint arXiv:2210.09245, 2022.2,3

[19] Han Liang, Wenqian Zhang, Wenxuan Li, Jingyi Yu, and Lan Xu. Intergen: Diffusion-based multi-human motion generation under complex interactions. arXiv preprint arXiv:2304.05684, 2023.2,5

[20] Arsalan Mousavian, Clemens Eppner, and Dieter Fox. 6-dof graspnet: Variational grasp generation for object manipulation. In ICCV, 2019. 2

[21] Mathis Petrovich, Michael J Black, and Gül Varol. Temos: Generating diverse human motions from textual descriptions. In ECCV, 2022. 1, 2

[22] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 3

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 6

[24] Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: modeling and capturing hands and bodies together. ACM Transactions on Graphics (TOG), 2017. 3

[25] Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas. Grab: A dataset of whole-body human grasping of objects. In ECCV, 2020. 2, 6

[26] Omid Taheri, Yi Zhou, Dimitrios Tzionas, Yang Zhou, Duygu Ceylan, Soren Pirk, and Michael J Black. Grip: Generating interaction poses using latent consistency and spatial cues. arXiv preprint arXiv:2308.11617, 2023.5

[27] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In ICLR, 2023. 1, 2, 4, 5, 6, 7

[28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NIPS, 2017. 3

[29] He Zhang, Yuting Ye, Takaaki Shiratori, and Taku Komura. Manipnet: neural manipulation synthesis with a hand-object spatial representation. ACM Transactions on Graphics (ToG), 2021.2,3,6

[30] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, and Ying Shan. Generating human motion from textual descriptions with discrete representations. In CVPR, 2023. 1, 2

[31] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv preprint arXiv:2208.15001, 2022. 1,2

[32] Juntian Zheng, Qingyuan Zheng, Lixing Fang, Yun Liu, and Li Yi. Cams: Canonicalized manipulation spaces for category-level functional hand-object manipulation synthesis. In CVPR, 2023. 3

[33] Chongyang Zhong, Lei Hu, Zihao Zhang, and Shihong Xia. Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In ICCV, 2023. 2

[34] Keyang Zhou, Bharat Lal Bhatnagar, Jan Eric Lenssen, and Gerard Pons-Moll. Toch: Spatio-temporal object-to-hand correspondence for motion refinement. In ECCV, 2022. 2

[35] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019. 3