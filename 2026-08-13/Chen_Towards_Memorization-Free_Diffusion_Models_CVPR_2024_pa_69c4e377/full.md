# Towards Memorization-Free Diffusion Models

Chen Chen Daochang Liu Chang Xu School of Computer Science, Faculty of Engineering, The University of Sydney {cche0711@uni., daochang.liu@, c.xu@}sydney.edu.au

## Abstract

Pretrained diffusion models and their outputs are widely accessible due to their exceptional capacity for synthesizing high-quality images and their open-source nature. The users, however, may face litigation risks owing to the models’ tendency to memorize and regurgitate training data during inference. To address this, we introduce Anti-Memorization Guidance (AMG), a novel framework employing three targeted guidance strategies for the main causes of memorization: image and caption duplication, and highly specific user prompts. Consequently, AMG ensures memorization-free outputs while maintaining high image quality and text alignment, leveraging the synergy of its guidance methods, each indispensable in its own right. AMG also features an innovative automatic detection system for potential memorization during each step of inference process, allows selective application of guidance strategies, minimally interfering with the original sampling process to preserve output utility. We applied AMG to pretrained Denoising Diffusion Probabilistic Models (DDPM) and Stable Diffusion across various generation tasks. The results demonstrate that AMG is the first approach to successfully eradicates all instances of memorization with no or marginal impacts on image quality and text-alignment, as evidenced by FID and CLIP scores.

## 1. Introduction

Diffusion models [12, 23, 34] have attracted substantial interest, given their superiority in terms of diversity, fidelity, scalability [28] and controllability [24] over previous generative models including VAEs [17], normalizing flows [29], and GANs [10, 14–16]. With guidance techniques [7, 11], diffusion models can be further improved by the strategical diversity-fidelity trade-off. State-of-the-art diffusion models trained on vast web-scale datasets are widespreadly used and have seen deployment at a commercial scale [1, 30, 31].

Such widespread adoption, however, has significantly heightened the litigation risks for companies using these models, particularly due to allegations that the models memorize and reproduce training data during inference without informing the data owners and the users of diffusion models. This potentially violates copyright laws and introduces ethical dilemmas, further complicated by the fact that the extensive size of training sets impedes detailed human review, leaving the intellectual property rights of the data sources largely undetermined. An ongoing example is that a legal action contends that Stable Diffusion is a 21st-century collage tool that remixes the copyrighted works of millions ofartists whose work was used as training data [32].

![](images/fff88c02c22a9f3ce63b65ce840246cf311088f611ff91b80e7f2078957cfeca.jpg)  
Figure 1. Stable Diffusion’s capacity to memorize training data, manifested as pixel-level memorization (left) and object-level memorization (right). Our approach successfully guides pretrained diffusion models to produce memorization-free outputs.

Prior studies [4, 35, 36] have observed memorization in pretrained diffusion models, particularly during unconditional CIFAR-10 [18] and text-conditional LAION dataset [33] generations. While previous research proposed strategies to reduce memorization, these often lead to only modest improvements and fail to fully eliminate the issue. The effectiveness often come with reduced output quality and text-alignment [36], the need for retraining models [4], and extensive manual intervention [19]. Moreover, these strategies lack an automated way to differentiate potential memorization cases for targeted mitigation. For example, [19] relies on a predefined list of text prompts prone to causing memorization, and [36] applies randomization mechanisms uniformly without distinguishing between scenarios.

In this paper, we undertake the following systematic efforts to address the issue of memorization. Firstly, we have identified and detailed the primary causes of memorization, pinpointing image and text duplication in training datasets, along with the high specificity of user prompts for text conditioning, as key contributors. Secondly, we propose a novel unified framework, Anti-Memorization Guidance (AMG), which comprises three distinct guidance strategies, namely, desspecification guidance $( G _ { s p e } )$ , caption deduplication guidance $( G _ { d u p } )$ , and dissimilarity guidance $( G _ { s i m } )$ with each meticulously crafted to address one of these identified causes. Each strategy within AMG effectively guides generations away from memorized training images, offering unique benefits. $G _ { s p e }$ and $G _ { d u p }$ excel in maximally preserving the quality of generated images, while $G _ { s i m }$ provides a definitive assurance against memorization. The absence of any one of these strategies would compromise the delicate balance between privacy and utility, underscoring the indispensability of each of the three methods in the framework. To further enhance the privacy-utility trade-off, AMG features an automatic detection mechanism that continuously assesses the similarity between the current prediction and its nearest training data during the inference process to identify potential instances of memorization. This allows AMG to apply guidance selectively rather than uniformly, ensuring that the original sampling process of the pretrained diffusion model is maximally preserved.

We conducted experiments with AMG on pretrained Denoising Diffusion Probabilistic Models (DDPM) and Stable Diffusion, spanning various generation tasks such as unconditional, class-conditional, and text-conditional generations. The outcomes, both qualitative and quantitative, demonstrate that AMG is the first method that effectively eradicates all memorization instances with minimal impact on image quality and text-alignment. In summary, our contributions through AMG are multifaceted and significant: 1) AMG introduces three guidance strategies, each meticulously designed to address one of the primary causes of memorization, providing a comprehensive solution that effectively balances privacy and utility. 2) AMG is equipped with an automatic detection system for potential memorization during each step of the inference process. This allows for the selective application of guidance strategies, maximizing the preservation of output utility. 3) Expanding upon previous research that focused only on unconditional and text-conditional generations, our study is the first to identify and address memorization in class-conditional diffusion model generations, and is the first that successfully achieves memorization-free generations with minimal compromise on image quality and text-alignment.

## 2. Related Work

Memorization in Diffusion Models has received increased scrutiny over the past year. [35] found that pretrained Stable Diffusions and unconditional DDPMs trained on small datasets like CelebA [22] and Oxford Flowers [25] often replicate training data. [4] reported memorization in pretrained DDPMs on CIFAR-10.Our work focus on pretrained diffusion models, which are extensively used and directly expose their users to litigation risks. We also pioneer in studying class-conditional model memorization, proposing a unified framework that successfully eradicates memorization in various generation tasks including unconditional, class-conditional, and text-conditional generations.

Mitigation Strategies. Training data deduplication, initially effective in language models [13, 20], was adapted for diffusion models [4], who removed 5,275 similar images from CIFAR-10 and retrained the model, achieving a reduction in memorization. Yet, it offers limited improvement and requires retraining the entire model, which is computa tionally intensive, especially for advanced diffusion models with large datasets. Concept ablation [19], implemented for Stable Diffusion, involved fine-tuning the pre-trained model on two sets of text-image pairs (one prone to memorization and the other not), curated using ChatGPT-generated para phrases, to minimize their output disparity. While effective, this method demands extensive manual effort and relies heavily on the crafted prompts’ quality. Also, it assumes the availability of a predefined list of memorization-prone prompts, which is unrealistic in many cases. Randomizing text conditioning [36] during training or inference can also reduce memorization but has limitations. It mitigates, rather than fully prevents, memorization and lacks guaranteed effectiveness in untested scenarios. It results in significant declines in CLIP [27] score, indicating a poorly balanced trade-off between reducing memorization and maintaining text-alignment of generated images. Furthermore, its uni form application across all text conditions, without considering their potential for causing memorization, further diminishes the images’ practical value. Our approach, AMG, adopts a distinct strategy. Instead of altering text condition to indirectly affect image generation, we aim to directly modify the generated image, thereby providing a guarantee of reduced similarity and addressing the issue of mem orization more effectively. Moreover, AMG uses real-time similarity metrics to selectively apply guidance to likely duplicates during inference, ensuring a targeted approach that leaves unaffected cases unaltered, in contrast to the indis criminate application of randomization techniques, also bypassing the need for manual crafting of concept ablation.

Other Privacy-Preservation Strategies encompass the adoption of Differential Privacy (DP) [9] in training generative models [5, 8, 40], primarily through the use of the differentially-private stochastic gradient descent (DP-SGD) algorithm [2]. Although effective on smaller datasets like MNIST and CelebA, [4] noted that implementing DP-SGD in diffusion models tends to result in divergence during training on datasets of CIFAR-10 scale. The feasibility of applying DP-SGD to even larger datasets, such as LAION, remain unexplored. In addition, methods such as dataset distillation [21, 39] offer a means to prevent raw data being directly used in the training of generative models, thereby aiding in privacy preservation. Yet, there has been no exploration of these methods on the LAION dataset to date.

## 3. Preliminaries

## 3.1. Diffusion Models

Denoising Diffusion Probabilistic Models (DDPMs) [12, 23] consist of two processes, firstly, a forward process is required to gradually add Gaussian noise to an image sampled from a real-data distribution $x _ { 0 } \sim q ( x )$ over $\bar { T }$ timesteps such that $x _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } )$ . The Diffusion Kernel then enables sampling $x _ { t }$ at arbitrary timestep t in a closed form:

$$
q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )\tag{1}
$$

where $x _ { t }$ is the noised version of $x _ { 0 }$ at timestep t, $\begin{array} { r l } { \alpha _ { t } } & { { } = } \end{array}$ $1 - \beta _ { t }$ is the noise schedule controls the amount of noise injected into the data at each step, and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ . Then, in the backward process, generative modelling can be realized by learning a denoiser network $\epsilon _ { \theta }$ to predict the noise $\epsilon _ { t }$ instead of the image $x _ { t - 1 }$ at any arbitrary step t:

$$
\mathcal { L } = \mathbb { E } _ { t \in [ 1 , T ] , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \| \epsilon _ { t } - \epsilon _ { \theta } ( x _ { t } , t ) \| _ { 2 } ^ { 2 } ]\tag{2}
$$

where the denoiser network can be easily reformulated as a conditional generative model $\epsilon _ { \theta } ( x _ { t } , y , t )$ by incorporating additional class or text conditioning $y .$

Score-based formulation [38] aims to construct a continuous time diffusion process, where $t \in [ 0 , T ]$ is continuous. The reverse processes can be formulated as:

$$
d x _ { t } = \left[ - \frac { 1 } { 2 } \beta ( t ) x _ { t } - \beta ( t ) \nabla _ { x _ { t } } \log q _ { t } ( x _ { t } ) \right] d t + \sqrt { \beta ( t ) } d \bar { w } _ { t }\tag{3}
$$

where $\beta ( t )$ is a time-dependent function that allows different step sizes $\beta _ { t } = \beta ( t ) \Delta t$ along the process t. A denoiser network $\nabla _ { x _ { t } }$ log $p _ { \theta } ( x _ { t } )$ is then learned to approximate the score function $\nabla _ { x _ { t } } \log { q _ { t } ( x _ { t } ) }$ in the reverse process using a denoising score matching objective, which can be derived to be the same objective as in Eq. (2), leveraging the connection between diffusion models and score matching:

$$
\nabla _ { x _ { t } } \log { p _ { \theta } ( x _ { t } ) } = - \frac { 1 } { \sqrt { 1 - \alpha _ { t } } } \epsilon _ { \theta } ( x _ { t } )\tag{4}
$$

## 3.2. Guidance in Diffusion Models

Classifier guidance (CG) and classifier-free guidance (CFG) are methods used in diffusion models to steer image generation towards higher likelihood outcomes as determined by an explicit or implicit classifier $p _ { \phi } ( \boldsymbol { y } | \boldsymbol { x } _ { t } )$ . In the scorebased framework [38], CG and CFG involve learning the gradient of the log probability for the conditional model, $\nabla _ { x _ { t } } \log { p _ { \theta } ( x _ { t } | y ) }$ , rather than the score of unconditional model, $\nabla _ { x _ { t } } \log p _ { \theta } ( x _ { t } )$ . The conditional score can be easily derived using Bayes’ rule as the sum of the unconditional score and the gradient of the log classifier probability:

$$
\nabla _ { x _ { t } } \log { p _ { \theta } ( x _ { t } | y ) } = \nabla _ { x _ { t } } \log { p _ { \theta } ( x _ { t } ) } + \nabla _ { x _ { t } } \log { p _ { \phi } ( y | x _ { t } ) }\tag{5}
$$

Classifier Guidance (CG) [7] involves training an explicit classifier $p _ { \phi } ( y | x _ { t } )$ on perturbed images $x _ { t }$ and then employing its gradients $\nabla _ { x _ { t } } \log { p _ { \phi } ( y | x _ { t } ) }$ to direct the diffusion sampling process towards a class label $y .$ Inserting Eq. (4) into $\operatorname { E q . }$ . (5), [7] shows a new epsilon prediction ϵˆ corresponds to the score presented in Eq. (5):

$$
\hat { \epsilon } : = \epsilon _ { \theta } ( x _ { t } ) - \sqrt { 1 - \bar { \alpha } _ { t } } \nabla _ { x _ { t } } \log p _ { \phi } ( y | x _ { t } )\tag{6}
$$

Classifier-Free Guidance (CFG) [11] eliminates the need of an explicit classifier for computing Eq. (5). It requires concurrent training on conditional and unconditional objectives. At inference, the epsilon prediction is linearly directed towards the conditional prediction and away from the unconditional, and $s _ { 0 } ~ > ~ 1$ controls the degree of adjustment:

$$
\hat { \epsilon }  \epsilon _ { \theta } ( x _ { t } ) + s _ { 0 } \cdot ( \epsilon _ { \theta } ( x _ { t } , y ) - \epsilon _ { \theta } ( x _ { t } ) ) .\tag{7}
$$

## 4. Memorization in Diffusion Models

Memorization in generative models is identified when generated images exhibit extreme similarity to certain training samples. The strictest definition of memorization relates to high pixel-level similarity, often qualitatively represented as the generated image being near-copies of training samples as in Fig. 4 and left of Fig. 1. To quantify this similarity, the negative normalized Euclidean L2-norm distance (nL2) is employed as a pixel-level metric [4]. For a generated representation $\scriptstyle { \hat { x } } _ { 0 }$ , this involves first identifying its nearest neighbor $n _ { 0 }$ using the $\ell _ { 2 }$ norm, and then normalizing the norm as follows:

$$
\sigma _ { t } = - \frac { \ell _ { 2 } ( \hat { x } _ { 0 } , n _ { 0 } ) } { \alpha \cdot \frac { 1 } { k } \sum _ { z _ { 0 } \in S _ { \hat { x } _ { 0 } } } \ell _ { 2 } ( \hat { x } _ { 0 } , z _ { 0 } ) }\tag{8}
$$

where $S _ { \hat { x } _ { 0 } }$ is a set of $k = 5 0$ nearest neighbors of $\scriptstyle { \hat { x } } _ { 0 }$ , and α is a scaling constant with a default value of 0.5.

A broader definition of memorization encompasses reconstructive memory in diffusion models, where the models reassemble various elements from memorized training images, such as foreground and background objects. These reconstructions might include transformations like shifting, scaling, or cropping. Consequently, the reconstructed outputs do not necessarily match any training image on a pixelby-pixel basis, yet they exhibit a high degree of similarity to certain training images at the object level. In right section of Fig. 1, we observe that the outputs generated by Stable Diffusion are not pixel-level identical to the training images. However, they demonstrate significant object-level similarity. To quantify such object-level similarity, a commonly used metric is the dot product of embeddings of $\scriptstyle { \hat { x } } _ { 0 }$ and $n _ { 0 } \colon$

![](images/a467c51d1c93fad014c2de04de288d116451ca4336af15bdb32808039c42148b.jpg)  
Figure 2. Geometric interpretation of different guidance methods and generations. The center O represents a scenario where the generated image is identical to the memorized training image. The distance of any point from O reflects its degree of dissimilarity to the memorized image. The surface of the sphere signifies the threshold that defines the presence of a memorization issue. The arrows represent different types of guidance strategies.

$$
\sigma _ { t } = E ( \hat { x } _ { 0 } ) ^ { T } \cdot E ( n _ { 0 } )\tag{9}
$$

where $E ( \cdot )$ represents the embedding obtained via a feature extractor, with Self-supervised Copy Detection (SSCD) [26] being the preferred method for identifying object-level memorization [35], $n _ { 0 }$ represents the nearest neighbor of $x _ { 0 } ,$ , identified using this metric.

In later studies [19, 36] that examine memorization issues within the diverse LAION dataset, where numerous object-level memorization instances are identified, SSCD has been established as the standard metric. On the other hand, for the CIFAR-10 dataset, the negative nL2 is found to be an efficient measure [4], attributable to the dataset’s smaller size and consistency in image presentation.

## 4.1. Causes of Memorization

The main causes of memorization in diffusion models are identified as follows: 1) Overly specific user prompts act as a “key” to the pretrained model’s memory, potentially retrieving a specific training image corresponding to this “key”, as observed by [36]. 2) Duplicated training images are more inclined to be memorized by diffusion models as noted by [4, 35], likely due to overfitting. 3) Duplicated captions across those duplicated images can exacerbate the memorization issue [36] by overfitting the textimage pairs to text-conditional diffusion models, turning the caption into a “key” that consistently retrieves the “value” of the associated image. Therefore, when a user employs such repetitive captions or closely related text prompts as the conditioning, the model is prone to generate the corresponding duplicated training image.

## 5. Anti-Memorization Guidance

Leveraging insights into the primary causes of memorization, for diffusion models, we present Anti-Memorization Guidance (AMG), a unified framework integrating a comprehensive suite of three distinct guidance strategies, namely, dissimilarity guidance, desspecification guidance, and caption deduplication guidance. Each strategy within AMG is meticulously crafted to address and effectively eliminate specific causes of memorization in these models. As illustrated in Fig. 2, in the original diffusion models employing classifier-free guidance (CFG), the unconditional generation (point A) is linearly guided towards its text-conditional generation (point B), which may falls inside the sphere, indicating that it has become a memorized case. In the AMG framework, all three guidance methods are capable of steering the generation process away from memorization, represented by moving the generation outside the sphere in the geometric representation. From an implementation standpoint, each guidance strategy is readily integrable with different types of pretrained diffusion models, such as conditional and unconditional Denoising Diffusion Probabilistic Models (DDPMs) and Latent Diffusion Models (LDMs), without the necessity for additional re-training or fine-tuning. This compatibility extends to different sampling methods, including DDPM sampler and accelerated sampling method such as DDIM [37]. The framework solely requires updating the epsilon prediction ϵˆ in accordance with each specific guidance strategy, which is then combined with $x _ { t }$ to estimate the previous step representation $x _ { t - 1 }$ in the reverse process of diffusion models:

$$
\hat { \epsilon }  \hat { \epsilon } + 1 _ { \{ \sigma _ { t } > \lambda _ { t } \} } \cdot ( G _ { s p e } + G _ { d u p } + G _ { s i m } )
$$

$$
x _ { t - 1 } \gets \sqrt { \bar { \alpha } _ { t - 1 } } \left( \frac { x _ { t } - \sqrt { 1 - { \bar { \alpha } _ { t } } } \hat { \epsilon } } { \sqrt { \bar { \alpha } _ { t } } } \right) + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \hat { \epsilon }\tag{10}
$$

(11)

To minimize alterations to the original sampling process and thus preserve output utility to the greatest extent, we introduce an indicator function $1 _ { \{ \sigma _ { t } > \lambda _ { t } \} }$ that activates guidance only when the current similarity $\sigma _ { t }$ exceeds a pre-set threshold $\lambda _ { t } .$ . Importantly, we have designed $\lambda _ { t }$ as a dynamic, rather than static, threshold. This dynamic nature accounts for the observed fluctuations of $\sigma _ { t }$ throughout the inference process. For instance, in the early denoising stages, when t is large, the prediction $\scriptstyle { \hat { x } } _ { 0 }$ is generally less precise than at later stages when t approaches 0. Consequently, $\sigma _ { t }$ tends to be lower at higher t values and increases as t decreases. To effectively manage this variation, we adopt a parabolic scheduling for $\lambda _ { t }$ in alignment with the characteristics of the denoising stages. As a result, this design of conditional guidance with parabolic scheduling operates as an automatic mechanism to selectively activates guidance only when necessary. Such a configuration enables AMG to optimize the balance between reducing memorization and maintaining high-quality outputs.

Moreover, AMG ensures that the generated images, during inference time, diverge from the memorized training image at either pixel or object level. The type and strength of this divergence can be flexibly tailored based on the user’s specific application and objectives.

## 5.1. Despecification Guidance

As previously discussed, a primary cause of memorization in text-conditional diffusion models is the overly specific nature of user prompts, acting as a “key” to the pretrained model’s memory [36]. To reduce caption specificity by instructing the inference process, we firstly employ a desspecification guidance. Given an noised image or latent-space representation $x _ { t }$ and predicted noise ϵˆ at time t, we can obtain its prediction of $\scriptstyle { \hat { x } } _ { 0 }$ using the Diffusion Kernel (Eq. (1)):

$$
\hat { x } _ { 0 } = \frac { x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \cdot \hat { \epsilon } } { \sqrt { \bar { \alpha } _ { t } } }\tag{12}
$$

Then, we search its nearest neighbor $n _ { 0 }$ in the training set and compute the similarity $\sigma _ { t }$ between $\scriptstyle { \hat { x } } _ { 0 }$ and $n _ { 0 }$ . Depending on the user’s goal to prevent pixel-level or object-level memorization, the similarity measure $\sigma _ { t }$ can be computed accordingly using nL2 (Eq. (8)) or SSCD (Eq. (9)).

This method aligns with the principles of CFG but pursues the inverse goal: to attenuate the original CFG scale to linearly adjust the epsilon prediction to be less aligned with the prompt-conditional prediction:

$$
s _ { 1 } = \operatorname* { m a x } ( \operatorname* { m i n } ( c _ { 1 } \sigma _ { t } , s _ { 0 } - 1 ) , 0 )\tag{13}
$$

$$
G _ { s p e } = - s _ { 1 } ( \epsilon _ { \theta } ( x _ { t } , y ) - \epsilon _ { \theta } ( x _ { t } ) )\tag{14}
$$

where $\epsilon _ { \theta } ( x _ { t } , y )$ represents the pretrained diffusion model’s prediction conditioned on user’s text prompt, while $\epsilon _ { \theta } ( x _ { t } )$ denotes the unconditional prediction. $s _ { 0 }$ is the original scale of $\mathrm { C F G } , c _ { 1 }$ is a constant and $c _ { 1 } \sigma _ { t }$ defines the guidance scale at step t, which is directly proportional to the similarity $\sigma _ { t }$ at step t. This enables the algorithm to adaptively adjust the scale of desspecification guidance throughout the sampling process, corresponding to the current level of the memorization, as indicated by the value of $\sigma _ { t }$ at any step t. The function maxp¨, 0q guarantees the guidance scale $- s _ { 1 }$ to be nonpositive, thus diminishing caption specificity, while minp¨q function bounds $c _ { 1 } \sigma _ { t }$ to not exceed $s _ { 0 } - 1$ , safeguarding against excessive low text-image alignment.

From a geometric perspective as in Fig. 2, Despecification guidance $( G _ { s p e } )$ is capable of steering the generation process away from memorization, start from point B, $G _ { s p e }$ directs the prediction in the exact opposite direction of the CFG, exiting the sphere at point X on the surface.

## 5.2. Caption Deduplication Guidance

As outlined in Sec. 4, duplicated captions can act as precise “keys” to retrieve memorized data from the training set, so why not turn this to our advantage? By intentionally using them as prompts for pretrained diffusion models to generate predictions that replicates these memorized images, we can then apply classifier-free guidance techniques to steer the generation away from these images:

$$
s _ { 2 } = \operatorname* { m a x } ( \operatorname* { m i n } ( c _ { 2 } \sigma _ { t } , s _ { 0 } - s _ { 1 } - 1 ) , 0 )\tag{15}
$$

![](images/1dd2784647a6cd65406d0734e66328d5a4d699b0eeb9fdd88278be688d8089af.jpg)  
Figure 3. Comparison of similarity scores throughout the inference process, with and without the application of $G _ { d u p }$

$$
G _ { d u p } = - s _ { 2 } ( \epsilon _ { \theta } ( x _ { t } , y _ { N } ) - \epsilon _ { \theta } ( x _ { t } ) )\tag{16}
$$

where $y _ { N }$ denotes the caption of $n _ { 0 } ,$ , which is the nearest neighbor of current prediction ${ \hat { x } } _ { 0 }$ as defined in Eq. (12). In case where $n _ { 0 }$ is a duplicated image prone to memorization and accompanied by a duplicated caption, $y _ { N }$ would correspond to this replicated caption. Consequently, $\epsilon _ { \theta } ( x _ { t } , y _ { N } )$ reflects the conditional prediction based on the duplicated caption, serving as an ideal antithesis to the prediction we aim to achieve. The function $m a x ( \cdot , 0 )$ again guarantees non-positive guidance scale $- s _ { 2 } .$ , directs ϵˆaway from conditional prediction compared to the unconditional prediction $\epsilon _ { \theta } ( x _ { t } )$ , while minp¨q bounds the total scale of $s _ { 1 } + c _ { 2 } \sigma _ { t }$ to not exceed $s _ { 0 } - 1$ for preserving text-image alignment.

From a geometric standpoint as in Fig. 2, caption deduplication guidance $( G _ { d u p } )$ runs parallel to line OA, representing the guidance direction when using perfect textconditioning that leads to memorization as a negative prompt. This method exits the sphere at point Y on the surface, thereby moving the generation out of the memorization zone. Fig. 3 further illustrates the efficiency of $G _ { d u p }$ It demonstrates that when the similarity score exceeds the dashed parabolic threshold line $\lambda _ { t } ,$ , as defined in Eq. (10), $G _ { d u p }$ is activated. This activation prompts $G _ { d u p }$ to guide the generations in a direction opposite to that of the training image, effectively preventing memorization.

## 5.3. Dissimilarity Guidance

Distinct from the first two strategies, which linearly adjust the epsilon prediction ϵˆ along the vector direction between conditional and unconditional predictions, dissimilarity guidance identifies another dimension that offer effective guidance to reduce, or even eliminate, memorization in diffusion models. It extends the discrete class label y in classifier guidance Eq. (6) to a continuous embedding represented by the similarity score. This approach assures that our generated images are actively directed towards reducing their similarity score, thereby ensuring persistent dissimilarity from their closest counterparts in the training set, as measured by metrics such as nL2 and SSCD.

$$
G _ { s i m } = c _ { 3 } \sqrt { 1 - \bar { \alpha } _ { t } } \cdot \nabla _ { x _ { t } } \sigma _ { t }\tag{17}
$$

where we use the similarity metric $\sigma _ { t }$ instead of log classifier probability log $p _ { \phi } ( y | x _ { t } )$ to compute gradient and guide the inference process. We also invert the sign preceding the guidance term from Eq. (6) to indicate our new objective: to minimize similarity as opposed to maximizing it. An additional scaling factor $c _ { 3 }$ is also employed, which functions as a hyperparameter to control the intensity of the guidance, thereby managing the privacy-utility trade-off and tailoring user’s specific goals. A larger $c _ { 3 }$ permits greater deviation of the generated data from the nearest training image, but at the expense of reduced quality and text alignment.

Geometrically as in Fig. 2, dissimilarity guidance $( G _ { s i m } )$ steers the generation direction directly away from point $\mathrm { O } ,$ which represents the perfectly memorized case (i.e., the training data), eventually exit at point Z on the sphere.

## 5.4. Unpacking AMG’s Threefold Guidance

We present a detailed analysis of the indispensable role and synergies of each of the three guidance methods in AMG for achieving the optimal balance between privacy and utility.

Impact on quality and text-alignment. Similar to CFG, our $G _ { s p e }$ method linearly combines the unconditional prediction $\epsilon _ { \theta } ( x _ { t } )$ and user-prompt based conditional predictions $\epsilon _ { \theta } ( x _ { t } , y )$ Altering the guidance scale modifies the weights of these components, but as both are derived from pretrained diffusion model’s high-quality outputs, overall output quality remains largely consistent. However, text alignment decreases with lower weights on $\epsilon _ { \theta } ( x _ { t } )$ , necessitating a minimum weight of one to preserve text alignment. Similarly, for $G _ { d u p } ,$ assuming using a neighbor’s caption $y _ { N }$ leads to an exact replication of training image, the output maintains high quality, thus its linear combination with $\epsilon _ { \theta } ( x _ { t } )$ also yields quality on par with the pretrained model. Geometrically, results within the $\mathbb { A B Y } \mathbb { p l a n e }$ , formed by lines AB and BY, maintain quality, but those closer to A (along line AB) show reduced text alignment. $G _ { s i m } ,$ however, does not align with the ABY plane, so its scale affects the quality and must be minimally set to preserve quality.

Importance of dissimilarity guidance $( G _ { s i m } ) .$ G<sub>sim</sub>, as shown in Fig. 2, is vital despite its possible quality impact. The necessity stems from the fact that $G _ { s p e }$ and $G _ { d u p } \mathrm { ' s }$ combined scale, capped at $s _ { 0 } \mathrm { ~ - ~ } 1$ , cannot assure moving the generation outside the sphere, potentially leaving it within the BXY sector. $G _ { s i m } .$ , on the other hand, reliably ensures the generation is guided out of the sphere, effectively addressing memorization.

Importance of caption deduplication guidance $( G _ { d u p } ) .$ Text-conditioning heightens specificity, thus reducing diversity in generations. In severe cases, such as when point B near center O, $G _ { s i m }$ would need a high scale to prevent memorization without $G _ { s p e }$ and $G _ { d u p } ,$ risking artifacts. $G _ { s p e }$ and $G _ { d u p }$ mitigate this by lowering the needed scale for $G _ { s i m }$ , thus preserving output quality. Comparing $G _ { s p e }$ and $G _ { d u p } ,$ , both maintain quality within the ABY plane, but $G _ { d u p }$ involves less text alignment sacrifice due to the shorter linear projection of BY on the AB line $( i . e . , \mathrm { B Y ^ { \prime } } < \mathrm { B X } )$ , thus more beneficial than $G _ { s p e }$

Importance of despecification guidance $( G _ { s p e } )$ . The inclusion of $G _ { s p e } ,$ despite $G _ { d u p } \mathrm { ' s }$ apparent advantages, is justified by $G _ { d u p } \mathrm { ' s }$ practical limitations as it idealizes deduplication guidance, requiring access to perfect “keys” for pretrained model memories, approximated here by captions of memorized training images. Such ideal “keys” are uncommon, and caption-based approximations may not always be as precise as prompt-based methods in $G _ { s p e }$ , particularly when memorized images’ captions are not duplicated in the dataset. Consequently, our approach in AMG integrates both $G _ { s p e }$ and $G _ { d u p }$ to emulate these ideal “keys” for effective negative guidance.

Unconditional and class-conditional generations. Our research reveals that in the absence of text-conditioning, memorization in diffusion models is reduced but not eliminated, aligning with our prior findings. This significantly lessens memorization, leaving image duplication as the only remaining issue. In such cases, $G _ { s p e }$ is highly effective in eliminating memorization due to two factors: 1) Early Detection: During initial stages of reverse diffusion, our conditional guidance with a parabolic schedule efficiently identifies potential replication. 2) Increased Diversity: Without specific text-conditioning, the generation process yields greater diversity. This enables $G _ { d i s }$ to effectively steer generations away from memorized modes to un-memorized ones in the initial stages, ensuring they don’t revert. Once memorization ceases to be detected, further guidance application is discontinued. As a result, this method predominantly impacts the coarse structure, with guidance typically applied only during the early stages of the denoising process. The finer details in later stages remain intact, thus ensuring the overall quality of the output is preserved.

## 6. Experiments

## 6.1. Experimental Setup

Scope. In the realm of pretrained diffusion models, studies [4, 35, 36] have identified memorization in unconditional DDPMs on CIFAR-10 and text-conditional Stable Diffusion on LAION datasets. While [35] found no memorization in latent diffusion models using ImageNet, our analysis of DDPMs on ImageNet [6] and LSUN Bedroom [41] also showed no memorization cases. Beyond these scenarios, we explored class-conditional generation and identified memorization cases on CIFAR-10. Our findings confirm the successful elimination of memorization in all tested scenarios, highlighting the wide applicability of our approach.

![](images/f172959168f93cba4296c2c9ca0f7598d5fb0996a8c56c671d046068d93b4458.jpg)  
Figure 4. Applying AMG to iDDPM on CIFAR-10. Left: Classconditional generation. Right: Unconditional generation.

Evaluation metrics. To assess text-conditional generations (Tab. 1), we employ three metric types: memorization, quality, and text-alignment. Memorization metrics include the 95th percentile [36] of similarity scores of all generated images, determined using pixel-level nL2 norm (Eq. (8)) or object-level SSCD embedding similarity (Eq. (9)). We contend that relying solely on this metric might be misleading, especially if the distribution of the generated data exhibits a heavy upper tail beyond the 95th percentile. In such scenarios, the similarity score could be significantly understated. We propose to additionally examine the maximum similarity score within the distribution, to gauge the worst-case scenario regarding memorization. Additionally, the proportion of images exceeding certain similarity thresholds, thus flagged as memorized, is a key metric [4]. Notably, thresholds vary; for CIFAR-10, an nL2 below 1.4 normally indicates pixel-level memorization, while for LAION, an SSCD above 0.5 suggests object-level memorization following the convention of previous work [19, 35, 36]. We use FID to measure the fidelity and diversity of generated images, and CLIP score to measure the generated images’ alignment with the input text prompts.

Implementational details. Experimental results depend on variables such as text prompts, number of generated images, training image scope (e.g., LAION-10k to LAION-5B), choice of diffusion model, and sampling steps. Since baseline methods often employ different settings, we have reimplemented these baselines to ensure a fair and comparable evaluation. In our text-conditional generation experiments on the LAION5B dataset, we utilized [3]’s system for replicates identification. They used a CLIP embeddingbased index for LAION5B, with an efficient retrieval system for identifying k-nearest neighbors. Our approach differed in seeking the singular nearest neighbor based on

<table><tr><td rowspan="2"></td><td colspan="4">Memorization Metrics by SSCD ↓</td><td colspan="2"></td></tr><tr><td>Top5%</td><td>Top1</td><td>%&gt;0.5</td><td>%&gt;0.4</td><td>FID ↓</td><td>CLIP↑</td></tr><tr><td>SD [30]</td><td>0.91</td><td>0.93</td><td>44.85</td><td>59.23</td><td>106.41</td><td>28.04</td></tr><tr><td>Ablation [19]</td><td>=</td><td>-</td><td>0.30*</td><td>-</td><td>-</td><td></td></tr><tr><td>GNI [36]</td><td>0.91</td><td>0.94</td><td>42.75</td><td>58.18</td><td>97.81</td><td>27.79</td></tr><tr><td>RT [36]</td><td>0.61</td><td>0.84</td><td>15.07</td><td>26.75</td><td>101.69</td><td>22.63</td></tr><tr><td>CWR [36]</td><td>0.79</td><td>0.85</td><td>26.45</td><td>40.93</td><td>96.25</td><td>25.96</td></tr><tr><td>RNA [36]</td><td>0.75</td><td>0.82</td><td>17.78</td><td>29.05</td><td>99.68</td><td>23.37</td></tr><tr><td>Ours(Main)</td><td>0.41</td><td>0.47</td><td>0.00</td><td>7.07</td><td>99.12</td><td>26.98</td></tr><tr><td>Ours(Strong)</td><td>0.34</td><td>0.39</td><td>0.00</td><td>0.00</td><td>100.45</td><td>26.72</td></tr></table>

Table 1. Comparisons on text-conditional generation of LAION5B based on SSCD similarity. AMG successfully eliminates memorization with minimal impact on quality and text-alignment.
<table><tr><td rowspan="2"></td><td colspan="4">Memorization Metrics by nL2</td><td rowspan="2">FID↓</td></tr><tr><td>Top5%↑</td><td>Top1↑</td><td>%&lt;1.4↓</td><td>%&lt;1.6↓</td></tr><tr><td>iDDPM [23]</td><td>1.58</td><td>0.51</td><td>0.93</td><td>5.78</td><td>7.44</td></tr><tr><td>Ours(Main)</td><td>1.61</td><td>1.47</td><td>0.00</td><td>4.34</td><td>7.25</td></tr><tr><td>Ours(Strong)</td><td>1.71</td><td>1.68</td><td>0.00</td><td>0.00</td><td>6.98</td></tr></table>

Table 2. Comparisons on unconditional generation of CIFAR-10 based on nL2 similarity. AMG effectively eliminates memorization without affecting image quality.
<table><tr><td rowspan="2"></td><td colspan="4">Memorization Metrics by nL2</td><td rowspan="2">FID↓</td></tr><tr><td>Top5%↑</td><td>Top1↑</td><td>%&lt;1.4↓</td><td>%&lt;1.6↓</td></tr><tr><td>iDDPM [23]</td><td>1.53</td><td>0.51</td><td>1.53</td><td>9.77</td><td>11.81</td></tr><tr><td>Ours(Main)</td><td>1.56</td><td>1.46</td><td>0.00</td><td>8.70</td><td>11.54</td></tr><tr><td>Ours(Strong)</td><td>1.71</td><td>1.68</td><td>0.00</td><td>0.00</td><td>11.44</td></tr></table>

Table 3. Comparisons on class-conditional generation of CIFAR-10 based on nL2 similarity. AMG effectively eliminates memorization without affecting image quality.

SSCD embedding similarity. To approximate this, we first identified 1,000 images with the lowest CLIP embedding similarities, then computed SSCD similarities to find the highest match, using it for memorization metrics. Stable Diffusion v1.4 with a DDIM sampler and 50 sampling steps was our model of choice. For unconditional and class-conditional generations on CIFAR-10, which comprises 50,000 images, we calculated SSCD similarities for each generated image with the entire training set. OpenAI’s iDDPM [23] with a DDPM sampler and 250 sampling steps was used. Further details are in the supplementary material.

## 6.2. Comparison with Baselines

Text-conditional generations on LAION. Table 1 shows that AMG outperforms all baselines in memorization metrics by a huge margin, eliminating all memorization cases defined by a similarity score over 0.5. It also shows a minimal loss in text-alignment, evidenced by having the secondhighest CLIP score among mitigation strategies, thus maintaining strong alignment with user intentions. Notably, the strategy with the highest CLIP score, GNI, performs poorly in memorization metrics, closely resembling the original Stable Diffusion without mitigation. AMG matches baselines in FID, indicating comparable quality. Overall, AMG leads to memorization-free generations and stands out in balancing quality and utility. AMG’s flexibility allows users to adjust guidance strength based on their definition of memorization. While the standard threshold is 0.5, increasing AMG’s guidance scale (i.e., the strong version of AMG) effectively prevents memorization even at a 0.4 threshold, at a minimal extra cost of quality and text-alignment.

<table><tr><td></td><td colspan="2">Mem. by SSCD ↓ Top5%  $\% > 0 . 5$ </td><td>FID ↓</td><td>CLIP ↑</td></tr><tr><td>Baseline [30]</td><td>0.9133</td><td>44.85</td><td>106.41</td><td>28.04</td></tr><tr><td> $\overline { { G _ { s i m } + G _ { s p e } } }$ </td><td>0.4072</td><td>0.00</td><td>119.13</td><td>26.67</td></tr><tr><td> $\overline { { G _ { s i m } + G _ { d u p } } }$ </td><td>0.4073</td><td>0.00</td><td>120.48</td><td>26.17</td></tr><tr><td> $\overline { { G _ { s p e } + G _ { d u p } } }$ </td><td>0.7396</td><td>31.62</td><td>87.10</td><td>27.18</td></tr><tr><td>Full</td><td>0.4066</td><td>0.00</td><td>99.12</td><td>26.98</td></tr></table>

Table 4. Ablation studies on text-conditional generation based on SSCD. Grey-colored font denotes areas of sacrifice.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Mem. by nL2Top5%↑  %&lt;1.4↓</td><td rowspan=1 colspan=1>FID ↓</td></tr><tr><td rowspan=1 colspan=1>Baseline [23]</td><td rowspan=1 colspan=1>1.58     0.93</td><td rowspan=1 colspan=1>7.44</td></tr><tr><td rowspan=1 colspan=1>w/o conditional guidance</td><td rowspan=1 colspan=1>1.49     0.00</td><td rowspan=1 colspan=1>257.27</td></tr><tr><td rowspan=1 colspan=1>w constant schedule</td><td rowspan=1 colspan=1>1.59     0.04</td><td rowspan=1 colspan=1>7.44</td></tr><tr><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>1.61     0.00</td><td rowspan=1 colspan=1>7.25</td></tr></table>

Table 5. Ablation studies on unconditional generation based on nL2. Grey-colored font denotes areas of sacrifice.

Unconditional and class-conditional generations on CIFAR-10. Tab. 2 and Tab. 3 illustrate AMG’s effectiveness in transitioning from text-conditional to classconditional or unconditional generation tasks, and from preventing object-level to pixel-level memorization. AMG consistently outperforms, even when compared to [4], who reported a 23% reduction in memorization by retraining diffusion models on a deduplicated CIFAR-10 dataset. AMG ensures memorization-free outputs and even slightly exceeds the original diffusion model’s quality, as reflected in FID scores, likely due to the increased diversity of its generated images compared to the original model’s replicated outputs. This success is attributed to two main factors: 1) AMG’s early memorization detection during reverse sampling, utilizing a conditional guidance with a parabolic schedule; 2) The absence of text-conditioning eliminates key memorization causes, like specific user prompts and caption duplication, enhancing output diversity. Solely using dissimilarity guidance in AMG can be very effective in preventing memorization whilst preserving output quality, since it only alters the coarse structure of images in early stages of reverse sampling when potential memorization is detected. Guidance ceases once memorization is no longer detected, thus preserving sample quality.

## 6.3. Ablation Studies

Table 4 underlines the crucial role of AMG’s tripartite guidance in optimizing the privacy-utility trade-off. Key observations include: 1) All AMG versions notably enhance memorization metrics, particularly those incorporating $G _ { s i m }$ , which eradicate memorization entirely with proper guidance scale tuning. This underscores $\vec { G } _ { s i m } \mathrm { \hat { s } }$ theoretical guarantee against memorization, albeit with slight impacts on quality and text-alignment. Ablation of $G _ { s i m }$ improves FID and CLIP scores but results in 31.62% of generated images being marked as memorized. Thus, $G _ { s i m }$ inclusion significantly boosts privacy with minimal utility loss. 2) Comparisons between the full AMG version and variants lacking either $G _ { d u p } \ : \mathrm { o r } \ : G _ { s p e }$ demonstrate that while achieving similar privacy levels, the ablated versions yield inferior FID and CLIP scores. This confirms the importance of both $G _ { d u p }$ and $G _ { s p e }$ in the guidance ensemble.

Table 5 highlights the efficacy of our conditional guidance with parabolic scheduling. Applying guidance indiscriminately during sampling, rather than selectively based on potential memorization, guarantees elimination of memorization but degrades output quality, evidenced by significantly higher FID scores. This is due to excessive alteration of both coarse structures (early sampling stages) and finer details (later stages). The parabolic schedule aligns with denoising stages: initially, predictions are highly noised and dissimilar to training images, becoming more accurate and revealing potential memorization cases with higher similarity as denoising progresses. This schedule enables early detection and effective resolution of memorization issues. A constant schedule would fail to provide this early detection, leading to 0.04% of generations being memorized, which could be eliminated by increasing the guidance scale but at the cost of quality. Therefore, our conditional guidance strategy enhances the privacy-utility trade-off by negating the need for this additional quality compromise.

## 7. Conclusion

We introduce AMG, a unified framework featuring three specialized guidance strategies, each addressing a specific cause of memorization in diffusion models. Theoretical analysis and empirical ablation studies confirm the essential role of each strategy in achieving an optimal privacy-utility trade-off. AMG’s strategic guidance scheduling and innovative automatic detection enable conditional application, further refining this balance. Our experiments demonstrate that AMG reliably generates images during inference that are distinct from memorized training images, maintaining high quality and text-alignment. Furthermore, AMG offers the flexibility to adapt to various user requirements by allowing customization in the type of memorization prevented (pixel-level or object-level) through adjustments in the similarity metrics employed in its guidance. Additionally, it provides options for guidance intensity (main or strong version) by adjusting the guidance scale, catering to a wide range of applications and user preferences.

## 8. Acknowledgement

This work was supported in part by the Australian Research Council under Projects DP210101859 and FT230100549.

## References

[1] Midjourney. https://www.midjourney.com/. 1

[2] Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pages 308–318, 2016. 2

[3] Romain Beaumont. Clip retrieval: Easily compute clip embeddings and build a clip retrieval system with them. https://github.com/rom1504/clipretrieval, 2022. 7

[4] Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne\` Ippolito, and Eric Wallace. Extracting training data from diffusion models. In USENIX Security Symposium, pages 5253–5270, 2023. 1, 2, 3, 4, 6, 7, 8

[5] Chen Chen, Daochang Liu, Siqi Ma, Surya Nepal, and Chang Xu. Private image generation with dual-purpose auxiliary classifier. In CVPR, 2023. 2

[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255. IEEE, 2009. 7

[7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, pages 8780–8794, 2021. 1, 3

[8] Tim Dockhorn, Tianshi Cao, Arash Vahdat, and Karsten Kreis. Differentially Private Diffusion Models. Transactions on Machine Learning Research, 2023. 2

[9] Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4):211–407, 2014. 2

[10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014. 1

[11] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv:2207.12598, 2022. 1, 3

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, pages 6840–6851, 2020. 1, 3

[13] Nikhil Kandpal, Eric Wallace, and Colin Raffel. Deduplicating training data mitigates privacy risks in language models. In ICML, 2022. 2

[14] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017. 1

[15] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, pages 4401–4410, 2019.

[16] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In CVPR, pages 8110–8119, 2020. 1

[17] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 1

[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. 1

[19] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In ICCV, 2023. 1, 2, 4, 7

[20] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In ACL, pages 8424–8445, 2022. 2

[21] Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang. Dataset distillation via factorization. Advances in neural information processing systems, 35:1100– 1113, 2022. 3

[22] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 2

[23] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 1, 3, 7, 8

[24] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. 2021. 1

[25] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, pages 722–729, 2008. 2

[26] Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, and Matthijs Douze. A self-supervised descriptor for image copy detection. In CVPR, pages 14532–14542, 2022. 4

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 2

[28] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. 2022. 1

[29] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In ICML, pages 1530–1538, 2015. 1

[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 7, 8

[31] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1

[32] Joseph Saveri and Butterick Matthew. Stable diffusion litigation, 2023. 2023. 1

[33] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. 2022. 1

[34] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256– 2265, 2015. 1

[35] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In CVPR, pages 6048–6058, 2023. 1, 2, 4, 6, 7

[36] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. In NeurIPS, 2023. 1, 2, 4, 5, 6, 7

[37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 4

[38] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 3

[39] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A Efros. Dataset distillation. arXiv preprint arXiv:1811.10959, 2018. 3

[40] Liyang Xie, Kaixiang Lin, Shu Wang, Fei Wang, and Jiayu Zhou. Differentially private generative adversarial network. arXiv preprint arXiv:1802.06739, 2018. 2

[41] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. In arXiv preprint arXiv:1506.03365, 2015. 7