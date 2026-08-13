# Unleashing the Potential of SAM for Medical Adaptation via Hierarchical Decoding

Zhiheng Cheng<sup>1</sup> Qingyue Wei<sup>2</sup> Hongru Zhu<sup>3</sup> Yan Wang<sup>1</sup> Liangqiong Qu<sup>4</sup> Wei Shao<sup>5</sup> Yuyin Zhou<sup>6</sup>

<sup>1</sup>Shanghai Key Laboratory of Multidimensional Information Processing, East China Normal University <sup>2</sup>Stanford University <sup>3</sup>Johns Hopkins University <sup>4</sup>The University of Hong Kong <sup>5</sup>University of Florida <sup>6</sup>UC Santa Cruz

## Abstract

The Segment Anything Model (SAM) has garnered significant attentionfor its versatile segmentation abilities and intuitive prompt-based interface. However, its application in medical imaging presents challenges, requiring either substantial training costs and extensive medical datasetsfor full model fine-tuning or high-quality prompts for optimal performance. This paper introduces H-SAM: a prompt-free adaptation ofSAM tailoredfor efficientfine-tuning ofmedical images via a two-stage hierarchical decoding procedure. In the initial stage, H-SAM employs SAM’s original decoder to generate a prior probabilistic mask, guiding a more intricate decoding process in the second stage. Specifically, we propose two key designs: 1) A class-balanced, mask-guided self-attention mechanism addressing the unbalanced label distribution, enhancing image embedding; 2) A learnable mask cross-attention mechanism spatially modulating the interplay among different image regions based on the prior mask. Moreover, the inclusion of a hierarchical pixel decoder in H-SAM enhances its proficiency in capturingfine-grained and localized details. This approach enables SAM to effectively integrate learned medical priors, facilitating enhanced adaptation for medical image segmentation with limited samples. Our H-SAM demonstrates a 4.78% improvement in average Dice compared to existing prompt-free SAM variantsfor multi-organ segmentation using only 10% of 2D slices. Notably, without using any unlabeled data, H-SAM even outperforms state-of-the-art semisupervised models relying on extensive unlabeled training data across various medical datasets. Our code is available at https://github.com/Cccccczh404/H-SAM.

## 1. Introduction

Accurate delineation of tissues, organs, and regions of interest through medical image segmentation is pivotal in aiding medical professionals’ diagnostic precision and treatment planning processes [13, 15]. Furthermore, it plays a fundamental role in propelling disease research and discovery [53]. Nonetheless, a significant challenge in this field lies in the demand for deep learning models to undergo extensive training on large annotated datasets, a resource often challenging to procure within the medical domain.

![](images/587c3eda4dbfe0fd1f9addb1b9f8742608e332e0634768e5d25551daa7c9276a.jpg)  
Figure 1. H-SAM is advantageous in few-shot medical image segmentation. It achieves over 80% in average Dice using only 10% slices for multi-organ segmentation, outperforming existing prompt-free SAM adaptation methods. Without using any unlabeled data at all, it even outperforms state-of-the-art semisupervised models that use extensive unlabeled training data for prostate segmentation.

Recently, Segment Anything Model (SAM) [37], trained with over a billion masks from diverse natural images, demonstrates remarkable zero-shot learning capabilities. This breakthrough presents an avenue for significant advancements in medical image segmentation, especially considering the limited availability of extensive datasets in the medical realm. However, SAM’s performance on medical images diminishes notably in zero-shot settings, exhibiting reduced accuracy and robustness [25, 29, 35, 43, 50]. This decline can be attributed to SAM’s lack of exposure to medical images during training, as its extensive training revolves around natural images [32, 71].

While training SAM exclusively on medical datasets is a potential solution, it incurs substantial training costs and risks of over-fitting to single datasets [49]. Efforts to bridge the gap between medical and natural image domains involve adapting SAM to specific medical datasets [14, 23, 69]. Previous works primarily focus on inserting adapter layers into the image encoder with minimal decoder changes [23, 70]. Most of these efforts employ prompted SAM adaptation, generating prompts using point or bounding boxes from ground truth during testing [21, 64, 72]. However, creating accurate prompts demands domain knowledge from medical experts, which is often limited, time-consuming, and prone to noise, compromising segmentation accuracy. In response, prompt-free SAM adaptation methods [9, 56, 70] have emerged, yet they typically yield inferior results compared to prompted methods due to the lack of medical knowledge that prompts provide.

We present H-SAM, a prompt-free variant of the Segment Anything Model (SAM), aimed at integrating medical knowledge via a streamlined two-stage hierarchical mask decoder while maintaining the image encoder frozen. Initially, input images are processed by a LoRA-adapted image encoder. H-SAM employs SAM’s original lightweight mask decoder in the first stage to generate a prior probabilistic mask, guiding a more intricate second decoding stage. Two key designs underpin this process: 1) A classbalanced, mask-guided self-attention mechanism recalibrates the image embedding using self-attention from the prior mask, ensuring balanced representation across categories with noise augmentation. 2) A learnable mask cross-attention mechanism employs the prior mask to modulate cross-attention spatially within the subsequent Transformer decoder, attenuating less relevant background noise. Moreover, a hierarchical pixel decoder complements the hierarchical Transformer decoder, enhancing the model’s precision and ability to capture localized details. Figure 2 illustrates the overall pipeline.

In both fully-supervised and few-shot multi-organ segmentation tasks, our H-SAM surpasses existing prompt-free SAM variants. Specifically, using 10% and 100% of 2D slices, H-SAM achieves an average Dice score improvement of 4.78% and 3.48% on the Synapse dataset, respectively. This superior performance is also evident in other few-shot medical image segmentation tasks including the left atrial (LA) dataset and PROMISE2012 dataset. Notably, as illustrated in Figure 1, H-SAM excels without the need for any unlabeled data, surpassing state-of-theart semi-supervised models that rely on extensive unlabeled training datasets. The promising performance of 87.27% and 89.22% achieved on the prostate and left atrial segmentation using only 3 and 4 cases highlights H-SAM’s potential in medical imaging applications.

## 2. Related Work

Medical Foundation Models Foundation models are pretrained, large-scale models that allow for rapid customization through fine-tuning or in-context learning, as exemplified in [18, 55]. Despite these notable signs of progress, challenges persist in complex tasks like image segmentation, primarily due to the difficulty of obtaining annotated masks. Segment Anything Model (SAM) [37], with extensive training on a dataset of over 1 billion natural images, showcases impressive performance in image segmentation. Particularly, in diverse real-world scenarios, SAM demonstrates powerful capabilities for zero-shot generalization, signifying its potential to address intricate computer vision challenges.

With the surge of Segment Anything Model [37], previous work seeks to apply SAM on medical images [5, 16, 73]. However, empirical studies of SAM’s zero-shot capability on medical images [12, 17, 27, 60] reveal a significant decline when dealing with unseen medical features [52, 61]. Consequently, recent work explores effective adaptations of prompted SAM on medical datasets. Due to the computational intensity associated with training all parameters of SAM, researchers primarily concentrate on updating a subset of SAM’s parameters. Many works require prompts to generalize SAM on medical datasets. For instance, MedSAM [49] curates a large medical image dataset to adapt bound box prompted SAM. Medical SAM Adapter [64] fine-tunes point-prompted SAM using Adaption modules. Several studies transfer SAM from 2D to 3D by adding layers to support volumetric inputs [24, 39, 40, 46]. Other prompted SAM adaptations employ exemplar learning [21] and new prompt mechanisms [72]. Apart from prompted SAM adaptations, which require prompts sampled from ground truth during testing, prompt-free SAM adaptation methods are proposed for medical segmentation where prompts are not necessarily available. AutoSAM [30] fine-tunes SAM without prompts by freezing the SAM encoder and adding prediction heads to generate segmentation masks. SAMed [70] achieves competitive prompt-free results by introducing LoRA [28] layers into the image encoder while using the original decoder.

This study also introduces a prompt-free version of SAM, designed to enhance efficient finetuning with limited medical data. It differentiates from existing research by introducing a novel hierarchical decoding process that incorporates medical prior knowledge.

![](images/0090ae9142ee78f6feda04b8ed7ab7ba5f533b8431431b8f15eb5291248b9cfc.jpg)  
Figure 2. The H-SAM framework integrates a LoRA-adapted image encoder and a sophisticated 2-stage hierarchical decoder. We finetune the prompt encoder with default embeddings under a prompt-free setting. A key innovation lies in our hierarchical mask decoder, which strategically utilizes predictions from the stage-1 decoder as priors to achieve nuanced segmentation with 2 implementations: Class-Balanced Mask-Guided Self-Attention (CMAttn), and Learnable Mask Cross-Attention. And a hierarchical pixel decoder is employed to complement the enriched object queries derived from the transformer decoder.

Model Fine-Tuning Efficient fine-tuning of foundation models is crucial, with adapters becoming key for integrating new knowledge into pretrained models and providing a more efficient alternative to complete finetuning [10, 26]. Low Rank Adaptation (LoRA) [28] advocates for a gradual parameter update within transformer blocks, aiming for a low-rank approximation to refine large-scale models. In our proposed H-SAM, we incorporate LoRA adapters into SAM image encoder to avoid over-fitting in medical image dataset adaptation.

Medical Image Segmentation Medical image segmentation refers to partitioning or dividing a medical image into the dense prediction of pixels corresponding to lesions or organs in imaging modalities such as CT [22, 62, 74, 75] and MRI [36, 68]. With the rapid progress of deep learning, U-Net [58] with its elaborate network design ushers a new era for medical image segmentation. Building on U-Net’s foundation, a series of works emerge to enhance segmentation performance with U-shaped models [19, 33, 76]. Recent advancements of vision transformers in natural image analysis also prompt exploration in medical image segmentation [20]. A prevalent network design strategy involves integrating transformer blocks into the U-Net framework, resulting in novel architectures such as TransUnet [7] and Swin-Unet [4]. MISSFormer [31] is a u-shaped encoderdecoder network with enhanced transformer blocks. Recent works also explore medical image segmentation in a fewshot setting. RP-Net [59] proposed to iteratively capture context relationships using a U-shaped network. CAT-Net and 3D TransUNet [8, 44] design a cross masked attention

Transformer to focus only on foreground regions between the support image and query image.

Unlike existing studies, this study is focused on the finetuning of large foundational models, specifically SAM, to facilitate more efficient adaptation for medical image segmentation tasks. To achieve this, we introduce a hierarchical decoding strategy designed to optimize SAM’s capabilities, thereby unlocking its full potential for efficient and effective fine-tuning on medical tasks.

## 3. Methodology

## 3.1. H-SAM Overview

Given an image I of size W × H, our goal is to predict its corresponding segmentation map of $W \times H$ . Each pixel in this map is assigned to a category from a predefined class list, aiming for maximal alignment with the ground truth gt. Our segmentation framework H-SAM is built up upon SAM, integrating a LoRA-adapted image encoder and a simple but effective 2-stage hierarchical decoder.

LoRA-adapted Image Encoder As illustrated in Figure 2, H-SAM utilizes the original image encoder of SAM and freezes all layers to preserve pre-learned knowledge. Then, we adopt the same LoRA implementation as SAMed [70] to add a smaller, trainable bypass composed of two low-rank matrices. In line with LoRA, these bypasses first compress the transformer features into a lowrank space. Subsequently, they reproject these condensed features to match the output feature channels of the frozen transformer blocks. Only these bypass matrices are updated during training, allowing for minor yet effective model adjustments. For the prompt encoder, H-SAM does not need any prompt and simply updates a default embedding during training.

Mask Decoder The original SAM mask decoder consists of a Transformer decoder and a pixel decoder. The Transformer decoder processes image embeddings extracted from the image encoder, employing self-attention mechanisms to evaluate the significance of various image regions and cross-attention mechanisms to focus on relevant areas for segmentation. Subsequently, the pixel decoder refines this output, generating a detailed segmentation map, and assigning a class or category to each pixel.

Hierarchical Decoding Our H-SAM introduces a more intricate two-stage hierarchical decoding procedure. In the first stage, H-SAM employs SAM’s original decoder to create a prior (probabilistic) mask, which will be used to guide more intricate decoding in the second stage, as illustrated in Figure 2. This second stage mirrors the original one with both a Transformer decoder and a pixel decoder. To enhance image embedding input and optimize cross-attention in the second Transformer decoder, we introduce two novel modules. Firstly, a class-balanced, maskguided self-attention mechanism is proposed to rectify the issue of unbalanced label distribution, thereby enhancing the image embedding for the second-stage Transformer decoder (Sec. 3.2). Secondly, we incorporate a learnable mask cross-attention mechanism within the second Transformer decoder. This mechanism adeptly modulates the spatial dynamics among various image regions, guided by the information from the prior mask, thereby enhancing the segmentation process (Sec. 3.3). Collectively, these decoders constitute a hierarchical Transformer decoder framework. Further, we propose a hierarchical pixel decoder, inspired by U-Net architecture, to supplement the hierarchical Transformer decoder and further refine the segmentation outcome. Concretely, the pixel decoder in the second stage integrates features from the first-stage pixel decoder through skip connections, enabling the generation of high-resolution predictions (Sec. 3.4).

## 3.2. Enhanced Image Embedding via Class-Balanced Mask-Guided Self-Attention

As shown in Figure 3, we incorporate a Class-Balanced Mask-Guided Self-Attention (CMAttn) block to enhance the image embeddings as input for the second-stage Transformer decoder. This is particularly helpful in the case where we have an imbalance between the abundant instances in the head categories and the scarcity of instances in the tail categories. We use a mask feature which is acquired by directly multiplying image embedding without upsampling from the first decoder as the input mask feature for CMAttn. Before the self-attention block, we adopt a class-balanced augmentation to introduce more variations to tail categories. Inspired by previous approaches utilizing logit adjustment in long-tail problems [41], we perturb the mask feature with Gaussian noise whose variance is inversely proportional to category sample frequencies:

![](images/3eaf42c04381f341a5be0e02166602ce9f07770921ed0334f75e6ed3e6ad4b04.jpg)  
Figure 3. The illustration of Class-Balanced Mask-Guided Self-Attention (CMAttn) block.

$$
P ( g t = i ) \mathrm { + } = \mathcal { N } ( 0 , v a r ( i ) ) ,\tag{1}
$$

where $P \in \mathbb { R } ^ { N \times C \times H \times W }$ is the normalized input mask feature. gt is the ground truth mask resized to the same size. N is the added Gaussian noise. The variance list is calculated offline and stored as var.

After self-attention, we adopt a linear layer to compress the channel dimension, and incorporate the resulting mask feature to input image embeddings using Hadamard product ⊙. A residual path is designed to retain information from the initial image embedding.

## 3.3. Learnable Mask Cross Attention

Mask-attention is a variant of cross-attention first proposed in Mask2Former [11]. Unlike cross-attention which attends to the global context, mask-attention operates only on the area within the predicted mask. Original mask-attention adds a transformed binary mask to the cross attention operation via

$$
X = s o f t m a x ( t ( M ) + K Q ^ { T } ) V + X ,\tag{2}
$$

where X is the input query feature of the transformer block. $K , Q .$ , and V are the key, query and value in cross attention. t(M) is a function mapping binarized input {0, 1} to $\{ - \infty , 0 \}$ . This mask formulation has two limitations: (1) the gradient of mask M vanishes through t(M); (2) binarized mask M treats all foreground pixels without differentiation, limiting its ability to interpret further information from the mask prior.

![](images/dbefeab11030391ce53e7c9f011b4e79ff9c5e5e6413418743d0a416480ccf31.jpg)  
Figure 4. The illustration of learnable mask-attention.

To address these limitations, we propose to use learnable mask cross-attention in the second decoding stage as shown in Figure 4, which can be formulated as:

$$
X = M \odot s o f t m a x ( K Q ^ { T } ) V + X ,\tag{3}
$$

which employs an untransformed probabilistic map M resized to the same spatial resolution as the saliency map in cross attention. With element-wise product between the mask and the saliency map, masked regions will be ignored by multiplying a near-zero probability. This new formulation mitigates the aforementioned limitations and facilitates rapid convergence and better performance. Our learnable mask cross-attention within the second-stage Transformer decoder leverages more information from the probabilistic map and can assign varying degrees of importance to different foreground regions.

## 3.4. Hierarchical pixel decoder

Complementing the Transformer decoder, SAM’s original pixel decoder directly upsamples the image embedding into a segmentation map of $H / 4 \times W / 4$ . We argue that this resolution is not capable of capturing some intricate local details and small-scale medical objects in medical image segmentation necessitates the use of U-shaped networks with multiple skip connections. To enhance the details in the segmentation output, we adopt U-shaped architectures only in pixel decoders with skip connections for effectively handling multi-scale objects in medical images with acceptable computation cost. Within our H-SAM hierarchical decoding, the second-stage transformer decoder undergoes meticulously crafted mask-guided designs to propagate the prior mask from the first stage to the next stage (as shown in Sec. 3.2 and Sec. 3.3). Here we propose the hierarchical pixel decoder designed to complement the enriched object queries derived from the transformer decoder. Similar to the hierarchical Transformer decoder, the hierarchical pixel decoder also consists of two successive pixel decoders, strategically incorporates skip connections to integrate features from the first pixel decoder to the second, and further upsample the resolution from $H / 4 \times W / 4$ to the full resolution $H \times W$ . Benefiting from skip-connected localized features, the hierarchical pixel decoder will be supplemented with the Transformer decoder to output an enriched representation with enhanced resolution.

Training Loss The training loss combines pixel-wise classification loss and binary mask loss for each segmented prediction:

$$
\mathcal { L } = \lambda _ { c e } \mathcal { L } _ { c e } + \lambda _ { d i c e } \mathcal { L } _ { d i c e } ,\tag{4}
$$

where the pixel-wise classification loss $\mathcal { L } _ { c e }$ and $\mathcal { L } _ { d i c e }$ denote binary cross-entropy loss and dice loss, respectively [51]. For our 2-stage hierarchical structure, there is also a $\lambda _ { w }$ for each loss in the 2 stage. The final loss $\mathcal { L } _ { t o t a l }$ a sum of $\lambda _ { w } \mathcal { L } _ { s t a g e 1 }$ and $( 1 - \lambda _ { w } ) \mathcal { L } _ { s t a g e 2 }$ . The parameter $\lambda _ { w }$ is set to gradually decrease from 0.8 in a way of exponential decay, at a decay coefficient of 0.005. The first decoder output is supervised by 1/4 resolution ground truth, and the second output by full resolution. The final output is ensembled by taking the average of probabilities from the two outputs.

Deep Supervision The training loss will be applied to every stage in our hierarchical decoding procedure. The supervisory signals for the stage-1 mask decoder stem from ground truth masks of $H / 4 \times W / 4$ , while the stage-2 mask decoder is directly supervised using original high-resolution ground truth. This ensures thorough supervision and enhances the overall effectiveness of the model.

## 4. Experiments

## 4.1. Dataset and Evaluation

We conduct experiments of multi-organ semantic segmentation on three medical datasets, including Synapse multi-organ CT [38], the left atrial (LA) dataset [6], and PROMISE12 [45]. We utilize Dice coefficient and the average Hausdorff distance (HD) as evaluation metrics.

Synapse Multi-Organ CT. The Synapse dataset is from MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge, which contains 3779 axial contrast enhanced abdominal CT images in total. The training set contains 2212 axial slices. We strictly follow TransUnet [7] and SAMed [70] for dataset split and data preprocessing. The dataset is split into 18 training cases and 12 test cases. The CT volumes for Synapse dataset of each volume contain 85 to 198 slices. The resolution of Synapse dataset is 512×512 during fewshot training, and 224×224 during fully-supervised training. We evaluate eight abdominal organs (aorta, gallbladder, spleen, left kidney, right kidney, liver, pancreas, stomach), following TransUnet [7].

<table><tr><td>Training set</td><td>Method</td><td>Spleen</td><td>Right Kidney</td><td>Left Kidney</td><td>Gallbladder</td><td>Liver</td><td>Stomach</td><td>Aorta</td><td>Pancreas</td><td>Mean Dice↑ (%)</td><td>HD↓</td></tr><tr><td rowspan="4">10%</td><td>AutoSAM [30]</td><td>68.80</td><td>77.44</td><td>76.53</td><td>24.87</td><td>88.06</td><td>52.70</td><td>75.19</td><td>34.58</td><td>55.69</td><td>31.67</td></tr><tr><td>SAM Adapter [9]</td><td>72.42</td><td>68.38</td><td>66.77</td><td>22.38</td><td>89.69</td><td>53.15</td><td>66.74</td><td>26.76</td><td>58.28</td><td>54.22</td></tr><tr><td>SAMed [70]</td><td>85.82</td><td>82.25</td><td>82.62</td><td>63.15</td><td>92.72</td><td>67.20</td><td>78.72</td><td>52.12</td><td>75.57</td><td>23.02</td></tr><tr><td>H-SAM (ours)</td><td>90.21</td><td>84.16</td><td>85.65</td><td>70.70</td><td>94.29</td><td>76.10</td><td>85.54</td><td>56.17</td><td>80.35</td><td>15.54</td></tr><tr><td rowspan="9">Fully Supervised</td><td>TransUnet [7]</td><td>87.23</td><td>63.13</td><td>81.87</td><td>77.02</td><td>94.08</td><td>55.86</td><td>85.08</td><td>75.62</td><td>77.48</td><td>31.69</td></tr><tr><td>SwinUnet [4]</td><td>85.47</td><td>66.53</td><td>83.28</td><td>79.61</td><td>94.29</td><td>56.58</td><td>90.66</td><td>76.60</td><td>79.13</td><td>21.55</td></tr><tr><td>TransDeepLab [1]</td><td>86.04</td><td>69.16</td><td>84.08</td><td>79.88</td><td>93.53</td><td>61.19</td><td>89.00</td><td>78.40</td><td>80.16</td><td>21.25</td></tr><tr><td>DAE-Former [2]</td><td>88.96</td><td>72.30</td><td>86.08</td><td>80.88</td><td>94.98</td><td>65.12</td><td>91.94</td><td>79.19</td><td>82.43</td><td>17.46</td></tr><tr><td>MERIT [57]</td><td>92.01</td><td>84.85</td><td>87.79</td><td>74.40</td><td>95.26</td><td>85.38</td><td>87.71</td><td>71.81</td><td>84.90</td><td>13.22</td></tr><tr><td>AutoSAM [30]</td><td>80.54</td><td>80.02</td><td>79.60</td><td>41.37</td><td>89.24</td><td>61.14</td><td>82.56</td><td>44.22</td><td>62.08</td><td>27.56</td></tr><tr><td>SAM Adapter [9]</td><td>83.68</td><td>79.00</td><td>79.02</td><td>57.49</td><td>92.67</td><td>69.48</td><td>77.93</td><td>43.07</td><td>72.80</td><td>33.08</td></tr><tr><td>SAMed [70]</td><td>87.77</td><td>69.11</td><td>80.45</td><td>79.95</td><td>94.80</td><td>72.17</td><td>88.72</td><td>82.06</td><td>81.88</td><td>20.64</td></tr><tr><td>H-SAM (ours)</td><td>93.34(8.22)</td><td>89.93(9.59)</td><td>91.88(8.25)</td><td>73.49(23.01)</td><td>95.72(1.49)</td><td>87.10(7.14)</td><td>89.38(2.82)</td><td>71.11(10.27)</td><td>86.49</td><td>8.18</td></tr></table>

Table 1. Comparison to state-of-the-art models on Synapse multi-organ CT dataset with both few-shot and fully-supervised settings. Our model shows outstanding results in both of the training settings. The value in (·) is standard deviation.

LA. The left atrial (LA) dataset is from 2018 Atrial Segmentation Challenge [6]. We strictly follow UA-MT [67] and BCP [3] for data split and data preprocess. Specifically, LA dataset is split into 80 scans for training and 20 scans for evaluation. And we keep 4(5%) scans as labeled data while the rest scans in the training set are treated as unlabeled data for these two settings respectively. And we resize each 2D slice into 512×512 during training. Note that we do not use the unlabeled data for training H-SAM. Instead, we only use these selected 4 labeled scans to train our model.

PROMISE12. PROMISE2012 dataset is from the Prostate MR Image Segmentation 2012 [45]. It contains 50 3D transversal T2-weighted MR images of the prostate with manual binary prostate gland segmentation and is obtained from multiple centers with different acquisition protocols. During the experiments, we strictly follow MLB-Seg [63] for data split and data preprocessing. Specifically, we split it into 40 / 10 cases for training / evaluation. 3 out of 40 are selected as data with labels while the rest 37 scans are used as unlabeled data. The resolution of PROMISE12 dataset is 512×512. Again, note that only the 3 labeled cases are used for training H-SAM.

## 4.2. Implementation details

All our implementation is in PyTorch and we train all our models on 4 NVIDIA RTX A5000 GPUs. During training, we adopt a data augmentation combination of elastic deformation, rotation, and scaling. The training loss is a combination of Cross-Entropy loss and Dice loss. We adopt the same LoRA settings as SAMed, in which the rank of LoRA is set to 4. A ViT-B and a ViT-L backbone are adopted separately for few-shot and fully-supervised training. For a fair comparison, we utilize the same resolution of 224×224 for fully supervised training on Synapse as other SAM variants and SOTA methods. The maximal training epoch is set to 300. The optimizer algorithm used for updating is based on the AdamW, and β1, β2, and weight decay are set to 0.9, 0.999, and 0.1.

## 4.3. Results

Synapse Multi-Organ CT In Table 1, H-SAM shows outstanding few-shot transferability with limited seen medical images (10%). Compared to other SAM promptfree variants: Auto SAM [30], SAM Adapter [9] and SAMed [70], our H-SAM reports results of 80.35% with limited training scans equal to only 1 volume, which outperforms other SAM adaptation variants by a large margin (≈5%). We also evaluate our H-SAM under a fullysupervised setting on Synapse multi-organ CT dataset. We evaluate our model in a fair comparison with the promptfree setting with several state-of-the-art methods, including TransUnet [7], SwinUnet [4], TransDeepLab [1], DAE-Former [2] and MERIT[57], along with other SAM promptfree variants: Auto SAM [30], SAM Adapter [9] and SAMed [70]. Our method achieves promising results in multi-organ segmentation with an 86.49% Mean Dice Coefficient, which is higher than newly released medical segmentation networks DAE-Former (82.43%) and MERIT (84.90%). H-SAM also easily outperforms other promptfree SAM variants (86.24% vs. 81.88%).

LA As shown in Table 2, we conduct the few-shot semantic segmentation experiment on LA dataset. Here, we present results using 4 labeled scans (5% of the dataset) for training. We compare our method with two categories of approaches: 1) SAM efficient adaptation methods including AutoSAM [30], SAM Adapter, and SAMed [70] and 2) semi-supervised methods such as UA-MT [67], MC-Net [65], SS-Net [66], and BCP [3]. We also compare our results with nnUnet [34], a well-known baseline for medical image segmentation. To ensure a fair comparison, we adhered to the data split protocols established in UA-MT [67] and BCP [3], employing identical sets of 4 labeled scans for our few-shot experiments. The semisupervised methods also utilized the remaining 76 unlabeled scans during their training. And all the methods were evaluated on the same test dataset. Our proposed H-SAM method outperforms both the SAM efficient adaptation and semi-supervised methods under both split settings. This underscores the ability of H-SAM to achieve superior or comparable results with significantly less data compared to semi-supervised methods. Under the few-shot setting, H-SAM also demonstrates superior improvements in dice coefficients compared to SAMed [70] (89.22% vs. 87.72%), indicating the effectiveness of H-SAM in few-shot learning.

<table><tr><td>Methods</td><td colspan="2">Scans used Labeled Unlabeled</td><td>Mean Dice (%)↑</td></tr><tr><td>UA-MT [67]</td><td rowspan="6"></td><td rowspan="6">76(95%)</td><td>82.26</td></tr><tr><td>SASSNet [42]</td><td>81.60</td></tr><tr><td>DTC [47] 4(5%)</td><td>81.25</td></tr><tr><td>URPC [48]</td><td>82.48</td></tr><tr><td>MC-Net [65]</td><td>83.59</td></tr><tr><td>SS-Net [66]</td><td>86.33</td></tr><tr><td>BCP [3]</td><td rowspan="6">4(5%)</td><td rowspan="6">0(0%)</td><td>88.02</td></tr><tr><td>nnUnet [34]</td><td>64.02</td></tr><tr><td>AutoSAM [30]</td><td>74.73</td></tr><tr><td>SAM Adapter [70]</td><td>82.79</td></tr><tr><td>SAMed [70]</td><td>87.72</td></tr><tr><td>Ours</td><td>89.22</td></tr></table>

Table 2. Results of LA dataset under semi-supervision using 4 labeled scans.

PROMISE12 We also conducted the few-shot semantic segmentation experiment on PROMISE12 dataset. Similar to the LA dataset, we compared H-SAM against SAM efficient adaptation methods (AutoSAM [30], SAM Adapter [9], and SAMed [70]), nnUnet [34], and semisupervised methods (UA-MT [67], MC-Net [65], SS-Net [66], and MLB-Seg [63]). For a fair comparison with the semi-supervised methods, we strictly follow MLB-Seg [63] and use the same 3 labeled cases as our training set under the few-shot setting, while the semi-supervised methods incorporated an additional 37 unlabeled cases. And all the methods are tested on the same test dataset. As shown in Table 3, H-SAM achieved a significant improvement in Dice coefficients (approximately 10.94%) over the semisupervised method MLB-Seg, despite being trained on only three labeled cases. This result highlights the efficiency of H-SAM in leveraging limited labeled data. Compared to the SAM efficient adaptation method SAMed [70], our method also demonstrated superior performance (87.27% vs. 86.00%), further establishing its effectiveness in fewshot semantic segmentation tasks.

<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Scans usedLabeled  Unlabeled</td><td rowspan=1 colspan=1>Mean Dice (%)↑</td></tr><tr><td rowspan=1 colspan=1>UA-MT [67]DTC [47]SASSNet [42]MC-Net [65]SS-Net [66]Self-Paced [54]MLB-Seg [63]</td><td rowspan=1 colspan=1>3(7.5%) 37(92.5%)</td><td rowspan=1 colspan=1>65.0563.4473.4372.6673.1974.0278.27</td></tr><tr><td rowspan=1 colspan=1>nnUnet [34]AutoSAM [30]SAM Adapter [9]SAMed [70]Ours</td><td rowspan=1 colspan=1>3(7.5%)    0(0%)</td><td rowspan=1 colspan=1>84.2268.4075.4586.0087.27</td></tr></table>

Table 3. Results of PROMISE12 dataset under semi-supervision.

<table><tr><td>Learnable Mask-Attention</td><td>Hierarchical Pixel Decoder</td><td>CM Self-Attention</td><td>Mean Dice (%)</td></tr><tr><td>X</td><td>X</td><td>X</td><td>75.57</td></tr><tr><td>√</td><td>X</td><td>X</td><td>77.68</td></tr><tr><td>√</td><td>√</td><td>X</td><td>78.58</td></tr><tr><td>X</td><td>√</td><td>X</td><td>77.05</td></tr><tr><td>X</td><td>√</td><td>√</td><td>79.03</td></tr><tr><td>X</td><td>X</td><td>√</td><td>77.71</td></tr><tr><td>√</td><td>X</td><td>√</td><td>78.76</td></tr><tr><td>√</td><td>√</td><td>√</td><td>80.35</td></tr></table>

Table 4. Effectiveness of the key contributions in H-SAM: Learnable Mask-Attention, CMAttn and Hierarchical Pixel Decoder.

<table><tr><td>Methods</td><td>Mean Dice (%)</td></tr><tr><td>w/o. mask-attention</td><td>75.57</td></tr><tr><td>original mask-attention</td><td>75.61</td></tr><tr><td>learnable mask-attention (ours)</td><td>77.68</td></tr></table>

Table 5. Effectiveness of our learnable mask-attention against original mask-attention and baseline.

## 4.4. Ablation study

Effectiveness of Learnable Mask Cross Attention As shown in Table 4, our learnable mask attention improves the baseline by 2.1%. Within the hierarchical decoding structure of H-SAM, our learnable mask cross attention shows no signal of performance saturation with other two key contributions: CMAttn and Hierarchical Pixel Decoder. We also compare learnable mask cross attention against normal cross attention and unlearnable mask attention in Table 5. Note that, unlearnable mask attention operation brings little improvement due to the lack of gradient backpropagation. Inversely, our proposed learnable mask cross attention brings an immediate performance increase of 2.1% with jointing training and inheritance of mask-guided prior.

Effectiveness of CMAttn In Table 4, we present the ablation of Class-Balanced Mask-Guided Self-Attention (CMAttn). The CMAttn alone brings a 1.2% improvement to the baseline, showing that SAM benefits from a more informed image embedding as input for the mask decoder. Combined with Learnable Mask-Attention, the two mask-guided implementations improve the baseline model by 3.2% in terms of Mean Dice.

Effectiveness of Hierarchical Pixel Decoder In Table 4, the hierarchical pixel decoder promotes the dice coefficient by 2.2%. Then we evaluate the effectiveness of a combination of mask-guided implementations with the hierarchical pixel decoder by adding or disabling one of them each time. Table 4 shows an improvement of 3.0% and 3.5%, respectively combined with Learnable Mask-Attention and CMAttn.

## 4.5. Efficiency Analysis

To show that the advantages of our hierarchical decoding are not achieved by additional training cost in the decoding section, here we conduct ablation experiments between our model and other prompt-free SAM variants in terms of total parameters and performance. AutoSAM [30] is not listed in the table because it freezes the image encoder with no adapters. The original lightweight SAM mask decoder consists of only 2 transformer layers. SAMed [70] adds LoRA adapter in the image encoder, while remaining a default SAM mask decoder with 2 transformer layers. SAM Adapter [9] injects adapter layers in both the image encoder and mask decoder. In general, the training cost of our H-SAM is equal to SAMed with 1 additional lightweight mask decoder. For a fair comparison, we double and dribble the number of SAMed transformer layers into 4 and 6, to achieve a comparable and even larger parameter scale than the training cost in our H-SAM. As shown in Table 6, SAMed [70] benefits from increased transformer layers. However, the performance promotion comes with huge computation costs. H-SAM consistently outperforms SAMed. Compared with SAM Adapter, H-SAM shows a 7.5% performance improvement with a 20M lesser parameter scale. The results prove the superiority of our H-SAM in wisely generating finer medical segmentation with little extra computation cost.

## 4.6. Qualitative Results

As shown in Figure 5, we compare our H-SAM to other prompt-free medical SAM variants, including AutoSAM [9], SAM Adapter [30] and SAMed [70]. As pointed out in the first row and the last row, compared to other SAM variants, H-SAM provides a precise mask prediction with lesser noise. In the second row, where other methods misrecognize Aorta and Pancreas to Stomach and

<table><tr><td>Methods</td><td>Transformer Layers</td><td>Total Parameter</td><td>Mean Dice (%)</td></tr><tr><td rowspan="3">SAMed [70]</td><td>2</td><td>108.8M</td><td>75.57</td></tr><tr><td>4</td><td>112.5M</td><td>76.80</td></tr><tr><td>6</td><td>116.2M</td><td>78.05</td></tr><tr><td>SAM Adapter [9]</td><td>2</td><td>131.5M</td><td>72.80</td></tr><tr><td>H-SAM(ours)</td><td>4</td><td>112.3M</td><td>80.35</td></tr></table>

Table 6. Efficiency analysis of our H-SAM against deeper default SAM mask decoder. H-SAM shows better performance with fewer parameters.

![](images/ea2f9d0ecc92a2c0bb86d37e4c8820bf6666be74681c3cf1fbc007dd49480c2b.jpg)  
Figure 5. The qualitative results of H-SAM and other SAM variants, including SAMed, SAM Adapter, and AutoSAM.

Aorta, H-SAM provides correctly attributes each organ to their categories. H-SAM also performs superiorly with small-scale organs. In the third row, while all other variants miss Spleen, only H-SAM provides correct prediction for all of the organs.

## 5. Conclusion

We present H-SAM, which is a simple and efficient hierarchical mask decoder for adaptation of Segment Anything Model on medical image segmentation. Using a probabilistic map from a default decoder as prior to guide finer medical segmentation in the sequential decoding unit, our H-SAM puts forward a new direction of SAM adaptation. Notably, H-SAM achieves this superior performance without relying on any unlabeled data, surpassing even state-ofthe-art semi-supervised models that use extensive unlabeled datasets in various medical imaging contexts. This underscores H-SAM’s significant potential in advancing the field of medical image segmentation, offering a robust, efficient, and data-economic solution.

## References

[1] Reza Azad, Moein Heidari, Moein Shariatnia, Ehsan Khodapanah Aghdam, Sanaz Karimijafarbigloo, Ehsan Adeli, and Dorit Merhof. Transdeeplab: Convolution-free transformerbased deeplab v3+ for medical image segmentation. In International Workshop on PRedictive Intelligence In MEdicine, pages 91–102. Springer, 2022. 6

[2] Reza Azad, Rene Arimond, Ehsan Khodapanah Aghdam,´ Amirhossein Kazerouni, and Dorit Merhof. Dae-former: Dual attention-guided efficient transformer for medical image segmentation. In International Workshop on PRedictive Intelligence In MEdicine, pages 83–95. Springer, 2023. 6

[3] Yunhao Bai, Duowen Chen, Qingli Li, Wei Shen, and Yan Wang. Bidirectional copy-paste for semi-supervised medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11514–11524, 2023. 6, 7

[4] Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, and Manning Wang. Swin-unet: Unet-like pure transformer for medical image segmentation. In European conference on computer vision, pages 205–218. Springer, 2022. 3, 6

[5] Shurong Chai, Rahul Kumar Jain, Shiyu Teng, Jiaqing Liu, Yinhao Li, Tomoko Tateyama, and Yen-wei Chen. Ladder fine-tuning approach for sam integrating complementary network. arXiv preprint arXiv:2306.12737, 2023. 2

[6] Chen Chen, Wenjia Bai, and Daniel Rueckert. Multi-task learning for left atrial segmentation on ge-mri. In Statistical Atlases and Computational Models ofthe Heart. Atrial Segmentation and LV Quantification Challenges: 9th International Workshop, STACOM 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers 9, pages 292–301. Springer, 2019. 5, 6

[7] Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L Yuille, and Yuyin Zhou. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021. 3, 5, 6

[8] Jieneng Chen, Jieru Mei, Xianhang Li, Yongyi Lu, Qihang Yu, Qingyue Wei, Xiangde Luo, Yutong Xie, Ehsan Adeli, Yan Wang, et al. 3d transunet: Advancing medical image segmentation through vision transformers. arXiv preprint arXiv:2310.07781, 2023. 3

[9] Tianrun Chen, Lanyun Zhu, Chaotao Ding, Runlong Cao, Shangzhan Zhang, Yan Wang, Zejian Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam fails to segment anything? – samadapter: Adapting sam in underperformed scenes: Camouflage, shadow, and more, 2023. 2, 6, 7, 8

[10] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. arXiv preprint arXiv:2205.08534, 2022. 3

[11] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceed-

ings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 4

[12] Dongjie Cheng, Ziyuan Qin, Zekun Jiang, Shaoting Zhang, Qicheng Lao, and Kang Li. Sam on medical images: A comprehensive study on three prompt modes. arXiv preprint arXiv:2305.00035, 2023. 2

[13] Junlong Cheng, Shengwei Tian, Long Yu, Chengrui Gao, Xiaojing Kang, Xiang Ma, Weidong Wu, Shijia Liu, and Hongchun Lu. Resganet: Residual group attention network for medical image classification and segmentation. Medical Image Analysis, 76:102313, 2022. 1

[14] Can Cui, Ruining Deng, Quan Liu, Tianyuan Yao, Shunxing Bao, Lucas W Remedios, Yucheng Tang, and Yuankai Huo. All-in-sam: from weak annotation to pixel-wise nuclei segmentation with prompt-based finetuning. arXiv preprint arXiv:2307.00290, 2023. 2

[15] Jeffrey De Fauw, Joseph R Ledsam, Bernardino Romera-Paredes, Stanislav Nikolov, Nenad Tomasev, Sam Blackwell, Harry Askham, Xavier Glorot, Brendan O’Donoghue, Daniel Visentin, et al. Clinically applicable deep learning for diagnosis and referral in retinal disease. Nature medicine, 24 (9):1342–1350, 2018. 1

[16] Guoyao Deng, Ke Zou, Kai Ren, Meng Wang, Xuedong Yuan, Sancong Ying, and Huazhu Fu. Sam-u: Multi-box prompts triggered uncertainty estimation for reliable sam in medical image. arXiv preprint arXiv:2307.04973, 2023. 2

[17] Ruining Deng, Can Cui, Quan Liu, Tianyuan Yao, Lucas W Remedios, Shunxing Bao, Bennett A Landman, Lee E Wheless, Lori A Coburn, Keith T Wilson, et al. Segment anything model (sam) for digital pathology: Assess zeroshot segmentation on whole slide imaging. arXiv preprint arXiv:2304.04155, 2023. 2

[18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2

[19] Foivos I Diakogiannis, Franc¸ois Waldner, Peter Caccetta, and Chen Wu. Resunet-a: A deep learning framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing, 162:94–114, 2020. 3

[20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3

[21] Weijia Feng, Lingting Zhu, and Lequan Yu. Cheap lunch for medical image segmentation by fine-tuning sam on few exemplars. arXiv preprint arXiv:2308.14133, 2023. 2

[22] Yabo Fu, Yang Lei, Tonghe Wang, Walter J Curran, Tian Liu, and Xiaofeng Yang. A review of deep learning based methods for medical image multi-organ segmentation. Physica Medica, 85:107–122, 2021. 3

[23] Yifan Gao, Wei Xia, Dingdu Hu, and Xin Gao. Desam: Decoupling segment anything model for generalizable medical image segmentation. arXiv preprint arXiv:2306.00499, 2023. 2

[24] Shizhan Gong, Yuan Zhong, Wenao Ma, Jinpeng Li, Zhao Wang, Jingyang Zhang, Pheng-Ann Heng, and Qi Dou. 3dsam-adapter: Holistic adaptation of sam from 2d to 3d for promptable medical image segmentation. arXiv preprint arXiv:2306.13465, 2023. 2

[25] Sheng He, Rina Bao, Jingpeng Li, P Ellen Grant, and Yangming Ou. Accuracy of segment-anything model (sam) in medical image segmentation tasks. arXiv preprint arXiv:2304.09324, 2023. 2

[26] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019. 3

[27] Chuanfei Hu and Xinde Li. When sam meets medical images: An investigation of segment anything model (sam) on multi-phase liver tumor segmentation. arXiv preprint arXiv:2304.08506, 2023. 2

[28] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 3

[29] Mingzhe Hu, Yuheng Li, and Xiaofeng Yang. Skinsam: Empowering skin cancer segmentation with segment anything model. arXiv preprint arXiv:2304.13973, 2023. 2

[30] Xinrong Hu, Xiaowei Xu, and Yiyu Shi. How to efficiently adapt large segmentation model (sam) to medical images. arXiv preprint arXiv:2306.13731, 2023. 2, 6, 7, 8

[31] Xiaohong Huang, Zhifang Deng, Dandan Li, and Xueguang Yuan. Missformer: An effective medical image segmentation transformer. arXiv preprint arXiv:2109.07162, 2021. 3

[32] Yuhao Huang, Xin Yang, Lian Liu, Han Zhou, Ao Chang, Xinrui Zhou, Rusi Chen, Junxuan Yu, Jiongquan Chen, Chaoyu Chen, et al. Segment anything model for medical images? arXiv preprint arXiv:2304.14660, 2023. 2

[33] Fabian Isensee, Jens Petersen, Andre Klein, David Zimmerer, Paul F Jaeger, Simon Kohl, Jakob Wasserthal, Gregor Koehler, Tobias Norajitra, Sebastian Wirkert, et al. nnu-net: Self-adapting framework for u-net-based medical image segmentation. arXiv preprint arXiv:1809.10486, 2018. 3

[34] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021. 7

[35] Ge-Peng Ji, Deng-Ping Fan, Peng Xu, Ming-Ming Cheng, Bowen Zhou, and Luc Van Gool. Sam struggles in concealed scenes–empirical study on” segment anything”. arXiv preprint arXiv:2304.06022, 2023. 2

[36] Yuanfeng Ji, Haotian Bai, Chongjian Ge, Jie Yang, Ye Zhu, Ruimao Zhang, Zhen Li, Lingyan Zhanng, Wanling Ma, Xiang Wan, et al. Amos: A large-scale abdominal multiorgan benchmark for versatile medical image segmentation. Advances in Neural Information Processing Systems, 35: 36722–36732, 2022. 3

[37] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 1, 2

[38] Bennett Landman, Zhoubing Xu, J Igelsias, Martin Styner, T Langerak, and Arno Klein. Miccai multi-atlas labeling beyond the cranial vault–workshop and challenge. In Proc. MICCAI Multi-Atlas Labeling Beyond Cranial Vault—Workshop Challenge, page 12, 2015. 5

[39] Wenhui Lei, Xu Wei, Xiaofan Zhang, Kang Li, and Shaoting Zhang. Medlsam: Localize and segment anything model for 3d medical images. arXiv preprint arXiv:2306.14752, 2023. 2

[40] Chengyin Li, Prashant Khanduri, Yao Qiang, Rafi Ibn Sultan, Indrin Chetty, and Dongxiao Zhu. Auto-prompting sam for mobile friendly 3d medical image segmentation. arXiv preprint arXiv:2308.14936, 2023. 2

[41] Mengke Li, Yiu-ming Cheung, and Yang Lu. Long-tailed visual recognition via gaussian clouded logit adjustment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6929–6938, 2022. 4

[42] Shuailin Li, Chuyu Zhang, and Xuming He. Shape-aware semi-supervised 3d semantic segmentation for medical images. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part I 23, pages 552–561. Springer, 2020. 7

[43] Yuheng Li, Mingzhe Hu, and Xiaofeng Yang. Polypsam: Transfer sam for polyp segmentation. arXiv preprint arXiv:2305.00293, 2023. 2

[44] Yi Lin, Yufan Chen, Kwang-Ting Cheng, and Hao Chen. Few shot medical image segmentation with cross attention transformer. arXiv preprint arXiv:2303.13867, 2023. 3

[45] Geert Litjens, Robert Toth, Wendy Van De Ven, Caroline Hoeks, Sjoerd Kerkstra, Bram Van Ginneken, Graham Vincent, Gwenael Guillard, Neil Birbeck, Jindang Zhang, et al. Evaluation of prostate segmentation algorithms for mri: the promise12 challenge. Medical image analysis, 18(2):359– 373, 2014. 5, 6

[46] Yihao Liu, Jiaming Zhang, Zhangcong She, Amir Kheradmand, and Mehran Armand. Samm (segment any medical model): A 3d slicer integration to sam. arXiv preprint arXiv:2304.05622, 2023. 2

[47] Xiangde Luo, Jieneng Chen, Tao Song, and Guotai Wang. Semi-supervised medical image segmentation through dualtask consistency. In Proceedings of the AAAI conference on artificial intelligence, pages 8801–8809, 2021. 7

[48] Xiangde Luo, Wenjun Liao, Jieneng Chen, Tao Song, Yinan Chen, Shichuan Zhang, Nianyong Chen, Guotai Wang, and Shaoting Zhang. Efficient semi-supervised gross target volume of nasopharyngeal carcinoma segmentation via uncertainty rectified pyramid consistency. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part II 24, pages 318– 329. Springer, 2021. 7

[49] Jun Ma and Bo Wang. Segment anything in medical images. arXiv preprint arXiv:2304.12306, 2023. 2

[50] Maciej A Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, Nicholas Konz, and Yixin Zhang. Segment anything model for medical image analysis: an experimental study. Medical Image Analysis, 89:102918, 2023. 2

[51] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016. 5

[52] S Mohapatra, A Gosai, and G Schlaug. Sam vs bet: A comparative study for brain extraction and segmentation of magnetic resonance images using deep learning. arXiv preprint arXiv:2304.04738, 2:4, 2023. 2

[53] David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curtis P Langlotz, Paul A Heidenreich, Robert A Harrington, David H Liang, Euan A Ashley, et al. Video-based ai for beat-to-beat assessment of cardiac function. Nature, 580(7802):252–256, 2020. 1

[54] Jizong Peng, Ping Wang, Christian Desrosiers, and Marco Pedersoli. Self-paced contrastive learning for semisupervised medical image segmentation with meta-labels. Advances in Neural Information Processing Systems, 34: 16686–16699, 2021. 7

[55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2

[56] Md Mostafijur Rahman and Radu Marculescu. G-cascade: Efficient cascaded graph convolutional decoding for 2d medical image segmentation. arXiv preprint arXiv:2310.16175, 2023. 2

[57] Md Mostafijur Rahman and Radu Marculescu. Multi-scale hierarchical vision transformer with cascaded attention decoding for medical image segmentation. arXiv preprint arXiv:2303.16892, 2023. 6

[58] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 3

[59] Hao Tang, Xingwei Liu, Shanlin Sun, Xiangyi Yan, and Xiaohui Xie. Recurrent mask refinement for few-shot medical image segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3918–3928, 2021. 3

[60] Tassilo Wald, Saikat Roy, Gregor Koehler, Nico Disch, Maximilian Rouven Rokuss, Julius Holzschuh, David Zimmerer, and Klaus Maier-Hein. Sam. md: Zero-shot medical image segmentation capabilities of the segment anything model. In Medical Imaging with Deep Learning, short paper track, 2023. 2

[61] An Wang, Mobarakol Islam, Mengya Xu, Yang Zhang, and Hongliang Ren. Sam meets robotic surgery: An empirical study in robustness perspective. arXiv preprint arXiv:2304.14674, 2023. 2

[62] Yan Wang, Yuyin Zhou, Wei Shen, Seyoun Park, Elliot K Fishman, and Alan L Yuille. Abdominal multi-organ segmentation with organ-attention networks and statistical fusion. Medical image analysis, 55:88–102, 2019. 3

[63] Qingyue Wei, Lequan Yu, Xianhang Li, Wei Shao, Cihang Xie, Lei Xing, and Yuyin Zhou. Consistency-guided metalearning for bootstrapping semi-supervised medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 183–193. Springer, 2023. 6, 7

[64] Junde Wu, Rao Fu, Huihui Fang, Yuanpei Liu, Zhaowei Wang, Yanwu Xu, Yueming Jin, and Tal Arbel. Medical sam adapter: Adapting segment anything model for medical image segmentation. arXiv preprint arXiv:2304.12620, 2023. 2

[65] Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai, and Lei Zhang. Semi-supervised left atrium segmentation with mutual consistency training. In Medical Image Computing and ComputerAssisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27– October 1, 2021, Proceedings, Part II 24, pages 297–306. Springer, 2021. 6, 7

[66] Yicheng Wu, Zhonghua Wu, Qianyi Wu, Zongyuan Ge, and Jianfei Cai. Exploring smoothness and class-separation for semi-supervised medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 34–43. Springer, 2022. 6, 7

[67] Lequan Yu, Shujun Wang, Xiaomeng Li, Chi-Wing Fu, and Pheng-Ann Heng. Uncertainty-aware self-ensembling model for semi-supervised 3d left atrium segmentation. In Medical Image Computing and Computer Assisted Intervention– MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part II 22, pages 605–613. Springer, 2019. 6, 7

[68] Chenyi Zeng, Lin Gu, Zhenzhong Liu, and Shen Zhao. Review of deep learning approaches for the segmentation of multiple sclerosis lesions on brain mri. Frontiers in Neuroinformatics, 14:610967, 2020. 3

[69] Jingwei Zhang, Ke Ma, Saarthak Kapse, Joel Saltz, Maria Vakalopoulou, Prateek Prasanna, and Dimitris Samaras. Sam-path: A segment anything model for semantic segmentation in digital pathology. arXiv preprint arXiv:2307.09570, 2023. 2

[70] Kaidong Zhang and Dong Liu. Customized segment anything model for medical image segmentation. arXiv preprint arXiv:2304.13785, 2023. 2, 3, 5, 6, 7, 8

[71] Lian Zhang, Zhengliang Liu, Lu Zhang, Zihao Wu, Xiaowei Yu, Jason Holmes, Hongying Feng, Haixing Dai, Xiang Li, Quanzheng Li, et al. Segment anything model (sam) for radiation oncology. arXiv preprint arXiv:2306.11730, 2023. 2

[72] Yiming Zhang, Tianang Leng, Kun Han, and Xiaohui Xie. Self-sampling meta sam: Enhancing few-shot medical image segmentation with meta-learning. arXiv preprint arXiv:2308.16466, 2023. 2

[73] Yizhe Zhang, Tao Zhou, Peixian Liang, and Danny Z Chen. Input augmentation with sam: Boosting medical image segmentation with segmentation foundation model. arXiv preprint arXiv:2304.11332, 2023. 2

[74] Yuyin Zhou, Zhe Li, Song Bai, Chong Wang, Xinlei Chen, Mei Han, Elliot Fishman, and Alan L Yuille. Prior-aware

neural network for partially-supervised multi-organ segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10672–10681, 2019. 3

[75] Yuyin Zhou, Yan Wang, Peng Tang, Song Bai, Wei Shen, Elliot Fishman, and Alan Yuille. Semi-supervised 3d abdominal multi-organ segmentation via deep multi-planar cotraining. In 2019 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 121–140. IEEE, 2019. 3

[76] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. Unet++: A nested u-net architecture for medical image segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4, pages 3–11. Springer, 2018. 3