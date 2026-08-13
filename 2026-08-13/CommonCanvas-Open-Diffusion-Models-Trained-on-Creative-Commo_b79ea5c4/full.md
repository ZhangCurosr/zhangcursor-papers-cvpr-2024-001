# CommonCanvas: Open Diffusion Models Trained on Creative-Commons Images

Aaron Gokaslan<sup>1∗</sup>, A. Feder Cooper<sup>1,2</sup>, Jasmine Collins<sup>3</sup>, Landan Seguin<sup>3</sup>, Austin Jacobson<sup>3</sup>, Mihir Patel<sup>3</sup>, Jonathan Frankle<sup>3</sup>, Cory Stephenson<sup>3</sup>, Volodymyr Kuleshov<sup>1</sup> <sup>∗</sup>akg87@cornell.edu

<sup>1</sup>Cornell University <sup>2</sup>The GenLaw Center <sup>3</sup>Mosaic Research Databricks

an oil painting of a tall ship sailing through a field of wheat at sunset

![](images/e620be4e1df1ba79083a8d18786a473dbffdc2682656203d67f5f501c40ed4a2.jpg)  
Figure 1. We achieve comparable performance to public Stable Diffusion 2 (SD2), using entirely Creative-Commons images and a synthetic captioning approach that requires only <3% of the amount of the data used to train previous models. We include results for two CommonCanvas architectures, small (S) and large (L), and two CC-image datasets, commercial (C) and non-commercial (NC).

## Abstract

We train a set of open, text-to-image (T2I) diffusion models on a dataset ofcurated Creative-Commons-licensed (CC) images, which yields models that are competitive with Stable Diffusion 2 (SD2). This task presents two challenges: (1) high-resolution CC images lack the captions necessary to train T2I models; (2) CC images are relatively scarce. To address these challenges, we use an intuitive transfer learning technique to produce a set ofhigh-quality synthetic captions paired with our assembled CC images. We then develop a data- and compute-efficient training recipe that requires as little as 3% of the LAION data (i.e., roughly 70 million examples) needed to train existing SD2 models, but obtains the same quality. These results indicate that we have a sufficient number ofCC images (also roughly 70 million) for training high-quality models. Our recipe also implements a variety of optimizations that achieve 2.71× training speed-ups, enabling rapid model iteration. We leverage this recipe to train several high-quality T2I models, which we dub the CommonCanvas family. Our largest model achieves comparable performance to SD2 on human evaluation, even though we use a synthetically captioned CC-image dataset that is only <3% the size of LAIONfor training. We release our models, data, and code on GitHub.

## 1. Introduction

Most high-quality text-to-image (T2I) models are trained using large-scale, web-scraped datasets, like LAION-

2B [34]. Even though this is a very common practice, U.S. courts have yet to definitively rule if this is permissible under copyright law [15, 17, 24, 25, 69]. In response, recent work in ML has begun to investigate alternative methods of navigating copyright concerns in text generation [44], code completion [18, 57], and image generation [28]. Nevertheless, matching the performance of state-of-the-art models remains a challenge. In this work, we study the following natural question: is it possible to efficiently produce a high-quality T2I model by training only on Creative-Commons-licensed data?

We suggest a path forward, training a suite of T2I architectures using only open-licensed, Creative-Commons (CC) images (Figures 1 & 2). This task brings to light two significant challenges. The first problem is data incompleteness: almost all CC images lack the captions necessary to train a high-quality T2I model. The second is data scarcity: there are relatively few high-resolution CC images — roughly 70 million, compared to LAION-2B’s roughly 2 billion [30].

We address the data incompleteness problem by using a pre-trained BLIP-2 model [39] to produce high-quality, synthetic captions for a set of curated, open-licensed CC images. This is an intuitive transfer-learning solution: we leverage a powerful pre-trained generative model to produce synthetic labels for an unlabeled dataset, which we can then use to train a different multimodal generative model. To deal with data scarcity, we propose a data- and computeefficient training recipe that obtains the same quality as Stable Diffusion 2 (SD2) [64], but, perhaps surprisingly,

an image of elsa from frozen

(a) Prompt  
![](images/292b193d43503d18bfea0add41d799782a243866c149297d4ce43e1ddbf856d0.jpg)  
(b) SD2

![](images/a797edd3add6aef40ef3fe049daeb0701723b09432e16a6edbbde9806019228c.jpg)  
(c) CommonCanvas-S-C  
the lion king

(d) Prompt  
![](images/3f3b3d88594478470fbebaab7ca9e3ffad56001122610faf22c49d0300813d19.jpg)  
(e) SD2

![](images/fbdfc1e8ea4576b6443fc175bfa61b371161111d90051bdc0a386a6777e6b52c.jpg)  
(f) CommonCanvas  
Figure 2. Prompting with Disney concepts (a, d). SD2 generates a recognizable image of Elsa from Frozen (b) and an image with a misshapen Disney logo and characters resembling those from The Lion King (e); CommonCanvas-S-C (small, commercial) does not (c, f).

requires as little as 3% of the LAION-2B data (i.e., roughly 70 million examples) originally used to train SD2. We call this model SD2-90M. These results indicate that we have a sufficient number of CC images (also roughly 70 million) for training high-quality models. Our training recipe also implements a variety of optimizations that achieve 2.71× training speed-ups, enabling rapid model iteration.

The above methods enable us to create CommonCanvas, a suite of latent diffusion model (LDM) architectures trained on our curated dataset of CC images and synthetic captions, which we denote CommonCatalog. For one of our architectures, we swap SD2’s UNet for SDXL’s larger network to demonstrate how, even with less data, larger models do not overfit to this smaller dataset. Our largest model (CommonCanvas-L-NC) achieves performance comparable to SD2-90M on human evaluation of Parti Prompts [75], even though our CommonCatalog training dataset is 3% the size of LAION and has synthetically generated captions. Although this is a larger and more capable model architecture than SD2, we find it surprising and important that it is possible to train an SD2-quality model at all based on such a limited dataset with synthetic captions. This reveals a promising path forward for future research on highly capable, open T2I models. In summary, we:

• Curate CommonCatalog, a multimodal training dataset of roughly 70 million open-licensed CC images (Section 4) for which we synthesize a set of high-quality captions. We note that synthesizing training data using generative models is an increasingly common transferlearning technique, and we give it the shorthand name telephoning (Sections 3).

• Train CommonCanvas, a suite of LDM architectures trained on CommonCatalog. The largest of these models, CommonCanvas-L-NC, produces qualitative results that are competitive with public SD2 (Section 6). To make this analysis tractable, we implement training optimizations that achieve 2.71× speed-ups in training SD2-90M (Section 5).

• We will release our CommonCatalog dataset along with our trained CommonCanvas models at https: //github.com/mosaicml/diffusion/blob/ main/assets/common-canvas.md.

## 2. Preliminaries and Motivation

In this section, we present background on training the T2I Stable Diffusion model, which was originally trained on the web-scraped LAION-2B dataset. We then discuss copyright and reproducibility with respect to LAION datasets. This discussion motivates the creation of an alternative dataset composed of open-licensed CC images with synthetic captions, which we introduce in Section 4.

## 2.1. Text-to-image generative models

Text-to-image (T2I) generative models are neural networks trained on image-caption pairs. One family of T2I models is Stable Diffusion (SD) [53]: a latent diffusion model (LDM) that converts images to latent representations and back again using Variational Autoencoders (VAEs) [27], and which uses an iterative sampling procedure [63] to train an underlying UNet [54]. The architecture also includes a text encoder, such as the Contrastive Language-Image Pre-training (CLIP) model [49] — the original OpenAI CLIP [51] or its open-source counterpart, OpenCLIP [11, 22].

Stable Diffusion 2 (SD2)’s UNet has approximately 865 million trainable parameters; Stable Diffusion XL (SDXL) has 2.6 billion parameters and other advancements involving aspect ratio bucketing, micro-conditioning, and multiple text encoders and tokenizers. In terms of training data, SD models and OpenCLIP are both trained on subsets of the LAION-5B dataset [3, 59]. The exact training dataset for CLIP is unknown, but it is likely web-scraped data [51].

## 2.2. Copyright, reproducibility, & LAION datasets

LAION-5B is a dataset derived from a snapshot of the Common Crawl, a massive corpus of data scraped from the web. From this snapshot, the LAION organization curated pairs of image URLs and their corresponding alt-text captions for the intended use of training T2I and image-to-text (I2T) generative models [3, 59]. In practice, T2I models are typically trained on filtered subsets of the full LAION-5B dataset (e.g. LAION-2B [30]). Training T2I models on this dataset requires visiting the URLs and downloading the associated images. There are two elements of LAION datasets that are relevant to our work:

Copyright. The images associated with LAION datasets have unclear provenance: it is often not known what the original image sources are [34]. Although LAION datasets are released under the open MIT license, some experts note that it is unclear if this is sufficient to allow for training on the underlying images and captions, which often have their own copyrights [12, 19, 33–35]. Courts have not yet decided if training on these datasets is “fair use” — an important exception in copyright [33, 35, 38, 56, 62]. There are several copyright lawsuits for the alleged use of LAION-5B subsets to train generative models [1, 17, 24, 70, e.g.].

Reproducibility. Since LAION datasets only contain the image URLs, and not the images themselves, they are plagued with link rot [31].<sup>1</sup> When accessing LAION-5B, there is no guarantee the images still exist at their URLs, making it impossible to fully reproduce the dataset and opening up the possibility of data poisoning attacks [9]. A natural alternative is to not use LAION datasets for training. Instead, one could independently curate a dataset of CC-licensed images with known provenance that explicitly allow for copying, adaptation, and commercial use. As constituent images can be stored and distributed, this would also solve the link-rot problem, enabling greater reproducibility. (Further, LAION datasets are no longer public because they contain CSAM [6, 67].) We defer our discussion of sourcing CC-licensed images to Section 4, where we detail CommonCatalog: our new, open dataset. While CC images are an attractive alternative to LAION-5B, we note that CC images rarely contain the captions necessary to train T2I models. Therefore, we first need a method for captioning CC images.

## 3. Transfer Learning for Image Captioning

Our solution for handling the lack of captions in CC images is an intuitive type of transfer learning for producing high-quality synthetic labels. We describe this method, and note that there are various similar methods in prior literature on generative modeling. Altogether, these methods indicate that this type of transfer learning has become an increasingly common pattern: producing synthetic labels that later serve as inputs to training other generative models. We therefore give this method a shorthand name: telephoning.

## 3.1. Telephoning

Telephoning (Figure 3) proceeds in two steps. First, shown in Figure 3b, it takes inputs from a high-dimensional modality (e.g., images) and effectively performs a “lossy compression” to a (scarce) low-dimensional modality (e.g., shorttext captions). Second, shown in Figure 3d, it takes the “lossy compression” and decompresses back to the highdimensional modality. Because the intermediate compression step is “lossy,” the ultimate output often does not remotely resemble the original input, just like a game of telephone [43]. We derive the term telephoning from the above intuition and use it as shorthand to denote instances of transfer learning that solve data-scarcity problems in multimodal generative modeling.

![](images/558b64d3ac72bdc090a8dd1bf18ae2b41842197156dc97a3ecd9a0c6f3c018a0.jpg)  
Figure 3. (a) We use the LAION-400M-pre-trained, I2T BLIP-2 model to produce synthetic captions for our uncaptioned CC images (e.g., the Wikipedia CC-licensed image of Snoopy). The synthetic captions are “lossy compressions” of the input images (e.g., a black and white cartoon dog with black ears has no mention of Snoopy). (b) We compile the resulting synthetic image-caption pairs into CommonCatalog, which (c) we use to train our open, T2I CommonCanvas models. (d) When we supply “lossy” captions to a T2I model, like a game of telephone, it produces outputs that no longer resemble the original images (e.g., CommonCanvas produces an image that matches the caption, but does not look like Snoopy).

In this work, CC images are the high-dimensional inputs, and we use a pre-trained BLIP-2 model [39] for “lossy compression” to short-text captions (Figure 3a). Together, these CC-image-caption pairs comprise the CommonCatalog dataset (Section 4), which we use to train our CommonCanvas T2I models (Figure 3b). While BLIP-2 was pre-trained on LAION-400M [58], we emphasize that, for training CommonCanvas, we only ever have access to the captions — to the “lossy compressions” it produces. We never have direct access to LAION-400M or, importantly, anything that is similar to the images that BLIP-2 was trained on. Instead, we only have access to the mapping in the model, which, given an image input, produces “lossy” output text.

Telephoning & Copyright We defer to experts about fair use (Section 2.2) — namely, regarding models like BLIP-2, and LAION-5B’s images and alt-text captions. Generally, these experts seem to think that many cases will fall under fair use [33, 37, 56], especially when model outputs do not resemble their inputs (i.e., the use is “nonexpressive” or “non-consumptive” [12]). This is the case with our use of BLIP-2 to produce “lossy” captions.

Nevertheless, it is possible that BLIP-2 could produce captions that resemble those in its LAION training data. This might seem to present a copyright concern similar to those that others have expressed about T2I generations that resemble LAION images. However, according to the U.S. Copyright Office, short phrases (like captions) may often not be copyrightable: “short phrases” often contain “an insufficient amount of authorship” to meet the threshold for copyright protection [66]. So, even if hypothetically BLIP-2 were to regurgitate captions from LAION verbatim, according to legal experts [33], the copyright considerations are likely to be different than they are for generated images or generated long-form text. We defer to experts for more precise legal arguments, but note that this is another reason why we believe it is reasonable for us to rely on BLIP-2 for captioning our CC images.

## 3.2. Related work on telephoning

Our work aligns with the trend of using advanced generative models to address data scarcity. This is evident in various modalities, such as producing audio captions from image-text pairs [73] and text from audio [52]. Similar approaches have also been used to generate instruction-tuning datasets for both text and images [40, 42]. Concurrent work, e.g. LLaVA [42], has used visual question-answer models to augment existing caption datasets, such as the ones used in training DALLE·3 [4] and Chen et al. [10]. Our model is one of the first works to train on a dataset without any ground-truth captions, and one of the first to release our dataset along with a fully trained diffusion model. The caption upsampling approaches described in these other works could be used to further improve the captions of Common-Catalog in future work.

Captioning models have also been used to create descriptive captions to guide a diffusion model to create an image visually similar to a specific image. In concurrent work, SynthCap [7] generates a synthetic captioning dataset using a diffusion model to generate images from captions — the inverse of our problem statement. We coin the term telephoning to short-hand processes like these, which include our work and prior work, and which we believe will become more prevalent as generative-model capabilities advance.

## 4. A CC-Image, Synthetic-Caption Dataset

We now introduce our open dataset, CommonCatalog. First, we describe the collection and curation process for the open-licensed, CC images. This process brings to light two challenges: caption-data incompleteness and imagedata scarcity. To address the lack of CC captions, we show concretely how we use telephoning to produce high-quality synthetic captions to accompany our set of curated images. We investigate the topic of data scarcity in the next section, where we also discuss necessary systems-level training optimizations that enable efficient model iteration.

## 4.1. Sourcing licensed images for CommonCatalog

We focus on locating high-resolution Creative-Commons images that have open licenses. We began with the YFCC100M dataset, which consists of 100 million CClicensed images and multimedia files, as well as Flickr IDs linking to the original data [68]. The images in the dataset associated with the original paper exhibit two issues that make it ill-suited for direct use to train Stable Diffusion: they are low-resolution, and many of them have licenses that do not expressly allow for the distribution of derivative works — a use that is in unsettled copyright law in the context of model training [33].

We therefore re-scraped these images from Flickr, based on the IDs provided in the YFCC100M metadata. Our scraped images are of very high resolution (exceeding 4K), which makes them more suitable for T2I training. We exclude images with non-derivative (ND) licenses. The remaining images can be further divided into those that can be used for commercial (C) purposes and those that cannot (NC). As shown in Table 4, we accordingly construct two datasets, CommonCatalog-C and CommonCatalog-NC. We defer additional details about licenses to Appendix B.1.1, but emphasize that all of the included images have open licenses: individuals are free to use, adapt, and remix the images, so long as they attribute them. In total, CommonCatalog contains roughly 70 million images that can be used non-commercially, of which a approximately 25 million images can also be used commercially.

Directly sourcing CommonCatalog avoids some concerns (Section 2.2); however, it also comes with its own challenges. For one, CC images rarely have the alt-text captions necessary to train a T2I model like Stable Diffusion (Figure 4); those that do have associated text often just include the image title or a URL. For another, we could only find roughly 70 million usable CC images, which pales in comparison to the billions of images in LAION used to train SD2 (Section 5). We take each of these challenges in turn. First, in the next subsection, we show how we instantiate telephoning (Section 3) to produce high-quality, synthetic captions for CC images.

## 4.2. Synthesizing captions with telephoning

We compared several captioning models and chose the pretrained BLIP-2 OPT2.5B model for synthesizing Common-

Figure 4. CommonCatalog-C contains images licensed only for commercial use; -NC contains -C as well as images licensed for non-commercial use.
<table><tr><td>Dataset</td><td># Images</td><td>% Alt Text</td></tr><tr><td>CommonCatalog-C</td><td>26,232,417</td><td>30.76%</td></tr><tr><td>CommonCatalog-NC</td><td>67,015,331</td><td>31.22%</td></tr></table>

<table><tr><td></td><td>Source</td><td>Caption</td></tr><tr><td><img src="images/16dc1f27281a39473c17e13d83e9a625d872539e7aa4e6acc5e4e2cc6d2110ee.jpg"/></td><td>Alt-Text (LAION-2B)</td><td>Latest 1PC Transparent Gradient Color Voile Window Curtain</td></tr><tr><td></td><td>BLIP2-OPT- 2.7B</td><td>A living room with a white couch and curtains</td></tr></table>

Figure 5. Original vs. BLIP-2-generated captions for an image from LAION-2B. In this example. BLIP-2’s caption better aligns with what a human would write. See appendix for more examples.

Catalog’s captions [39], based on qualitative analysis and state-of-the-art performance on MS COCO. BLIP-2 consists of three components: a pre-trained, frozen (i.e., fixed) visual encoder, a learned transformer network that converts the visual embeddings into a text prompt, and a frozen large language model (LLM) that takes in the prompt. The only trainable variables in the transformers are between the frozen visual encoder and the frozen LLM layers.

Given a LAION-2B image as input, we found that the resulting BLIP-2 caption is often qualitatively more descriptive than the corresponding LAION-2B ground-truth alt-text caption. LAION-2B captions often contain product names, irrelevant details, or poor grammar and syntax (Figure 5). This finding is corroborated by Nguyen et al. [48], which quantitatively shows that (in terms of CLIP Score) BLIP-2 captions are higher quality than ground-truth captions, at the cost of caption diversity. Based on these preliminary results, we captioned all of the YFCC100M Creative-Commons images, which required about 1,120 GPU A100 hours. We center-cropped and resized all of the images to a maximum size of 512x512 pixels, since captioning images at native resolution would be very expensive. At training time for CommonCanvas models, we use the highresolutation images.

We release our commercial (CommonCatalog-C) and non-commercial (CommonCatalog-NC) CC-image and synthetic-caption datasets with associated data cards. As an evaluation set, we also release the BLIP-2 captions that we produced for the non-derivative (ND) CC images that we did not use for training.

## 5. Optimizations and Data-Scarcity Analysis

High-resolution CC images are indeed much less abundant than web-scraped images; however, it is unclear if this scarcity presents a problem for training. Prior work has not studied in depth how much data is actually necessary to train high-quality SD2 models. We set out to quantify this amount by training multiple SD2 models on differentlysized subsets of LAION-2B. However, training a single SD2 model, even with hundreds of GPUs, can take several days. So, to make our data scarcity analysis more tractable, we first implemented several efficiency optimizations.

![](images/0bdf5a6a5ce4d47f89097bafd3a35753853b65e846fae3b7d137c7233ce0b874.jpg)  
Figure 6. Cumulative effect of various speed-ups (totalling 2.71×) in our SD2 training pipeline evaluated on 128 A100s.

## 5.1. Software and hardware speed-ups

Stability AI reports an estimated 200,000 A100 hours to train SD2 [65]. Depending on hardware, a single SD2 training run could take anywhere from a few weeks to over a month. We sought out multiple avenues to reduce this training-time constraint. We applied Flash Attention [13] with the xFormers library [36], pre-computed VAE and text encoder latents over the entire training dataset, cast all GroupNorm [72] and LayerNorm [2] to float16 precision, and applied fully-sharded data parallelism (FSDP) to our training run. Finally we opted to only keep an exponential moving average of the weights for the final 3.5% of training. Altogether, we are able to achieve a 2.71X speedup in A100 hours over our SD2 baseline implementation.

We found that latent pre-computation helped the most at low resolutions, while FSDP also provided significant gains, especially at scale. The other optimizations helped reduce total memory usage, allowing us to increase the microbatch size for better hardware utilization. Figure 6 summarizes each of the proposed methods and the cumulative speedup that results from their application. Equipped with an optimized training setup, it is more feasible for us to study the effect of varying training-dataset size. More details can be found in Appendix D.

## 5.2. Investigating data scarcity

YFCC100M contains 100 million images, about 10% the size of the 1.1B LAION examples we could access (due to link rot) — about 5% of the original LAION-2B dataset. An interesting question remains: how much data is actually needed to train these diffusion models effectively; do we really need billions of images to get high-quality results?

To answer this question, we train multiple SD2 architectures on increasingly smaller, random subsets of data from our LAION-1.1B dataset: 1.1B, 90M, 10M, and 1M sample subsets. While human evaluation remains the gold standard for evaluating generative models, we use proposed automated metrics like Frechet-Inception Distance [21], Kernal Inception Distance [5] and caption-alignment metrics such as CLIP Score [20] (Section 6). We find that performance (FID and KID on MS COCO) does not degrade until training with as few as 1 million images; our models trained on 10M and 90M subsets perform comparably to the entire 1.1B dataset (Appendix Figure 16). Figure 7 further compares our SD2 variants trained on 10M and 90M LAION subsets across different guidance scales. We also plot the effect of using the original LAION captions vs. BLIP-2 synthetic captions at these size regimes (discussed further in Section 6.1). These findings suggest that SD2 models may be underparameterized. We hypothesize about why this might be the case and how much data is actually necessary to saturate the model in the appendix.

![](images/cdac616bbb1da00ac076a6a073eacbe62af4d95b79aab2ca4ab81eebadeb4924.jpg)

![](images/f74b4cfea086108f65f0600d6176d0a32aa48ba148e0bbe2bc3152a4d1973c5b.jpg)

![](images/144f86e55df9565aeee185db1c45d8aa49554258e188cfe6e51966cd2ebb975d.jpg)  
Figure 7. For different SD2 models trained on subsets of LAION (90M, 10M using either original captions or synthetic BLIP-2 captions), we compute FID [21], KID [5], CLIP-FID [29], and CLIP-Score [20] on 30K samples from MS COCO. We compute these metrics across a text-guidance scale of 1-8, with higher values indicating the model should respect the text prompt more. Lower FID, KID, and CLIP-FID indicate higher quality; higher CLIP-Score indicates higher quality. Together, these plots show that increasing the amount of training data from 10M to 90M samples does not lead to quantitative improvements. BLIP-2 re-captions provide nearly identical performance to LAION in terms of FID and KID; the re-captions indicate slightly better performance when using CLIP-FID as the quality metric.

## 6. Experiments

In this section, our model evaluations use automated, quantitative image-quality metrics from the literature. We measure performance with three metrics on the commonly used MS COCO dataset [41]: Frechet Inception Distance (FID) [21], Kernel Inception Distance (KID) [5], and CLIP-FID [29]. Each captures a slightly different measures of generated-image quality and diversity, in relation to statistics in the training data, with lower values corresponding to higher quality. Additionally, we evaluated CLIP-Score [20], which can help us understand the alignment between captions and their respective images, with higher values signaling better alignment. While these automated metrics are intended to be efficient proxies for human preferences in image quality, they often fall short; the gold standard for T2I model evaluation still remains human evaluation. Since synthetic captions differ so much from human-designed ones [48], we also set up a pairwise preference rating task to measure the relative quality of our trained models.

## 6.1. Training with Synthetic Captions

First, we look at the effect of training with synthetic captions instead of ground-truth captions from LAION. Interestingly, we observe that synthetic captions can enhance the alignment of our model. For instance, the CLIP-Score for synthetic captions exceeded that of ground-truth captions as seen in Figure 7 (for CLIP-FID).

To get a more nuanced perspective on the effect of our synthetic captions, we assess CLIP-FID for image generations from different models on human- and computergenerated captions (Fig. 8). In Figure 8, we compute CLIP-FID for various models trained using LAION, CommonCatalog, or LAION images re-captioned with BLIP-2; CLIP-FID is computed based on generating for prompts from MS COCO and the Conceptual Captions dataset. Unlike other caption datasets, MS COCO captions are human written. Most captions from web-based datasets (like LAION) are computer-generated [48]. BLIP-2 captions are also generated, but the BLIP-2 model is then fine-tuned to align with human-written captions. Given the higher quality of our synthetic captions, it is unsurprising that CommonCanvas’s CLIP-FID is better (i.e., lower) for MS COCO (i.e., aligns better with human-written captions).

![](images/a8f8475bb567479cfc5e28162ad21ddd6098c74f72196c95f52d0be3ad7a570c.jpg)  
Figure 8. Evaluating models at 256 resolution on different subsets of the Conceptual Captions dataset and MS COCO. LAION models are trained on 1.1 billion, 90 million (SD2-90M), and 10 million subsets. We also train a model with a 90 million subset re-captioned with BLIP-2 to evaluate distribution shift. The last two models are trained on on the CommonCatalog-C, and CommonCatalog-NC. We observe a domain shift between MS COCO and web-scraped Conceptual Captions. CLIP-FID may exhibit a preference for SD2 models, given that CLIP has been trained on a text style akin to that found in LAION. Subsampling the LAION dataset from 1.13B to 10M images does not seem to affect quantative performance. Using synthetic captions causes a significant performance drop on the LAION dataset when evaluated on Conceptual Caption test datasets, but not MS COCO.

![](images/6613ebdfabae0bd9aac5011da8da6d34fe81eea35addaa48dc8b0dc85e97496b.jpg)  
Figure 9. Using entirely Creative-Commons images and our synthetic captioning approach, we achieve comparable qualitative performance to public SD2, as seen in CommonCanvas generations, while only requiring a small fraction (< 3%) of the amount of training data. We include results for two CommonCanvas architectures, small (S) and large (L) (Section 6), and two CC-image datasets, commercial (C) and non-commercial (NC) (Section 4). We label our results accordingly as CommonCanvas-<architecture>-<dataset>.

However, like any model, ours has limitations. CommonCanvas under-performed in several categories, including faces, general photography, and paintings. These datasets all originated from the Conceptual Captions dataset [61], which relies on web-scraped data. These websourced captions, while abundant, may not always align with human-generated language nuances [4, 7, 48]. Although transitioning to synthetic captions introduces certain performance challenges, the drop in performance is not as dramatic as one might assume. Moreover, we speculate that the model will perform better if users provide their more specialized datasets to the model, such as FFHQ [26].

## 6.2. CommonCanvas vs. LAION-trained SD2

Given that our data-scarcity analysis suggests that CommonCatalog is large enough to train a high-quality SD2 model and that synthetic captions can perform well (Section 6.1), we train two different CommonCanvas models: one trained on commercial (CommonCatalog-C) images, another on non-commercial (CommonCatalog-NC). For a fair comparison with SD2, we use the OpenCLIP text encoder. Like BLIP-2, OpenCLIP is trained on LAION captions (Section 2.2). For example generations, see Figure 9.

We also note that, although we train on Creative-Commons images, it is still possible for an adversarial prompt to produce content that includes iconic characters. In Figure 10, we subject our model to ambiguous prompts that are suggestive of such characters. Examples include visuals closely resembling Elsa from Frozen, Indiana Jones resembling Harrison Ford, and even a likeness to Harry Potter (Figure 10). Qualitatively, our model deviated more from these characters than SD2.

## 6.3. Reaching SD2 quality with CommonCanvas-L

We also did a human study measuring pairwise preference ratings for the 512x512 resolution CommonCanvas models compared to SD2 (Figure 12). In this experiment, human raters were shown a prompt (selected randomly from the PartiPrompts prompts set [75]) along with two generated images in randomized order, one from the reference model (public SD2) and the other from a CommonCanvas model. We report the fraction of the time users selected the image generated by the CommonCanvas model over the corresponding generation from SD2 as the user preference rate for that model. We find that our CommonCanvas models are slightly less preferred than SD2-90M, with preference rates of 37% for CommonCanvas-S-C and 38% for CommonCanvas-S-NC, which we find surprisingly high considering the smaller and synthetic nature of the dataset. Figure 9 displays the results from our human study.

Our previous results suggest that SD2 may be underparameterized. We additionally train a larger variant of CommonCanvas-N-C (CommonCanvas-L-NC) that has a significantly larger U-Net (the U-Net architecture from SDXL ([49], see the appendix). When we use CommonCanvas-L-NC, we achieve competitive performance with SD2 on user preferences (Figure 9). For the largest model, CommonCanvas-L-NC, we do not measure a statistically significant difference in user preference between this model and SD2.

![](images/8aae962625d04c1263c2211714c18f7de173c0ce0f626cbea50ff55aa4c2b5a0.jpg)

![](images/c889387978d8e90e5b2b24dba62ee2f9bc3172ff24a52273c0df8ad92c756f1d.jpg)  
Kim Kardashian

Hillary Clinton  
![](images/093613dc89a9d26ba69026c15d108cf5ceeacc85ce79002c088f3c97f4d68988.jpg)  
Figure 10. We compare CommonCanvas-S-NC (Ours) to SD2. Our model is less likely to generate iconic characters given suggestive prompts (drawn from Lee et al. [33]).

## 7. Discussion and Related Work

In this paper, we train the CommonCanvas family of text-to-image, latent diffusion models using only Creative-Commons images and synthetic captions. We discuss and address data incompleteness and scarcity issues associated with CC images. For data incompleteness, we propose telephoning, an intuitive type of transfer learning (Section 3), which we instantiate with BLIP-2 to produce synthetic captions for CC images (together, the CommonCatalog dataset; Section 4). Regarding data scarcity, we hypothesize that only a small fraction of the data contained in LAION-2B is actually necessary to saturate SD2, and that the examples in CommonCatalog should be sufficient for training. To make testing this hypothesis more efficient, we implement a variety of ML-systems optimizations, which achieve a 2.71× speed-up over our SD2 baseline.

Ultimately, we find that we can train the SD2 model on <3% of LAION-2B (i.e., roughly 70 million images; Section 5), yielding a model we call SD2-90M. This encourages us to train on CommonCatalog’s commercially usable (also roughly 70 million) and non-commercially usable (roughly 25 million) examples. Compared to SD2, our CommonCanvas models under-perform in some categories, like faces, but CommonCanvas-L-NC demonstrates statistically equivalent performance with SD2 on human evaluation (Section 6).

While several recent works similarly address ML topics

![](images/d7ebf6910d753831cab37ad5c0c9d0a826cad05d30329d3e3fcaeb655067efcd.jpg)

![](images/48b4a194f54d6493a7dc850ce53a8fb2cd890fcd623cf2548fe11ea17f999fb5.jpg)  
Barack Obama

![](images/c2e3cb0bc6f2cd043bd19881d942d0cd579834788fe39955a79dda655be2b976.jpg)  
Richard Feynman

Figure 11. Using CommonCanvas-SNC (Ours) to generate celebrities. Our model is worse at synthesizing individual people than SD2, but is capable of generating some noteworthy public figures. This result demonstrates how our model struggles to generate specific celebrities, which may be desirable from a privacy perspective.

relating to copyright, the literature tends to concern textto-text training data [44], be primarily theoretical [57, 71], involve ablation studies [28], or only handle verbatim memorization [8, 47] through the use of generation-time content filters [18], which has been shown to be an incomplete solution [23]. To the best of our knowledge, no prior open work attempts to train T2I models on only open-licensed data. Most prior work on image-caption-dataset creation has extracted caption data from Common Crawl [14, 16, 32]. We instead focus on synthesizing captions directly by using a pre-trained BLIP-2 model. Nguyen et al. [48] demonstrates that existing caption datasets can be improved by using BLIP-2 to replace low-quality image captions (e.g., in Datacomp), but does not focus on creating a new dataset of synthetic captions.

Another limitation is that the YFCC100M data is about a decade old; its CC images are not as current as those in LAION-2B. In the future, we plan to augment CommonCatalog with Creative-Commons images from other sources, as well as test larger model architectures and more advanced captioning models, like LLaVA [42].

![](images/79cd22ada56a914333c9e1e1260b29d7abf37493f14c293e39bd62139654812b.jpg)  
Figure 12. User preference study using Parti prompts. Preference rate (compared to SD2, the thick black horizontal line). CommonCanvas-L-NC matches the performance of SD2.

## References

[1] Anderson v. Stability AI, Ltd., 2023. No. 3:23-cv-00201 (N.D. Cal. Jan. 13, 2023). 3

[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 5

[3] Romain Beaumont. LAION-5B: A New Era of Large-Scale Multi-Modal Datasets. LAION Blog, 2022. 2

[4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. 2023. 4, 7

[5] Mikołaj Binkowski, Danica J Sutherland, Michael Arbel, and´ Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 6

[6] Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes, 2021. 3

[7] Davide Caffagni, Manuele Barraco, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Synthcap: Augmenting transformers with synthetic data for image captioning. In International Conference on Image Analysis and Processing, pages 112–123. Springer, 2023. 4, 7

[8] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew\` Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and<sup>´</sup> Colin Raffel. Extracting Training Data from Large Language Models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650. USENIX Association, 2021. 8

[9] Nicholas Carlini, Matthew Jagielski, Christopher A. Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramer.\` Poisoning Web-Scale Training Datasets is Practical, 2023. 3

[10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023. 4

[11] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning, 2022. 2

[12] A. Feder Cooper, Katherine Lee, James Grimmelmann, Daphne Ippolito, Christopher Callison-Burch, Christopher A. Choquette-Choo, Niloofar Mireshghallah, Miles Brundage, David Mimno, Madiha Zahrah Choksi, Jack M. Balkin, Nicholas Carlini, Christopher De Sa, Jonathan Frankle, Deep Ganguli, Bryant Gipson, Andres Guadamuz, Swee Leng Harris, Abigail Z. Jacobs, Elizabeth Joh, Gautam Kamath, Mark Lemley, Cass Matthews, Christine McLeavey, Corynne McSherry, Milad Nasr, Paul Ohm, Adam Roberts, Tom Rubin, Pamela Samuelson, Ludwig Schubert, Kristen Vaccaro, Luis Villa, Felix Wu, and Elana Zeide. Report of the 1st Workshop on Generative AI and Law. arXiv preprint arXiv:2311.06477, 2023. 3

[13] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and memory-efficient´ exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022. 5, 2

[14] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431, 2021. 8

[15] Doe 1 v. GitHub, Inc., 2022. No. 4:22-cv-06823 (N.D. Cal. November 3, 2022). 1

[16] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. DataComp: In search of the next generation of multimodal datasets, 2023. 3, 8

[17] Getty Images (US), Inc. v. Stability AI, Inc., 2023. No. 1:23- cv-00135 (D. Del. February 3, 2023). 1, 3

[18] GitHub. Configuring github copilot in your environment, 2023. 1, 8

[19] Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A. Lemley, and Percy Liang. Foundation Models and Fair Use, 2023. 3

[20] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. 6

[21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6

[22] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Open-CLIP, 2021. 2

[23] Daphne Ippolito, Florian Tramer, Milad Nasr, Chiyuan\` Zhang, Matthew Jagielski, Katherine Lee, Christopher A. Choquette-Choo, and Nicholas Carlini. Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy, 2023. 8

[24] J.L. v. Alphabet Inc., 2023. No. 3:23-cv-03440-LB (N.D. Cal July 11, 2023). 1, 3

[25] Kadrey v. Meta Platforms, Inc., 2023. No. 3:23-cv-03417 (N.D. Cal. July 7, 2023). 1

[26] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 7

[27] Dirk P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In International Conference on Learning Representations, 2014. 2

[28] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating Concepts in Text-to-Image Diffusion Models, 2023. 1, 8

[29] Tuomas Kynka¨anniemi, Tero Karras, Miika Aittala, Timo¨ Aila, and Jaakko Lehtinen. The role of imagenet

classes in fr\’echet inception distance. arXiv preprint arXiv:2203.06026, 2022. 6

[30] LAION-2Ben, 2022. Accessed September 23, 2023. 1, 2

[31] Viktor Lakic, Luca Rossetto, and Abraham Bernstein. Link-Rot In Web-Sourced Multimedia Datasets. In MultiMedia Modeling: 29th International Conference, MMM 2023, Bergen, Norway, January 9–12, 2023, Proceedings, Part I, page 476–488, Berlin, Heidelberg, 2023. Springer-Verlag. 3

[32] Hugo Laurenc¸on, Lucile Saulnier, Leo Tronchon, Stas Bek-´ man, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, and Victor Sanh. OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents, 2023. 8

[33] Katherine Lee, A. Feder Cooper, and James Grimmelmann. Talkin’ ’Bout AI Generation: Copyright and the Generative-AI Supply Chain. arXiv preprint arXiv:2309.08133, 2023. 3, 4, 8

[34] Katherine Lee, A. Feder Cooper, James Grimmelmann, and Daphne Ippolito. AI and Law: The Next Generation, 2023. 1, 3

[35] Katherine Lee, A. Feder Cooper, and James Grimmelmann. Talkin’ ’Bout AI Generation: Copyright and the Generative-AI Supply Chain (The Short Version). In Proceedings of the Symposium on Computer Science and Law, page 48–63, New York, NY, USA, 2024. Association for Computing Machinery. 3

[36] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xFormers: A modular and hackable Transformer modelling library. https://github.com/ facebookresearch/xformers, 2022. 5, 2

[37] Mark A. Lemley. How Generative AI Turns Copyright Law on its Head, 2023. 3

[38] Pierre N. Leval. Toward a Fair Use Standard. Harvard Law Review, 103(5):1105, 1990. 3

[39] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 3, 5

[40] Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Selfalignment with instruction backtranslation. arXiv preprint arXiv:2308.06259, 2023. 4

[41] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6

[42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 4, 8

[43] Susan Box Mann. The Telephone Game, 2019. Accessed September 27, 2023. 3

[44] Sewon Min, Suchin Gururangan, Eric Wallace, Hannaneh Hajishirzi, Noah A. Smith, and Luke Zettlemoyer. SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore, 2023. 1, 8

[45] The Mosaic ML Team. composer. https://github. com/mosaicml/composer/, 2021. 2

[46] The Mosaic ML Team. streaming. <https://github. com/mosaicml/streaming/>, 2022. 2

[47] Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramer, and Kather-\` ine Lee. Scalable Extraction of Training Data from (Production) Language Models. arXiv preprint arXiv:2311.17035, 2023. 8

[48] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving multimodal datasets with image captioning. arXiv preprint arXiv:2307.10350, 2023. 5, 6, 7, 8

[49] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis, 2023. 2, 8

[50] Jacob Portes, Alexander R Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, and Jonathan Frankle. Mosaicbert: How to train bert with a lunch money budget. In Workshop on Efficient Systemsfor Foundation Models@ ICML2023, 2023. 2

[51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, 2021. 2, 3

[52] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492– 28518. PMLR, 2023. 4

[53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-Resolution Image¨ Synthesis with Latent Diffusion Models. In 2022 IEEE Conference on Computer Vision and Pattern Recognition, 2022. 2

[54] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention, pages 234–241, 2015. 2

[55] Matthew Sag. Copyright Safety for Generative AI. Houston Law Review, 2023. Forthcoming. 3

[56] Pamela Samuelson. Generative AI meets copyright. Science, 381(6654):158–161, 2023. 3

[57] Sarah Scheffler, Eran Tromer, and Mayank Varia. Formalizing Human Ingenuity: A Quantitative Framework for Copyright Law’s Substantial Similarity. In Proceedings of the Symposium on Computer Science and Law, pages 37–49, 2022. 1, 8

[59] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2

[60] Charles M. Schultz. Snoopy Peanuts, 2020. Accessed September 26, 2023. 3

[61] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 7

[62] Benjamin L.W. Sobel. Artificial Intelligence’s Fair Use Crisis. Columbia Journal of Law and The Arts, 41:45, 2017. 3

[63] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathany, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, 2015. 2

[64] Stability AI. Stable Diffusion 2.0 Release, 2022. 1

[65] Stability AI. Stable Diffusion v2-base Model Card, 2022. 5

[66] The US Copyright Office. Works Not Protected by Copyright (Circular 33), 2021. 4

[67] David Thiel. Identifying and eliminating csam in generative ml training data and models. Technical report, Technical report, Stanford University, Palo Alto, CA, 2023. URL https://purl . . . , 2023. 3

[68] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications ofthe ACM, 59(2):64–73, 2016. 4, 2

[69] Tremblay v. OpenAI, Inc., 2023. No. 3:23-cv-03223 (N.D. Cal. June 28, 2023). 1

[70] James Vincent. Getty Images is suing the creators of AI art tool Stable Diffusion for scraping its content. The Verge, 2023. 3

[71] Nikhil Vyas, Sham Kakade, and Boaz Barak. On Provable Copyright Protection for Generative Models, 2023. 8

[72] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018. 5

[73] Feiyang Xiao, Qiaoxi Zhu, Jian Guan, Xubo Liu, Haohe Liu, Kejia Zhang, and Wenwu Wang. Synth-ac: Enhancing audio captioning with synthetic supervision. arXiv preprint arXiv:2309.09705, 2023. 4

[74] Yuanzhong Xu, HyoukJoong Lee, Dehao Chen, Hongjun Choi, Blake Hechtman, and Shibo Wang. Automatic crossreplica sharding of weight update in data-parallel training. arXiv preprint arXiv:2004.13336, 2020. 2

[75] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregres-