# Flexible Biometrics Recognition: Bridging the Multimodality Gap through Attention, Alignment and Prompt Tuning

Leslie Ching Ow Tiong\*<sup>,1</sup> Dick Sigmund\*<sup>,2</sup> Chen-Hui Chan<sup>3</sup> Andrew Beng Jin Teoh<sup>†,4</sup>

<sup>1</sup>Samsung Electronics <sup>2</sup>AIDOT Inc. <sup>3</sup>Korea Institute of Science and Technology <sup>4</sup>Yonsei University

<sup>1</sup>leslie.tiong@samsung.com <sup>2</sup>dsigmund@aidot.ai <sup>3</sup>chchan@kist.re.kr <sup>4</sup>bjteoh@yonsei.ac.kr

## Abstract

Periocular and face are complementary biometrics for identity management, albeit with inherent limitations, notably in scenarios involving occlusion due to sunglasses or masks. In response to these challenges, we introduce Flexible Biometric Recognition (FBR), a novel framework designed to advance conventional face, periocular, and multimodal face-periocular biometrics across both intra- and cross-modality recognition tasks. FBR strategically utilizes the Multimodal Fusion Attention (MFA) and Multimodal Prompt Tuning (MPT) mechanisms within the Vision Transformer architecture. MFA facilitates the fusion of modalities, ensuring cohesive alignment between facial and periocular embeddings while incorporating soft-biometrics to enhance the model’s ability to discriminate between individuals. The fusion of three modalities is pivotal in exploring interrelationships between different modalities. Additionally, MPT serves as a unifying bridge, intertwining inputs and promoting cross-modality interactions while preserving their distinctive characteristics. The collaborative synergy of MFA and MPT enhances the shared features of theface and periocular, with a specific emphasis on the ocular region, yielding exceptional performance in both intraand cross-modality recognition tasks. Rigorous experimentation across four benchmark datasets validates the noteworthy performance of the FBR model. The source code is available at https://github.com/MIS-DevWorks/FBR.

## 1. Introduction

Facial recognition has attained ubiquitous applications across diverse domains today [18, 19, 35]. However, they struggle with challenges arising from cosmetic changes, plastic surgery, and particularly obstructions like face masks. On the other hand, periocular recognition, which focuses on the region around the eyes, has gained traction as an alternative to face recognition [1, 23–25]. Despite the significant progress, however, challenges remain, especially with glasses or sunglasses, impacting the accuracy of periocular recognition.

The fusion of facial and periocular [15, 32, 37], holds promise for enhancing recognition performance. However, traditional multimodal biometrics present fresh challenges in managing and storing templates of all biometric modalities, which can result in computational and storage overhead. Moreover, ensuring all modalities are available for recognition is critical for seamless deployment. In response to these challenges, conditional biometrics [11, 22], along with cross-modality biometrics recognition [20, 33], offer promising avenues to mitigate the constraints of unimodal biometric systems, i.e., sole face or periocular recognition as well as multimodal biometric systems. Conditional biometrics enhance a single modality by incorporating information from another, such as periocular recognition conditioned by the face or vice versa. On the other hand, crossmodality biometrics encompasses the task of matching biometric samples across distinct biometric modalities, such as face vs. periocular matching.

This paper introduces Flexible Biometrics Recognition (FBR), designed to support intra- and cross-modality biometric matching, as illustrated in Figure 1. The FBR model is initially trained to align facial, periocular, and softbiometric attributes. The latter encompasses social or physical descriptive traits of individuals such as gender, age, or ethnicity, which have proven to enhance the discriminative power of the embedding [6]. During deployment, the trained FBR model serves as a feature extractor, acquiring facial or periocular embeddings based on the input modality. In contrast to unimodal and multimodal biometric systems, FBR produces modality-invariant embeddings, facilitating both intra- and cross-modality matching. Additionally, FBR can address scenarios where only facial or periocular data is stored, enabling exclusive reliance on the available modality during recognition.

The adaptability of FBR ensures robust performance and operational reliability, even in incomplete or temporarily unavailable biometric data. As depicted in Figure 1, the proposed FBR model outperforms the unimodal biometrics baseline and a competing model [10], which is also trained with three modalities and equips with a prompt tuning mechanism. While FBR offers exceptional adaptability, attaining decent performance in intra- and cross-modality matching is a non-trivial challenge. Enhancing one facet could potentially undermine the other and vice versa. Striking the right balance between optimizing intra- and crossmodality recognition is crucial.

![](images/f867a567c15d932cb2d08143c59279b738ceda4575b13ca80db45adfaf6a01c8.jpg)  
Figure 1. An overview of the Flexible Biometrics Recognition (FBR). This approach encompasses flexible recognition tasks such as intra- and cross-modality biometric recognition, demonstrating its versatility in biometric systems. Our model exhibits enhanced performance relative to benchmarks, including unimodal biometrics baseline and a competing model [10].

To substantiate FBR, we introduce a Multimodal Fusion Attention Vision Transformer (MFA-ViT) and a multimodal-prompt tuning (MPT) mechanism. MFA-ViT is crafted to establish a cohesive alignment between facial and periocular features within a ViT. Furthermore, integrating soft-biometric attributes enhances the model’s ability to discern differences between identities effectively. The multimodal fusion attention (MFA) module achieves the fusion of three modalities, which is pivotal in exploring interrelationships between different modalities. This endeavor proves advantageous in mitigating the trade-off between intra- and cross-modality recognition tasks.

In the realm of robust embedding learning across diverse modalities, prompt tuning has proven effective for modality alignment [10, 39]. Nevertheless, standard prompt tuning (SPT) often falls short of effectively utilizing multimodal information. For FBR problems, SPT lacks proper guidance for intra- and cross-modality integration, especially when dealing with multiple modalities. The proposed MPT addresses this challenge by providing modality-specific guidance for aligning multimodal information within MFA-ViT. Unlike SPT, MPT establishes a unified bridge for three modalities, intertwining input sequences while promoting cross-modality interaction while preserving their distinct characteristics.

To sum up, we leverage the MFA and MPT within the ViT architecture to address FBR problems. These mechanisms are devised for effective multiple modalities alignment, especially to accentuate the common distinctive features of the face and periocular, with a specific focus on the ocular region, as they play a pivotal role in achieving exceptional performance in intra- and cross-modality recognition tasks. Additionally, our approach derives advantages from incorporating soft-biometric attributes as supplementary information.

The contributions of this paper are summarized as follows:

• A novel FBR framework is introduced to address intraand cross-modality recognition by integrating face and periocular modalities with soft-biometric attributes.

• A MFA-ViT is designed to substantiate the FBR notion, enabling the effective fusion of three modalities and systematically examining the inter-dependencies between intra- and cross-modality relationships while establishing a modality-invariant embedding to represent identities.

• The MPT mechanism is proposed, crucially guiding the integration of multimodal data to produce rich, coherent embeddings, capturing intricate relationships between different biometric modalities.

## 2. Related Work

Conditional biometrics has garnered attention as a promising solution to address the limitations of multimodal biometrics [8, 9, 11]. These studies illustrate the performance improvements achieved by conditioning the face modality with soft-biometrics, particularly in challenging environmental variations and occlusions. A previous study has utilized knowledge distillation techniques [12] to enhance periocular modality performance by incorporating face biometrics. In a similar vein, [22] introduces an approach for face-conditioned periocular recognition. This approach employs a conditional biometrics contrastive loss within a shared-parameter convolutional network.

Nevertheless, FBR offers a more robust solution due to its inherent flexibility to handle intra- and cross-modality recognition tasks without specific conditioning, making it a better choice for demanding real-world applications.

Cross-modality biometrics. Prior studies in crossmodality biometrics have primarily focused on face-voice recognition [3, 13, 20, 36], while others like [5, 34] focused on bridging visible light face images with alternative modalities such as infrared or depth images. Most recently, [33] introduces HA-ViT, utilizing a face-periocular contrastive learning approach for cross-modality recognition. This approach effectively demonstrates how contrastive learning can proficiently align and differentiate these modalities, enhancing cross-modality recognition task performance. However, previous works tend to neglect the wider utility, especially in tasks related to intra-modality recognition. Furthermore, while the importance of face and periocular biometrics in cross-modality recognition is acknowledged, effectively capturing their intricate relationships remains challenging. [5, 22, 33].

Our approach goes beyond the boundaries of crossmodality biometric recognition by harnessing the MFA and MPT modules within the ViT architecture. Supplementary Material Section 8.1 highlights that these modules emphasize the shared salient features of the face and periocular, particularly the eyes, which are crucial for excelling in intraand cross-modality recognition tasks.

Prompt Tuning involves generating task-specific continuous vectors using gradient descent [16]. These vectors are designed to guide a pre-trained transformer model to perform specific tasks without requiring extensive finetuning of the entire model. This technique has recently been extended to image classification tasks, as explored by [10, 39]. In this context, the learnable visual prompt tuning (VPT) is used to adapt pre-trained ViT models, resulting in improved performance compared to conventional finetuning.

However, applying VPT directly to FBR poses challenges. These challenges arise from the inherent limitations of guiding alignment within VPT methodologies, mainly when dealing with the complexities of multiple modalities. To this end, our proposed MPT addresses the alignment of multiple modality embeddings within a shared embedding space that captures multimodal information. Importantly, this unified alignment is a novel contribution not explored in existing literature.

## 3. Methodology

## 3.1. Network Architecture

MFA-ViT is built upon a shared-parameter network architecture that accommodates input from three sources: face $( \mathbf { I } _ { f } )$ , periocular $( \mathbf { I } _ { p } )$ , and soft-biometric attributes $\left( \mathbf { I } _ { a } \right)$ , as illustrated in Figure 2. The MFA-ViT is based on the ViT architecture [4] due to its effectiveness in handling multimodality fusion without explicit modifications [38]. Further justifications are provided in Supplementary Material Section 8.3.

Given a pair of image patches $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p } ,$ , each is tokenized to yield embeddings $\mathbf { Z } _ { f }$ and $\mathbf { Z } _ { p }$ with dimension d, and it is set to 1, 024. To incorporate ${ \mathbf I } _ { a }$ into the network, we utilize feature tokenizer [7], in order to transform the input ${ \mathbf I } _ { a }$ into embeddings $\mathbf { Z } _ { a } \in \mathbb { R } ^ { 1 \times d }$ . Feature tokenizer enables the seamless integration of categorical-based softbiometric attributes with image-based face and periocular modalities.

The network takes in biometric token embedding $\mathbf { Z } _ { \ast }$ where denotes f, p, or $^ { a , }$ a learnable class token embedding T as well as a learnable prompt token embedding $\mathbf { P } _ { * }$ . Subsequently, these embeddings are directed to the multimodal fusion attention (MFA) block, $B _ { m }$ where $m = 1 , \cdots , M$ . Each MFA block comprises MFA layers $F _ { n }$ with $n = 0 , 1 , \cdots , N - 1$ . Each $F _ { n }$ is constructed by a $3 \times 3$ depth-wise Conv (DWS-Conv) layer, a depth-wise fusion Conv-based multi-head self-attention (DWFC-MSA) layer, a $1 \times 1$ Conv layer, and a LeakyReLu $( R _ { \mathrm { L e a k y } } )$ activation layer. In this paper, $M = 2$ and $N = 4$ are adopted.

The embeddings $\mathbf { T } _ { * } , \mathbf { Z } _ { * }$ , and $\mathbf { P } _ { * }$ are concatenated and subsequently fed into the $F _ { n }$ layer within each $B _ { m }$ , which is outlined as follows:

$$
\begin{array} { r } { F _ { n + 1 } ( \mathbf { K } _ { * , n } ) = R _ { \mathrm { L e a k y } } ( \mathrm { C o n v } ( [ \mathrm { D W S - C o n v } ( \mathbf { K } _ { * , n } ) , } \\ { \mathrm { D W F C - M S A } ( \mathbf { K } _ { * , n } ) ] ) ) + \mathbf { K } _ { * , n } , } \end{array}\tag{1}
$$

where ${ \bf K } _ { \ast , n } = [ { \bf T } _ { \ast , n } , { \bf Z } _ { \ast , n } , { \bf P } _ { \ast , n } ] \in \mathbb { R } ^ { S \times H \times W }$ and S, H, W denote the number of input embeddings, height, width, respectively. The DWS-Conv layer specializes in extracting distinct local features from the $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ patches while simultaneously considering the associations encoded by the ${ \mathbf I } _ { a }$ . These associations substantiate the learning of multimodal embeddings, particularly facilitated by the $\mathbf { P } _ { * }$

The DWFC-MSA layer is tailored to capture the relationships within and across modalities within the $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ patches. The presence of ${ \mathbf I } _ { a }$ enriches this layer, allowing it to achieve a holistic comprehension. The DWFC-MSA layer collaborates seamlessly with $\mathbf { P } _ { * }$ to support the development of this understanding. The DWFC-MSA (E) layer is structured as follows:

$$
E _ { n + 1 } ^ { \prime } ( \mathbf { K } _ { * , n } ) = \mathbf { C } \mathbf { \mathrm { - } } \mathbf { M S A } ( \mathbf { N o r m } ( \mathbf { K } _ { * , n } ) ) + \mathbf { K } _ { * _ { n } } ,\tag{2}
$$

$$
\begin{array} { r l } & { E _ { n + 1 } ( \mathbf K _ { * , n } ) = } \\ & { \qquad \mathrm { M L P } ( \mathrm { N o r m } ( E _ { n + 1 } ^ { \prime } ( { \mathbf K } _ { * , n } ) ) ) + E _ { n + 1 } ^ { \prime } ( { \mathbf K } _ { * , n } ) , } \end{array}\tag{3}
$$

where Norm( ) denotes layer normalization, C-MSA( ) refers to $3 \times 3$ depth-wise Conv-based MSA layer, and MLP( ) represents multi-layer perception. Noted that the input tokens for DWS-Conv and DWFC-MSA layers are reshaped for spatial dimensions.

As the final step, the network encodes joint embeddings J by aggregating the MPT embeddings $\mathbf { P } _ { * , N } ^ { \prime }$ from each

![](images/4456abf84b8b34e3cd13853c656fedfbb4e5f534d907ac270310149078a96928.jpg)  
Figure 2. Network architecture of Multimodal Fusion Attention (MFA) Vision Transformer with Multimodal-Prompt Tuning (MPT).

$B _ { m }$ via addition and followed by an average pooling (Avgpool) operation. For classification, we utilize J as the input to the softmax layer, resulting in the softmax vector $\mathbf { Y } _ { * }$ . We opt to employ J instead of a class token T as commonly utilized in ViT, is driven by the hypothesis that the MPT, explicitly designed for the multimodal fusion task, can offer a more effective way to capture and leverage cross-modality information. We explore the impacts of this option in Section 4.4.4. The final formulation is defined as follows:

$$
{ \bf J } _ { * } = \mathrm { A v g p o o l } ( \sum _ { m = 1 } ^ { M } { \bf P } _ { * , N , M } ^ { \prime } ) ,\tag{4}
$$

$$
\mathbf { Y } _ { * } = \mathrm { S o f t m a x } ( \mathbf { J } _ { * } ) .\tag{5}
$$

## 3.2. Multimodal-Prompt Tuning

MPT is incorporated at the input space after the embedding layers, which are attached to $\mathbf { T } _ { * }$ and $\mathbf { Z } _ { \ast }$ , as illustrated in Figure 2. The MPT embeddings $( \mathbf { P } _ { * } ^ { \prime } )$ play a pivotal role in guiding the process of multimodal feature prompt learning in each $F _ { n }$ layer. Utilizing this guidance enables a subtle and detailed exploration of relationships among the diverse modalities and attributes under consideration.

Specifically, MPT is integrated at the input space of each layer in $B _ { m }$ . The structure of MPT involves a 1  1 Conv layer, followed by a ReLU layer, applied to $\mathbf { P } _ { * , n }$ embeddings. We designate the set of learnable $\mathbf { P } _ { * , n } ^ { \prime }$ embeddings with dimension size of d. Section 4.4.2 emphasizes MPT embeddings’ role in discerning intricate details within multimodal features, with Figure 5 demonstrating their efficacy in capturing eye regions across facial and periocular images. Additional studies can be found in Supplementary Material Section 8.2. The input space with MPT embeddings can be computed as:

$$
\begin{array} { r l } & { [ \mathbf { T } _ { * , n + 1 } , \mathbf { Z } _ { * , n + 1 } , \mathbf { P } _ { * , n + 1 } ^ { \prime } ] = } \\ & { \qquad F _ { n + 1 } ( [ \mathbf { T } _ { * , n + 1 } , \mathbf { Z } _ { * , n + 1 } , L _ { n + 1 } ( \mathbf { P } _ { * , n } , \mathbf { P } _ { * , n + 1 } ) ] ) , } \end{array}\tag{6}
$$

where $\mathbf { P } _ { * , n }$ and $\mathbf { P } _ { * , n + 1 }$ represent the previous and current input prompt embeddings. $L _ { n + 1 }$ is calculated using a ReLU activation applied to the output of $1 \times 1$ Conv as defined by:

$$
L _ { n + 1 } ( \mathbf { P } _ { * , n } , \mathbf { P } _ { * , n + 1 } ) = \mathrm { R e L u } ( \mathrm { C o n v } ( [ \mathbf { P } _ { * , n } , \mathbf { P } _ { * , n + 1 } ] ) ) .\tag{7}
$$

## 3.3. Multimodal Contrastive Loss Function

For training, we employ a dual strategy that combines both large-margin softmax loss $( \mathcal { L } _ { \mathrm { L M } } )$ [14] and contrastive loss $( \mathcal { L } _ { \mathrm { C L } } )$ [33]. This strategy aims to learn intramodality relationships within face (f) and periocular $( p ) .$ and cross-modality relationships encompassing f, p, and soft-biometric attributes (a).

Specifically, the $\mathcal { L } _ { \mathrm { L M } }$ contributes to shaping the embedding space to better discriminate between different identities, which is essential for intra-modality recognition. The $\mathcal { L } _ { \mathrm { L M } }$ is given as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { L M _ { \ddag } } } = } \\ { \displaystyle \ - \log ( \Phi _ { \ddag } ) + \frac { \lambda } { 2 } \sum _ { c \neq l } \left[ \Psi _ { \ddag , c } - \frac { 1 } { C - 1 } \cdot \log ( \Psi _ { \ddag , c } ) \right] , } \end{array}\tag{8}
$$

where   denotes softmax function, $\Psi _ { \ddag , c }$ is to maximize the margin of discriminated embedding between the identity of the same modalities, C is the number of identities,   is to control the degree of degrading labels, and l is the label.   is set to 0.3 same as [14]. The $\mathcal { L } _ { \mathrm { L M } , \ddag }$ is computed individually to f and p modalities, with representing either $f \ \mathrm { o r } \ p .$

On the other hand, the ${ \mathcal { L } } _ { \mathrm { C L } }$ accentuates the embedding of the same identity samples, fostering closer proximity in the embedding space while simultaneously pushing apart samples from different identities across three modalities, $\mathrm { i . e . , } f ,$ $p ,$ and $a .$ The loss function is implemented following the formulation introduced by [33] and [40] to establish significant connections between cross-modality relationships. The ${ \mathcal { L } } _ { \mathrm { C L } }$ can be expressed as:

$$
\mathcal { L } _ { \mathrm { C L } } = \mathcal { L } _ { \mathrm { C M } } ( f , p ) + \mathcal { L } _ { \mathrm { C M } } ( f , a ) + \mathcal { L } _ { \mathrm { C M } } ( p , a ) ,\tag{9}
$$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C M } } ( x _ { u } , y _ { u } ) = } \\ & { \quad - \log \frac { \sigma ( x _ { u } , y _ { u } ) } { \sigma ( x _ { u } , y _ { u } ) + \alpha \left( \delta ( x _ { u } , x _ { v } ) \right) + \delta ( x _ { u } , y _ { v } ) } , } \end{array}\tag{10}
$$

where $\begin{array} { r } { \sigma ( x _ { u } , y _ { u } ) = \exp ( \frac { { \mathbf J } _ { x _ { u } } ^ { T } { \mathbf J } _ { y _ { u } } } { \theta } ) } \end{array}$ serves to map the embeddings J, of distinct modalities x and $y ,$ yet sharing the same identity $u ,$ into a shared embedding space. The hyperparameter ✓ is introduced to extend the range of $\mathbf { J } _ { x _ { u } } ^ { T } \mathbf { J } _ { y _ { u } }$ to facilitate the model’s convergence.

Furthermore, $\begin{array} { r } { \delta ( x _ { u } , x _ { v } ) ~ = ~ \sum _ { x _ { v } \in { \bf N } _ { \neq } ^ { X } } \sigma ( x _ { u } , x _ { v } ) } \end{array}$ represents pairs of data samples sharing the same modality but differing in identity $v .$ Here, $x _ { v } \ \in \ \mathbf { N } _ { u } ^ { X }$ designates intra-modality pairs, which are pairs of different samples within the same modality. Similarly, $\delta ( x _ { u } , y _ { v } ) ~ =$ $\textstyle \sum _ { y _ { u } \in \mathbf { N } _ { * } ^ { Y } } \sigma ( x _ { u } , y _ { v } )$ characterizes pairs of data samples from different modalities and identities, thereby ensuring they remain distinguishable in the shared embedding space. Here, $y _ { v } \in \mathbf { N } _ { u } ^ { Y }$ refers to cross-modality pairs between different samples. Leveraging $\delta ( x _ { u } , x _ { v } )$ and $\delta ( x _ { u } , y _ { v } )$ empowers the model to effectively discern between the high similarities in cross-modalities from different identities.

In our study, ↵ is set to 0.8 same as [40]. Additionally, we set ✓ to 0.03 for the $\mathcal { L } _ { \mathrm { C M } } ( f , p )$ term, and to 0.04 for both ${ \mathcal { L } } _ { \mathrm { C M } } ( f , a )$ and ${ \mathcal { L } } _ { \mathrm { C M } } ( p , a )$ terms.

The total loss $( \mathcal { L } _ { \mathrm { t o t a l } } )$ is given as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { L M } _ { f } } + \mathcal { L } _ { \mathrm { L M } _ { p } } + \mathcal { L } _ { \mathrm { C L } } .\tag{11}
$$

## 3.4. Flexible Recognition

To determine a person’s identity, the trained model is designed flexibly to accept inputs from the f or $p$ modalities. Specifically, the softmax layer is detached for recognition, and the modality-invariant embedding is extracted from $\mathbf { J } _ { \ddag }$ where  denotes $f$ or $p .$ The identity of $\mathbf { J } _ { \ddag }$ can be decided based on

$$
\psi = \operatorname* { m a x } _ { k } \left[ s ( \mathbf { G } _ { k , \pm } , \mathbf { J } _ { \mp } ) \right]\tag{12}
$$

where $s ( \cdot )$ calculates a similarity score between the unknown identity $\mathbf { J } _ { \ddag }$ and the gallery sets $\mathbf { G } _ { k , \ddag }$ , k refers to the number of identities in gallery set.

## 4. Experiment

## 4.1. Dataset

## 4.1.1 Training dataset

The training dataset comprises modalities $\mathbf { I } _ { f } , \mathbf { I } _ { p } ,$ and ${ \mathbf I } _ { a } ,$ which are sampled from the VGGFace2 dataset [2] and the MAAD-Face dataset [31]. After a comprehensive dataset review, we have selected 1.49 million samples with 9,131 identities. For each identity, we have paired face $\mathbf { I } _ { f }$ and periocular $\mathbf { I } _ { p }$ images with 47 attributes ${ \mathbf I } _ { a }$ representing the identities, detailed in [31]. We randomly partitioned the identities, allocating them into training and validation sets in an 80:20 ratio.

## 4.1.2 Evaluation dataset

To have a fair comparison, we have selected four public datasets, namely Ethnic [32], FaceScrub [21], IMDB [26], and Cross-Modal DB [33]. Further details can be found in Supplementary Material Section 7. These datasets are benchmarks for evaluating our network’s intra- and crossmodality matching performance. We adhere to the protocol in [32], which involves matching a probe from the gallery sets. In this evaluation, all trained models serve as feature extractors for $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ across both gallery and probe sets. The matching process is executed using cosine similarity.

## 4.2. Experimental Setup

MFA-ViT is trained over 50 epochs with a batch size of 64. The input sizes for $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ are both set to $3 \times 1 1 2 \times 1 1 2$ while ${ \mathbf I } _ { a }$ is $1 \times 4 7$ . Each $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ image is divided into 14 patches, and the size of each patch is $8 \times 8 .$ . We minimize the total loss using the AdamW Optimizer [17]. We employ a learning rate of 1e-4, a weight decay parameter 1e-5, and a dropout rate of 0.1.

## 4.3. Experimental Results

To assess the FBR performance, we conduct a comprehensive comparison by re-implementing a baseline model trained solely on $\mathbf { I } _ { f }$ and another solely on $\mathbf { I } _ { p }$ . Furthermore, we examined several relevant models designed for intra- and cross-modality recognition problems, i.e., [32], [22], and [5]. We also aim to provide a broader comparison encompassing recent studies such as HA-ViT [33] and ViT/VPT [10] to examine the effectiveness of our proposed model. Noted that we have re-implemented these models to adapt input embeddings, customizing them for fair comparisons, while adhering to the same experimental settings and protocols. In Table 1, we present the performance comparisons for intra- and cross-modality recognition tasks, with the primary metric being rank-1 recognition accuracy. Specifically, the cross-modality recognition tasks encompass scenarios such as face gallery vs. periocular probe $( f - p )$ and periocular gallery vs. face probe $( p { - } f )$ while intra-modality recognition focuses on face gallery vs. face probe $( f { - } f )$ and periocular gallery vs. periocular probe $\left( p { - } p \right)$

## 4.3.1 Intra-modality Recognition

As indicated in Table 1, the MFA-ViT/MPT model exhibits exceptional recognition performance on both the Ethnic and FaceScrub datasets, achieving high accuracy rates of 94.82% and 95.71% for the $f { - } f$ and 89.98% and 93.06% for the $p { - } p _ { ; }$ , respectively. Furthermore, even when confronted with more challenging datasets such as IMDB and Cross-Modal DB, MFA-ViT/MPT maintains its competitive advantage, yielding mean accuracy rates of 86.03% and 85.88% for $f { - } f$ and 80.53% and 76.54% for $p { \mathrm { - } } p .$ . These outcomes underscore the model’s consistent outperformance of baseline and competing models.

The performance of both the baseline and [32] is suboptimal, primarily attributed to their extensive dependence on the individual $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ modalities. In contrast, the models presented by [5] and [22] have exhibited satisfactory results by adeptly capturing adaptive relational knowledge between $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ embeddings. Nevertheless, the necessity for both face and periocular inputs in these models restricts their adaptability.

Notably, the experiments reveal that models with prompt tuning, such as ViT/VPT, generally outperform those without, highlighting the efficacy of the prompt tuning. However, ViT/VPT still exhibits a marginal performance gap compared to our model. This observation justifies that the MFA-ViT and MPT are pivotal in exploring complex relationships among the modalities.

## 4.3.2 Cross-modality Recognition

In Table 1, the MFA-ViT/MPT model exhibits notable average recognition accuracies for both the $f { - } p$ and $p { - } f$ scenarios. Specifically, in the $f { - } p$ , the model achieved accuracies of 86.70% on the Ethnic dataset, 90.38% on Face-Scrub, 75.28% on IMDB, and 72.01% on Cross-Modal DB. Conversely, in the $p { - } f$ configuration, it attains impressive average accuracies of 89.06% on Ethnic, 92.02% on FaceScrub, 76.36% on IMDB, and 75.96% on Cross-Modal DB. The outcomes underscore the consistent superiority of our model over existing methods, highlighting its effectiveness in addressing cross-modality recognition across four datasets.

The baseline among benchmark methods significantly underperformed, scoring below 1% in evaluations, indicating the distinct nature of face and periocular modalities despite periocular being a face subset. Hence, modality alignment is crucial for cross-modal matching. In contrast, [5] and [22] employed contrastive learning for enhanced performance, introducing a trade-off between intraand cross-modality recognition. As intra-modality performance improved, cross-modality recognition degraded, and vice versa, compared to our model. Their focus on broad embedding space alignment overshadowed nuances in $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ attributes, potentially causing suboptimal recognition across modalities.

As shown in Table 1, ViT/VPT and HA-ViT, even trained with ${ \mathbf I } _ { a } ,$ did not perform as effectively as our approach. This can be attributed to HA-ViT primarily focusing on utilizing cross-modality loss functions, neglecting the benefits of prompt tuning for enhancing multimodal features. On the other hand, ViT/VPT, while utilizing prompt tuning, encountered challenges in effectively handling multimodal features.

In summary, results demonstrate MFA adeptly aligns cross-modality relations, enabling precise matching without contrastive learning trade-offs. Simultaneously, MPT efficiently guides the model, capturing cross-modality and multimodal dependencies. Collaborative synergy between MPT and ${ \mathbf I } _ { a }$ enhances the understanding, contributing to superior cross-modal recognition. This underscores the efficacy of our approach to FBR challenges.

## 4.4. Ablation Study

## 4.4.1 Effects of Soft-biometric Attributes

We undertook an ablation study to evaluate the effectiveness of ${ \mathbf I } _ { a }$ using MFA-ViT and HA-ViT. Notably, both networks were trained to employ the MPT approach. For a fair comparison, soft-biometric attributes are integrated into the input embeddings of HA-ViT with a feature tokenizer during training. Table 1 and Figure 3 demonstrate that MFA-ViT outperforms other models when equipped with $\mathbf { I } _ { a } .$ . Surprisingly, HA-ViT, whether with or without ${ \mathbf I } _ { a } ,$ lagged behind MFA-ViT, even though both models employed the identical $\mathcal { L } _ { \mathrm { t o t a l } }$ loss function.

One plausible rationale behind the enhanced utilization of ${ \mathbf I } _ { a }$ by MFA-ViT lies in its architectural refinement, designed for seamless integration of features derived from diverse modalities with ${ \mathbf I } _ { a }$ . This empowers the network to process multiple information sources efficiently, equipping it to capture the rich information encapsulated within $\mathbf { I } _ { a } .$ . In contrast, HA-ViT, despite sharing a similar network structure, appears to lack an effective fusion mechanism, relying primarily on simple concatenation for aggregating multimodal features. This primitive fusion approach in HA-ViT may account for its comparatively suboptimal utilization of ${ \mathbf I } _ { a }$ compared to MFA-ViT.

Table 1. FBR performance comparisons on intra-modality $( f - f$ and $p { - } p )$ and cross-modality $( f - p$ and $p { - } f )$ recognition tasks in terms of rank-1 recognition (%). The best accuracy is written in bold.
<table><tr><td rowspan="2">Model</td><td colspan="4">Ethnic</td><td colspan="4">FaceScrub</td><td colspan="4">IMDB</td><td colspan="4">Cross-Modal DB</td></tr><tr><td> $\overline { { f - f } }$ </td><td> $\underline { { p {  } p } }$ </td><td> $\underline { { f - p } }$ </td><td> $\underline { { { p \mathrm { - } } f } }$ </td><td> $f { - } f$ </td><td> $\underline { { p { \mathrm { - } } p } }$ </td><td>f-p</td><td> $p { - } f$ </td><td> $\overline { { f - f } }$ </td><td> $\underline { { p { \mathrm { - } } p } }$ </td><td> $\underline { { f - p } }$ </td><td>p f</td><td> $\underline { { f - f } }$ </td><td>p-p</td><td>f-p</td><td>p f</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Model trained independently on Face or Periocular</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>80.61</td><td>61.79</td><td>0.39</td><td>0.33</td><td>76.10</td><td>63.29</td><td>0.47</td><td>0.42</td><td>62.08</td><td>55.31</td><td>0.10</td><td>0.09</td><td>60.43</td><td>51.43</td><td>0.09</td><td>0.07</td></tr><tr><td>Model trained with Face and Periocular</td><td></td><td colspan="9"></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tiong et al. [32]</td><td>80.30</td><td>65.42</td><td>0.44</td><td>0.51</td><td>76.46</td><td>72.13</td><td>0.57</td><td>0.61</td><td>65.31</td><td>54.91</td><td>0.18</td><td>0.19</td><td>64.70</td><td>50.18</td><td>0.11</td><td>0.13</td></tr><tr><td>George et al. [5]</td><td>85.98</td><td>66.90</td><td>54.18</td><td>59.46</td><td>89.80</td><td>77.76</td><td>62.31</td><td>66.82</td><td>72.58</td><td>54.72</td><td>34.91</td><td>43.57</td><td>70.56</td><td>47.16</td><td>29.50</td><td>38.72</td></tr><tr><td>Ng et al. [22] ViT/VPT [10]</td><td>92.55</td><td>79.29</td><td>71.54</td><td>75.36</td><td>91.82</td><td>85.74</td><td>75.15</td><td>78.36</td><td>77.19</td><td>71.99</td><td>58.56</td><td>61.19</td><td>76.21</td><td>65.63</td><td>58.13</td><td>61.35</td></tr><tr><td>HA-ViT [33]</td><td>91.80</td><td>83.94</td><td>76.68</td><td>77.74</td><td>93.11</td><td>89.26</td><td>83.35</td><td>85.17</td><td>78.57</td><td>72.15</td><td>60.46</td><td>62.32</td><td>80.97</td><td>69.01</td><td>63.13</td><td>66.42</td></tr><tr><td></td><td>91.36</td><td>84.32</td><td>76.34</td><td>77.65</td><td>92.13 88.52</td><td></td><td>80.27</td><td>81.80</td><td>78.23</td><td>71.92</td><td>59.49</td><td>59.68</td><td>78.76</td><td>66.87</td><td>59.93</td><td>61.87</td></tr><tr><td>Model trained with Face, Periocular, and Soft-biometric Attributes</td><td></td><td colspan="9"></td><td colspan="4"></td><td></td><td></td></tr><tr><td>ViT/VPT [10] HA-ViT [33]</td><td>92.95</td><td>85.68</td><td>83.37</td><td>86.10</td><td>93.57</td><td>90.84</td><td>87.43</td><td>88.61</td><td>81.97</td><td>74.59</td><td>69.40</td><td>70.21</td><td>82.76</td><td>70.92</td><td>65.99</td><td>69.11</td></tr><tr><td></td><td>91.72</td><td>85.10</td><td>80.03</td><td>81.61</td><td>92.46</td><td>88.70</td><td>84.33</td><td>85.63</td><td>78.81</td><td>71.42</td><td>64.13</td><td>65.49</td><td>78.81</td><td>67.22</td><td>62.34</td><td>64.03</td></tr><tr><td>MFA-ViT/MPT</td><td>94.82</td><td>89.98</td><td>86.70</td><td>89.07</td><td>95.71</td><td>93.06</td><td>90.38</td><td>92.02</td><td>86.03</td><td>80.53</td><td>75.28</td><td>76.36</td><td>85.88</td><td>76.54</td><td>72.01</td><td>75.96</td></tr></table>

![](images/785583a4c5966ed243b8f77e04facf80181c8d19efb096f1b6f737456dd4052a.jpg)  
Figure 3. Performance comparison on the proposed model trained with or without ${ \mathbf { I } } _ { a }$ against HA-ViT.

## 4.4.2 With and Without MPT

To gauge the efficacy of the MPT strategy, we compare the MFA-ViT’s performance with and without the incorporation of MPT. As illustrated in Figure 4, a progressive enhancement in performance is observed when MFA-ViT is integrated with MPT, compared to its operation without MPT. However, it is crucial to underscore that the performance of MFA-ViT without MPT lags in delivering substantial rank-1 results in intra- and cross-modality recognition tasks.

![](images/77058bfacd95e7e557bf65fe37610e22d9e34bdf7ad9a58ea7969a93c5484207.jpg)  
Figure 4. Performance comparison on MPT, VPT, and no prompt. This analysis exclusively focuses on the evaluation of MFA-ViT trained using $\mathbf { I } _ { f } , \mathbf { I } _ { p } ,$ , and ${ \mathbf { I } } _ { a }$

To gain deeper insights into the advantages of the MPT strategy, we present visualization results using the Grad-CAM [28]. As shown in Figure 5, MFA-ViT/MPT focuses intensively on the shared areas of the eye regions in the $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ images. Conversely, MFA-ViT does not employ a prompt strategy, further limiting the model’s ability to discern less specific features compared to the comprehensive approach offered by the MPT strategy.

These observations vividly showcase the robustness of MPT. Without the guidance of MPT, MFA-ViT struggles to discern and accentuate specific features across modalities precisely. However, the integration of MPT enables MFA-ViT to manage input sequences adeptly across its multimodal fusion attention layers. This goes beyond just aligning different modalities; it is also about discerning and preserving the unique attributes of each while fostering cohesive cross-modality collaboration.

Table 2. Performance comparisons on classification head inputs (CLS and PRM) in terms of rank-1 recognition (%). The best accuracy is indicated in bold.
<table><tr><td colspan="2">Head Input</td><td colspan="4">Ethnic</td><td colspan="4">FaceScrub</td><td colspan="4">IMDB</td><td colspan="4">Cross-Modal DB</td></tr><tr><td>CLS PRM</td><td>ff</td><td>p-p</td><td></td><td>p f</td><td></td><td>f-f</td><td></td><td>f-p</td><td>p-f</td><td>f-f</td><td>p-P</td><td>f-p</td><td>p-f</td><td>f-f</td><td>p-p</td><td>f-p</td><td>p f</td></tr><tr><td rowspan="3">√</td><td></td><td>94.57</td><td>89.18</td><td>86.24 88.34</td><td></td><td>95.26</td><td>92.63</td><td>89.81</td><td>91.51</td><td>85.32</td><td>79.43</td><td>74.52</td><td>76.36</td><td>85.11</td><td>75.53</td><td>71.06</td><td>74.35</td></tr><tr><td>√ 94.82</td><td></td><td>89.98</td><td>86.70 89.07</td><td></td><td>95.71</td><td>93.06</td><td>90.38</td><td>92.02</td><td>86.03</td><td>80.53</td><td>75.28</td><td>77.37</td><td>85.88</td><td>76.54</td><td>72.01</td><td>75.96</td></tr></table>

![](images/5e344ea87f464558fe23a6d4ce4b38069ec6d041bebbe27485a87dc9715cecba.jpg)  
Figure 5. Visualization of activation maps for MFA-ViT with MPT, VPT, and no prompt strategies.

## 4.4.3 MPT vs VPT

We further investigate the effectiveness of MPT compared to VPT within the MFA-ViT architecture. Since VPT was initially designed only for ViT, we extend it to demonstrate the performance of MPT in utilizing attributes in contrast to VPT. As illustrated in Figure 4, we observe a gradual increase in the performance of MFA-ViT/MPT compared to MFA-ViT/VPT. However, it is essential to note that the performance of MFA-ViT/VPT still falls short of achieving significant rank-1 results in intra- and cross-modality recognition tasks.

Delving deeper, the visualizations in Figure 5 elucidate that MFA-ViT/MPT is impressive because it adeptly attends the ocular region on the $\mathbf { I } _ { f }$ image and simultaneously on the $\mathbf { I } _ { p }$ modalities. In contrast, MFA-ViT/VPT appears to have difficulty differentiating specific features between $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ modalities. This is because the VPT strategy was initially designed for task-specific features, lacking guidance for the model to understand the intricate association between multimodal features.

These findings underscore the prowess of MPT in facilitating a seamless interplay between $\mathbf { I } _ { f }$ and $\mathbf { I } _ { p }$ modalities with $\mathbf { I } _ { a } .$ . The nuanced guidance steers the model towards a delicate understanding of the modalities, distinguishing it from VPT. Moreover, the results indicate that VPT’s functionality is impacted by neglecting important multimodal features within each layer of the learning process.

## 4.4.4 Impacts of Classification Head Inputs

In this subsection, we investigate the influence of different classification head inputs on our model’s performance in FBR. We consider two types of classification head input: the class token embedding T (CLS) follow the standard practice as in ViT [4] and the joint embeddings J from multimodal-prompt token embedding (PRM). As shown in Table 2, the result reveals a significant performance degradation when the model is trained with CLS. In contrast, training with PRM consistently improves performance across all benchmark datasets.

The superiority of PRM over CLS is underscored by these findings. PRM exhibits effective alignment with FBR tasks, enabling the model to acquire a more discriminative embedding. This is achieved through prompt embeddings to guide attention toward specific features and relationships within multimodal data. In contrast, CLS lacks cross-modality feature alignment, potentially leading to a failure in capturing delicate relationships crucial for addressing FBR problems.

## 5. Discussion and Conclusion

This paper introduces the MFA-ViT approach to address the intricate challenges of flexible biometric recognition (FBR). MFA-ViT leverages an MFA and an MPT to capture crossmodality and multimodal dependencies in embedding. The MPT is crucial in facilitating the acquisition of intermodal associations, which is vital in intra- and cross-modality recognition tasks. The integration enhances the comprehensive understanding of data, particularly in challenging real-world scenarios, significantly boosting recognition performance. Additionally, incorporating soft-biometric attributes provides further contextual insights, strengthening the discriminative potential of our embeddings. For future work, we see the potential to extend FBR to encompass other biometric modalities, paving the way for exceptional accuracy and efficiency in diverse recognition tasks.

## 6. Acknowledgement

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIP) (NO. NRF-2022R1A2C1010710).

## References

[1] Elisa Barroso, Gil Santos, Luis Cardoso, Chandrashekhar Padole, and Hugo Proenc¸a. Periocular recognition: How much facial expressions affect performance? Pattern Anal. Appl., 19:517–530, 2016.

[2] Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, and Andrew Zisserman. VGGFace2: A dataset for recognising faces across pose and age. In Int. Conf. Autom. Face Gesture Recog., pages 67–74, 2018.

[3] Kai Cheng, Xin Liu, Yiu-ming Cheung, Rui Wang, Xing Xu, and Bineng Zhong. Hearing like seeing: Improving voice-face interactions and associations via adversarial deep semantic matching network. In ACM MM, page 448–455, 2020.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[5] Anjith George and Sebastien Marcel. Cross modal focal loss for RGBD face anti-spoofing. In CVPR, pages 7878–7887, 2021.

[6] Ester Gonzalez-Sosa, Julian Fierrez, Ruben Vera-Rodriguez, and Fernando Alonso-Fernandez. Facial soft biometrics for recognition in the wild: Recent works, annotation, and cots evaluation. IEEE Trans. Inf. Forensics Secur., 13(8):2001– 2014, 2018.

[7] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In NeurIPS, pages 18932–18943, 2021.

[8] JongWon Hwang, Leslie Ching Ow Tiong, and Andrew Beng Jin Teoh. Towards face representation learning conditioned on the soft biometrics. In Int. Conf. Mach. Vis. Appl., page 1–7, 2022.

[9] Seyed Mehdi Iranmanesh, Ali Dabouei, and Nasser Nasrabadi. Attribute adaptive margin softmax loss using privileged information. In BMVC, pages 1–13, 2020.

[10] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In ECCV, page 709–727, 2022.

[11] Luo Jiang, Juyong Zhang, and Bailin Deng. Robust RGB-D face recognition using attribute-aware loss. IEEE TPAMI, 42 (10):2552–2566, 2020.

[12] Yoon Gyo Jung, Cheng Yaw Low, Jaewoo Park, and Andrew Beng Jin Teoh. Periocular recognition in the wild with generalized label smoothing regularization. IEEE Sign. Process. Letters, 27:1455–1459, 2020.

[13] Changil Kim, Hijung Valentina Shin, Tae-Hyun Oh, Alexandre Kaspar, Mohamed Elgharib, and Wojciech Matusik. On learning associations of faces and voices. In ACCV, pages 276–292, 2018.

[14] Takumi Kobayashi. Large margin in softmax cross-entropy loss. In BMVC, pages 1–12, 2019.

[15] Nagashri N Lakshminarayana, Nishant Sankaran, Srirangaraj Setlur, and Venu Govindaraju. Multimodal deep feature aggregation for facial action unit recognition using vis-

ible images and physiological signals. In Int. Conf. Autom. Face Gesture Recog., pages 1–4, 2019.

[16] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9):1–35, 2023.

[17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, pages 1–10, 2019.

[18] Iacopo Masi, Yue Wu, Tal Hassner, and Prem Natarajan. Deep face recognition: A survey. In Conf. Graph. Patterns Images, pages 471–478, 2018.

[19] Shervin Minaee, Amirali Abdolrashidi, Hang Su, Mohammed Bennamoun, and David Zhang. Biometrics recognition using deep learning: A survey. Artif. Intell. Rev., 56 (8):8647–8695, 2023.

[20] Arsha Nagrani, Samuel Albanie, and Andrew Zisserman. Learnable PINS: Cross-modal embeddings for person identity. In ECCV, page 71–88, 2018.

[21] Hong-Wei Ng and Stefan Winkler. A data-driven approach to cleaning large face datasets. In ICIP, pages 343–347, 2014.

[22] Tiong-Sik Ng, Cheng-Yaw Low, Jacky Chen Long Chai, and Andrew Beng Jin Teoh. Conditional multimodal biometrics embedding learning for periocular and face in the wild. In ICPR, pages 812–818, 2022.

[23] Chandrashekhar N. Padole and Hugo Proenc¸a. Periocular recognition: Analysis of performance degradation factors. In Int. Conf. Biometrics (ICB), pages 439–445, 2012.

[24] Unsang Park, Arun Ross, and Anil K. Jain. Periocular biometrics in the visible spectrum: A feasibility study. In Int. Conf. Biometrics Theory Appl. Syst. (BTAS), pages 1–6, 2009.

[25] Hugo Proenc¸a and Joao C. Neves. Deep-PRWIS: Periocular˜ recognition without the iris and sclera using deep learning frameworks. IEEE Trans. Inf. Forensics Secur., 13(4):888– 896, 2018.

[26] Rasmus Rothe, Radu Timofte, and Luc Van Gool. Deep expectation of real and apparent age from a single image without facial landmarks. IJCV, 126(2):144–157, 2018.

[27] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR, pages 4510–4520, 2018.

[28] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, pages 618–626, 2017.

[29] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, pages 1–14, 2015.

[30] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In Int. Conf. Mach. Learn. (ICML), page 6105–6114, 2019.

[31] Philipp Terhorst, Daniel F ¨ ahrmann, Jan Niklas Kolf, Naser ¨ Damer, Florian Kirchbuchner, and Arjan Kuijper. Maadface: A massively annotated attribute dataset for face images. IEEE Trans. Inf. Forensics Secur., 16:3942–3957, 2021.

[32] Leslie Ching Ow Tiong, Seong Tae Kim, and Yong Man Ro. Multimodal facial biometrics recognition: Dual-stream convolutional neural networks with multi-feature fusion layers. Image Vis. Comput., 102:103977, 2020.

[33] Leslie Ching Ow Tiong, Dick Sigmund, and Andrew Beng Jin Teoh. Face-periocular cross-identification via contrastive hybrid attention vision transformer. IEEE Sign. Process. Letters, 30:254–258, 2023.

[34] Hanrui Wang, Xingbo Dong, Zhe Jin, Jean-Luc Dugelay, and Massimo Tistarelli. Cross-spectrum face recognition using subspace projection hashing. In ICPR, pages 615–622, 2021.

[35] Mei Wang and Weihong Deng. Deep face recognition: A survey. Neurocomputing, 429:215–244, 2021.

[36] Rui Wang, Xin Liu, Yiu-ming Cheung, Kai Cheng, Nannan Wang, and Wentao Fan. Learning discriminative joint embeddings for efficient face and voice association. In ACM SIGIR Conf. Res. Develop. Inf. Retrieval, page 1881–1884, 2020.

[37] Gou Wei, Li Jian, and Sun Mo. Multimodal(audio, facial and gesture) based emotion recognition challenge. In Int. Conf. Autom. Face Gesture Recognit., pages 908–911, 2020.

[38] Peng Xu, Xiatian Zhu, and David A. Clifton. Multimodal learning with transformers: A survey. IEEE TPAMI, 45(10): 12113–12132, 2023.

[39] Yuhao Zhu, Min Ren, Hui Jing, Linlin Dai, Zhenan Sun, and Ping Li. Joint holistic and masked face recognition. IEEE Trans. Inf. Forensics Secur., 18:3388–3400, 2023.

[40] M. Zolfaghari, Y. Zhu, P. Gehler, and T. Brox. CrossCLR: Cross-modal contrastive learning for multi-modal video representations. In ICCV, pages 1430–1439, 2021.