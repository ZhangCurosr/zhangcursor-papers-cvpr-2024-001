# Data-Free Quantization via Pseudo-label Filtering

Chunxiao Fan<sup>1,2</sup>, Ziqi Wang<sup>1</sup>, Dan Guo<sup>1,2\*</sup>, Meng Wang<sup>1,2</sup>

<sup>1</sup>School of Computer Science and Information Engineering, Hefei University of Technology, Hefei, 230009, Anhui, China

<sup>2</sup>Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, 230088, Anhui, China

fanchunxiao@hfut.edu.cn, zackiewang29@gmail.com, guodan@hfut.edu.cn, eric.mengwang@gmail.com

## Abstract

Quantization for model compression can efficiently reduce the network complexity and storage requirement, but the original training data is necessary to remedy the performance loss caused by quantization. The Data-Free Quantization (DFQ) methods have been proposed to handle the absence of original training data with synthetic data. However, there are differences between the synthetic and original training data, which affects the performance of the quantized network, but none of the existing methods considers the differences. In this paper, we propose an efficient data-free quantization via pseudo-label filtering, which is the first to evaluate the synthetic data before quantization. We design a new metric for evaluating synthetic data using self-entropy, which indicates the reliability of synthetic data. The synthetic data can be categorized with the metric into high- and low-reliable datasets for the following training process. Besides, the multiple pseudo-labels are designed to label the synthetic data with different reliability, which can provide valuable supervision information and avoid misleading training by low-reliable samples. Extensive experiments are implemented on several datasets, including CIFAR-10, CIFAR-100, and ImageNet with various models. The experimental results show that our method can perform excellently and outperform existing methods in accuracy.

## 1. Introduction

Deep Neural Networks (DNNs) [26] have shown tremendous potential in a number of fields, but their high computing costs and storage requirements make the implementation complex, especially on embedded systems and edge devices [9, 36]. In recent years, many model compression methods [7] have been proposed to decrease the computational and memory requirements while keeping the performance. The existing methods can be divided into network pruning [41, 45], model quantization [2, 21], knowledge distillation [18, 31], and neural architecture search (NAS) [33, 39]. Among these techniques, quantization uses finite approximations to represent the full-precision values in the pre-trained network, which needs to be quantized. It can efficiently reduce the network complexity for acceleration and storage, but the approximate operation inevitably affects the network performance, resulting in accuracy drops after quantization.

![](images/bb7c68a0f38a99e34f6f592f8948f24e710ad8cfbcc85f973f45cd1101e52b14.jpg)  
Figure 1. The self-entropy of the prediction results on the original and synthetic data using pre-trained network. The prediction results on the synthetic dataset always have a higher self-entropy than that on the original dataset, owing to different reliability.

To reduce the performance loss caused by quantization, many methods propose to optimize the quantizer [1, 2, 10, 11, 35] or retrain the pre-trained network with quantization constraint [4, 15, 27]. In these methods, the original training data is very helpful for maintaining model performance, but it may not be feasible due to data privacy concerns in specific scenarios. The Data-Free Quantization (DFQ) methods [3, 5, 28, 30, 42, 47, 48] have been proposed to deal with the absence of original training data. In principle, the batch normalization (BN) [20] layers in the pre-trained model contain the statistical information of the original training data, i.e. mean and variance, which can be regarded as prior information for data synthesis [46]. Thus, the synthetic data can be generated from random initial data by guiding its distribution closer to the original data with the full-precision pre-trained model. Generally, exsiting DFQ methods can be divided into Generator-Based Approach (GBA) [5, 22, 37, 42] and Distill-Based Approach (DBA) [3, 28, 30, 47].

The GBA trains a specific generative network (e.g., GAN [12] or VAE [24]) to generate the synthetic data, which requires a complex training process for the generative network. The DBA regards the synthetic data as trainable, and iterates them to fit the original data distribution using the back-propagation of the pre-trained model, avoiding the training process for the generative network. However, although many exquisite technologies are designed in these existing methods, a noticeable difference still exists between the synthetic and original training data, which affects the performance of the quantized network. Thus, it is necessary to evaluate the synthetic training data before quantization.

![](images/bdedceacabbf530b217ded93986c36aaab5fe0b51eaff9e713d2b4b36dab558c.jpg)  
Figure 2. The difference between existing data-free methods and our method. The existing methods optimize the quantizer or retrain the pre-trained network with quantization constraint directly without evaluating the synthetic data. We propose to evaluate the synthetic data with self-entropy and divide the synthetic data into high- and low-reliable datasets before training the quantized network for better performance.

In this paper, we propose to evaluate the synthetic data before quantization as shown in Figure 2. To complete this goal, we aim to deal with three problems: 1) How to evaluate the synthetic data? In existing frameworks, the synthetic training data is generated with the information in the pre-trained model, and it lacks clear evaluation indicators for the synthetic data. 2) How to label the evaluated synthetic data? After evaluating the synthetic data, it is necessary to assign suitable labels to the synthetic data according to different evaluation results, which aims to further improve the reliability of synthetic data and avoid misleading caused by inappropriate labels. 3) How to design the training process with evaluated synthetic data? The training process needs to be able to learn supervision information from the synthetic data with different labels, and more importantly, avoid misleading caused by the data with low evaluation results.

To overcome the problems above, we propose an efficient Data-Free Quantization via Pseudo-label Filtering as follows: 1) The self-entropy [23] is used as a metric for synthetic data to evaluate the reliability of the pre-trained network on it. We notice that the pre-trained network has a relatively higher self-entropy on the synthetic dataset than the original dataset, as shown in Figure.1, owing to different reliability performance, so the self-entropy can be used as a metric to evaluate the reliability of synthetic data before quantization. 2) The synthetic data is labeled using multiple pseudo-labels, which include major and auxiliary pseudolabels, to reflect its reliability. Major pseudo-labels are assigned to high-reliable samples, providing valuable supervision information. In contrast, low-reliable samples are labeled with auxiliary pseudo-labels to give a soften supervised learning for enhancing the robustness of the quantized network and avoiding misled by low-reliable samples. 3) The pseudo-label training is designed to integrate the supervision information provided by major and auxiliary pseudolabels. The knowledge distillation is selected as the basic framework to train the quantized model, which has similar performance and intermediate features as the pre-trained network using the proposed synthetic data evaluation.

The main contributions of this work can be summarized as follows:

• We propose an efficient data-free quantization via pseudo-label filtering, which is the first to evaluate the synthetic data before quantization, so that the samples with different reliability can provide different information and improve the performance.

• The self-entropy is used as the metric for the synthetic data evaluation, which represents the reliability of synthetic data under a specific label. With the metric, the synthetic data can be categorized into high- and low-reliable samples for the following training process.

• The multiple pseudo-labels are designed in our method to label the synthetic data with different reliability. It incorporates major and auxiliary pseudo-labels for the high- and low-reliable samples, providing valuable supervision information and avoiding misleading quantization by low-reliable samples.

• Extensive experiments are conducted on the CIFAR-10, CIFAR-100, and ImageNet in various models. Our method can achieve superior performances compared with existing DFQ methods.

## 2. Related Work

Quantization. Quantization is widely used in the model compression for neural networks to represent the fullprecision model with low-bit approximations. To alleviate the performance loss caused by approximate operations, many methods have been proposed, which can be grouped into post-training quantization (PTQ) [1, 13, 34, 35] and quantization-aware training (QAT) [2, 4, 10, 11]. The PTQ methods take calibration data to optimize the parameters in the quantizer, and the QAT methods retrain the pre-trained network with a quantization constraint. However, the original training data is necessary for these methods, it may not be feasible in specific scenarios. To deal with the challenge without original training data, the Data-Free Quantization (DFQ) is proposed.

![](images/d5fa420b76bb909214bb1e9e0e7d2aadc78fb99b3197004ab72d8b73f436fc5b.jpg)  
Figure 3. The framework of the proposed method. In the proposed method, synthetic data is evaluated with self-entropy of the pre-trained network on the synthetic data, which is divided into high- and low-reliable data. Then, the multiple pseudo-labels are used to label the evaluated synthetic data according to its reliability. The high-reliable samples are assigned with major pseudo-labels, and the low-reliable samples use auxiliary pseudo-labels. With these multiple pseudo-labels, the pseudo-labels cross-entropy loss can be obtained and combined with the MSE loss function to guide the quantized network has similar prediction ability with pre-trained network.

Data-free Quantization. Many works [14, 34, 43] attempt to optimize the quantized network solely relying on the information inherent in the pre-trained network itself without the demand for training data. D-FQ [34] proposes the method of weight equalization and bias correction to make the network more suitable to quantize. SQuant [14] adopts a new rounding metric based on a diagonal Hessian approximation to improve the performance of the quantized network. However, owing to the absence of training data to adjust the pre-trained network, these methods can hardly achieve a significant performance improvement.

Many methods propose to utilize generated data, and can be divided into Generator-Based Approach (GBA) [5, 32, 37, 38, 42, 44, 48] and Distill-Based Approach (DBA) [3, 30, 30, 47]. (1) GBA proposes to use a generator for synthesizing training data. GDFQ [42] constructs informative data from the pre-trained model and generates data approximating the original dataset. Qimera [5] proposes to use superposed latent embeddings to generate higher-quality samples. These methods always demand much time and resources to generate high-quality data. (2) DBA utilizes the pre-trained model to directly optimize the generated data, thereby eliminating the need for the generator. ZeroQ [3] matches the statistics of batch normalization to optimize for a distilled dataset. IntraQ [47] attempts to synthesize images with intra-class heterogeneity. HAST [28] generates more hard samples to enhance model training effectiveness.

However, even with many ingenious designs, there is a difference between the generated synthetic data and the original training data, but none of the existing methods takes this into account, and the synthetic data is directly used for the training of the quantized network. In the proposed method, we propose to evaluate the generated synthetic data and label different samples according to the evaluation results to enhance the performance of the quantized network.

## 3. Proposed Method

The framework of our proposed method is illustrated in Figure 3, including Reliability Filtering, Multiple Pseudo-Labels, and Pseudo-Label Training.

## 3.1. Reliability Filtering

In principle, the mean and variance in BN layers of the pre-trained model are affected by the original training data. Thus, the synthetic data can be generated from random initial samples by adapting the output distribution of BN layers the same as that stored in the pre-trained model with:

$$
\mathcal { L } _ { \mathrm { B N } } = \sum _ { l = 1 } ^ { L } \big ( | | \pmb { \mu } _ { l } ^ { p } - \pmb { \mu } _ { l } ^ { s } | | _ { 2 } ^ { 2 } + | | \pmb { \sigma } _ { l } ^ { p } - \pmb { \sigma } _ { l } ^ { s } | | _ { 2 } ^ { 2 } \big ) ,\tag{1}
$$

where L denotes the layer number of the model. $\mu _ { l } ^ { p }$ and $\pmb { \sigma } _ { l } ^ { p }$ are the mean and variance stored in l-th BN layer of the pretrained model, and $\pmb { \mu } _ { l } ^ { s }$ and $\pmb { \sigma } _ { l } ^ { s }$ are the mean and variance of synthetic data batch in l-th layer. The synthetic data is generated by minimize ${ \mathcal { L } } _ { \mathrm { D A T A } }$ as,

$$
\mathcal { L } _ { \mathrm { { D A T A } } } = \mathcal { L } _ { \mathrm { { B N } } } + \gamma \cdot \mathcal { L } _ { \mathrm { { I L } } } ,\tag{2}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { I L } } = ~ \sum _ { i = 1 } ^ { N } \mathrm { C E } \left( P ( \hat { \mathbf { x } } _ { i } ) , \hat { y } _ { i } \right) } \end{array}$ is the loss to improve the predicted probability of pre-trained network for the assigned label [16]. N denotes the number of samples for the synthetic data, and $\operatorname { C E } ( { \mathrm { \cdot } } )$ represents the cross-entropy loss. $\hat { \mathbf { x } } _ { i }$ denotes the generated synthetic sample and $P \left( \hat { \pmb x } _ { i } \right)$ represents its predicted probability after the softmax layer. $\hat { y } _ { i }$ denotes the assigned label. γ is a hyper-parameters to balance two losses of $\mathcal { L } _ { \mathrm { B N } }$ and ${ \mathcal { L } } _ { \mathrm { I L } }$

However, the synthetic data can hardly have precisely the same features as the original data. As discussed above, we notice that the synthetic dataset always has a higher selfentropy compared with the original dataset as shown in Figure 1. The reason is that the pre-trained network is trained on the original dataset, which can have high reliability on most samples. Owing to the difference between the original and synthetic data, it is impossible for the pre-trained network to have high reliability on the whole synthetic data, and the low-reliable data may lead to incorrect guidance during training and seriously influence the quantization performance.

Thus, we apply self-entropy as a metric to evaluate the reliability of the pre-trained network on the synthetic data as in Eq. 3:

$$
\mathcal { H } _ { \mathrm { s e l f } } ( \hat { \pmb x } _ { i } ) = - \frac { 1 } { \log N _ { c } } \sum _ { c = 1 } ^ { N _ { c } } ( \mathbb { P } ( \hat { \pmb x } _ { i } , c ) \cdot \log \left( \mathbb { P } ( \hat { \pmb x } _ { i } , c ) \right) ) ,\tag{3}
$$

where $N _ { c }$ refers to the number of prediction classes, and $\mathbb { P } \left( \hat { \pmb { x } } _ { i } , c \right)$ denotes the predicted probability of the class c obtained by the pre-trained network.

In principle, $\mathcal { H } _ { \mathrm { s e l f } }$ indicates the uncertainty, whereas a low self-entropy for the predicted results can refer to a reliable prediction with high confidence [23]. In other words, low $\mathcal { H } _ { \mathrm { s e l f } }$ means the pre-trained network can have a high possibility for the specific label on the synthetic data, which means it has high reliability for the samples, so it can provide adequate supervision information for training the quantized network to fit the performance of the pre-trained network.

Based on the reliability evaluation, the samples in the synthetic dataset $\hat { X }$ can be divided into high-reliable dataset ${ \hat { \hat { X } } } ^ { h }$ and low-reliable dataset $\hat { X } ^ { l }$ with a reliable threshold t as,

$$
\begin{array} { r l r } & { } & { \hat { X } ^ { h } = \left\{ \hat { \pmb { x } } ^ { h } \Big | \hat { \pmb { x } } ^ { h } \in \hat { X } , \mathcal { H } _ { \mathrm { s e l f } } \left( \hat { \pmb { x } } ^ { h } \right) \leqslant t \right\} , } \\ & { } & { \hat { X } ^ { l } = \left\{ \hat { \pmb { x } } ^ { l } \Big | \hat { \pmb { x } } ^ { l } \in \hat { X } , \mathcal { H } _ { \mathrm { s e l f } } \left( \hat { \pmb { x } } ^ { l } \right) > t \right\} , } \end{array}\tag{4}
$$

where t is a dynamic threshold parameter for fast convergence and learning more supervision information. To converge fast, the requirement for high-reliable samples can be relaxed at the beginning of training to provide more samples and quickly improve network performance; As the training progresses, the network performance gradually improves, which raises the standard for high-reliable samples. It is necessary to use high-reliable samples to provide better supervision information and avoid the impact of low-reliable samples on network performance. At the same time, using more low-reliable samples can also improve the robustness of the network. Thus, t continuously decrease as the training epoch increases:

$$
\scriptstyle t = T _ { u } - f _ { t } ( e p o c h ) ( T _ { u } - T _ { l } ) ,\tag{5}
$$

where $T _ { l }$ and $T _ { u }$ are the lower and upper boundaries, and $t \in [ T _ { l } , \ T _ { u } ]$ , which continuously decrease as the training epoch increase. $\begin{array} { r } { f _ { t } ( e p o c h ) ~ = ~ \frac { e \dot { p } o c h } { E } } \end{array}$ , epoch and E denote the current and whole epoch for training, respectively.

## 3.2. Multiple Pseudo-Labels

With the reliability filtering, the samples in ${ \hat { X } } ^ { h }$ have high reliability and can provide more supervision information for the quantized model. In the proposed method, major pseudo-labels are designed for these samples, and the pseudo-label $\hat { y } _ { i } ^ { m }$ for the high-reliable sample $\hat { \pmb { x } } _ { i } ^ { h }$ is assigned as,

$$
\hat { y } _ { i } ^ { m } = \arg \operatorname* { m a x } _ { c \in \mathbb { C } } \mathbb { P } ( \hat { \pmb { x } } _ { i } ^ { h } , c ) ,\tag{6}
$$

where $\mathbb { C } = \{ 1 , 2 , . . . , N _ { c } \}$ denotes the set of all the possible prediction classes. We define the loss of $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ for supervised learning with high-reliable dataset:

$$
\mathcal { L } _ { \mathrm { C E } } ^ { h } = \frac { 1 } { N _ { h } } \sum _ { i = 1 } ^ { N _ { h } } \mathrm { C E } \left( P _ { q } ( \hat { \pmb x } _ { i } ^ { h } ) , \hat { \pmb y } _ { i } ^ { m } \right) ,\tag{7}
$$

where $N _ { h }$ refers to the number of high-reliable samples, and $P _ { q } \left( \hat { \pmb x } _ { i } ^ { h } \right)$ denotes the predicted probability for all classes with the quantized network on high-reliable samples $\hat { \pmb { x } } _ { i } ^ { h }$

Generally, the high-reliable samples account for a small portion of the total samples [23]. In quantization, we aim to train the quantized network, which can fit the performance of the pre-trained network. The low-reliable synthetic data can also reflect the features of the pre-trained network and provide some information for robustness. But owing to the low predicted probability for the labels with low reliability, they may mislead the training if applied directly in training.

To deal with it, auxiliary pseudo-labels are designed to give a soften supervised learning with the low-reliable samples, which consists of the primary label $\hat { y } _ { i } ^ { p }$ and secondary label $\hat { y } _ { i } ^ { s }$ . The primary label $\hat { y } _ { i } ^ { p }$ represents the label with the highest predicted probability, and the secondary label $\hat { y } _ { i } ^ { s }$ represents that with the second highest probability. In the prediction with high self-entropy, the labels excluding $\hat { y } _ { i } ^ { p }$ can also have a decent probability, especially the secondary label $\hat { y } _ { i } ^ { s }$ . Thus, the pseudo-labels from $\hat { y } _ { i } ^ { p }$ and $\hat { y } _ { i } ^ { s }$ can be leveraged to enhance the training and mitigate the risk of misleading for these samples [29]. The primary label $\hat { y } _ { i } ^ { p }$ and secondary label $\hat { y } _ { i } ^ { s }$ can be obtained as,

$$
\begin{array} { r l } & { \hat { y } _ { i } ^ { p } = \underset { c \in \mathbb { C } } { \arg \operatorname* { m a x } } ~ \mathbb { P } ( \hat { \pmb { x } } _ { i } ^ { l } , c ) , } \\ & { \hat { y } _ { i } ^ { s } = \underset { c \in \hat { \mathbb { C } } } { \arg \operatorname* { m a x } } ~ \mathbb { P } ( \hat { \pmb { x } } _ { i } ^ { l } , c ) , } \end{array}\tag{8}
$$

where $\hat { \mathbb { C } }$ denotes the set of classes removing the class with the highest predicted probability.

The auxiliary cross-entropy $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ loss are defined with the primary and secondary labels $\hat { y } _ { i } ^ { p } , \hat { y } _ { i } ^ { s }$ as,

$$
\mathcal { L } _ { \mathrm { C E } } ^ { l } = \frac { 1 } { N _ { l } } \sum _ { i = 1 } ^ { N _ { l } } \big ( \lambda _ { i } ^ { p } \cdot \mathrm { C E } \left( P _ { q } ( \hat { \pmb { x } } _ { i } ^ { l } ) , \hat { \pmb { y } } _ { i } ^ { p } \right)\tag{9}
$$

where $N _ { l }$ refers to the number of low-reliable samples, $\lambda _ { i } ^ { p }$ and $\lambda _ { i } ^ { s }$ represent the weights of primary and secondary labels to balance their importance, and can be obtained as,

$$
\begin{array} { r l } & { \lambda _ { i } ^ { p } = \frac { \mathbb { P } ( \hat { x } _ { i } ^ { l } , p ) } { \mathbb { P } ( \hat { x } _ { i } ^ { l } , p ) + \mathbb { P } ( \hat { x } _ { i } ^ { l } , s ) } , } \\ & { \lambda _ { i } ^ { s } = \frac { \mathbb { P } ( \hat { x } _ { i } ^ { l } , s ) } { \mathbb { P } ( \hat { x } _ { i } ^ { l } , p ) + \mathbb { P } ( \hat { x } _ { i } ^ { l } , s ) } , } \end{array}\tag{10}
$$

where $\mathbb { P } ( \hat { \pmb x } _ { i } ^ { l } , \boldsymbol { p } )$ and $\mathbb { P } ( \hat { \pmb x } _ { i } ^ { l } , s )$ denote the predicted probabilities of the primary and secondary labels with the pre-trained model, respectively.

With the designed major and auxiliary pseudo-labels, the high- and low-reliable samples can be simultaneously used to provide the supervision information for model training:

$$
\mathcal { L } _ { \mathrm { C E } } ^ { t o t a l } = \mathcal { L } _ { \mathrm { C E } } ^ { h } + \beta \cdot \mathcal { L } _ { \mathrm { C E } } ^ { l } ,\tag{11}
$$

where $\beta$ is the hyper-parameter to balance the importance of samples with different reliability, which is less than 1.

## 3.3. Pseudo-Label Training

To train a high-performance quantized model, we design a pseudo-label training based on the knowledge distillation framework [18], whereby the pre-trained network is employed as the teacher for the quantized network. In order to learn the supervision information from the designed major and auxiliary pseudo-labels, the designed cross-entropy loss $\mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ is used in training and combined with $\mathcal { L } _ { \mathrm { M S E } }$ to form the prediction similarity loss $\mathcal { L } _ { \mathrm { P } }$ , which aims to guide the quantized network has similar prediction ability with pretrained network as,

$$
{ \mathcal { L } } _ { \mathrm { P } } = { \mathcal { L } } _ { \mathrm { C E } } ^ { t o t a l } + \mu \cdot { \mathcal { L } } _ { \mathrm { M S E } } ,\tag{12}
$$

where $\mu$ is the hyperparameter to balance the two losses. $\mathcal { L } _ { \mathrm { M S E } }$ denotes the Mean Squared Error (MSE) between the intermediate feature layers of the two networks. Owing to the reduction of bits number for weights and activations in the quantized network, the features extracted by intermediate layers are significantly affected, resulting in performance degradation. $\mathcal { L } _ { \mathrm { M S E } }$ is utilized to minimize the difference between the intermediate feature layers of the two networks:

$$
\mathcal { L } _ { \mathrm { M S E } } = \sum _ { k = 1 } ^ { L } \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( f _ { k } \left( \hat { \pmb x } _ { i } \right) - f _ { k } ^ { q } \left( \hat { \pmb x } _ { i } \right) \right) ^ { 2 } \right) ,\tag{13}
$$

where $f _ { k } \left( \hat { \pmb x } _ { i } \right)$ and $f _ { k } ^ { q } \left( \hat { \pmb x } _ { i } \right)$ represent the outputs in the k-th layer of the full-precision model and the quantized model, respectively.

In common, the Kullback-Leibler (KL) divergence ${ \mathcal { L } } _ { \mathrm { K I } }$ [3] between the outputs of the pre-trained and quantized networks can be used to minimize the discrepancy between the outputs of the two networks, thereby ensuring the performance of the quantized network.

$$
\mathcal { L } _ { \mathrm { K L } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigg ( P ( \hat { \pmb { x } } _ { i } ) \cdot l o g \frac { P ( \hat { \pmb { x } } _ { i } ) } { P _ { q } ( \hat { \pmb { x } } _ { i } ) } \bigg ) ,\tag{14}
$$

where $P \left( \hat { \pmb x } _ { i } \right)$ and $P _ { q } \left( \hat { \pmb x } _ { i } \right)$ denotes predicted probability using the pre-trained and quantized networks, respectively.

Thus, the total loss function for the entire training process can be obtained as,

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { K L } } + \tau { \mathcal { L } } _ { \mathrm { P } } ,\tag{15}
$$

where hyperparameter τ is used to balance the weights of the two losses.

## 4. Experiment

## 4.1. Experiment Setup

Datasets and Networks. Data-Free Quantization is typically evaluated on CIFAR-10/100 [25] and ImageNet (ILSVRC2012) [8] datasets. The proposed method is implemented and examined with ResNet [17] and MobileNet [19], i.e. ResNet-20 on CIFAR-10/100; ResNet-18/50, and MobileNet-V1 on ImageNet (ILSVRC2012).

Implementation Details. For a fair comparison, we synthesize 5,120 images as the synthetic data, which is the same as the settings of IntarQ [47] and HAST [28], and these synthetic samples are optimized with 1000 iterations. The hyper-parameters γ are set as 10 for CIFAR-10/100 and 0.1 for ImageNet. We adopt the SGD optimizer with a weight decay of $1 0 ^ { - 4 }$ and momentum of 0.9 for model training. The initial learning rate is set as 0.001 for CIFAR-10/100 and $1 0 ^ { - 5 }$ for ImageNet. The hyper-parameters T<sub>l</sub>, $T _ { u } , \beta , \mu$ and τ are respectively set to 0.2, 0.5, 0.3, 100 and 1 for CIFAR-10/100 and 0.1, 0.4, 0.5, 4000 and 1 for ImageNet. All full-precision pre-trained models are provided by pytorchcv [42]. All layers are quantized, including the first and last layers of the model. Our implementation is conducted with PyTorch on a GPU Nvidia GTX 3090 Ti workstation, CUDA 11.4, and Ubuntu 18.04.

## 4.2. Comparisons with State-of-the-Art Methods

The proposed method is compared with state-of-the-art data-free quantization methods on CIFAR-10/100 and ImageNet as listed in Tables 1∼4. W-bit/A-bit represent the bits number of weights and activations after quantization, respectively. Among these methods, GDFQ [42], DSG [44], ZAQ [32], Qimera [5], ARC [6], ARC+AIT [48], AdaSG [38] and AdaDFQ [37] belong to GBA. ZeroQ [3], IntraQ [47] and HAST [28] belong to DBA.

CIFAR-10/100. For ResNet-20 on CIFAR-10/100, the model is quantized into 4-bit and 3-bit. As shown in Table 1, the proposed method achieves the superior performances at 92.47% (4-bit), 88.04% (3-bit) on CIFAR-10 and 66.94% (4-bit), 57.03% (3-bit) on CIFAR-100. So our method can get the higher performance compared with most of exsiting methods, i.e. +0.98% over IntraQ, +0.37% over AdaSG, +0.16% over AdaDFQ, and +0.11% over HAST on CIFAR-10 (4-bit). The performance improvement is higher on CIFAR-100 than CIFAR-10, i.e. +1.96% (4-bit) and +8.78% (3-bit) over IntraQ, +0.52% (4-bit) and +4.27% (3-bit) over AdaSG, +0.13% (4-bit) and +4.29% (3-bit) over AdaDFQ, and +0.26% (4-bit) and +1.36% (3-bit) over HAST. Generally, the proposed method can get the best performance except in 3-bit on CIFAR-10 dataset (0.30% lower than HAST). By analysis, HSAT aims to improve the quality of synthetic data by increasing the proportion of hard samples, resulting in more high quality synthetic data, but our method emphasises on the synthetic data evaluation for better utilization of synthetic data.

Table 1. Top-1 accuracy (%) comparison with the state-of-theart methods on CIFAR-10, CIFAR-100 for 3/4-bit ResNet-20. \* represents the results reimplemented in paper [37].
<table><tr><td rowspan=1 colspan=7>Dataset       Method        Venue</td><td rowspan=1 colspan=1>W4A4</td><td rowspan=1 colspan=1>W3A3</td></tr><tr><td rowspan=1 colspan=7>ZeroQ[3]     CVPR&#x27;20</td><td rowspan=1 colspan=1>84.68</td><td rowspan=1 colspan=1>29.32</td></tr><tr><td rowspan=10 colspan=7>GDFQ[42]     ECCV&#x27;20Qimera[5]    NeurIPS’21CIFAR-10      ARC[6]      IJCAI&#x27;21ForResNet-20  ARC+AIT[48](FP:94.03)    IntraQ[47]     CVPR’22AdaSG[38]     AAAI&#x27;23AdaDFQ[37]    CVPR&#x27;23HAST[28]     ICCV’23Ours            一</td><td rowspan=1 colspan=1>90.25</td><td rowspan=2 colspan=1>71.1032.90</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3>DSG[44]CVPR’21</td><td rowspan=1 colspan=1>88.74</td></tr><tr><td rowspan=1 colspan=1>91.26</td><td rowspan=3 colspan=1>74.43*</td></tr><tr><td rowspan=1 colspan=1>88.55</td></tr><tr><td rowspan=1 colspan=1>CVPR'22</td><td rowspan=1 colspan=1>90.49</td></tr><tr><td rowspan=1 colspan=1>91.49</td><td rowspan=1 colspan=1>77.07</td></tr><tr><td rowspan=1 colspan=1>92.10</td><td rowspan=1 colspan=1>84.14</td></tr><tr><td rowspan=1 colspan=1>92.31</td><td rowspan=1 colspan=1>84.89</td></tr><tr><td rowspan=1 colspan=1>92.36</td><td rowspan=1 colspan=1>88.34</td></tr><tr><td rowspan=1 colspan=1>92.47</td><td rowspan=1 colspan=1>88.04</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td rowspan=1 colspan=3>ZeroQ[3]     CVPR&#x27;20</td><td rowspan=1 colspan=1>58.42</td><td rowspan=1 colspan=1>15.38</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td rowspan=1 colspan=3>GDFQ[42]     ECCV’20</td><td rowspan=1 colspan=1>63.58</td><td rowspan=1 colspan=1>43.87</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td rowspan=1 colspan=3>DSG[44]     CVPR&#x27;21</td><td rowspan=1 colspan=1>62.36</td><td rowspan=1 colspan=1>25.48</td></tr><tr><td rowspan=2 colspan=3>CIFAR-100</td><td></td><td rowspan=2 colspan=3>Qimera[5]    NeurIPS’21ARC[6]      IJCAI&#x27;21</td><td rowspan=1 colspan=1>65.10</td><td rowspan=1 colspan=1>46.13*</td></tr><tr><td></td><td rowspan=1 colspan=1>62.76</td><td rowspan=1 colspan=1>40.15*</td></tr><tr><td rowspan=2 colspan=3>ResNet-20</td><td></td><td rowspan=2 colspan=3>ARC+AIT[48]   CVPR&#x27;22IntraQ[47]     CVPR&#x27;22</td><td rowspan=1 colspan=1>61.05</td><td rowspan=1 colspan=1>41.34*</td></tr><tr><td rowspan=1 colspan=3>(FP:70.33</td><td></td><td rowspan=1 colspan=1>64.98</td><td rowspan=2 colspan=1>48.2552.76</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>AdaSG[38]     AAAI&#x27;23</td><td rowspan=1 colspan=1>66.42</td></tr><tr><td rowspan=3 colspan=7>HAST[28]     ICCV’23Ours            一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AdaDFQ[37]CVPR’23</td></tr><tr><td rowspan=1 colspan=1>66.68</td><td rowspan=1 colspan=1>55.67</td></tr><tr><td rowspan=1 colspan=1>66.94</td><td rowspan=1 colspan=1>57.03</td></tr></table>

Table 2. Top-1 accuracy (%) comparison with the state-of-the-art methods on ImageNet for 4/5-bit ResNet-18.
<table><tr><td rowspan=1 colspan=2>Dataset       Method        Venue</td><td rowspan=1 colspan=1>W5A5</td><td rowspan=1 colspan=1>W4A4</td></tr><tr><td rowspan=7 colspan=2>ZeroQ[3]     CVPR&#x27;20GDFQ[42]    ECCV’20DSG[44]     CVPR&#x27;21ZAQ[32]     CVPR&#x27;21ImageNetQimera[5]    NeurIPS’21ForARC[6]      IJCAI&#x27;21ResNet-18(FP:71.47)  ARC+AIT[48]   CVPR&#x27;22</td><td rowspan=1 colspan=1>69.65</td><td rowspan=1 colspan=1>60.68</td></tr><tr><td rowspan=1 colspan=1>66.82</td><td rowspan=1 colspan=1>60.60</td></tr><tr><td rowspan=1 colspan=1>69.53</td><td rowspan=1 colspan=1>60.12</td></tr><tr><td rowspan=1 colspan=1>64.54</td><td rowspan=1 colspan=1>52.64</td></tr><tr><td rowspan=1 colspan=1>69.29</td><td rowspan=1 colspan=1>63.84</td></tr><tr><td rowspan=1 colspan=1>68.88</td><td rowspan=1 colspan=1>61.32</td></tr><tr><td rowspan=1 colspan=1>70.28</td><td rowspan=1 colspan=1>65.73</td></tr><tr><td rowspan=1 colspan=2>IntraQ[47]    CVPR&#x27;22</td><td rowspan=1 colspan=1>69.94</td><td rowspan=1 colspan=1>66.47</td></tr><tr><td rowspan=1 colspan=2>AdaSG[38]    AAAI&#x27;23</td><td rowspan=1 colspan=1>70.29</td><td rowspan=1 colspan=1>66.50</td></tr><tr><td rowspan=2 colspan=2>AdaDFQ[37]   CVPR&#x27;23</td><td rowspan=1 colspan=1>70.29</td><td rowspan=1 colspan=1>66.53</td></tr><tr><td rowspan=1 colspan=1>HAST[28]ICCV'23</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.91</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>70.35</td><td rowspan=1 colspan=1>67.02</td></tr></table>

ImageNet. To further verify the effectiveness of our method on the large-scale dataset, we compare the performance on ImageNet using different networks with stateof-the-art methods in Tables 2∼4. (1) The ResNet-18 is implemented with the proposed method, and the results are listed in Table 2. It can be observed that the proposed method achieves the best performance among these methods, which are 70.35% (5-bit) and 67.02% (4- bit). Especially in the case of 4-bit, compared with GDFQ (60.60%), Qimera (63.84%), AdaDFQ (66.53%), and HAST (66.91%), our method can get much better performance. (2) The MobileNet-V1 is selected for the evaluation of the proposed method on the light-weighted network as shown in Table 3. Generally, light-weighted networks always have a heavy accuracy drop after quantization, but the proposed method can also outperform existing methods, which get 0.92% and 1.81% higher accuracy than the advanced method HAST for 5-bit and 4-bit, respectively. (3) The ResNet-50 is used to evaluate the performance of the proposed method on the network with complex and deeper structure as Table 4. The proposed method achieves superior performances at 76.08% (5-bit) and 68.97% (4-bit), especially compared with the latest methods AdaSG (68.58%) and AdaDFQ (68.38%) at 4-bit setup. Thus, the proposed method can efficiently improve performance on various network structures and datasets, proving the effectiveness of our proposed evaluation for synthetic data.

Table 3. Top-1 accuracy (%) comparison with the state-of-theart methods on ImageNet for 4/5-bit MobileNet-V1. ’IL’ denotes using the inception loss [16].
<table><tr><td>Dataset</td><td>Method</td><td>Venue</td><td>W5A5</td><td>W4A4</td></tr><tr><td rowspan="5">ImageNet For MobileNet-V1 (FP:73.39)</td><td>ZeroQ[3]+IL[16] GDFQ[42]</td><td>CVPR&#x27;20</td><td>67.11 59.76</td><td>25.43</td></tr><tr><td></td><td>ECCV&#x27;20</td><td></td><td>28.64</td></tr><tr><td>DSG[44]+IL[16]</td><td>CVPR&#x27;21</td><td>66.61</td><td>42.19</td></tr><tr><td>SQuant[13]</td><td>ICLR&#x27;22</td><td>64.20</td><td>10.32</td></tr><tr><td>IntraQ[47] HAST[28]</td><td>CVPR&#x27;22 ICCV’23</td><td>68.17 68.52</td><td>51.36 57.70</td></tr></table>

Table 4. Top-1 accuracy (%) comparison with the state-of-the-art methods on ImageNet for 4/5-bit ResNet-50.
<table><tr><td>Dataset</td><td>Method</td><td>Venue</td><td>W5A5</td><td>W4A4</td></tr><tr><td>ImageNet For ResNet-50 (FP:77.73)</td><td>GDFQ[42] ZAQ[32] Qimera[5] ARC[6] ARC+AIT[48] AdaSG[38] AdaDFQ[37] Ours</td><td>ECCV’20 CVPR&#x27;21 NeurIPS&#x27;21 IJCAI&#x27;21 CVPR&#x27;22 AAAI&#x27;23 CVPR&#x27;23</td><td>71.63 73.38 75.32 74.13 76.00 76.03 76.03</td><td>54.16 53.02 66.25 64.37 68.27 68.58 68.38</td></tr></table>

## 4.3. Ablation Study

Effect of Different Components in ${ \mathcal { L } } _ { \mathrm { P } }$ . To evaluate the effect of each component in the proposed method, we test each item in the designed prediction similarity loss ${ \mathcal { L } } _ { \mathrm { P } }$ . Table 5 lists different networks $( \mathcal { N } _ { 1 } \sim \mathcal { N } _ { 7 } )$ trained with different loss combinations (the common ${ \mathcal { L } } _ { \mathrm { K I } }$ is used in all the networks to provide the basic performance of quantized network). Symbol $\checkmark$ means the component is used for quantization training, and symbol × indicates that the component is removed for training. $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ denote the crossentropy loss functions using high-reliable and low-reliable samples, respectively. $\mathcal { L } _ { \mathrm { M S E } }$ represents the usage of the MSE loss function. There are two observations. (1) Multiple Pseudo-labels can provide supervision information for model training, and improve the performance of the quantized network. In the comparison for $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ (i.e. comparing $\mathcal { N } _ { 1 }$ with ${ \mathcal { N } } _ { 3 }$ , and comparing ${ \mathcal { N } } _ { 4 }$ with $\mathcal { N } _ { 7 } )$ , it is obvious that the designed $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ can improve the quantized network performance, which can prove the efficiency of proposed reliability filtering. By comparing the network with different combinations of $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ (i.e. comparing $\mathcal { N } _ { 2 }$ with ${ \mathcal { N } } _ { 3 }$ , comparing ${ \mathcal { N } } _ { 5 }$ with $\mathcal { N } _ { 7 }$ , and comparing ${ \mathcal { N } } _ { 6 }$ with $\mathcal { N } _ { 7 } ) .$ , the network trained using multiple pseudo-labels can have better performance than using only one of them, which shows the designed multiple pseudolabels works and can provide more supervision information for improving the performance of the quantized network. Especially, the performance of ${ \mathcal { N } } _ { 5 }$ is higher than that of ${ \mathcal { N } } _ { 6 }$ , which can show better training efficiency using highreliable data than low-reliable data. (2) Mean Squared $E r \mathrm { - }$ ror (MSE) can make significant performance gain for quantized network. In the comparison with and w/o $\mathcal { L } _ { \mathrm { M S E } }$ (i.e. comparing $\mathcal { N } _ { 1 }$ with ${ \mathcal { N } } _ { 4 } ,$ comparing $\mathcal { N } _ { 2 }$ with ${ \mathcal { N } } _ { 5 } .$ , and comparing ${ \mathcal { N } } _ { 3 }$ with $\mathcal { N } _ { 7 } ) .$ , the performance can have a stable improvement with $\mathcal { L } _ { \mathrm { M S E } }$ , which can prove the efficiency for minimizing the difference between the intermediate feature layers of the two networks.

![](images/374500f13343011013780b5501473b415dbc2a6cf6b4b1363a6f415c690ace9d.jpg)  
(a)

![](images/52534201bf07fd978a300d5a626650f5b4cbad4dcc48f51190e7ec1ba53ebf3a.jpg)

![](images/ea2024d1d1d5691c35a4622670cc862a71dacbf094ba11dea5fde5e58ba335e1.jpg)

![](images/3248163a505d6c1e379f2dc80dc02248768b274afc3d850866352db3b261e938.jpg)  
(b)

![](images/677aafb9e549e1070aea1c2401cd2649b95d2bad41d1bc8fae8e2773d27d0458.jpg)

(e)  
![](images/cc642401a95c1ea4d54672322574473d2f07b84b823ffa975f295c06c856d9ae.jpg)  
(f)

(c)  
![](images/fe12b67fdf8a6e79a08b6678d64bdad82e523f2b36a8f779b45896d525d8aa02.jpg)  
(g)

(d)  
![](images/6ecc701cc864b039d6a51d25a648c45cb6eb147fa6329872e4bb7e444f78225b.jpg)  
(h)

Figure 4. Feature visualization using t-SNE. Figure 4 (a∼d) show the feature distributions of the network ${ \mathcal { N } } _ { 4 }$ (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l } ) .$ , N<sub>5</sub> (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { l } )$ , N<sub>6</sub> (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { h } )$ and $\mathcal { N } _ { 7 }$ (Ours) on the original test dataset. Figure 4 (e ∼ h) show the feature distributions of the pre-trained network on the original dataset, the synthetic dataset, the evaluated high-reliable dataset ${ \hat { X } } ^ { h }$ and low-reliable dataset $\hat { X } ^ { l }$ , respectively.  
![](images/1b749f3f7d9fdd3b12cb7104c77f3ed8bb34ab80a02185c279e6349c09167949.jpg)  
(a) high-reliable data in CIFAR-10

![](images/7d3d4fb7f6997df51e2721167be18c64cdbad8e19f824310900c2abdb252a29e.jpg)  
(b) low-reliable data in CIFAR-10

![](images/b770772c5af3476c09a576a15f09dbad5110f7359765ecee6ae05ba15af57198.jpg)  
(c) high-reliable data in CIFAR-100

![](images/6502e75e7bd880da9e717d3ae10d6d546069b05ec59cb4d3f27e573c3c935755.jpg)  
(d) low-reliable data in CIFAR-100  
Figure 5. The highest and secondary highest predicted probabilities of pre-trained network on the high- and low-reliable datasets. The highest predicted probabilities on the high-reliable dataset are far higher than the secondary highest predicted probabilities, since the pretrained network has high performance on these data. While on the low-reliable dataset, the pre-trained network cannot perform well, which can be reflected by the lower highest predicted probabilities and higher secondary highest predicted probabilities.

Table 5. Ablation studies of losses with 3/4-bit ResNet-20 on the CIFAR-100.
<table><tr><td></td><td> $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ </td><td> $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ </td><td> $\mathcal { L } _ { \mathrm { M S E } }$ </td><td>W4/A4</td><td>W3/A3</td></tr><tr><td> ${ \mathcal { N } } _ { 1 }$ </td><td>×</td><td>X</td><td>X</td><td>65.81</td><td>50.87</td></tr><tr><td>N2</td><td>√</td><td>X</td><td>X</td><td>66.30</td><td>53.22</td></tr><tr><td> ${ \mathcal { N } } _ { 3 }$ </td><td>√</td><td>√</td><td>X</td><td>66.52</td><td>54.14</td></tr><tr><td> ${ \mathcal { N } } _ { 4 }$ </td><td>X</td><td>×</td><td>√</td><td>66.32</td><td>54.59</td></tr><tr><td> ${ \mathcal { N } } _ { 5 }$ </td><td>√</td><td>×</td><td>√</td><td>66.68</td><td>56.43</td></tr><tr><td> ${ \mathcal { N } } _ { 6 }$ </td><td>×</td><td>√</td><td>√</td><td>66.57</td><td>55.86</td></tr><tr><td> $\mathcal { N } _ { 7 }$ </td><td>√</td><td>√</td><td>√</td><td>66.94</td><td>57.03</td></tr></table>

Table 6. Top-1 accuracy (%) of the combination of GDFQ [42], IntraQ [47] and our training method on CIFAR-100 for 4-bit ResNet-20 and ImageNet for 4-bit ResNet-18. The combination with our multiple pseudo-labels is denoted as $+ \mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$
<table><tr><td>Dataset</td><td>Method</td><td>Acc</td><td>Acc Up</td></tr><tr><td rowspan="4">Cifar100 (FP:70.33)</td><td>GDFQ[42]</td><td>63.58</td><td></td></tr><tr><td> $\mathrm { G D F Q } [ 4 2 ] + \mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$  IntraQ[47]</td><td>64.01 64.98</td><td>0.43 ↑</td></tr><tr><td> $\mathrm { I n t r a Q [ 4 7 ] } { + } \mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ </td><td>65.49</td><td>0.51 ↑</td></tr><tr><td>Ours</td><td>66.94</td><td>-</td></tr><tr><td rowspan="4">ImageNet (FP:73.09)</td><td>GDFQ[42]</td><td>60.60</td><td></td></tr><tr><td> $\mathrm { G D F Q } [ 4 2 ] + \mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ </td><td>61.36</td><td>0.76↑</td></tr><tr><td>IntraQ[47]  $\mathrm { I n t r a Q [ 4 7 ] } { + } \mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ </td><td>66.47</td><td></td></tr><tr><td>Ours</td><td>66.79 67.02</td><td>0.32 ↑ 一</td></tr></table>

To give visualizations for the improvement of network performance with the designed loss function, the feature distributions of the network ${ \mathcal { N } } _ { 4 }$ (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l } ) , \mathcal { N } _ { 5 }$ (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { l } ) , \mathcal { N } _ { 6 }$ (w/o $\mathcal { L } _ { \mathrm { C E } } ^ { h } )$ and $\mathcal { N } _ { 7 }$ (Ours) on the original test dataset are shown in Figure 4 (a∼d) with t-SNE [40], which can display the distribution for classification by transferring data from high dimension into the two-dimensional space. It is evident that the designed $\mathcal { L } _ { \mathrm { C E } } ^ { h }$ and $\mathcal { L } _ { \mathrm { C E } } ^ { l }$ can improve the classification performance of the trained quantized network, which can be observed from the aggregation effect of different classes (comparing ${ \mathcal { N } } _ { 4 }$ with ${ \mathcal { N } } _ { 5 }$ in Figure 4 (a) and (b), and comparing ${ \mathcal { N } } _ { 4 }$ with ${ \mathcal { N } } _ { 6 }$ in Figure 4 (a) and (c)), so the designed reliability filtering can improve the training of quantized network. In addition, $\mathcal { N } _ { 7 }$ in Figure 4 (d) has the best aggregation effect among these four networks, which can prove the efficiency of the designed multiple pseudolabels in the proposed method.

Effect of Reliability Filtering. To show the efficiency of proposed self-entropy metric for evaluating the reliability of samples in the synthetic data, we present the feature distributions of the pre-trained network on the original training dataset, the synthetic dataset, the evaluated high-reliable dataset ${ \hat { X } } ^ { h }$ and low-reliable dataset $\hat { X } ^ { l }$ as shown in Figure 4 $( \mathrm { e } \sim \mathrm { h } )$ . The feature distributions of the pre-trained network on the original training dataset are shown in Figure 4 (e), which means the pre-trained network can have a good classification performance on the original training dataset. However, the feature distributions in Figure 4 (f) show the pre-trained network can hardly efficiently classify the synthetic data. From the results in Figure 4 (g) and (h), it is evident that the pre-trained network can have an excellent classification performance on ${ \hat { X } } ^ { h }$ . This means the application of designed reliability filtering can efficiently filter the high-reliable samples, which is suitable to provide supervision information and train quantized network.

Effect of Multiple Pseudo-Labels. The highest and secondary highest predicted probabilities of pre-trained network on the high- and low-reliable datasets are plotted in Figure 5. It can be observed that the highest predicted probabilities on the high-reliable dataset are far higher than the secondary highest predicted probabilities, since the pretrained network has high reliability on these data. While on the low-reliable dataset, the pre-trained network cannot perform well, which can be reflected by the lower highest predicted probabilities and higher secondary highest predicted probabilities. Therefore, the designed secondary label $\hat { y } _ { i } ^ { s }$ can also provide information for fitting the performance of the pre-trained network, which can enhance the training and mitigate the risk of misleading for the high-reliable samples.

Effect of Pseudo-Label Training. Our designed multiple pseudo-labels $\mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ can also combined with other training framework. To evaluate the proposed training framework, two classic DFQ methods (GDFQ and IntraQ) are selected as baselines, and we combine the designed $\mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ with these two baselines. We examine the performance on CIFAR-100 for 4-bit ResNet-20 and ImageNet for 4- bit ResNet-18 as listed in Table 6. From the experimental results, the combination with our designed $\mathcal { L } _ { \mathrm { C E } } ^ { t o t a l }$ can efficiently improve the performance of the quantized network for the existing methods, which can prove the effectiveness of our designed multiple pseudo-labels. In addition, it is also observed that the proposed training framework can still have the best accuracy among these methods, which can prove that the proposed pseudo-label training can guide the quantized network to have a similar prediction ability with the pre-trained network and learn supervision information from the synthetic data with different labels.

## 5. Conclusion

This work proposes an efficient data-free quantization method via pseudo-label filtering, which is the first to evaluate the synthetic data before training the quantized network. In the proposed method, self-entropy is selected as an evaluation metric to divide the synthetic data into highreliable and low-reliable data. The multiple pseudo-labels are designed to label the evaluated samples, which can further improve the reliability and avoid misleading caused by low-reliable data. The pseudo-label training is designed to integrate the supervision information provided by multiple pseudo-labels. Extensive experiments are implemented and evaluated on CIFAR-10/100 and ImageNet datasets, demonstrating that the proposed framework performs better than existing methods.

Acknowledgement. This work is supported in part by the National Key R&D Program of China (No. 2022ZD0118201), the National Natural Science Foundation of China (No.61802105, 62272144, 72188101, 62020106007, and U20A20183), the University Synergy Innovation Program of Anhui Province (No. GXXT-2021- 005 and GXXT-2022–033), the Fundamental Research Funds for the Central Universities (No. JZ2022HGTB0250 and PA2023IISL0096), and the Major Project of Anhui Province (202203a05020011).

## References

[1] Ron Banner, Yury Nahshan, Elad Hoffer, and Daniel Soudry. Aciq: Analytical clipping for integer quantization of neural networks. International Conference on Learning Representations, 2019. 1, 2

[2] Yash Bhalgat, Jinwon Lee, Markus Nagel, Tijmen Blankevoort, and Nojun Kwak. Lsq+: Improving low-bit quantization through learnable offsets and better initialization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 696– 697, 2020. 1, 2

[3] Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. Zeroq: A novel zero shot quantization framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13169–13178, 2020. 1, 3, 5, 6

[4] Ting-An Chen, De-Nian Yang, and Ming-Syan Chen. Alignq: Alignment quantization with admm-based correlation preservation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12538–12547, 2022. 1, 2

[5] Kanghyun Choi, Deokki Hong, Noseong Park, Youngsok Kim, and Jinho Lee. Qimera: Data-free quantization with synthetic boundary supporting samples. Advances in Neural Information Processing Systems, 34:14835–14847, 2021. 1, 3, 5, 6

[6] Kanghyun Choi, Hye Yoon Lee, Deokki Hong, Joonsang Yu, Noseong Park, Youngsok Kim, and Jinho Lee. It’s all in the teacher: Zero-shot quantization brought closer to the teacher. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8311–8321, 2022. 5, 6

[7] Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Sarangapani. A comprehensive survey on model compression and acceleration. Artificial Intelligence Review, 53:5113–5155, 2020. 1

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255. Ieee, 2009. 5

[9] Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compression and hardware acceleration for neural networks: A comprehensive survey. Proceedings of the IEEE, 108(4):485–532, 2020. 1

[10] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. In International Conference on Learning Representations, 2020. 1, 2

[11] Ruihao Gong, Xianglong Liu, Shenghu Jiang, Tianxiang Li, Peng Hu, Jiazhen Lin, Fengwei Yu, and Junjie Yan. Differentiable soft quantization: Bridging full-precision and lowbit neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4852–4861, 2019. 1, 2

[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and

Yoshua Bengio. Generative adversarial networks. Communications ofthe ACM, 63(11):139–144, 2020. 1

[13] Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, and Minyi Guo. Squant: On-the-fly data-free quantization via diagonal hessian approximation. In International Conference on Learning Representations, 2021. 2, 6

[14] Cong Guo, Yuxian Qiu, Jingwen Leng, Xiaotian Gao, Chen Zhang, Yunxin Liu, Fan Yang, Yuhao Zhu, and Minyi Guo. Squant: On-the-fly data-free quantization via diagonal hessian approximation. arXiv preprint arXiv:2202.07471, 2022. 3

[15] Tiantian Han, Dong Li, Ji Liu, Lu Tian, and Yi Shan. Improving low-precision network quantization via bin regularization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5261–5270, 2021. 1

[16] Matan Haroush, Itay Hubara, Elad Hoffer, and Daniel Soudry. The knowledge within: Methods for data-free model compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8494– 8502, 2020. 3, 6

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. 5

[18] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 1, 5

[19] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 5

[20] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448–456. pmlr, 2015. 1

[21] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 2704–2713, 2018. 1

[22] Yongkweon Jeon, Chungman Lee, and Ho-young Kim. Genie: Show me the data for quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12064–12073, 2023. 1

[23] Youngeun Kim, Donghyeon Cho, Kyeongtak Han, Priyadarshini Panda, and Sungeun Hong. Domain adaptation without source data. IEEE Transactions on Artificial Intelligence, 2(6):508–518, 2021. 2, 4

[24] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1

[25] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 5

[26] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015. 1

[27] Jung Hyun Lee, Jihun Yun, Sung Ju Hwang, and Eunho Yang. Cluster-promoting quantization with bit-drop for minimizing network quantization loss. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5370–5379, 2021. 1

[28] Huantong Li, Xiangmiao Wu, Fanbing Lv, Daihai Liao, Thomas H Li, Yonggang Zhang, Bo Han, and Mingkui Tan. Hard sample matters a lot in zero-shot quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24417–24426, 2023. 1, 3, 5, 6

[29] Xinhao Li, Jingjing Li, Lei Zhu, Guoqing Wang, and Zi Huang. Imbalanced source-free domain adaptation. In Proceedings ofthe 29th ACM International Conference on Multimedia, pages 3330–3339, 2021. 4

[30] Yuhang Li, Feng Zhu, Ruihao Gong, Mingzhu Shen, Xin Dong, Fengwei Yu, Shaoqing Lu, and Shi Gu. Mixmix: All you need for data-free compression are feature and data mixing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4410–4419, 2021. 1, 3

[31] Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, and Jian Yang. Curriculum temperature for knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1504– 1512, 2023. 1

[32] Yuang Liu, Wei Zhang, and Jun Wang. Zero-shot adversarial quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1512– 1521, 2021. 3, 5, 6

[33] Vasco Lopes, Fabio Maria Carlucci, Pedro M Esperanc¸a, Marco Singh, Antoine Yang, Victor Gabillon, Hang Xu, Zewei Chen, and Jun Wang. Manas: Multi-agent neural architecture search. Machine Learning, pages 1–24, 2023. 1

[34] Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. Data-free quantization through weight equalization and bias correction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1325–1334, 2019. 2, 3

[35] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In International Conference on Machine Learning, pages 7197–7206, 2020. 1, 2

[36] Kalin Ovtcharov, Olatunji Ruwase, Joo-Young Kim, Jeremy Fowers, Karin Strauss, and Eric S Chung. Accelerating deep convolutional neural networks using specialized hardware. Microsoft Research Whitepaper, 2(11):1–4, 2015. 1

[37] Biao Qian, Yang Wang, Richang Hong, and Meng Wang. Adaptive data-free quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7960–7968, 2023. 1, 3, 5, 6

[38] Biao Qian, Yang Wang, Richang Hong, and Meng Wang. Rethinking data-free quantization as a zero-sum game. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023. 3, 5, 6

[39] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Xiaojiang Chen, and Xin Wang. A comprehensive

survey of neural architecture search: Challenges and solutions. ACM Computing Surveys (CSUR), 54(4):1–34, 2021. 1

[40] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (11), 2008. 8

[41] Wenxiao Wang, Minghao Chen, Shuai Zhao, Long Chen, Jinming Hu, Haifeng Liu, Deng Cai, Xiaofei He, and Wei Liu. Accelerate cnns from three dimensions: A comprehensive pruning framework. In International Conference on Machine Learning, pages 10717–10726. PMLR, 2021. 1

[42] Shoukai Xu, Haokun Li, Bohan Zhuang, Jing Liu, Jiezhang Cao, Chuangrun Liang, and Mingkui Tan. Generative lowbitwidth data free quantization. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part XII 16, pages 1–17. Springer, 2020. 1, 3, 5, 6, 7

[43] Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, and Kevin Bailly. Spiq: Data-free per-channel static input quantization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3869–3878, 2023. 3

[44] Xiangguo Zhang, Haotong Qin, Yifu Ding, Ruihao Gong, Qinghua Yan, Renshuai Tao, Yuhang Li, Fengwei Yu, and Xianglong Liu. Diversifying sample generation for accurate data-free quantization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15658–15667, 2021. 3, 5, 6

[45] Yuyao Zhang and Nikolaos M Freris. Adaptive filter pruning via sensitivity feedback. IEEE Transactions on Neural Networks and Learning Systems, 2023. 1

[46] Yunshan Zhong, Mingbao Lin, Mengzhao Chen, Ke Li, Yunhang Shen, Fei Chao, Yongjian Wu, and Rongrong Ji. Finegrained data distribution alignment for post-training quantization. In European Conference on Computer Vision, pages 70–86. Springer, 2022. 1

[47] Yunshan Zhong, Mingbao Lin, Gongrui Nan, Jianzhuang Liu, Baochang Zhang, Yonghong Tian, and Rongrong Ji. Intraq: Learning synthetic images with intra-class heterogeneity for zero-shot network quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12339–12348, 2022. 1, 3, 5, 6, 7

[48] Baozhou Zhu, Peter Hofstee, Johan Peltenburg, Jinho Lee, and Zaid Alars. Autorecon: Neural architecture searchbased reconstruction for data-free compression. In 30th International Joint Conference on Artificial Intelligence, IJCAI 2021, pages 3470–3476. International Joint Conferences on Artificial Intelligence, 2021. 1, 3, 5, 6