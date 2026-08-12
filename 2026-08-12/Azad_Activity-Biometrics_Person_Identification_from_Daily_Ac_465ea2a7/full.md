# Activity-Biometrics: Person Identification from Daily Activities

Shehreen Azad Shehreen.Azad@ucf.edu

Yogesh Singh Rawat yogesh@ucf.edu

Center for Research in Computer Vision, University of Central Florida

## Abstract

In this work, we study a novel problem which focuses on person identification while performing daily activities. Learning biometric features from RGB videos is challenging due to spatio-temporal complexity and presence of appearance biases such as clothing color and background. We propose ABNet, a novel framework which leverages disentanglement of biometric and non-biometric features to perform effective person identification from daily activities. ABNet relies on a bias-less teacher to learn biometric features from RGB videos and explicitly disentangle nonbiometric features with the help of biometric distortion. In addition, ABNet also exploits activity prior for biometrics which is enabled by joint biometric and activity learning. We perform comprehensive evaluation of the proposed approach acrossfive different datasets which are derivedfrom existing activity recognition benchmarks. Furthermore, we extensively compare ABNet with existing works in person identification and demonstrate its effectiveness for activitybased biometrics across all five datasets. The code and dataset can be accessed at: https://github.com/ sacrcv/Activity-Biometrics/

## 1. Introduction

Person identification is an important task with a wide range of applications in security, surveillance, and various domains where recognizing individuals across different locations or time frames is essential [41]. We have seen a great progress in face recognition [2, 31], however scenarios exist where faces may not be visible, such as at long distances, with uncooperative subjects, under occlusion, or due to mask-wearing. This limitation prompts the exploration of whole-body-based person identification methods where most of the existing works are often restricted to image-based approaches [4, 14, 43], overlooking crucial motion patterns. Video-based methods for person identification is comparatively recent area where most of the work is focused on gait recognition; mostly silhouettebased [12, 13, 27] with some recent works on RGB frames [26, 44]. However these works are mainly focused on walking style of individuals (see Figure 1).

![](images/e5830d2309815442cf8b2e6efedd71dcc557b1a4768f4df96510913d6d9296b1.jpg)  
Figure 1. Different approaches for person identification: (left) samples for existing person identification problems such as face recognition (top: Celeb-A[30]), whole body recognition (middle: Market-1501[45]), and gait recognition (bottom: CASIA-B[42]). (right) we focus on person identification from daily activities which presents more challenges beyond learning walking or facial patterns. We show some samples from datasets we used to study this problem; (top: NTU RGB-AB, middle: Charades-AB, bottom: ACC-MM1-Activities<sup>1</sup>).

In this work, we study a novel problem which focuses on face-restricted person identification during routine activities. The current landscape of image-based and video-based whole-body person identification methods predominantly centers around analyzing human walking patterns from images or videos [15, 20, 33, 40]. However, in real-world scenarios, the individual requiring identification might not always be engaged in walking; instead, they could be involved in various daily activities. It is crucial to acknowledge the significance of capturing and understanding motion cues that extend beyond simple walking patterns to ensure accurate and reliable identification in diverse and complex situations. These activities may offer unique cues that can prove instrumental in identifying individuals even without explicit facial information, paving the way for diverse applications in real-world scenarios, like increased surveillance in public spaces, workplace security and productivity, assistance for people requiring special needs, and smart home automation.

Learning biometrics from videos of daily activities presents several inherent challenges. Learning from such diverse activities amplifies the difficulty in capturing essential biometrics features. Among the crucial challenges lies the necessity to prioritize biometrics features while mitigating appearance biases present in RGB video frames, including background variations, clothing color, and other external factors. Striking a balance between extracting pertinent biometrics cues and disregarding irrelevant appearancerelated biases is essential in developing robust and accurate video-based biometrics identification methods.

We propose a novel framework ABNet, which addresses some of these challenges and provides effective biometrics representation for person identification from videos of daily activities. It relies on two main components; 1) feature disentanglement, and 2)joint activity-biometrics learning. Feature disentanglement aims at avoiding appearance biases while learning the biometrics features. It explicitly learns biometrics and non-biometrics features with the help of, a) distillation from bias-less teacher, and b) bias learning using biometrics distortion. Joint activity-biometrics learning provides activity prior for biometrics where the knowledge of performed activity helps in person identification.

We present extensive evaluations on five different benchmarks using several metrics comparing the proposed approach with several state-of-the-art person identification methods including both image-based and video-based approaches. This comprehensive evaluation demonstrates the effectiveness and superiority of our proposed method in handling diverse datasets and scenarios for activity-based biometrics. Our main contributions can be summarized as,

• We study a novel problem of person identification from daily activities using RGB videos.

• We propose a simple and novel strategy to disentangle biometrics and non-biometrics features from videos for person identification.

• We show the benefits of activity-prior for biometrics.

• We present several benchmarks to study this problem; these datasets are dervied from existing activity recognition datasets specifically curated for person identification.

## 2. Related work

Image-based identification: Most of the existing person identification methods use image-based approach [4, 6, 19, 21, 39, 40]. Moreover, most of these methods are designed towards learning better features in-terms of body shape, clothes, appearance etc. In recent years, learning cloth invariant features is found to be a promising direction in person identification with several works trying to address this issue. For example, one of the most popular person identification approach, CAL [15] uses advarsarial loss to learn cloth invariant features. On the other hand, SCNet[16] uses a tri-stream network to learn semantically invariant features. Some works also attempt to use multiple modalities (e.g., silhouettes [23], skeletons [34], 3D shape [6]) etc. for better feature representation. Even though the image based methods can have better performance than some video-based methods, this performance is measured on very specific datasets, which might or might not generalize to more complex datasets where the person in consideration is performing some other activities rather than walking.

Video-based identification: The key for video-based person identification is to extract representations robust to spatial and temporal distractors. These methods incorporate temporal information in their learned features and generally have better performance than image based methods. Several previous works [10, 43] have exploited temporal cues by aggregating frames features via LSTM network. However, instead of using aggregated features extracted by RNNs, 3D CNNs perform better in terms of directly extracting spatio-temporal features that are more robust for person identification [5, 28]. Following current research direction [3, 20, 22, 33, 37], our work is also based on 3D CNN.

Gait recognition: Gait recognition is a very active area of research where the goal is to identify individuals using their walking style. Existing methods mostly utilize silhouettes to avoid interference of appearance [12, 13, 27] which limits their applicability on real-world RGB videos. There are some approaches making use of RGB for gait recognition [26, 44], but they do require silhouette in addition to RGB data. In our proposed method we only use silhouette during training and it is not required for inference.

Knowledge distillation: It is one of the most common techniques to transfer knowledge from a large model (teacher) to a smaller model (student) for compression and efficient learning [18]. It has also been found very effective for semisupervised learning where the models can learn from unlabelled samples under student teacher setup [36]. In some recent efforts it was also explored for person identification too for effective cross-view [33] and cross-scene [38] representation learning. It has been mostly explored within same modality, whereas we perform a cross-modal distillation to leverage the teacher’s knowledge of a different data modality to improve the performance of the student.

## 3. Method

Our goal is to identify an individual given an RGB video of that individual performing some activity. We are using a face restricted setting to perform this task, where the face of the individual is blurred so as to avoid learning any of the facial features. Avoiding the explicit learning of facial features is motivated by acknowledging potential issues like wearing accessory (masks, sunglasses), privacy concerns, and individuals’ unwillingness to reveal their faces.

Problem formulation: Given a dataset D containing elements of $v , y ^ { A } , y ^ { B }$ with N samples, we want to train a person identification model M which can provide a latent feature $F _ { A B }$ for each video v which can be used for matching it with the person id $y ^ { B }$ . Here $v \in \mathbb { R } ^ { n X C X H X W }$ represents an RGB video, where n is the number of frames, $C , H , W$ are the number of channels, height and width of the video, and $y ^ { B }$ is its ground truth actor label that is performing some activity $y ^ { \bar { A } }$ . Once trained, the model M will be evaluated on a gallery $G \in v , y ^ { b }$ and probe $P \in v , y ^ { b }$ The goal is to match the id of the person $y ^ { b }$ in probe video v with the correct id in videos from gallery.

![](images/5477b59088a10baecabc5b0a377e12e9d58618fa9c8c44d674d8a353ecad4219.jpg)  
Figure 2. Overview of our proposed method ABNet. RGB video is passed to a video encoder $S _ { \varphi } ( \cdot )$ for spatio-temporal feature $F _ { A B }$ extraction which is passed to the activity head $C ^ { A }$ and the actor head $\hat { C } ^ { B } . C ^ { B }$ captures both biometrics (in red) and appearance (in green) features in $F _ { B T }$ . To disentangle features, bias-less teacher encoder $T _ { \theta } ( \cdot )$ distills biometrics knowledge from corresponding silhouettes. The appearance feature bias is learned via a distortion network using encoder $A _ { \varphi } ( \cdot )$ on the distorted video input. Similar to $\check { C ^ { B } } , C ^ { D B }$ also captures both distorted biometrics (in red) and distorted appearance (in green) features in $F _ { B T } ^ { D }$ . Here, green and red denote positive and negative feature. Joint training is performed using both $C ^ { \hat { A } }$ and $C ^ { B }$ . During inference, only the dashed box highlighted branch is utilized.

Overview: We propose ABNet, Activity Biometrics Network, denoted as M to solve this problem. ABNet performs biometrics-bias disentanglement and make use of activity prior to learn a discriminative identity feature for person identification. Given a video $v ,$ the model M first extracts spatio-temporal features $F _ { A B }$ with the help of a video encoder $S _ { \varphi } ( \cdot )$ . The spatio-temporal feature $F _ { A B }$ is split into two segments and are passed to the actor head $C ^ { B }$ for person identification as well as the activity head $C ^ { A }$ for activity recognition. Joint biometrics and activity learning enables the use of activity-prior for biometrics. We get actor features $F _ { B T }$ from $\dot { C } ^ { \dot { B } }$ that contains both biometrics and appearance feature entangled with each other. Now to make the model robust to appearance bias while learning accurate biometrics features, we introduce two different components $\textsuperscript { - 1 ) }$ distillation from a bias-less teacher and learning the bias using biometrics distortion. The actor feature $F _ { B T }$ are disentangled into biometrics feature $f _ { b b }$ and appearance feature $f _ { b a }$ . This disentanglement for biometrics feature $f _ { b b }$ is performed using distillation from a bias-less teacher $T .$ . On the contrary, the disentanglement for appearance feature $f _ { b a }$ is done by constraining it using a distortion network A. An overview of the proposed method is shown in Figure 2.

## 3.1. Biometrics bias disentanglement

Appearance bias in biometrics arises when the models overly rely on superficial visual cues, such as clothing or specific accessories for identification. This leads to challenges such as limited generalization across appearances, vulnerability to adversarial attacks, and reduced robustness to environmental variations. This bias can result in biased matching decisions, and inconsistent performance across cameras. There has been extensive research done to avoid clothing features for person reidentification [15, 16, 40], however, appearance bias can come from features other than clothes as well. To deal with this issue of appearance bias, we introduce two different aspects; 1) bias-less distillation from a teacher network, and 2) learning the bias using negative mining through biometrics distortion.

Bias-less distillation: One split segment of the extracted feature $F _ { A B }$ is fed to the actor head $C ^ { B }$ , which contains $D _ { \omega } ^ { B }$ that is a standard transformer decoder. We get actor feature $F _ { B T }$ from $D _ { \omega } ^ { B }$ , which contains biometrics feature $f _ { b b }$ and appearance feature $f _ { b a } . \ D _ { \omega } ^ { B }$ uses self-attention to process the input sequence and then projects the attention output into $f _ { b b }$ and $f _ { b a }$ using separate linear layers. Now to disentangle the biometrics features from the appearance features, we propose the use of silhouette features to perform bias-less distillation using teacher network T. T is termed as bias-less because it is trained on binary silhouette video $b _ { s } \in \mathbb { R } ^ { n X C X H X W }$ that corresponds to RGB video $v ,$ and thus have no knowledge of appearance based features. T contains a silhouette encoder $T _ { \theta } ( \cdot )$ that takes $b _ { s }$ as input and extracts $F _ { S }$ features. Following [18] we use the standard Kullback-Leibler (KL) divergence loss to minimize the discrepancy between the probability distributions of the teacher $T$ and our model M. The distillation loss $\mathcal { L } _ { K D }$ is formulated as below:

$$
\mathcal { L } _ { K D } = \tau ^ { 2 } K L ( y _ { T } | | y _ { S } ) ,\tag{1}
$$

where, $y _ { T }$ and $y _ { S }$ are the probability distribution of the teacher $T$ and our model M. τ is the temperature parameter that controls the softness of the teacher’s output. Along with this distillation loss $\mathcal { L } _ { K D } , C ^ { B }$ has its own biometrics loss $\mathcal { L } _ { B i o }$ formulated as below:

$$
\begin{array} { r } { \mathcal { L } _ { B i o } = \mathcal { L } _ { c e } + \mathcal { L } _ { t r i } , } \end{array}\tag{2}
$$

where, $\mathcal { L } _ { c e }$ and $\mathcal { L } _ { t r i }$ are standard triplet and cross-entropy losses for person identification formulated as below:

$$
\mathcal { L } _ { c e } = - y \log \hat { y } ,\tag{3}
$$

$$
\mathcal { L } _ { t r i } = \operatorname* { m a x } ( ( D ( f _ { a } , f _ { p } ) - D ( f _ { a } , f _ { n } ) + m ) , 0 ) ,\tag{4}
$$

where, y and $\hat { y }$ are the ground truth and predicted label, $f _ { p }$ and $f _ { n }$ are the positive and negative features for an anchor feature $f _ { a }$ within the same batch, $D ( \cdot )$ is the Euclidean distance function, and m is the margin of triplet loss.

Bias learning: To make the model robust to appearance bias, we introduce the distortion network A, which is identical to M and shares weights. It contains video encoder $A _ { \varphi } ( \cdot )$ that takes distorted video $\hat { v } \in \mathbb { R } ^ { n X C X H X W }$ that corresponds to the original video v. The key idea is to distort the identity of the person while preserving the appearance. We rely on elastic transform [1] which randomly transforms the morphology of objects in images and produces a seethrough-water-like effect in the image still preserving the appearance. It is used to generate “negative” or “distractor” samples in the training dataset where the distorted samples will have the same appearance while changing the identity. Some sample distorted images are shown in Fig. 3.

Similar to M, this distortion network A also extracts spatio-temporal feature $F _ { A B } ^ { D }$ using encoder $A _ { \varphi } ( \cdot )$ . Since this branch is designed for bias-learning, thus the activity head $C ^ { D A }$ of A is not utilized. On the contrary, A’s actor head $C ^ { D B }$ extracts distorted biometrics feature $f _ { b b } ^ { D }$ and distorted appearance feature $f _ { b a } ^ { D }$ . Due to the distortion, $f _ { b a }$ and $f _ { b a } ^ { D }$ are treated as positive samples, whereas, $f _ { b b }$ and $f _ { b b } ^ { D }$ as hard negative samples. The goal is to pull together positive pairs (i.e. similar features) and push apart negative pairs (i.e. dissimilar features). We use this distorted augmentation loss $\mathcal { L } _ { D i s }$ for bias learning and it is described as,

$$
\mathcal { L } _ { D i s } = m a x ( ( D ( f _ { b a } , f _ { b a } ^ { D } ) - D ( f _ { b b } , f _ { b b } ^ { D } ) + m ) , 0 ) ,\tag{5}
$$

where $D ( \cdot )$ is the Euclidean distance function and m is the margin for the contrastive loss.

## 3.2. Joint biometrics and activity learning

Jointly training a network for both activity recognition and person identification can benefit person identification when the training data includes activities by enabling the model to learn shared representations. By learning to understand contextual cues from activities alongside actor features, the network can develop richer embeddings, thereby enhancing the model’s ability to accurately identify individuals across varying activity contexts. Thus we perform joint learning of the activity and actor branch of ABNet. One segment of feature $F _ { A B }$ is fed to activity head $C ^ { A }$ that contains decoder $D _ { \Omega } ^ { A }$ that learns features $F _ { A c } . { \cal C } ^ { A }$ is trained using $\mathcal { L } _ { A c }$ which is a standard cross-entropy loss for the activity labels regardless of the actor labels. This joint training also enables ABNet to utilize activity priors for biometrics, where we use knowledge of activity for person identification. This is accomplished by concatenating the activity features $F _ { A C }$ with biometrics features $f _ { b b }$ during testing.

## 3.3. Overall learning objective

Finally the model M is optimized by combining all the losses which include, biometrics loss ${ \mathcal { L } } _ { B i o } ,$ distillation loss $\mathcal { L } _ { K D }$ , distortion loss $\mathcal { L } _ { D i s }$ and activity loss $\mathcal { L } _ { A c }$ and we get the total loss L formulated as,

$$
\mathcal { L } = \mathcal { L } _ { B i o } + \lambda _ { 1 } \mathcal { L } _ { A c } + \lambda _ { 2 } \mathcal { L } _ { K D } + \lambda _ { 3 } \mathcal { L } _ { D i s }\tag{6}
$$

where $\lambda _ { i } , i \in [ 1 , 2 , 3 ]$ are the weights for each of the losses.

## 4. Experiments and results

Datasets: We perform our experiments on five different datasets which are derived from existing activity recognition benchmarks. 1) NTU RGB-AB is derived from NTU RGB+D [29] which is a large-scale benchmark for activity recognition. We ignore mutual activities and consider 94 ac tivity classes with 88692 samples fro NTU RGB-AB. The activity classes are divided into daily activities and medical conditions performed by a total of 106 subjects across 32 different setups, 155 different views which are shown with 3 cameras. We use the official cross-subject split for the train test separation. 2) PKU MMD-AB is derived from PKU-MMD [8] which is another large scale benchmark for activity recognition. Similar to NTU RGB-AB, we ignore mutual activities from PKU-MMD and PKU MMD-AB has

![](images/4a9243c7ff8b87d4286caff25982a300e37c2bd1d9482863833fdda4c66dc81f.jpg)  
Figure 3. Biometrics distortion: here original samples are shown in the top row and their corresponding distorted samples in the bottom row. From left to right, every two columns contain samples from NTU RGB-AB, PKU MMD-AB, Charades-AB, ACC-MM1-Activities and BRIAR-BGC3 dataset respectively. The subjects from BRIAR-BGC3 and ACC-MM1-Activities consented to publication.

41 activity categories with almost 17,000 labeled activity instances. These activities are performed by 66 actors in 3 different camera views and we use the official cross-subject split for our experiments. 3) Charades-AB contains all the 9,848 annotated videos from Charades [35] with approximately 6.8 activities per video performed by 267 actors across 157 activity classes from a single viewpoint. We use the official train-test split for our experiments. 4) ACC-MM1-Activities [32] is a recently curated daily activities dataset which contains 1378 annotated videos where 7 daily activities are being performed by 200 subjects from a single view-point. These activities are - enter/exit car, pull/push door, walk upstairs/downstairs, and texting. We use the official train-test split for our experiments. 5) BRIAR-BGC3 [9] is a large-scale, in-the-wild person identification dataset containing samples across varying distances, environment conditions. It is mainly focused on walking/standing scenario and consists of 3 different walking conditions (structured walk, random walk and standing) performed by 1055 subjects in outdoor settings from different ranges and angle of elevation. BRIAR-BGC3 contains over 1300 hours of labeled training videos from 1055 subjects in indoor/outdoor settings. We use a 20K subset of this dataset for training with official face-restricted testing set for evaluation.

The videos from all five datasets undergo an arbitrarily chosen value of hue shifting. Training a model on hue-shifted data, even when appearance features are not explicitly utilized, serves to enhance the model’s robustness and generalization capabilities. To facilitate face restricted person identification the faces are blurred using Gaussian blur for both the test and train split of all datasets.

Implementation and training details: The proposed method is implemented using Pytorch. We use ResNet3D-50 [17] as the backbone of the video encoder $S _ { \varphi } ( \cdot )$ and GaitGL [27] for the teacher’s silhouette encoder $T _ { \theta } ( \cdot )$ The silhouettes of the RGB videos are extracted using Mask2Former [7] to use as input to $T _ { \theta } ( \cdot )$ . We create RGB video clips from each original video by randomly selecting

8 frames with a stride of 4. Every input frame undergoes resizing to dimensions of 256X128. We train the model with a batch size of 32 with each batch containing 8 person and 4 clips for each person. Adam [24] is used as the optimizer with weight decay of $5 x 1 0 ^ { - 4 }$ and learning rate of $3 . 5 X 1 0 ^ { - 4 }$ . The model is trained for 150 epochs with a decay factor 0.1 after every 40 epochs. The triplet loss margin m is set to 0.3 and $\lambda _ { i } , i \in [ 1 , 2 , 3 ]$ in Eq. (6) is set to 0.01. During inference the activity feature $F _ { A c }$ is concatenated with the biometrics feature $f _ { b b }$ that acts as the activity prior.

Evaluation protocol: For all datasets except BRIAR-BGC3, we randomly split the test set into gallery and probe (more details in supplementary). We use two different evaluation protocols; 1) same activity inclusive, and 2) crossactivity. For the first one, we use all the activities in the gallery whereas in cross-activity we exclude the activity in the probe while retrieval. Similarly, we also evaluate for same-view (View<sup>+</sup>) and cross-view (View<sup>−</sup>) for NTU RGB-AB and PKU MM-AB where view information is available. For BRIAR-BGC3, we use the official protocol for face-restricted evaluation.

Evaluation metrics: For a thorough assessment of the model’s performance, we employ rank 1 accuracy, rank 5 accuracy, mean average precision (mAP), and TAR @ 0.1% FAR. While the first three evaluation metrics are more popular to evaluate a person identification model, the latter metric is also crucial to check the model’s ability to minimize the false acceptance rate.

Baseline methods: We consider ResNet3D-50 [17], MViTv2 [25] and GaitGL [27] as baselines. To further demonstrate the effectiveness of our model, we compare it against several state-of-the-art image based (CAL [15], PSTR [4], SCNet [16] and AIM [40]) and video based (TSF [22], VKD [33], BiCnet-TKS [20], STMN [11], PSTA [37], SINet [3],Video-CAL[15]) person identification methods.

Table 1. Comparison with state-of-the-art person identification methods: Evaluation shown on NTU RGB-AB, PKU MMD-AB, Charades-AB, and ACC-MM1-Activities on same-activity, View<sup>+</sup> evaluation protocol . †: this model was trained on silhouettes.
<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td rowspan="2">Venue</td><td colspan="2">NTU RGB-AB</td><td colspan="2">PKU MMD-AB</td><td colspan="2">Charades-AB</td><td colspan="2">ACC-MM1-Activities</td></tr><tr><td>Rank 1</td><td>mAP</td><td>Rank 1</td><td>mAP</td><td>Rank 1</td><td>mAP</td><td>Rank 1</td><td>mAP</td></tr><tr><td rowspan="4">Image</td><td>CAL [15]</td><td>CVPR22</td><td>73.79</td><td>28.40</td><td>81.31</td><td>49.45</td><td>43.84</td><td>25.81</td><td>69.83</td><td>42.81</td></tr><tr><td>PSTR [4]</td><td>CVPR22</td><td>69.14</td><td>34.14</td><td>84.33</td><td>47.52</td><td>37.15</td><td>24.69</td><td>57.41</td><td>34.48</td></tr><tr><td>SCNet [16]</td><td>ACM MM23</td><td>69.89</td><td>31.47</td><td>79.53</td><td>43.55</td><td>31.73</td><td>21.89</td><td>64.68</td><td>39.79</td></tr><tr><td>AIM [40]</td><td>CVPR23</td><td>71.37</td><td>35.41</td><td>82.52</td><td>48.89</td><td>40.13</td><td>28.31</td><td>74.79</td><td>49.14</td></tr><tr><td rowspan="7">Video</td><td>TSF [22]</td><td>AAAI20</td><td>71.79</td><td>31.80</td><td>76.43</td><td>37.50</td><td>35.38</td><td>21.89</td><td>49.41</td><td>29.73</td></tr><tr><td>VKD [33]</td><td>ECCV20</td><td>67.41</td><td>35.63</td><td>78.35</td><td>38.54</td><td>36.31</td><td>20.71</td><td>55.38</td><td>29.57</td></tr><tr><td>BiCnet-TKS [20]</td><td>CVPR21</td><td>72.71</td><td>34.45</td><td>80.79</td><td>38.52</td><td>40.31</td><td>27.34</td><td>60.44</td><td>32.79</td></tr><tr><td>STMN [11]</td><td>ICCV21</td><td>72.98</td><td>35.08</td><td>76.55</td><td>47.92</td><td>38.72</td><td>24.49</td><td>59.44</td><td>39.68</td></tr><tr><td>PSTA [37]</td><td>ICCV21</td><td>67.41</td><td>34.78</td><td>77.44</td><td>50.42</td><td>42.89</td><td>28.32</td><td>71.41</td><td>50.31</td></tr><tr><td>SINet [3]</td><td>CVPR22</td><td>69.41</td><td>30.68</td><td>79.58</td><td>40.80</td><td>40.31</td><td>26.90</td><td>65.39</td><td>45.41</td></tr><tr><td>Video-CAL [15]</td><td>CVPR22</td><td>75.49</td><td>39.86</td><td>79.59</td><td>49.42</td><td>43.91</td><td>28.51</td><td>77.48</td><td>50.08</td></tr><tr><td rowspan="3">Baselines</td><td>GaitGL [27] †</td><td></td><td>61.51</td><td>28.89</td><td>65.38</td><td>33.78</td><td>18.43</td><td>6.81</td><td>39.41</td><td>18.51</td></tr><tr><td>ResNet3D-50 [17]</td><td></td><td>64.23</td><td>26.89</td><td>69.70</td><td>32.64</td><td>32.25</td><td>17.42</td><td>44.31</td><td>22.54</td></tr><tr><td>MViTv2 [25]</td><td></td><td>63.87</td><td>26.41</td><td>68.37</td><td>28.52</td><td>28.51</td><td>15.39</td><td>40.59</td><td>21.52</td></tr><tr><td></td><td>ABNet (ours)</td><td></td><td>78.76</td><td>40.31</td><td>86.83</td><td>57.31</td><td>45.84</td><td>31.58</td><td>80.43</td><td>52.71</td></tr></table>

## 4.1. Results

In Table 1, we present rank 1 accuracy and mAP metrics for different baselines and state-of-the-art person identification methods across NTU RGB-AB, PKU MMD-AB, Charades-AB, and ACC-MM1-Activities datasets, using the same activity View<sup>+</sup> evaluation protocol. ABNet consistently outperforms both the best SOTA models and baselines across all four datasets. Table 3 compares ABNet with top-performing identification methods and baselines on the BRIAR-BGC3 dataset.

For a detailed evaluation, Table 2 shows ABNet’s performance across NTU RGB-AB, PKU MMD-AB, Charades-AB, and ACC-MM1-Activities datasets. This includes both same activity and cross activity evaluation protocols, featuring View<sup>+</sup> and View<sup>−</sup> settings for NTU RGB-AB and PKU MMD-AB. As view information is unavailable for Charades-AB and ACC-MM1-Activities datasets, the evaluation focuses solely on same and cross activity protocols.

Comparisons: From Tables 1 and 3, it’s clear that existing methods are primarily focused on identifying individuals based on walking patterns in various settings, lacking optimization for diverse activities. Our proposed ABNet consistently outperforms existing models across all datasets. AB-Net demonstrates approximately 2% to 4% higher rank 1 accuracy compared to the best existing method. This consistent superiority highlights ABNet’s effectiveness in person identification across diverse activity scenarios.

In Table 2, ABNet shows relatively stable performance across different evaluation protocols, except for ACC-MM1-Activities, which has fewer activity classes leading to larger performance gaps. The presence of overlapping activities in Charades-AB video samples reduces its performance compared to other datasets. Despite these challenges, ABNet consistently delivers strong results. Even on the predominantly walking-focused BRIAR-BGC3 dataset, ABNet outperforms the best SOTA model by 4% in rank 1 accuracy. Overall, ABNet demonstrates robust performance, particularly on datasets with diverse activity classes.

## 4.2. Ablations

To verify the effectiveness of ABNet and each of its components, we perform ablation study on the NTU RGB-AB dataset in Table 4 on the same activity evaluation protocol. Refer to the supplementary for ablation study on the cross activity evaluation protocol. Here, B/L stands for the baseline which is just the backbone model taking RGB video as input. K/D stands for bias-less distillation, A/P stands for activity prior, and lastly F/D stands for the bias learning.

Effect of bias-less distillation: Introducing bias-less distillation, either independently (row 2) or with an activity prior (row 4), leads to notable performance improvements over the baseline. However, combining bias-less distillation and activity prior demonstrates superior performance over independent use of distillation, showcasing their synergistic effect on model enhancement.

Effect of bias learning: Incorporating bias learning through a distorted video encoder branch boosts model performance even more (row 5). Similar to bias-less distillation, combining bias learning with an activity prior yields the best overall performance (row 6), highlighting the importance of their synergy in enhancing model robustness and disentangling biometrics and appearance information.

Effect of activity prior: Incorporating activity and biometrics features during inference significantly enhances performance compared to using only the baseline model (row

Table 2. Comprehensive performance evaluation of ABNet: results shown on NTU RGB-AB, PKU MMD-AB, Charades and ACC-MM1-Activities. We observe that cross-view and cross-activity setup is the most challenging with some performance drop when compared with same activity and same view setup.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Evaluation Protocol</td><td colspan="2">R@1</td><td colspan="2">R@5</td><td colspan="2"> $\mathrm { m A P }$ </td><td colspan="2">TAR @ 0.1% FAR</td></tr><tr><td> $\mathrm { V i e w } ^ { + }$ </td><td> $\mathrm { { V i e w } ^ { - } }$ </td><td> $\mathrm { V i e w } ^ { + }$ </td><td> $\mathrm { { V i e w } ^ { - } }$ </td><td> $\mathrm { V i e w } ^ { + }$ </td><td> $\mathrm { { V i e w } ^ { - } }$ </td><td> $\mathrm { V i e w } ^ { + }$ </td><td> $\mathrm { { V i e w } ^ { - } }$ </td></tr><tr><td rowspan="2">NTU RGB-AB</td><td>Same activity</td><td>78.76</td><td>77.81</td><td>85.31</td><td>82.41</td><td>40.31</td><td>38.80</td><td>39.83</td><td>35.68</td></tr><tr><td>Cross activity</td><td>77.01</td><td>76.43</td><td>81.37</td><td>80.37</td><td>37.64</td><td>36.14</td><td>34.92</td><td>33.79</td></tr><tr><td rowspan="2">PKU MMD-AB</td><td>Same activity</td><td>86.83</td><td>81.41</td><td>91.37</td><td>87.73</td><td>57.31</td><td>51.74</td><td>42.79</td><td>40.31</td></tr><tr><td>Cross activity</td><td>81.44</td><td>79.41</td><td>89.31</td><td>84.83</td><td>51.79</td><td>46.30</td><td>37.31</td><td>34.38</td></tr><tr><td rowspan="2">Charades</td><td>Same activity</td><td>45.84</td><td>-</td><td>51.04</td><td>-</td><td>31.58</td><td>-</td><td>25.39</td><td>-</td></tr><tr><td>Cross activity</td><td>44.82</td><td>-</td><td>52.01</td><td>-</td><td>28.78</td><td></td><td>22.61</td><td>-</td></tr><tr><td rowspan="2">ACC-MM1-Activities</td><td>Same activity</td><td>80.43</td><td>-</td><td>89.31</td><td>-</td><td>52.71</td><td></td><td>43.72</td><td>-</td></tr><tr><td>Cross activity</td><td>68.31</td><td></td><td>76.39</td><td></td><td>38.83</td><td></td><td>35.32</td><td></td></tr></table>

Table 3. Performance comparison on BRIAR-BGC3 against best state-of-the-art person identification and baselines.
<table><tr><td>Model</td><td>R@1</td><td>mAP</td><td>TAR@ 0.1%FAR</td></tr><tr><td>Image-CAL [14]</td><td>30.57</td><td>17.44</td><td>25.38</td></tr><tr><td>Video-CAL [14]</td><td>28.32</td><td>15.43</td><td>24.16</td></tr><tr><td>PSTA [37]</td><td>27.75</td><td>13.78</td><td>21.54</td></tr><tr><td>GaitGL[27]</td><td>12.61</td><td>9.51</td><td>6.44</td></tr><tr><td>ResNet3D-50[17]</td><td>22.50</td><td>12.83</td><td>19.71</td></tr><tr><td>MViTv2[25]</td><td>11.78</td><td>10.21</td><td>8.44</td></tr><tr><td>ABNet (ours)</td><td>34.38</td><td>18.78</td><td>26.42</td></tr></table>

Table 4. Ablation studies of each component of ABNet on NTU RGB-AB on same activity evaluation protocol.
<table><tr><td rowspan="2">B/L</td><td rowspan="2">K/D A/P</td><td rowspan="2">F/D</td><td colspan="2"> $\overline { { \mathrm { { V i e w } } ^ { + } } }$ </td><td colspan="2"> $\overline { { \mathrm { { V i e w } } ^ { - } } }$ </td></tr><tr><td>R@1</td><td>mAP</td><td>R@1</td><td> $\mathrm { m A P }$ </td></tr><tr><td>√</td><td></td><td></td><td>64.23</td><td>26.89</td><td>62.10</td><td>22.45</td></tr><tr><td>√</td><td>√</td><td></td><td>69.31</td><td>28.01</td><td>66.57</td><td>24.29</td></tr><tr><td>√</td><td></td><td>√</td><td>69.43</td><td>27.97</td><td>67.37</td><td>24.77</td></tr><tr><td>√</td><td>√</td><td>√</td><td>72.89</td><td>32.38</td><td>70.17</td><td>30.68</td></tr><tr><td>√</td><td>√</td><td>√</td><td>76.70</td><td>36.21</td><td>73.82</td><td>33.18</td></tr><tr><td>√</td><td>√</td><td>√ √</td><td>78.76</td><td>40.31</td><td>77.81</td><td>38.80</td></tr></table>

3). This integration consistently improves model efficacy across various model configurations demonstrating the role of activity recognition for biometrics.

## 4.3. Discussion and analysis

Effect of distortion: Figure 4 presents t-SNE plots of the biometrics and appearance feature space for ten NTU RGB-AB individuals at $\alpha \in [ 0 , 5 0 , 1 0 0 , 1 5 0 , 2 0 0 , 2 5 0 , 3 0 0 , 3 5 0 ]$ where α represents the amount of distortion. For optimal α we want to find such a value where biometrics feature clusters are overlapped due to being negative, but appearance feature clusters still remain relatively same due to being positive. Increasing α causes more overlap in biometrics feature clusters; whereas up ${ \mathrm { ~ t o ~ } } \alpha = 2 5 0 .$ , appearance feature clusters remain relatively stable. However, beyond this point, excessive distortion causes overlapping appearance clusters. Thus $\alpha = 2 5 0$ is selected as the optimal value. From the quantitative results presented in Table 5 similar effect of α is observed on the model’s performance.

Table 5. Effect of distortion on model performance for NTU RGB-AB on the same activity evaluation protocol
<table><tr><td rowspan="2">Distortion amount</td><td colspan="2"> $\overline { { \mathrm { V i e w } ^ { + } } }$ </td><td colspan="2"> $\overline { { \mathrm { V i e w } ^ { - } } }$ </td></tr><tr><td>R@1</td><td>mAP</td><td>R@1</td><td> $\mathrm { m A P }$ </td></tr><tr><td> $\alpha = 2 0 0$ </td><td>78.23</td><td>38.31</td><td>76.81</td><td>37.91</td></tr><tr><td> $\alpha = 2 5 0$ </td><td>78.76</td><td>40.31</td><td>77.81</td><td>38.80</td></tr><tr><td> $\alpha = 3 0 0$ </td><td>75.24</td><td>31.42</td><td>73.17</td><td>29.84</td></tr></table>

Table 6. Effect offace restriction on model performance for NTU RGB-AB on same activity evaluation protocol
<table><tr><td rowspan="2">Face Restricted</td><td colspan="2"> $\overline { { \mathrm { { V i e w } ^ { + } } } }$ </td><td colspan="2"> $\overline { { \mathrm { ~ V i e w } ^ { - } } }$ </td></tr><tr><td>R@1</td><td> $\mathrm { m A P }$ </td><td>R@1</td><td> $\mathrm { m A P }$ </td></tr><tr><td>Yes</td><td>78.76</td><td>40.31</td><td>77.81</td><td>38.80</td></tr><tr><td>No</td><td>79.24</td><td>41.64</td><td>78.87</td><td>40.04</td></tr></table>

Performance analysis across activities: Figure 5 illustrates the comparison between our method and the baseline across selected activities, encompassing the top five best and bottom five worst instances in person identification performance. Notably, activities posing challenges for person identification, resulting in lower performance, also exhibit reduced accuracy in activity recognition, except for a few exceptional activity classes. This correlation underscores the consistent relationship between the difficulty of identifying individuals within activities and the corresponding accuracy of recognizing those activities.

Effect of face restriction: Table 6 illustrates the model’s performance on the same activity evaluation protocol, indicating a minimal increase in performance despite the presence of facial features. This suggests the model’s resilience to facial variations, showcasing its capability to identify individuals based on non-facial cues. ABNet demonstrates stability in performance even after the removal of facial appearance cues, highlighting its reliance on other distinguishing features, such as activity-related cues.

![](images/1247dc7734f1fe8a7fa6955f74b2097d3fabf9b84941ca6bb2944574e6f4a290.jpg)  
Figure 4. Effect of distortion on feature space: The t-SNE plots illustrate the impact of varying distortion amount α ∈ [0, 50, 100, 150, 250, 300, 350] on biometrics (top) and appearance (bottom) features of ABNet for ten random NTU RGB-AB identities. As α increases from left to right, the optimal results occur at α = 250 (shown in square) where biometrics changes while appearance remains consistent. Beyond α = 250, appearance gets distorted too, making it unsuitable for disentanglement.

![](images/9b2916e0156f4dfad02f8ac5da3c7a0c1f2c0940f09f2d3250c633ef469ec942.jpg)  
Figure 5. Performance analysis across activities: The bar plot on left axis shows rank 1 identification accuracy for given activity of ABNet against baseline on 10 activity (5 best and 5 worst) classes of NTU RGB-AB. The scatter plot with markers on right axis shows activity recognition accuracy for corresponding classes.

Qualitative results: In addition to the quantitative results, we show top 4 rank retrieval results in Figure 6. Each row in this figure corresponds to a probe (left) and the identities retrieved (right) by ABNet. The retrieval list shows accurate person identification across a variety of activities and appearance, effectively highlighting ABNet’s ability to learn from activity cues rather than appearance.

## 5. Conclusion

In this work we study a novel problem of person identification from videos of daily activities. We propose AB-Net, a simple approach to solve this problem which relies on feature disentanglement and activity prior for person identification. This approach incorporates feature disentanglement at both biometric and appearance levels, leveraging distinct strategies to enhance accuracy and mitigate biases. By distilling biometric knowledge from a bias-free silhouette-trained model and learning appearance biases via elastic distortion-based transformations, our framework ensures a comprehensive understanding of individuals’ inherent biometric traits while accounting for appearance variations. Moreover, the integration of an activity prior during inference further enriches the model’s capabilities. Through extensive evaluations on five benchmark datasets derived from large-scale activity recognition datasets, our approach consistently surpasses several state-of-the-art methods.

![](images/7ea73ed71c4b00fdb02b3dd510bd485a4646af8b245fec75bd09cb1b83e82c6e.jpg)  
Figure 6. Top 4 rank retrieval samples for ABNet on NTU RGB-AB, Charades-AB, PKU MMD-AB and ACC-MM1-Activities on row 1, 2, 3, 4 respectively. The left most column shows the probe and rest of the columns are the retrieved list. Accurate retrieval is shown with green box and inaccurate with red. The subjects from ACC-MM1-Activities consented to publication.

## 6. Acknowledgement

This research is based upon work supported in part by the Office of the Director of National Intelligence (IARPA) via 2022-21102100001. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the US Government. The US Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

## References

[1] Elastic transform. https://pytorch.org/vision/ main / generated / torchvision . transforms . ElasticTransform.html. [Online; accessed 08- November-2023]. 4

[2] Insaf Adjabi, Abdeldjalil Ouahabi, Amir Benzaoui, and Abdelmalik Taleb-Ahmed. Past, present, and future of face recognition: A review. Electronics, 9(8):1188, 2020. 1

[3] Shutao Bai, Bingpeng Ma, Hong Chang, Rui Huang, and Xilin Chen. Salient-to-broad transition for video person reidentification. In CVPR, pages 7339–7348, 2022. 2, 5, 6

[4] Jiale Cao, Yanwei Pang, Rao Muhammad Anwer, Hisham Cholakkal, Jin Xie, Mubarak Shah, and Fahad Shahbaz Khan. Pstr: End-to-end one-step person search with transformers. In CVPR, pages 9458–9467, 2022. 1, 2, 5, 6

[5] Guangyi Chen, Jiwen Lu, Ming Yang, and Jie Zhou. Learning recurrent 3d attention for video-based person reidentification. IEEE TIP, 29:6963–6976, 2020. 2

[6] Jiaxing Chen, Xinyang Jiang, Fudong Wang, Jun Zhang, Feng Zheng, Xing Sun, and Wei-Shi Zheng. Learning 3d shape feature for texture-insensitive person re-identification. In CVPR, pages 8146–8155, 2021. 2

[7] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2022. 5

[8] Liu Chunhui, Hu Yueyu, Li Yanghao, Song Sijie, and Liu Jiaying. Pku-mmd: A large scale benchmark for continuous multi-modal human action understanding. arXiv preprint arXiv:1703.07475, 2017. 4

[9] David Cornett, Joel Brogan, Nell Barber, Deniz Aykac, Seth Baird, Nicholas Burchfield, Carl Dukes, Andrew Duncan, Regina Ferrell, Jim Goddard, et al. Expanding accurate person recognition to new altitudes and ranges: The briar dataset. In WACV, pages 593–602, 2023. 5

[10] Ju Dai, Pingping Zhang, Dong Wang, Huchuan Lu, and Hongyu Wang. Video person re-identification by temporal residual learning. IEEE TIP, 28(3):1366–1377, 2018. 2

[11] Chanho Eom, Geon Lee, Junghyup Lee, and Bumsub Ham. Video-based person re-identification with spatial and temporal memory networks. In ICCV, pages 12036–12045, 2021. 5, 6

[12] Chao Fan, Yunjie Peng, Chunshui Cao, Xu Liu, Saihui Hou, Jiannan Chi, Yongzhen Huang, Qing Li, and Zhiqiang He. Gaitpart: Temporal part-based model for gait recognition. In CVPR, pages 14225–14233, 2020. 1, 2

[13] Chao Fan, Junhao Liang, Chuanfu Shen, Saihui Hou, Yongzhen Huang, and Shiqi Yu. Opengait: Revisiting gai recognition towards better practicality. In CVPR, pages 9707–9716, 2023. 1, 2

[14] Xinqian Gu, Hong Chang, Bingpeng Ma, Hongkai Zhang, and Xilin Chen. Appearance-preserving 3d convolution for video-based person re-identification. In ECCV, pages 228– 243. Springer, 2020. 1, 7

[15] Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan, and Xilin Chen. Clothes-changing person

re-identification with rgb modality only. In CVPR, pages 1060–1069, 2022. 1, 2, 3, 5, 6

[16] Peini Guo, Hong Liu, Jianbing Wu, Guoquan Wang, and Tao Wang. Semantic-aware consistency network for clothchanging person re-identification. In ACM MM, 2023. 2, 3, 5, 6

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 5, 6, 7

[18] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2, 4

[19] Peixian Hong, Tao Wu, Ancong Wu, Xintong Han, and Wei-Shi Zheng. Fine-grained shape-appearance mutual learning for cloth-changing person re-identification. In CVPR, pages 10513–10522, 2021. 2

[20] Ruibing Hou, Hong Chang, Bingpeng Ma, Rui Huang, and Shiguang Shan. Bicnet-tks: Learning efficient spatialtemporal representation for video person re-identification. In CVPR, pages 2014–2023, 2021. 1, 2, 5, 6

[21] Yan Huang, Qiang Wu, Jingsong Xu, and Yi Zhong. Celebrities-reid: A benchmark for clothes variation in longterm person re-identification. In IJCNN, pages 1–8. IEEE, 2019. 2

[22] Xinyang Jiang, Yifei Gong, Xiaowei Guo, Qize Yang, Feiyue Huang, Wei-Shi Zheng, Feng Zheng, and Xing Sun. Rethinking temporal fusion for video-based person re-identification on semantic and time aspect. In AAAI, pages 11133–11140, 2020. 2, 5, 6

[23] Xin Jin, Tianyu He, Kecheng Zheng, Zhiheng Yin, Xu Shen, Zhen Huang, Ruoyu Feng, Jianqiang Huang, Zhibo Chen, and Xian-Sheng Hua. Cloth-changing person reidentification from a single image with gait prediction and regularization. In CVPR, pages 14278–14287, 2022. 2

[24] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015. 5

[25] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In CVPR, pages 4804–4814, 2022. 5, 6, 7

[26] Junhao Liang, Chao Fan, Saihui Hou, Chuanfu Shen, Yongzhen Huang, and Shiqi Yu. Gaitedge: Beyond plain end-to-end gait recognition for better practicality. In ECCV, pages 375–390. Springer, 2022. 1, 2

[27] Beibei Lin, Shunli Zhang, and Xin Yu. Gait recognition via effective global-local feature representation and local temporal aggregation. In ICCV, pages 14648–14656, 2021. 1, 2, 5, 6, 7

[28] Chih-Ting Liu, Chih-Wei Wu, Yu-Chiang Frank Wang, and Shao-Yi Chien. Spatially and temporally efficient non-local attention network for video-based person re-identification. arXiv preprint arXiv:1908.01683, 2019. 2

[29] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+ d 120: A largescale benchmark for 3d human activity understanding. IEEE TPAMI, 42(10):2684–2701, 2019. 4

[30] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, 2015. 1

[31] Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou. Magface: A universal representation for face recognition and quality assessment. In CVPR, pages 14225–14234, 2021. 1

[32] K. O’Brien, M. Rybak, J. Huang, A. Stevens, M. Fredriksz, M. Chaberski, D. Russell, L. Castin, M. Jou, N. Gurrapadi, and M. Bosch. Accenture-mm1: A multimodal person recognition dataset. In WACVW, 2024. 5

[33] Angelo Porrello, Luca Bergamini, and Simone Calderara. Robust re-identification by multiple views knowledge distillation. In ECCV, pages 93–110. Springer, 2020. 1, 2, 5, 6

[34] Xuelin Qian, Wenxuan Wang, Li Zhang, Fangrui Zhu, Yanwei Fu, Tao Xiang, Yu-Gang Jiang, and Xiangyang Xue. Long-term cloth-changing person re-identification. In ACCV, 2020. 2

[35] Gunnar A Sigurdsson, Gul Varol, Xiaolong Wang, Ali¨ Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In ECCV, pages 510–526, 2016. 5

[36] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. NeurIPS, 30, 2017. 2

[37] Yingquan Wang, Pingping Zhang, Shang Gao, Xia Geng, Hu Lu, and Dong Wang. Pyramid spatial-temporal aggregation for video-based person re-identification. In ICCV, pages 12026–12035, 2021. 2, 5, 6, 7

[38] Ancong Wu, Wei-Shi Zheng, Xiaowei Guo, and Jian-Huang Lai. Distilled person re-identification: Towards a more scalable system. In CVPR, pages 1187–1196, 2019. 2

[39] Qize Yang, Ancong Wu, and Wei-Shi Zheng. Person reidentification by contour sketch under moderate clothing change. IEEE TPAMI, 43(6):2029–2046, 2019. 2

[40] Zhengwei Yang, Meng Lin, Xian Zhong, Yu Wu, and Zheng Wang. Good is bad: Causality inspired cloth-debiasing for cloth-changing person re-identification. In CVPR, pages 1472–1481, 2023. 1, 2, 3, 5, 6

[41] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven CH Hoi. Deep learning for person reidentification: A survey and outlook. IEEE TPAMI, 44(6): 2872–2893, 2021. 1

[42] Shiqi Yu, Daoliang Tan, and Tieniu Tan. A framework for evaluating the effect of view angle, clothing and carrying condition on gait recognition. In ICPR, pages 441–444, 2006. 1

[43] Dongyu Zhang, Wenxi Wu, Hui Cheng, Ruimao Zhang, Zhenjiang Dong, and Zhaoquan Cai. Image-to-video person re-identification with temporally memorized similarity learning. IEEE TCSVT, 28(10):2622–2632, 2017. 1, 2

[44] Ziyuan Zhang, Luan Tran, Xi Yin, Yousef Atoum, Xiaoming Liu, Jian Wan, and Nanxin Wang. Gait recognition via disentangled representation learning. In CVPR, pages 4710–4719, 2019. 1, 2

[45] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian. Scalable person re-identification: A benchmark. In ICCV, pages 1116–1124, 2015. 1