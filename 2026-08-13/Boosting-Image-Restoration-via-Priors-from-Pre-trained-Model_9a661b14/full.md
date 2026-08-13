# Boosting Image Restoration via Priors from Pre-trained Models

Xiaogang Xu<sup>1,2,3</sup> Shu Kong<sup>5,6,7</sup> Tao Hu<sup>3,8</sup> Zhe Liu<sup>1\*</sup> Hujun Bao<sup>1,4</sup> <sup>1</sup> Zhejiang Lab <sup>2</sup> CUHK <sup>3</sup> RealityEdge <sup>4</sup> Zhejiang University <sup>5</sup> University of Macau <sup>6</sup> Institute of Collaborative Innovation <sup>7</sup> Texas A&M University <sup>8</sup> National University of Singapore

xiaogangxu00@gmail.com, skong@um.edu.mo, yihouxiang@gmail.com zhe.liu@zhejianglab.com, bao@cad.zju.edu.cn

## Abstract

Pre-trained models with large-scale training data, such as CLIP and Stable Diffusion, have demonstrated remarkable performance in various high-level computer vision tasks such as image understanding and generation from language descriptions. Yet, their potential for low-level tasks such as image restoration remains relatively unexplored. In this paper, we explore such models to enhance image restoration. As off-the-shelffeatures (OSF)from pretrained models do not directly serve image restoration, we propose to learn an additional lightweight module called Pre-Train-Guided Refinement Module (PTG-RM) to refine restoration results ofa target restoration network with OSF. PTG-RM consists of two components, Pre-Train-Guided Spatial-Varying Enhancement (PTG-SVE), and Pre-Train-Guided Channel-Spatial Attention (PTG-CSA). PTG-SVE enables optimal short- and long-range neural operations, while PTG-CSA enhances spatial-channel attention for restoration-related learning. Extensive experiments demonstrate that PTG-RM, with its compact size (<1M parameters), effectively enhances restoration performance of various models across different tasks, including low-light enhancement, deraining, deblurring, and denoising.

## 1. Introduction

Image restoration plays a vital role in real-world scenarios, aiming to reconstruct high-quality images by eliminating degradations. It has broad applications in various fields, such as denoising [43, 44] and low-light enhancement [41, 42] for improving smartphone-captured photos. While effective restoration networks have been proposed [19, 52], the inherently ill-posed nature of image restoration makes it challenging to achieve significant improvements by merely modifying network structures. Simply increasing model parameters does not guarantee better results, as the model may tend to overfit to the training data.

![](images/c601bb14d1141d1b2a6aafb96eab2510a392444b057ea2911d6c6f182fd20469.jpg)  
Figure 1. Our method leverages pre-trained models, such as CLIP and Stable Diffusion, and significantly improves image restoration across various tasks. More results on different tasks/models can be seen in experiments. Pre-trained models are involved during the training and not required during the inference.

Restoration performance relies on strong image priors, such as the novel level of denoising [38] or the blur kernel in deblurring [14, 50]. However, estimating these priors is challenging, especially with real-world data. Some approaches utilize physical variables as priors, like depth information [46] and semantic features [1, 36, 41] derived from pre-trained networks. Nevertheless, these physical variables are not robust enough since the dense depth/semantic prediction networks do not have sufficient generalization ability among different scenes in restoration tasks. As a result, employing them requires complex and specific mechanisms, limiting their applicability across various tasks. In this paper, we propose a novel approach that extracts degradation-related information from pre-trained models (with various training objectives) exposed to different degradation during pre-training, all without requiring explicit annotations.

Motivation. Two types of pre-trained models may contain degradation-related information during training: restoration models, and pre-trained models on large-scale data (e.g., CLIP [27], BLIP [16], and BLIP2 [17]). Using the former is evident, but models trained with some types of degradation may not effectively help restore images with other types of degradation. Using the latter remains unexplored. CLIP-IQA [33] finds that CLIP features contain degradation-related information and so be useful for image assessment, while no restoration approaches have been proposed yet. Existing pre-trained multi-modality models may have been trained on various degraded images. Presumably, restoration-related annotations are unavailable during pretraining, their resulted features likely contain valuable information for image restoration. The key is to leverage such information to help the target restoration learning. However, the heterogeneity of pre-trained models and restoration models poses difficulties in using the off-the-shelf features extracted from pre-trained models.

![](images/377ae097f4f30a1e5871d3f83dcc405db92b360f6cbc4d34367b1786b289e134.jpg)  
Figure 2. We present a lightweight plugin, pre-training guided refining module (PTG-RM), to leverage pre-trained models for enhancing image restoration. The desired prior is the OFS G(I<sub>d</sub>). It has two components, PTG spatial varying enhancement (PTG-SVE), and PTG channel-spatial attention (PTG-CSA). Fig. 3 depicts their details. Our PTG-RM significantly improves restoration in various tasks as listed in the top-right (see quantitative results previewed in Fig. 1).

Technical novelty. We introduce a novel pre-training guided refinement module (PTG-RM) that leverages offthe-shelf features (OSF) computed by a pre-trained model G to improve image restoration tasks. The PTG-RM R is a lightweight plugin (Fig. 2) (additional R has <1M parameters in total). PTG-RM enables us to determine optimal operation ranges and spatial-channel attention, thus facilitating image restoration. It takes as input the initially enhanced image from F, the input image, and its OSF extracted by a pre-trained model. It is trained with F (using the same loss as F) and adaptively enhances it. PTG-RM R consists of two components: Pre-Train-Guided Spatial Varying Enhancement (PTG-SVE), and Pre-Train-Guided Channel-Spatial Attention (PTG-CSA).

PTG-SVE employs spatial-varying operations to refine the initially enhanced results differently from region to region. Unlike previous methods [42] that rely on fixed references to determine optimal operation ranges, we establish a spatial-aware learnable mapping for OSF and utilize the mapped features as spatial-wise guidance. This adaptively fuses the features extracted from short- and long-range operations, allowing different regions to be refined appropriatel and yielding more effective enhancement.

Following PTG-SVE, PTG-CSA further enhances the results by formulating effective channel- and spatial-attention with OSF. We note that different areas may require varying degrees of feature correctness via the attention mechanism. Hence, we propose to generate spatial-varying convolution kernels to synthesize the spatial weights. Our approach tailors the attention process to different regions.

Contributions. We make three major contributions.

• We present a novel and general method that leverages pretrained models to enhance various restoration tasks. Our work opens up possibilities for improving performance across various domains.

• We propose a novel paradigm that leverages pre-trained priors to formulate effective neural operation ranges and attention mechanisms.

• We validate our method through extensive experiments on different datasets, networks, and tasks, and show remarkable improvements over prior methods (cf. Fig. 1).

## 2. Related Work

Image Priors for Restoration. Different restoration tasks demand distinct image priors, such as noise levels for denoising and blurring kernels for deblurring. Due to the ill-posed nature of restoration, estimating priors is difficult. In real-world scenarios, these priors are typically intertwined, adding further complexity to the restoration process. Recent literature introduces several methods to improve restoration by leveraging multi-modal maps as unified priors. These methods predominantly rely on pre-computed physical multi-modal maps. For instance, SKF [41] uses semantic maps to optimize the feature space for low-light enhancement. SMG [46] employs a generative framework t o i nt e gr at e e d g e, d e pt h, a n d s e m a nti c i nf or m ati o n, e n h a n ci n g t h e i niti al a p p e ar a n c e m o d eli n g f or l o w-li g ht s c e n ari o s. A d diti o n all y, s o m e a p pr o a c h e s u s e N e ar-I nfr ar e d ( NI R) i nf or m ati o n t o r e fi n e i m a gi n g r e s ult s [1 2 3 2 ]. T h e s e pri or s ar e al s o a p pli e d t o ot h er r e st or ati o n t a s k s, s u c h a s i m a g e d e n oi si n g [ 2 0 ] a n d d er ai ni n g [1 8 ]. H o w e v er, ali g ni n g t h e s e pri or s wit h t h e i n p ut i m a g e c a n b e c h all e n gi n g, a n d err or s i n t h e pri or s m a y a d v er s el y i m p a ct p erf or m a n c e. Diff ere nt fr o m e xi sti n g w or k s, w e pr o p o s e t o l e v er a g e pr e-tr ai n e d m o d el s a s pri or s t o e n h a n c e i m a g e r e st or ati o n.

P r e- Tr ai n e d M o d el s f o r D o w n st r e a m T a s k s. R ec e ntl y, a s eri e s of pr e-tr ai n e d m o d el s wit h l ar g e- s c al e tr ai ni n g d at a s et s h a v e e m er g e d, p arti c ul arl y i n t h e f or m of m ulti- m o d al m o d el s s u c h a s C LI P [ 2 7 ], B LI P [1 6 ], a n d B LI P 2 [ 1 7 ]. T h e f e at ur e s p a c e l e ar n e d b y t h e s e m o d el s off er s ri c h k n o wl e d g e t h at c a n b e n e fit v ari o u s t a s k s. W hil e pr e vi o u s w or k h a s d e m o n str at e d t h e eff e cti v e n e s s of C LI P i n hi g h-l e v el t a s k s li k e z er o- s h ot cl a s si fi c ati o n [ 5 3 ], i ma g e e diti n g [ 2 5 ], o p e n- w orl d s e g m e nt ati o n [3 9 6 0 ], a n d 3 D cl a s si fi c ati o n [ 4 7 5 9 ], it s p ot e nti al f or ai di n g l o w-l e v el r e st or ati o n t a s k s r e m ai n s u n e x pl or e d. O nl y t h e c a p a bilit y of e m pl o yi n g s u c h f or i m a g e q u alit y a s s e s s m e nt, a s d e m o nstr at e d i n C LI P-I Q A, h a s b e e n e x pl or e d. We pr o p o s e a g e ner al fr a m e w or k t o l e v er a g e pr e-tr ai n e d m o d el s t o i m pr o v e v ari o u s r e st or ati o n t a s k s.

## 3. M et h o d s

B a c k g r o u n d. L et $I _ { d }$ r e pr e s e nt a d e gr a d e d i m a g e, a n d $I _ { c }$ d e n ot e t h e c orr e s p o n di n g gr o u n d-tr ut h ( wit h o ut d e gr a d ati o n). A r e st or ati o n n et w or k $\mathcal { F }$ pr o d u c e s r e st or e d i m a g e $\hat { I } _ { c } = \mathcal { F } ( I _ { d } )$ . D e s pit e t h e e xi st e n c e of v ari o u s eff e cti v e n etw or k str u ct ur e s $\bar { \mathcal { F } }$ t h at h a v e b e e n pr o p o s e d, t h er e ar e c urr e nt u p p er b o u n d s i n t h e s e t a s k s. Br e a ki n g t hr o u g h t h e s e b o u n d s oft e n r e q uir e s d e si g ni n g m or e c o m pl e x n et w or k s a n d tr ai ni n g str at e gi e s, w hi c h c a n b e ar d u o u s. A d diti o n all y, i n n o v ati o n s i n n et w or k ar c hit e ct ur e or tr ai ni n g str at e gi e s f or o n e t a s k mi g ht n ot tr a n sl at e t o a n ot h er. W hil e diff er e nt pri or s h a v e b e e n i ntr o d u c e d i nt o t h e r e st or ati o n pr o c e s s, i n cl u di n g i m a g e a n d p h y si c al pri or s, e sti m ati n g t h e s e pri or s i s dif fic ult.

M oti v ati o n. We h y p ot h e si z e t h at t h e pri or $g$ c a n b e eff e cti v el y r e pr e s e nt e d a s t h e f e at ur e e xtr a ct e d fr o m v ari o u s pr e-tr ai n e d m o d el s ${ \mathcal { G } } ,$ a s $g = \mathcal { G } ( I _ { d } )$ . N ot e t h at i s t y pi c all y n ot tr ai n e d wit h r e st or ati o n t ar g et s b ut mi g ht h a v e b e e n e xp o s e d t o i m a g e s wit h di v er s e d e gr a d ati o n s. S o it i s li k el y t o l e ar n u s ef ul i nf or m ati o n t o h el p i m a g e r e st or ati o n. We pr o p o s e a n o v el a p pr o a c h t h at u s e s $g$ t o i m pr o v e t h e i niti al r e st or ati o n b y ${ \mathcal { F } } ,$ , e v e n if t h e s e n et w or k s h a v e alr e a d r e a c h e d t h eir c urr e nt u p p er b o u n d s.

C h all e n g e. U si n g t o a s si st $\mathcal { F }$ i s n o n-tri vi al. Pri m aril y, t h e f e at ur e i s n ot i n h er e ntl y ali g n e d wit h t h e r e st or ati o n t a s k s b e c a u s e t h e y mi g ht r e pr e s e nt diff er e nt a s p e ct s. F or i n st a n c e, f e at ur e s fr o m C LI P f o c u s m or e o n s e m a nti c i nf orm ati o n, m a ki n g dir e ct ali g n m e nt t o r e st or ati o n c h all e n gi n g. M or e o v er, t h e s e pri or s e x hi bit v ar yi n g s h a p e s, s u c h a s t h e o n e- di m e n si o n al ( 1 D) f e at ur e s fr o m t h e C LI P m o d el, w hil e t h e f e at ur e s i n $\mathcal { F }$ ar e t y pi c all y 2 D. T o r e c o n cil e t h e di s cr e pa n ci e s i n b ot h r e pr e s e nt ati o n a n d s h a p e, w e pr o p o s e a r efi n e m e nt m o d ul e t o r e fi n e t h e i niti al r e st or ati o n b y ${ \mathcal F } .$ T hi s eli mi n at e s t h e n e e d t o ali g n $g$ t o di sti n ct f e at ur e s of $\mathcal { F }$ a n d all o w s f or a u ni fi e d 1 D r e pr e s e nt ati o n f or $g .$ F urt h erm or e, w e i ntr o d u c e a n o v el a p pr o a c h t o utili z e $g$ t o f or m ul at e o pti m al n e ur al o p er ati n g r a n g e s vi a a n eff e cti v e att e nti o n m e c h a ni s m i n $\scriptstyle { \mathcal { R } } .$ . T hi s i m pli citl y di still s r e st or ati o nr el at e d i nf or m ati o n, eff e cti v el y b o o sti n g t h e fi n al p erf orm a n c e.

## 3. 1. O v e r vi e w of R e fi n e m e nt M o d ul e

Fi g. d e pi ct s t h e r e st or ati o n pi p eli n e u si n g o ur m et h o d. Gi v e n a n i n p ut i m a g e $I _ { d } ,$ w e h a v e a n i niti al r e st or ati o n r es ult a s $\hat { I } _ { c } ~ \equiv ~ \mathcal { F } ( I _ { d } )$ . We ai m t o r e fi n e t h e r e s ult u si n g t h e pr o p o s e d pr e-tr ai ni n g g ui d e d r e fi n e m e nt m o d ul e ( P T G-R M) $\textstyle { \mathcal { R } } ,$ , r e s ulti n g i n $\bar { I } _ { c } = \mathcal { R } ( \hat { I } _ { c } , I _ { d } , g )$ . T h e k e y of t hi s a ppr o a c h i s t o di still r e st or ati o n-r el at e d i nf or m ati o n fr o m t h e pri or

i s a si m pl e e n c o d er- d e c o d er str u ct ur e. T h e e n c o d er a n d d e c o d er of ar e d e n ot e d a s $\mathcal { R } _ { e }$ a n d $\mathcal { R } _ { d } ,$ r e s p e cti v el y. T o e n s ur e li g ht w ei g ht i m pl e m e nt ati o n, di still ati o n o c c ur s i n t h e l at e nt s p a c e, a v oi di n g t h e n e e d t o ali g n $g$ wit h r e st or ati o n-r el at e d f e at ur e s. T h e l at e nt f e at ur e $f$ i s d eri v e d t hr o u g h a c o m p ari s o n b et w e e n t h e i niti al e n h a n c e d r e s ult s a n d t h e ori gi n al i n p ut i m a g e s, gi v e n a s $f = \mathcal { R } _ { e } ( \hat { I } _ { c } \oplus I _ { d } )$ w h er e d e n ot e s t h e c o n c at e n ati o n o p er ati o n. T h e r e s ulti n g $f$ i s i n $\mathbb { R } ^ { h \times w \times c }$ , wit h , a n d r e pr e s e nti n g f e at ur e h ei g ht, wi dt h, a n d c h a n n el n u m b er, r e s p e cti v el y. T h e pri or s ar e u s e d i n f urt h er l e ar ni n g t h e l at e nt f e at ur e a s $\bar { \boldsymbol { f } } = \mathcal { C } ( \boldsymbol { A } ( \boldsymbol { f } , \boldsymbol { g } ) , \boldsymbol { g } )$ , w h er e a n d r e pr e s e nt t h e Pr e- Tr ai n-G ui d e d S p ati al- Var yi n g E n h a n c e m e nt ( P T G- S V E) a n d Pr e-Tr ai n- G ui d e d C h a n n el- S p ati al Att e nti o n ( P T G- C S A) m o dul e s, r e s p e cti v el y. T h e fi n al e n h a n c e m e nt i s o bt ai n e d fr o m t h e d e c o d er a s $[ \hat { I } _ { m } , I _ { r } ] = \mathcal { R } _ { d } ( \hat { f } )$ , c o m pri si n g t w o c o m p on e nt s. T h e fir st c o m p o n e nt, $I _ { m } ,$ , r e pr e s e nt s t h e c orr e cti o n m a s k u s e d t o miti g at e err or s i n t h e i niti al e n h a n c e m e nt r es ult s. T h e s e c o n d c o m p o n e nt, $I _ { r } ,$ i s t h e r e si d u al r e fi n e m e nt t h at a d dr e s s e s artif a ct s a n d a d d s a d diti o n al d et ail s. T h e fi n al r e s ult i s d e n ot e d a s

$$
\bar { I } _ { c } = I _ { d } + ( \hat { I } _ { c } - I _ { d } ) \times I _ { m } + I _ { r } .\tag{1}
$$

3. 2. P r e- Tr ai n- G ui d e d S p ati al- V a r yi n g O p e r ati o n s I n P T G- S V E, w e ar g u e t h at $g = \mathcal { G } ( I _ { d } )$ m a y c o nt ai n i nf orm ati o n r e fl e cti n g t h e pi x el-l e v el i m a g e q u alit y of $I _ { d } .$ . F or are a s wit h p o or q u alit y, l o n g-r a n g e o p er ati o n s ar e u s e d t o c a pt ur e n o n-l o c al f e at ur e s, w hil e r e gi o n s wit h r el ati v el y g o o d q u alit y pri oriti z e l o c al f e at ur e s f or a c c ur at e r e st or ati o n.

![](images/c89d26b284d82ed39f83c3414fca15af252cea332d0635ab8199741cb39e2d12.jpg)  
Fi g ur e 3. T h e pi p eli n e of P T G- S V E a n d P T G- C S A. I n P T G- S V E, w e u s e t h e l e ar n a bl e s p ati al e m b e d di n g $S _ { m } ,  { \operatorname { O S F } } g ,$ a n d i n p ut f e at ur e t o a d a pti v el y f or m ul at e s p ati al w ei g ht s ( , E q. ) f or f u si n g s h ort- a n d l o n g-r a n g e pr o c e s s e d f e at ur e s $( f _ { s }$ a n d ) vi a o p er ati o n s $\mathcal { R } _ { s }$ a n d $\begin{array} { r } { \mathcal { R } _ { l } , } \end{array}$ , yi el di n g ( E q. ). I n P T G- C S A, O S F c o n diti o n s c h a n n el att e nti o n $M _ { c }$ f or t hr o u g h $\scriptstyle { \mathcal { R } } _ { c }$ ( E q. ). A d diti o n all y, c o m bi n e s wit h l e ar n a bl e s p ati al r e pr e s e nt ati o n $\scriptstyle { \mathcal { S } } _ { c }$ a n d $\hat { f }$ t o g e n er at e s p ati al att e nti o n m a p $M _ { s } ,$ , u si n g s p ati al- wi s e c o n v ol uti o n s $C _ { p }$ ( o bt ai n e d vi a $\mathcal { R } _ { p } )$ t o d eri v e $\hat { \mathcal { M } } _ { s }$ t h at i s f urt h er pr o c e s s e d wit h $\scriptstyle { \mathcal { R } } _ { o }$ ( E q s. a n d ). C h a n n el- a n d s p ati al- att e nti o n o ut p ut s $( \hat { f } _ { c }$ a n d $\hat { f } _ { s } )$ m er g e vi a $\mathcal { R } _ { f }$ t o e n h a n c e f e at ur e ( E q. ).

## 3. 3. P r e- Tr ai n- G ui d e d Att e nti o n

I n Fi g. , t h e pri m ar y o bj e cti v e i s t o pr e di ct t h e o pti m al n e ur al o p er ati o n r a n g e f or e a c h l o c ati o n of t h e f e at ur e m a p $f ,$ w hi c h w e r ef er t o a s t h e “r a n g e s c or e m a p ”, d e n ot e d a s . T o e n s ur e a g e n er al wit h u ni fi e d 1 D pri or s $g$ fr o m v ari o u s m o d el s, w e pr o p o s e a d di n g l o c ati o n- a w ar e e m b e ddi n g s f or t h e pri or s, t h er e b y a d a pti v el y di s c o v eri n g q u alit y i nf or m ati o n f or diff er e nt pi x el s. L et $S = \{ ( x , y ) | x \in$ $[ 1 , w ] , y \in [ 1 , h ] \}$ r e pr e s e nt t h e 2 D c o or di n at e m a p wit h dim e n si o n s $h \times w \times 2$ . We u s e a p o siti o n e m b e d di n g m o d ul e t o g e n er at e s p ati al r e pr e s e nt ati o n, d e n ot e d a s $S _ { m } = \mathcal { P } ( S )$ w h er e $\mathcal { S } \in \mathbb { R } ^ { h \times w \times c }$ . F urt h er m or e, t o d et er mi n e t h e a dmir e d n e ur al o p er ati o n r a n g e f or e a c h l o c ati o n of $f ,$ w e u s e a l e ar n a bl e m a p pi n g f u n cti o n $\mathcal { T } _ { m }$ t o tr a n sf or m t h e pri or s t o a n ot h er s p a c e t h at c a n m or e eff e cti v el y d e ci d e t h e o pti m al r a n g e. T o o bt ai n , w e u s e a r a n g e-l e ar ni n g m o d ul e $\mathcal { R } _ { m }$ w hi c h t a k e s t h e e n c o d er’s f e at ur e $f ,$ t h e pr e-tr ai n e d pri or a n d t h e s p ati al r e pr e s e nt ati o n $s _ { m }$ a s i n p ut s. T h e pr o c e d ur e i s d e n ot e d a s

$$
M = \mathcal { R } _ { m } ( f \oplus \mathcal { T } _ { m } ( g ) \oplus \mathcal { S } _ { m } ) .\tag{2}
$$

F oll o wi n g [ 4 2 ], w e u s e C N N f or t h e s h ort-r a n g e o p erati o n, d e n ot e d a s $\mathcal { R } _ { s } ,$ , a n d tr a n sf or m er f or t h e l o n g-r a n g e o p er ati o n, r e pr e s e nt e d a s $\mathcal { R } _ { l }$ . S p e ci fi c all y, w e e m pl o y t h e R e st or m er b a c k b o n e f or $\mathcal { R } _ { l }$ a n d R e s N et f or $\mathcal { R } _ { s }$ . S u p p o s e t h e f e at ur e s aft er t h e s h ort- a n d l o n g-r a n g e o p er ati o n ar e $f _ { s }$ a n d $f _ { l } ,$ , r e s p e cti v el y. We c a n o bt ai n t h e r e fi n e d f e at ur e $\hat { f }$ a s

$$
\begin{array} { r } { f _ { s } = \mathcal { R } _ { s } ( f ) , f _ { l } = \mathcal { R } _ { l } ( f ) , \hat { f } = M \times f _ { s } + ( 1 - M ) \times f _ { l } . } \end{array}\tag{3}
$$

T h e pr e vi o u s a p pr o a c h [ 4 2 ] r eli e s o n pr e- c o m p ut e d S N R v al u e s, w hi c h m a y n ot al w a y s b e a c c ur at e a n d c a n f ail t o e n h a n c e r e s ult s, e s p e ci all y w h e n t h e i niti al r e s ult s fr o m $\mathcal { F }$ h a v e r e a c h e d t h eir u p p er b o u n d. I n c o ntr a st, o ur s c or e r a n g e m a p i s l e ar n e d o nli n e b a s e d o n t h e i n p ut i m a g e, r e st or ati o nr el at e d pri or s, a n d e x pli cit s p ati al f e at ur e s t h at ar e l e ar n a bl e. T hi s fl e xi bilit y all o w s u s t o h a n dl e v ari o u s sit u ati o n s, r es ulti n g i n b ett er p erf or m a n c e a n d g e n er ali z ati o n ( a s d e m o nstr at e d i n t h e a bl ati o n st u d y).

A s s h o w n i n Fi g. , w e f urt h er i ntr o d u c e a li g ht w ei g ht c o mp o n e nt t h at utili z e s pr e-tr ai n e d pri or s t o cr e at e a n eff e cti v e att e nti o n m e c h a ni s m i n . O pti mi zi n g t h e f e at ur e att e nti o n i n i s cr u ci al f or eff e cti v el y i d e ntif yi n g h el pf ul f e at ur e s t o e n h a n c e t h e i niti al r e s ult s $\dot { \boldsymbol { I } } _ { c } .$ T hi s i n v ol v e s b ot h s p ati all e v el a n d c h a n n el-l e v el att e nti o n s. T h e hi d d e n r e st or ati o nr el at e d i nf or m ati o n i n c a n b e di s c o v er e d b y u si n g t o i m pr o v e t h e r e st or ati o n f e at ur e s i n c o n diti o n e d o n t h e m.

We b e gi n b y f or m ul ati n g t h e att e nti o n c o m p ut ati o n at t h e c h a n n el l e v el. We i ntr o d u c e a m a p pi n g f u n cti o n $\tau _ { c }$ t o tr a n sf or m $g$ i nt o t h e att e nti o n- pr e di cti o n s p a c e, a n d utili z e t h e c h a n n el att e nti o n c o m p ut ati o n m o d ul e $\mathcal { R } _ { c } .$ . T h e f or m ul ati o n of t h e c h a n n el att e nti o n i s

$$
M _ { c } = \mathcal { R } _ { c } ( \mathcal { O } ( \hat { f } ) \oplus \mathcal { T } _ { c } ( g ) ) , \ \hat { f } _ { c } = \hat { f } \times M _ { c } ,\tag{4}
$$

w h er e i s t h e p o oli n g o p er ati o n, a n d $\mathcal { M } _ { c } \in \mathbb { R } ^ { c }$

A s f or t h e s p ati al- att e nti o n c o m p ut ati o n, w e utili z e t h e 1 D pr e-tr ai n e d pri or t o pr e di ct l o c ati o n- wi s e att e nti o n b a s e d o n t h e f e at ur e di stri b uti o n of e a c h l o c ati o n i n ${ \hat { f } } .$ Si mpl y u si n g t h e s p ati al l o c ati o n i nf or m ati o n, a s s h o w n i n E q. r e s ult s i n e a c h pi x el’s f e at ur e c o n si d eri n g a si mil ar c o n diti o n f or n ei g h b ori n g f e at ur e s, li miti n g t h e eli mi n ati o n of s p ati al artif a ct s. T h er ef or e, w e pr o p o s e a n alt er n ati v e str ate g y b y pr e di cti n g t h e n e ur al o p er ati o n p ar a m et er s f or e a c h l o c ati o n, o pti mi zi n g t h e s p ati al att e nti o n b a s e d o n t h e v ar yi n g l o c ati o n- wi s e f e at ur e di stri b uti o n. We d e n ot e t h e s p ati al att e nti o n c o m p ut ati o n m o d ul e a s $\mathcal { R } _ { p }$ , a n d fir st f or m ul at e t h e l o c ati o n- wi s e c o n v ol uti o n m a p, a s

$$
\begin{array} { r } { \mathcal { C } _ { p } = \mathcal { R } _ { p } ( \hat { f } , \mathcal { T } _ { c } ( g ) , S _ { c } ) , } \end{array}\tag{5}
$$

w h er e t h e o bt ai n e d c o n v ol uti o n m a p $ { \mathcal { C } } _ { p } \in$ $\mathbb { R } ^ { h \times w \times ( k _ { h } \times k _ { w } \times c ) }$ $k _ { h }$ a n d $k _ { w }$ ar e t h e c o n v ol uti o n k ern el si z e, a n d $ { \mathcal { S } } _ { c }$ i s a n ot h er l e ar n a bl e p o siti o n e m b e d di n g h er e. T h e o bt ai n e d c o n v ol uti o n m a p s c a n b e utili z e d t o o pti mi z e t h e f e at ur e, a n d s p ati al att e nti o n c a n b e o bt ai n e d a s

$$
\hat { M } _ { s } = \hat { f } * \mathcal { C } _ { p } , M _ { s } = \mathcal { R } _ { o } ( \hat { M } _ { s } ) ,\tag{6}
$$

<table><tr><td rowspan="2">Datasets Method</td><td rowspan="2"></td><td rowspan="2">Original PSNR SSIM</td><td colspan="2">+Ours-c PSNR</td><td colspan="2">+Ours-b</td><td colspan="2">+Ours-s</td><td colspan="2">+Ours-r</td><td colspan="2">+Ours-f</td></tr><tr><td></td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td rowspan="3">LOL</td><td>UHD</td><td>19.87 0.706</td><td>22.91 (+3.04)</td><td>0.767 (+6.1)</td><td></td><td>21.83 (+1.96)</td><td>0.732 (+2.6)</td><td>22.35 (+2.48) 0.758 (+5.2)</td><td>21.71 (+1.84)</td><td>0.737 (+3.1)</td><td>22.74 (+2.87)</td><td></td><td>0.764 (+5.8)</td></tr><tr><td>URetinex</td><td>21.16</td><td>0.840</td><td>24.70 (+3.54)</td><td>0.878 (+3.8)</td><td>23.57 (+2.41)</td><td>0.869 (+2.9)</td><td>24.23 (+3.07)</td><td>0.866 (+2.6)</td><td>23.99 (+2.83)</td><td>0.862 (+2.2)</td><td>24.56 (+3.40)</td><td>0.870 (+3.0)</td></tr><tr><td>SNR</td><td>21.48</td><td>0.849</td><td>25.50 (+4.02)</td><td>0.892 (+4.3)</td><td>25.61 (+4.13)</td><td>0.891 (+4.2)</td><td>25.19 (+3.71)</td><td>0.874 (+2.5)</td><td>25.24 (+3.76)</td><td>0.887 (+3.8)</td><td>24.90 (+3.42)</td><td>0.888 (+3.9)</td></tr><tr><td rowspan="3">SID</td><td>UHD</td><td>20.46</td><td>0.614</td><td>20.99 (+0.53)</td><td>0.616 (+0.2)</td><td>21.06 (+0.60)</td><td>0.619 (+0.5)</td><td>22.34 (+1.88)</td><td>0.625 (+1.1)</td><td>21.11 (+0.65)</td><td>0.618 (+0.4)</td><td>21.08 (+0.62)</td><td>0.619 (+0.5)</td></tr><tr><td>URetinex</td><td>21.56</td><td>0.619</td><td>22.34 (+0.78)</td><td>0.623 (+0.4)</td><td>22.02 (+0.46)</td><td>0.621 (+0.2)</td><td>22.21 (+0.65)</td><td>0.623 (+0.4)</td><td>22.17 (+0.61)</td><td>0.625 (+0.6)</td><td>22.40 (+0.84)</td><td>0.626 (+0.7)</td></tr><tr><td>SNR</td><td>22.87</td><td>0.625</td><td>23.34 (+0.47)</td><td>0.630 (+0.5)</td><td>23.15 (+0.28)</td><td>0.627 (+0.2)</td><td>23.08 (+0.21)</td><td>0.631 (+0.6)</td><td>23.06 (+0.19)</td><td>0.632 (+0.7)</td><td>23.17 (+0.30)</td><td>0.636 (+1.1)</td></tr></table>

Ta bl e 1. C o m p ari s o n s o n L O L-r e al a n d SI D d at a s et. $- c , - b , - s ,$ , a n d r ef er t o u si n g C LI P, B LI P 2, St a bl e Diff u si o n, a n d r e st or ati o n m o d el s tr ai n e d o n S D S D, r e s p e cti v el y. d e n ot e s a p pl yi n g r e fi n e m e nt o n t h e f e at ur e s of . ( +) i n di c at e s i m pr o v e m e nt s f or P S N R a n d $\mathbf { S S I M } _ { ( \mathbf { x } 1 0 0 ) }$

<table><tr><td>Methods</td><td>SNR</td><td>+SKF</td><td>+SMG</td><td>+SMG(dep)</td><td>+Ours-c</td></tr><tr><td>PSNR</td><td>21.48</td><td>23.05</td><td>24.84</td><td>24.12</td><td>25.50</td></tr><tr><td>SSIM</td><td>0.849</td><td>0.853</td><td>0.880</td><td>0.851</td><td>0.892</td></tr><tr><td>Methods</td><td>URetinex</td><td>+SKF</td><td>+SMG</td><td>+SMG(dep)</td><td>+Ours-c</td></tr><tr><td>PSNR</td><td>21.16</td><td>23.51</td><td>23.74</td><td>23.25</td><td>24.70</td></tr><tr><td>SSIM</td><td>0.840</td><td>0.856</td><td>0.852</td><td>0.849</td><td>0.878</td></tr><tr><td>+Params</td><td>0</td><td>2.15M</td><td>16.76M</td><td>16.76M</td><td>0.67M</td></tr></table>

Ta bl e 2. Q u a ntit ati v e c o m p ari s o n o n t h e L O L-r e al d at a s et. + P ar a m s m e a n s t h e a d diti o n al p ar a m et er n u m b er c o m p ar e d wit h ori gi n al

w h er e i s t h e c o n v ol uti o n o p er ati o n f or e a c h l o c ati o n, a n d $\scriptstyle { \mathcal { R } } _ { o }$ i s a n ot h er l e ar n a bl e o p er ati o n w hi c h m a p p s t h e f e at ur e c h a n n el t o 1, eli mi n ati n g t h e i n fl u e n c e fr o m t h e c h a n n ell e v el d e p e n d e n c y. F urt h er, t h e f e at ur e aft er s p ati al att e nti o n c a n b e d e s cri b e d a s $\hat { f } _ { s } = \hat { f } \times M _ { s }$

T h e f e at ur e s aft er s p ati al a n d c h a n n el att e nti o n s c a n b e m er g e d vi a a f u si o n m o d ul e a s

$$
\bar { f } = \mathcal { R } _ { f } ( \hat { f } _ { c } \oplus \hat { f } _ { s } ) ,\tag{7}
$$

w h er e $\mathcal { R } _ { f }$ d e n ot e s t h e f u si n m o d ul e. T h e o bt ai n e d f e at ur e $\bar { f }$ c a n b e pr o c e s s e d vi a a d e c o d er $\textstyle { \mathcal { R } } _ { d }$ t o o bt ai n t h e r e si d u al r e fi n e m e nt $I _ { r }$ a n d t h e m a s k $I _ { m }$ a s i n di c at e d i n E q.

## 3. 4. L o s s F u n cti o n

O ur d e si g n e d c a n b e j oi ntl y tr ai n e d wit h t h e m o d el ${ \mathcal { F } } .$ S u p p o s e t h e p air e d gr o u n d tr ut h f or t h e i n p ut i m a g e $I _ { d }$ i s $\mathcal { T } _ { c } ,$ a n d t h e l o s s f u n cti o n f or t h e m o d el $\hat { \mathcal F }$ i s d e n ot e d a s $\mathcal { L } _ { g } ( \hat { I } _ { c } , \mathcal { T } _ { c } )$ (i s u s u all y t h e r e c o n str u cti o n l o s s i n t h e pi x el l e v el or p er c e pt u al l o s s, a n d c a n al s o b e t h e u n s u p er vi s e d l o s s), t h e n t h e l o s s f u n cti o n f or t h e r e fi n e m e nt m o d ul e c a n b e writt e n a s $\mathcal { L } _ { g } ( \bar { I } _ { c } , \mathcal { T } _ { c } )$ . I n s u m m ar y, t h e o v er all l o s s i s

$$
\mathcal { L } _ { g } ( \hat { I } _ { c } , \mathcal { T } _ { c } ) + \lambda _ { 1 } \mathcal { L } _ { g } ( \bar { I } _ { c } , \mathcal { T } _ { c } ) ,\tag{8}
$$

w h er e $\lambda _ { 1 }$ i s t h e l o s s w ei g ht a n d r e m ai n s r o b u st a cr o s s v ari o u s t a s k s a n d n et w or k s (i n o ur e x p eri m e nt s, i s al w a y s s et a s 1).

## 4. E x p e ri m e nt s

We fir st i ntr o d u c e t a s k s a n d d at a s et s u s e d i n e x p eri m e nt s, f oll o w e d b y a d et ail e d a n al y si s of o ur m et h d u si n g l o w-li g ht i m a g e e n h a n c e m e nt a s a n e x a m pl e. We al s o d e m o n str at e t h e eff e cti v e n e s s of o ur m et h o d o n ot h er t a s k s.

![](images/84ec7a5b5b2b01f74c6bd0ef562998a57ad11f57e73001b6eccd202e49587814.jpg)  
Fi g ur e 4. C o m p ari s o n s o n L O L-r e al (t o p) a n d SI D ( b ott o m). R es ult s wit h “ O ur s ” h a v e l e s s n oi s e a n d cl e ar er vi si bilit y.

## 4. 1. T a s k s a n d D at a s et s

F or l o w-li g ht e n h a n c e m e nt, w e u s e t h e SI D [ ] a n d L O Lr e al [4 9 ] d at a s et s. F or d er ai ni n g, w e u s e t h e R ai n 1 3 K [5 2 d at a s et f or tr ai ni n g a n d t e st o n R ai n 1 0 0 H [ 4 8 ], R ai n 1 0 0 L [ 4 8 ], Te st 1 0 0 [5 5 ], Te st 1 2 0 0 [5 4 ], a n d Te st 2 8 0 0 [ ] d at a s et s. F or g a u s si a n d e n oi si n g, w e u s e t w o s etti n g s: s y nt h eti c n oi s e o n S et 1 2 [5 6 ], B S D 6 8 [2 3 ], C B S D 6 8 [ 2 3 ], K o d a k [ ], M c M a st er [5 8 ], a n d Urb a n 1 0 0 [ 1 0 ]; a n d r e al- w orl d d e n oi si n g o n SI D D [ ]. F or si n gl e-i m a g e m oti o n d e bl urri n g, w e u s e t h e G o-Pr o [ 2 4 ] d at a s et f or tr ai ni n g a n d e v al u at e o n s y nt h eti c d at a s et s ( G o Pr o [ 2 4 ], HI D E [3 0 ]) a n d r e al- w orl d d at a s et s ( R e al Bl ur- R [2 8 ], R e al Bl ur- J [2 8 ]). F or d ef o c u s d e bl urri n g, w e u s e t h e D P D D [ ] tr ai ni n g d at a a n d t e st o n t h e E B D B [ 1 3 ] a n d J N B [3 1 ] d at a s et s.

## 4. 2. L o w-li g ht I m a g e E n h a n c e m e nt

C o m p a ri s o n. We c h o o s e c urr e nt S O T A l o w-li g ht i m a g e e n h a n c e m e nt m et h o d s a s t h e b a s eli n e s ( U H D [ 3 5 ], U R eti n e x [4 0 ], S N R [4 2 ]), a n d a p pl y o ur r e fi n e m e nt m o d ul e f or t h e s e b a s eli n e s t o s e e if t h eir p erf or m a n c e c a n b e i m pr o v e d. T h e pri or s ar e c h o s e n fr o m t h e C LI P [ 2 7 ], B LI P 2 [1 7 ], St a bl e Diff u si o n [ 2 9 ], a n d pr e-tr ai n e d r e st or ati o n m o d el s (tr ai n e d o n a n ot h er d at a s et, a s S D S D [3 4 4 5 ]). We d e n ot e t h e s e r e s ult s a s , a n d , r e s p e cti v el y. I n Tabl e , w e o b s er v e t h at c o m bi ni n g t h e s e pri or s wit h o ur r efi n e m e nt m o d ul e si g ni fi c a ntl y i m pr o v e s t h e p erf or m a n c e of t h e b a s eli n e s. A d diti o n all y, Fi g. pr o vi d e s vi s u al c o m p ari s o n s.

M or e o v er, w e c o n d u ct e d a n e x p eri m e nt b y a d di n g t h e r e fi n e m e nt m o d ul e t o t h e i nt er m e di at e l a y er of ${ \mathcal F } ,$ r e fi ning features of the target model. The refinement module is added to the deepest feature layer for efficiency, producing the residual feature map and the mask information for refinement. These results are denoted as −f. The improvement achieved by this operation is also evident as displayed in Table 1.

<table><tr><td rowspan="2"></td><td colspan="2">LOL-real</td><td colspan="2">SID</td></tr><tr><td>URetinex PSNR SSIM</td><td>SNR PSNR SSIM</td><td>URetinex PSNR SSIM</td><td>SNR PSNR SSIM</td></tr><tr><td>w/o SP, with CA and SA</td><td>23.45 0.868</td><td>24.25 0.886</td><td>21.98 0.619</td><td>23.02 0.620</td></tr><tr><td>with SP, w/o CA, with SA</td><td>22.100.856</td><td>24.050.875</td><td>22.050.623</td><td>22.930.624</td></tr><tr><td>with SP and CA, w/o SA</td><td>23.760.850</td><td>23.860.879</td><td>21.920.620</td><td>23.070.621</td></tr><tr><td>Large R w/o SP/CA/SA</td><td>22.740.857</td><td>24.51 0.881</td><td>22.060.621</td><td>23.040.627</td></tr><tr><td>w/o Position Embedding S</td><td>23.660.843</td><td>24.130.874</td><td>22.130.620</td><td>22.920.622</td></tr><tr><td>SNR Value as Mask</td><td>22.660.855</td><td>24.77 0.887</td><td>22.010.617</td><td>22.940.627</td></tr><tr><td>Use 1D Priors via Con.</td><td>23.010.853</td><td>23.83 0.878</td><td>22.070.622</td><td>22.93 0.628</td></tr><tr><td>Use 2D Priors via Con.</td><td>22.680.862</td><td>24.11 0.880</td><td>22.080.618</td><td>23.060.625</td></tr><tr><td>Full Setting</td><td>24.700.878</td><td>25.50 0.892</td><td>22.34 0.623</td><td>23.34 0.630</td></tr></table>

Table 3. Ablation study results. We adopt CLIP as the pre-trained model. “SP” denote PTG-SVE, “CA” and “SA” denote spatialand channel attentions in PTG-CSA. Con. means Concatenation.

<table><tr><td>Datasets</td><td colspan="3">LOL-real</td><td colspan="3">SID</td></tr><tr><td>Methods</td><td>ZeroDCE</td><td>RUAS</td><td>SCI</td><td>ZeroDCE</td><td>RUAS</td><td>SCI</td></tr><tr><td>PSNR</td><td>18.06</td><td>18.37</td><td>20.28</td><td>18.08</td><td>18.44</td><td>19.09</td></tr><tr><td>SSIM</td><td>0.580</td><td>0.723</td><td>0.752</td><td>0.576</td><td>0.581</td><td>0.585</td></tr><tr><td>Methods</td><td>+Ours-c</td><td>+Ours-c</td><td>+Ours-c</td><td>+Ours-c</td><td>+Ours-c</td><td>+Ours-c</td></tr><tr><td>PSNR</td><td>18.79</td><td>19.53</td><td>21.62</td><td>18.65</td><td>18.93</td><td>19.61</td></tr><tr><td>SSIM</td><td>0.614</td><td>0.747</td><td>0.781</td><td>0.593</td><td>0.590</td><td>0.598</td></tr></table>

Table 4. Quantitative comparison on the LOL-real and SID dataset for unsupervised methods. We adopt CLIP as the pre-trained model here.

<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td></td><td colspan="2">Test100</td><td colspan="2">Rain100H</td><td colspan="2">Rain100L</td></tr><tr><td>SPAIR</td><td>30.35</td><td>0.909</td><td>30.95</td><td>0.892</td><td>36.93</td><td>0.969</td></tr><tr><td>SPAIR+Ours-c</td><td>30.62</td><td>0.917</td><td>31.20</td><td>0.901</td><td>37.26</td><td>0.973</td></tr><tr><td>Restormer</td><td>32.00</td><td>0.923</td><td>31.46</td><td>0.904</td><td>38.99</td><td>0.978</td></tr><tr><td>Restormer+Ours-c</td><td>32.30</td><td>0.934</td><td>31.77</td><td>0.913</td><td>39.27</td><td>0.985</td></tr><tr><td></td><td colspan="2">Test2800</td><td colspan="2">Test1200</td><td colspan="2">Average</td></tr><tr><td>SPAIR</td><td>33.34</td><td>0.936</td><td>33.04</td><td>0.922</td><td>32.91</td><td>0.926</td></tr><tr><td>SPAIR+Ours-c</td><td>33.58</td><td>0.942</td><td>33.35</td><td>0.924</td><td>33.16</td><td>0.932</td></tr><tr><td>Restormer</td><td>34.18</td><td>0.944</td><td>33.19</td><td>0.926</td><td>33.96</td><td>0.935</td></tr><tr><td>Restormer+Ours-c</td><td>34.47</td><td>0.951</td><td>33.48</td><td>0.929</td><td>34.24</td><td>0.943</td></tr></table>

Table 5. Image deraining results.

Comparison with Other Priors. Some works, such as SKF [41] and SMG [46], utilize additional information like semantic maps, edge maps, and depth maps to enhance lowlight image enhancement results. However, these methods require supervision with paired multi-modal information, whereas our method does not. Additionally, as shown in Table 2, our approach achieves better performance improvement for a given target model. Notably, the improvements achieved by other methods are based on large additional parameters, while our approach only uses a lightweight refinement module < 1M.

Ablation Study: Ablation of Different Components. We first set experiments by deleting different components from our framework, including PTG-SVE (abbreviated as “SP”), and spatial-channel attentions with priors that are abbreviated as “CA” and “SA”, respectively. As shown in Table 3, deleting any component will lead to a performance drop.

![](images/38af599e68ab658c2a112771a383c6ec54c716029ba7b32f1998829afa3866ac.jpg)

![](images/b39f294d6960b30ec3ad1f7e79b9fb3f34425e05e79a0caab391167032c9f820.jpg)

Figure 5. Visual comparison on Rain100H showing the effects of our strategy.
<table><tr><td>Method</td><td>GoPro PSNR SSIM</td><td>HIDE PSNR SSIM</td><td>RealBlur-R PSNR SSIM</td><td>PSNR</td><td>RealBlur-J SSIM</td></tr><tr><td>MPRNet</td><td>32.66 0.959</td><td>30.96 0.939</td><td>35.99</td><td>0.952</td><td>28.70 0.873</td></tr><tr><td>MPRNet+Ours-c</td><td>32.87 0.964</td><td>31.19 0.943</td><td>36.25</td><td>0.960</td><td>28.98 0.881</td></tr><tr><td>Restormer</td><td>32.92 0.961</td><td>31.22 0.942</td><td>36.19</td><td>0.957</td><td>28.96 0.879</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Restormer+Ours-c</td><td>33.18 0.966</td><td>31.51 0.950</td><td>36.47</td><td>0.962</td><td>29.21 0.883</td></tr></table>

Table 6. Single-image motion deblurring results.

We conduct experiments without SP, CA, or SA to analyze whether additional parameters or priors take a prominent role. The short-range and long-range results are fused via a simple sum, and the spatial-channel attention is conducted using only the features themselves. Additionally, we increase the feature channel number fourfold to add more parameters. The results, denoted as “Large R w/o SP/CA/SA” in Table 3, are still lower than our full setting, indicating the effectiveness of our proposed approach over simply increasing parameters.

In addition, we perform an experiment by removing the learnable position embeddings $S _ { m }$ and $\boldsymbol { S } _ { c } ,$ , denoted as “w/o Position Embedding for Priors” in Table 3. This comparison highlights the importance of using spatial-aware representations for the pre-trained features.

Ablation Study: SNR Value as Mask. In comparison to previous methods that directly use the SNR value as the mask to fuse the short- and long-range results, our approach utilizes pre-trained priors to automatically discover restoration-related information and formulate the fusion mask adaptively. In this ablation study, we demonstrate that our strategy outperforms the direct SNR-based approach, as shown in Table 3.

Ablation Study: Alternatives of Using Priors. In this study, we demonstrate the difficulty of directly aligning priors to the restoration features. We conduct an experiment where the priors are concatenated with the features in the refinement module to implement different components. However, the improvement obtained with this direct approach is not as significant as our proposed method, as shown in Table 3. This is because the different features are heterogeneous with the restoration features, even when the priors are adopted as 2D feature maps. This study highlights the importance of our novel strategy of employing these priors.

![](images/64c3fa51689689f7e111faace212f627d1288925bcc33a9ba3bf77df34c9209c.jpg)  
Input

![](images/b2d203b799497f769df63404d774a92159ab1f4e5ec156d5fd16c3fc4c623d8b.jpg)

![](images/aab2fc62b916f098a87829a7eb37e0784902426ecce4359a205c916196237e0f.jpg)  
Restormer

Ground-truth  
![](images/b13835dccbd5805f386f2479c7cedd5201a96ca0abd4a1c3e2eb5e993e67a70d.jpg)  
Restormer+Ours

Figure 6. Visual comparison on HIDE.  
![](images/f132101549d730161e5b1d0d4631242d7e2cb253c1e991c4e2dc5379e88430b9.jpg)  
Input

![](images/20d9be4728d94bf63f2dd5d36e9ee63966e8183e7195c8c19fcdf034bfbf2a08.jpg)

![](images/18eb8f5c05efe9b6abd74c5f3080180fa787e1b3d74f6b2622c152afd6b289f3.jpg)  
GRL

Ground-truth  
![](images/e5303ee60adbf6c11d34790db1b27885c97e9a40b4b8a073a388db9b325db2b3.jpg)  
GRL+Ours  
Figure 7. Visual comparison on single-image defocus deblurring.

R for Unsupervised Approach. Different from existing refinement methods that need supervision for learning the additional features (e.g., SKF needs the semantic ground truth of the normal-light data, SMG needs the depth and edge information of the normal-light data), our approach does not require the feature of the normal-light data during both training and inference. We only need the feature that is extracted from $I _ { d }$ with the pre-trained model G during the training. Also, the loss function for training the refinement module can be set the same as that of the target model. Thus, the unsupervised training of the target model can also be adopted in our framework. As shown in Table 4, our method can successfully improve the performance of various unsupervised low-light image enhancement methods with different unsupervised loss terms, including En-GAN [11], ZeroDCE [9], RUAS [21], and SCI [22].

## 4.3. Other Restoration Tasks

In this section, we conduct experiments using CLIP as the pre-trained model (−c). CLIP is chosen for its efficiency and convenience compared to other pre-trained models.

<table><tr><td rowspan="3">Method</td><td>Indoor Scenes</td><td>Outdoor Scenes</td><td></td><td>Combined</td></tr><tr><td>PSNR SSIM LPIPS</td><td>PSNR SSIM LPIPS</td><td></td><td>PSNR SSIM LPIPS</td></tr><tr><td></td><td></td><td></td><td>0.217</td></tr><tr><td>IFANS</td><td>28.11 0.861 0.179</td><td>22.76 0.720</td><td>0.254</td><td>25.37 0.789</td></tr><tr><td>IFANS+Ours-c</td><td>28.32 0.870 0.171</td><td>23.08 0.727</td><td>0.248 25.72</td><td>0.795 0.213</td></tr><tr><td>RestormerS</td><td>28.87 0.882 0.145</td><td>23.24 0.743</td><td>0.209</td><td>25.980.811 0.178</td></tr><tr><td>Restormer§+Ours-c</td><td>29.17 0.890 0.141</td><td>23.43 0.749</td><td>0.206 26.13</td><td>0.816 0.165</td></tr><tr><td>GRLS-B</td><td>29.06 0.886 0.139</td><td>23.45 0.761</td><td>0.196 26.18</td><td>0.822 0.168</td></tr><tr><td>GRLS-B +Ours-c</td><td>29.30 0.894 0.133</td><td>23.67 0.768</td><td>0.189</td><td>26.45 0.828 0.161</td></tr><tr><td>IFAND</td><td>28.66 0.868 0.172</td><td>23.46 0.743</td><td>0.240</td><td>25.99 0.804 0.207</td></tr><tr><td>IFAND+Ours-c</td><td>28.94 0.875 0.167</td><td>23.70 0.748</td><td>0.235 26.20</td><td>0.811 0.203</td></tr><tr><td>RestormerD</td><td>29.48 0.895 0.134</td><td>23.97 0.773</td><td>0.175 26.66</td><td>0.833 0.155</td></tr><tr><td>RestormerD+Ours-c</td><td>29.79 0.902 0.131</td><td>24.23 0.778</td><td>0.155 26.89</td><td>0.840 0.153</td></tr><tr><td>GRLD-B</td><td>29.83 0.903 0.114</td><td>24.39 0.795</td><td>0.150 27.04</td><td>0.847 0.133</td></tr><tr><td>GRLD-B+Ours-c</td><td>29.96 0.911 0.110</td><td>24.62 0.803</td><td>0.145 27.27</td><td>0.855 0.128</td></tr></table>

Table 7. Defocus deblurring comparisons on the DPDD testset (containing 37 indoor and 39 outdoor scenes). S: single-image defocus deblurring. D: dual-pixel defocus deblurring.

<table><tr><td rowspan="2">Method</td><td colspan="3">Set12</td><td colspan="3">BSD68</td><td colspan="3">Urban100</td></tr><tr><td>σ=15</td><td>σ=25</td><td>σ=50</td><td>σ=15</td><td>σ=25</td><td>σ=50 σ=15</td><td>σ=25</td><td></td><td>σ=50</td></tr><tr><td>DRUNet</td><td>33.25</td><td>30.94</td><td>27.90</td><td>31.91</td><td>29.48</td><td>26.59</td><td>33.44</td><td>31.11</td><td>27.96</td></tr><tr><td>DRUNet+Ours-c</td><td>33.51</td><td>31.18</td><td>28.27</td><td>32.20</td><td>29.73</td><td>26.84</td><td>33.65</td><td>31.34</td><td>28.16</td></tr><tr><td>Restormer</td><td>33.35</td><td>31.04</td><td>28.01</td><td>31.95</td><td>29.51</td><td>26.62</td><td>33.67</td><td>31.39</td><td>28.33</td></tr><tr><td>Restormer+Ours-c</td><td>33.57</td><td>31.28</td><td>28.36</td><td>32.11</td><td>29.78</td><td>26.91</td><td>33.96</td><td>31.67</td><td>28.58</td></tr><tr><td>Restormer</td><td>33.42</td><td>31.08</td><td>28.00</td><td>31.96</td><td>29.52</td><td>26.62</td><td>33.79</td><td>31.46</td><td>28.29</td></tr><tr><td>Restormer+Ours-c</td><td>33.70</td><td>31.29</td><td>28.35</td><td>32.24</td><td>29.81</td><td>26.86</td><td>33.97</td><td>31.73</td><td>28.58</td></tr><tr><td>GRL-B</td><td>33.47</td><td>31.12</td><td>28.03</td><td>32.00</td><td>29.54</td><td>26.60</td><td>34.09</td><td>31.80</td><td>28.59</td></tr><tr><td>GRL-B+Ours-c</td><td>33.74</td><td>31.30</td><td>28.37</td><td>32.29</td><td>29.76</td><td>26.91</td><td>34.22</td><td>31.95</td><td>28.74</td></tr></table>

Table 8. Gaussian grayscale image denoising comparisons. Top super rows: learning a single model to handle various noise levels. Bottom super rows: training a separate model for each noise level.

Deraining. For deraining tasks, we use SOTA methods such as SPAIR [26] and Restormer [52] as baselines. We compute PSNR/SSIM values using the Y channel in the YCbCr color space, similar to existing methods. Table 5 demonstrates that our approach improves the performance of these existing methods and consistently achieves significant performance gains across all five datasets. The qualitative comparison results are shown in Fig. 5.

Motion Deblurring. We analyze our approach for deblurring tasks on synthetic datasets (GoPro, HIDE) and realworld datasets (RealBlur-R, RealBlur-J). The baselines include MPRNet [51] and Restormer [52]. Table 6 demonstrates that our approach improves the performance of all these methods on all four benchmark datasets. Although the enhanced network is trained only on the GoPro dataset, it shows more robust generalization to other datasets. Qualitative comparisons are shown in Fig. 6, further supporting our claim.

Defocus Deblurring. Table 7 presents the image fidelity scores of SOTA approaches on the DPDD dataset [3], including IFAN [15], Restormer [52], and GRL [19]. Our refinement module achieves significant performance improvement for these SOTA schemes in both single-image and dual-pixel defocus deblurring settings across all scene categories. The qualitative results are depicted in Fig. 7.

Gaussian Denoising. We conduct denoising experiments on synthetic benchmark datasets with additive white Gaussian noise. We choose DRUNet [57], Restormer [52], and

<table><tr><td rowspan="2">Method</td><td>CBSD68</td><td>Kodak24</td><td></td><td>McMaster</td><td></td><td>Urban100</td></tr><tr><td>σ=15 σ=25 σ=50</td><td>σ=15 σ=25</td><td>σ=50</td><td>σ=15 σ=25</td><td>σ=50</td><td>σ=15 σ=25 σ=50</td></tr><tr><td>DRUNet</td><td>34.30 31.69 28.51</td><td>35.31 32.89</td><td>29.86</td><td>35.40 33.14 30.08</td><td></td><td>34.8132.60 29.61</td></tr><tr><td>+Ours-c</td><td>34.54 31.97 28.76</td><td>35.58 33.15</td><td>29.97</td><td>35.71 33.50 30.25</td><td></td><td>35.10 32.82 29.83</td></tr><tr><td>Restormer</td><td>34.39 31.78 28.59</td><td>35.44 33.02</td><td>30.00</td><td>35.55 33.31</td><td>30.29</td><td>35.0632.91 30.02</td></tr><tr><td>+Ours-c</td><td>34.63 32.04 28.88</td><td>35.65 33.26</td><td>30.15</td><td>35.86 33.64 30.63</td><td></td><td>35.2633.22 30.21</td></tr><tr><td>Restormer</td><td>34.40 31.79 28.60</td><td>35.47</td><td>33.04 30.01</td><td>35.61</td><td>33.34 30.30</td><td>35.1332.96 30.02</td></tr><tr><td>+Ours-c</td><td>34.76 32.05 28.94</td><td>35.72 33.27</td><td>30.21</td><td>35.80</td><td>33.63 30.55</td><td>35.3233.14 30.27</td></tr><tr><td>GRL-B</td><td>34.45 31.82 28.62</td><td>35.43 33.02</td><td>29.93</td><td>35.73 33.4630.36</td><td></td><td>35.54 33.35 30.46</td></tr><tr><td>+Ours-c</td><td>34.73 32.07 28.90</td><td>35.71 33.24</td><td>30.18</td><td>35.9633.7530.62</td><td></td><td>35.70 33.57 30.64</td></tr></table>

Table 9. Gaussian color image denoising. Equivalent notation meanings (top and bottom rows) as those in Table 8.

<table><tr><td>Dataset</td><td>Method</td><td></td><td>+ Ours-c</td><td></td><td>+ Ours-c</td><td>MPRNet MPRNet Uformer Uformer Restormer Restormer</td><td>+ Ours-c</td></tr><tr><td>SIDD</td><td>PSNR ↑</td><td>39.71</td><td>39.93</td><td>39.77</td><td>39.94</td><td>40.02</td><td>40.22</td></tr><tr><td></td><td>SSIM↑</td><td>0.958</td><td>0.961</td><td>0.959</td><td>0.962</td><td>0.960</td><td>0.965</td></tr></table>

Table 10. Real image denoising on the SIDD dataset.

GRL [19] as baselines, which are SOTA approaches in denoising. Tables 8 and 9 present PSNR scores of different approaches on grayscale and color image denoising, respectively, for noise levels of 15, 25, and 50. We evaluate two experimental settings: (1) learning a single model to handle various noise levels and (2) learning separate models for each noise level. Our method achieves significant performance enhancement for all these methods under both experimental settings on different datasets and noise levels. The visual results are shown in Fig. 8, showing the effectiveness of our strategy.

Real Denoising. We also conduct denoising experiments on the real-world SIDD dataset, with MPRNet [51], Uformer [37], and Restormer [52] as baselines. Table 10 demonstrates that our refinement method improves both PSNR and SSIM metrics. Notably, on the SIDD dataset, our refinement enables the SOTA approach Restormer to achieve a PSNR surpassing 40.2 dB. The visual comparison is shown in Fig. 8.

User Study. Furthermore, we conducted a large-scale user study with an A/B test strategy involving 80 participants. Each participant is asked to simultaneously see two restored results, i.e., baseline and baseline+ours, and gauge which one is better. As shown in Fig. 9, the results combined with our strategy are more preferred by the participants.

## 5. Conclusion

In this work, we explore the utilization of features from a pre-trained model to enhance the performance of a restoration model. By unifying the shapes of the pre-trained features, we introduce a novel refinement module PTG-RM that employs PTG-SVE and PTG-CSA mechanisms. Unlike existing strategies, we focus on formulating optimal operation ranges and attention strategies guided by the pretrained features. The extensive experiments conducted on various tasks, datasets, and networks demonstrate the effectiveness and generalization ability of our approach. We believe that our proposed principle of discovering hidden useful information in pre-trained models can be applicable to other domains as well.

![](images/ffcfc0cab688349703aa58399b05b01b8bc5a4465275f48f61ce0b9173948a9b.jpg)  
Figure 8. Visual comparisons on Kodak (top) and SIDD (bottom).

![](images/2dc3961e86927f3d5323dd5d21645f86a1948ab5ce201ff77889e32ddbe82908.jpg)  
Figure 9. The user study results show that our strategy can effectively improve the performance of restoration approaches in terms of human subjective evaluation.

Limitation and Future Work. While our proposed strategy has exhibited significant effects in enhancing the performance of diverse restoration networks across various architectures with its lightweight module, the extent of improvement appears to vary across different experiments. Some instances showcase noticeable enhancement, while others do not. Such differences correlate with the capacity of the target network and the difficulty/complexity of the target task. In future endeavors, we intend to delve into more effective approaches that specifically aid target restoration tasks. We aim to employ a tailored distillation framework to derive refined restoration feature priors, ultimately making significant strides beyond existing upper boundaries. We also aim to develop corresponding technical products.

Acknowledgements. This work is supported by the Natural Science Foundation of Zhejiang Pvovince, China, under No. LD24F020002. SK is partially supported by University of Macau (SRG2023-00044- FST).

## References

[1] Andreas Aakerberg, Anders S Johansen, Kamal Nasrollahi, and Thomas B Moeslund. Semantic segmentation guided real-world super-resolution. In WACV, 2022. 1

[2] Abdelrahman Abdelhamed, Stephen Lin, and Michael S Brown. A high-quality denoising dataset for smartphone cameras. In CVPR, 2018. 5

[3] Abdullah Abuolaim and Michael S Brown. Defocus deblurring using dual-pixel data. In ECCV, 2020. 5, 7

[4] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In CVPR, 2022. 3

[5] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In CVPR, 2018. 5

[6] Sepideh Esmaeilpour, Bing Liu, Eric Robertson, and Lei Shu. Zero-shot out-of-distribution detection based on the pre-trained model clip. In AAAI, 2022. 3

[7] Rich Franzen. Kodak lossless true color image suite. http: //r0k.us/graphics/kodak/, 1999. Online accessed 24 Oct 2021. 5

[8] Xueyang Fu, Jiabin Huang, Delu Zeng, Yue Huang, Xinghao Ding, and John Paisley. Removing rain from single images via a deep detail network. In CVPR, 2017. 5

[9] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In CVPR, 2020. 7

[10] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In CVPR, 2015. 5

[11] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. TIP, 2021. 7

[12] Shuangping Jin, Bingbing Yu, Minhao Jing, Yi Zhou, Jiajun Liang, and Renhe Ji. Darkvisionnet: Low-light imaging via rgb-nir fusion with deep inconsistency prior. In AAAI, 2022. 3

[13] Ali Karaali and Claudio Rosito Jung. Edge-based defocus blur estimation with adaptive scale selection. TIP, 2017. 5

[14] Shu Kong and Charless Fowlkes. Image reconstruction with predictive filter flow. arXiv preprint arXiv:1811.11482, 2018. 1

[15] Junyong Lee, Hyeongseok Son, Jaesung Rim, Sunghyun Cho, and Seungyong Lee. Iterative filter adaptive network for single image defocus deblurring. In CVPR, 2021. 7

[16] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 1, 3

[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint, 2023. 1, 3, 5

[18] Yi Li, Yi Chang, Changfeng Yu, and Luxin Yan. Close the loop: a unified bottom-up and top-down paradigm for join image deraining and segmentation. In AAAI, 2022. 3

[19] Yawei Li, Yuchen Fan, Xiaoyu Xiang, Denis Demandolx, Rakesh Ranjan, Radu Timofte, and Luc Van Gool. Efficient and explicit modelling of image hierarchies for image restoration. In CVPR, 2023. 1, 7, 8

[20] Ding Liu, Bihan Wen, Xianming Liu, Zhangyang Wang, and Thomas S Huang. When image denoising meets high-level vision tasks: A deep learning approach. In IJCAI, 2018. 3

[21] Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement. In CVPR, 2021. 7

[22] Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image enhancement. In CVPR, 2022. 7

[23] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In ICCV, 2001. 5

[24] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In CVPR, 2017. 5

[25] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In ICCV, 2021. 3

[26] Kuldeep Purohit, Maitreya Suin, AN Rajagopalan, and Vishnu Naresh Boddeti. Spatially-adaptive image restoration using distortion-guided networks. In ICCV, 2021. 7

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 1, 3, 5

[28] Jaesung Rim, Haeyun Lee, Jucheol Won, and Sunghyun Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In ECCV, 2020. 5

[29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In CVPR, 2022. 5

[30] Ziyi Shen, Wenguan Wang, Xiankai Lu, Jianbing Shen, Haibin Ling, Tingfa Xu, and Ling Shao. Human-aware motion deblurring. In ICCV, 2019. 5

[31] Jianping Shi, Li Xu, and Jiaya Jia. Just noticeable defocus blur detection and estimation. In CVPR, 2015. 5

[32] Renjie Wan, Boxin Shi, Wenhan Yang, Bihan Wen, Ling-Yu Duan, and Alex C Kot. Purifying low-light images via nearinfrared enlightened image. TMM, 2022. 3

[33] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 2

[34] Ruixing Wang, Xiaogang Xu, Chi-Wing Fu, Jiangbo Lu, Bei Yu, and Jiaya Jia. Seeing dynamic scene in the dark: A highquality video dataset with mechatronic alignment. In ICCV, 2021. 5

[35] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Tong Lu. Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In AAAI, 2023. 5

[36] Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Recovering realistic texture in image super-resolution by deep spatial feature transform. In CVPR, 2018. 1

[37] Zhendong Wang, Xiaodong Cun, Jianmin Bao, and Jianzhuang Liu. Uformer: A general u-shaped transformer for image restoration. In CVPR, 2022. 8

[38] Zejin Wang, Jiazheng Liu, Guoqing Li, and Hua Han. Blind2unblind: Self-supervised image denoising with visible blind spots. In CVPR, 2022. 1

[39] Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, and Tongliang Liu. Cris: Clip-driven referring image segmentation. In CVPR, 2022. 3

[40] Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In CVPR, 2022. 5

[41] Yuhui Wu, Chen Pan, Guoqing Wang, Yang Yang, Jiwei Wei, Chongyi Li, and Heng Tao Shen. Learning semantic-aware knowledge guidance for low-light image enhancement. In CVPR, 2023. 1, 2, 6

[42] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Snr-aware low-light image enhancement. In CVPR, 2022. 1, 2, 4, 5

[43] Xiaogang Xu, Yitong Yu, Nianjuan Jiang, Jiangbo Lu, Bei Yu, and Jiaya Jia. Pvdd: A practical video denoising dataset with real-world dynamic scenes. arXiv preprint, 2022. 1

[44] Xiaogang Xu, Hengshuang Zhao, Philip Torr, and Jiaya Jia. General adversarial defense against black-box attacks via pixel level and feature level distribution alignments. arXiv preprint, 2022. 1

[45] Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and Jiaya Jia. Deep parametric 3d filters for joint video denoising and illumination enhancement in video super resolution. In AAAI, 2023. 5

[46] Xiaogang Xu, Ruixing Wang, and Jiangbo Lu. Low-light image enhancement via structure modeling and guidance. In CVPR, 2023. 1, 2, 6

[47] Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning unified representation of language, image and point cloud for 3d understanding. In CVPR, 2023. 3

[48] Wenhan Yang, Robby T Tan, Jiashi Feng, Jiaying Liu, Zongming Guo, and Shuicheng Yan. Deep joint rain detection and removal from a single image. In CVPR, 2017. 5

[49] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep Retinex network for robust low-light image enhancement. TIP, 2021. 5

[50] Yan Yang, Liyuan Pan, Liu Liu, and Miaomiao Liu. K3dn: Disparity-aware kernel estimation for dual-pixel defocus deblurring. In CVPR, 2023. 1

[51] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In CVPR, 2021. 7, 8

[52] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 1, 5, 7, 8

[53] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In CVPR, 2022. 3

[54] He Zhang and Vishal M Patel. Density-aware single image de-raining using a multi-stream dense network. In CVPR, 2018. 5

[55] He Zhang, Vishwanath Sindagi, and Vishal M Patel. Image de-raining using a conditional generative adversarial network. TCSVT, 2019. 5

[56] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. TIP, 2017. 5

[57] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. TPAMI, 2021. 7

[58] Lei Zhang, Xiaolin Wu, Antoni Buades, and Xin Li. Color demosaicking by local directional interpolation and nonlocal adaptive thresholding. JEI, 2011. 5

[59] Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. Pointclip: Point cloud understanding by clip. In CVPR, 2022. 3

[60] Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, and Yifan Liu. Zegclip: Towards adapting clip for zero-shot semantic segmentation. In CVPR, 2023. 3