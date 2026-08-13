# Affine Equivariant Networks Based on Differential Invariants

Yikang Li<sup>1</sup>, Yeqing Qiu<sup>2,3</sup>, Yuxuan Chen<sup>1</sup>, Lingshen He<sup>1</sup>, Zhouchen Lin<sup>1,4,5</sup>\* <sup>1</sup>National Key Lab of General AI, School of Intelligence Science and Technology, Peking University <sup>2</sup>The Chinese University of Hong Kong, Shenzhen <sup>3</sup>Shenzhen Research Institute of Big Data <sup>4</sup>Institute for Artificial Intelligence, Peking University <sup>5</sup>Peng Cheng Laboratory

liyk18@pku.edu.cn, yeqingqiu@link.cuhk.edu.cn, edmondx.chen@gmail.com lingshenhe@pku.edu.cn, zlin@pku.edu.cn

## Abstract

Convolutional neural networks benefit from translation equivariance, achieving tremendous success. Equivariant networks further extend this property to other transformation groups. However, most existing methods require discretization or sampling of groups, leading to increased model sizes for larger groups, such as the affine group. In this paper, we build affine equivariant networks based on differential invariants from the viewpoint of symmetric PDEs, without discretizing or sampling the group. To address the division-by-zero issue arising from fractional differential invariants of the affine group, we construct a new kind of affine invariants by normalizing polynomial relative differential invariants to replace classical differential invariants. For further flexibility, we design an equivariant layer, which can be directly integrated into convolutional networks of various architectures. Moreover, our framework for the affine group is also applicable to its continuous subgroups. We implement equivariant networks for the scale group, the rotation-scale group, and the affine group. Numerical experiments demonstrate the outstanding performance of our framework across classification tasks involving transformations ofthese groups. Remarkably, under the out-of-distribution setting, our model achieves a 3.37% improvement in accuracy over the main counterpart affConv on the affNIST dataset.

## 1. Introduction

The success of convolutional neural networks (CNNs) can be attributed to their utilization of translation symmetry. This profound insight emphasizes the significance of incorporating symmetry priors into the design of models. With this insight, equivariant networks extend the exploitation of more symmetries, leading to great improvement on performance and efficiency. Development of equivariant networks begins with the approach of group convolutions, which views feature maps as functions defined on a group and conducts convolution operation over the group [3, 64]. Further advancements in equivariant networks effectively achieve equivariance on the Euclidean group and its subgroups [4, 11, 12, 62, 63, 68, 69]. However, existing methods have certain limitations when dealing with more complicated groups. One representative group is the affine group. While group convolutions typically require discretization of continuous groups, it becomes impractical for the affine group due to its high dimension. Finzi et al. [16] conduct group convolutions by sampling from Haar measure on the group. But it relies on easy access to Haar measure, which is unsuitable for the affine group. MacDonald et al. [37] overcome the limitation by computing the integral on the Lie algebra, thereby obtaining the affine equivariant model, affConv. Nevertheless, this approach still requires sampling from the group and encounters an exponential growth in memory requirements as the number of convolutional layers increases. A recent work [39] addresses the issue of exponential memory growth, but it also relies on sampling based on specific measures. In particular, sampling from the GL(n, R)-invariant measure of positive definite matrices is infeasible, and the authors adopt the log-normal distribution as a substitute, leading to imperfect equivariance theoretically.

In another branch, some works adopt partial differential operators (PDOs) to design equivariant networks [25, 27, 50]. They achieve equivariance on Euclidean groups by imposing constraints on the weights of PDOs. In fact, specific functional combinations of partial derivatives remain constant under group actions — a concept known as “differential invariants.” Under the guidance of the differential invariant theory, Liu et al. [33, 34] design a shift and rotationally equivariant system of learnable partial differential equations (PDEs) with linear combinations of differential invariants. The evolution process of PDEs can be used to solve multiple vision problems. Subsequent works extend the approach to more tasks and further develop the models [14, 47, 49, 78]. However, these efforts also concentrate on equivariance of Euclidean groups, and the full potential of differential invariants in handling more general groups has yet to be explored.

![](images/7acd1e5c2991859c49682f09a23c95f5f1771ea95f0d2312dceb13fe6d484a2f.jpg)  
Figure 1. InvarPDEs-Net consists of iterative processes of multiple symmetric PDEs constructed with invariants. We link them by linearly combining the output of one PDE to match the dimension of the subsequent one, which can be implemented with 1 1 convolutions.

In this paper, we construct affine equivariant networks based on differential invariants from the viewpoint of symmetric PDEs, without discretizing or sampling the group. Inspired by learnable PDEs [33, 34], we regard image data as smooth functions on the 2D plane and model the equivariant inference process of feature extraction as an evolving system governed by symmetric PDEs. The differential invariant theory reveals that, given a group G, a PDE admits G as a symmetry group if and only if the PDE consists of fundamental differential invariants of the group G [42]. To construct learnable symmetric PDEs, we can precompute a complete set of fundamental differential invariants of the group, and then employ multilayer perceptrons (MLPs) to combine them into equations, leveraging the universal approximation capability of neural networks. However, differential invariants of the affine group may take the form of fractional polynomials, potentially leading to the divisionby-zero issue in practice. Nonetheless, we notice that affine differential invariants can be represented by polynomial relative differential invariants. Building on this observation, we propose a technique to construct a new kind of affine invariants by normalizing polynomial relative differential invariants with a special norm, thus replacing the fundamental differential invariants. These new invariants not only avoid the division-by-zero issue but also retain more information. To discretize the symmetric PDE (not the affine group), we approximate the temporal derivatives by forward difference and approximate the spatial derivatives by Gaussian derivatives, resulting in an iterative process that can be viewed as a feed-forward deep equivariant network.

To equip our network with adaptability to varying channel numbers, similar to other modern networks, we sequentially stack iterative processes of multiple learnable symmetric PDEs with different dimensions. We connect them by linearly combining the output channels of one PDE to match channel numbers of the subsequent PDE. Thus, the output of one PDE can serve as the input of the subsequent one. This approach allows us to create an equivariant network with varying channel numbers, which consists of multiple symmetric PDEs constructed with invariants. We name it InvarPDEs-Net (see Figure 1). For further flexibility, we extract a block from the iterative process and modify it into an equivariant layer, offering the freedom to specify input and output channel numbers. The layer can serve as a drop-in replacement for convolutional networks of various architectures. We name it InvarLayer. Our framework for constructing equivariant networks of the affine group is also applicable to its continuous subgroups. We implement equivariant networks for the scale group, the rotation-scale group and the affine group. Empirical experiments on classification tasks involving transformations of these groups demonstrate the outstanding performance of our method.

We summarize our main contributions as follows:

• From the viewpoint of symmetric PDEs, we construct affine equivariant networks based on differential invariants. It is the first time that affine equivariance for networks is achieved without discretizing or sampling the group. Consequently, we overcome the limitation on network depth encountered by affConv [37].

• We propose a technique to construct a new kind of affine invariants by normalizing polynomial relative differential invariants with a special norm, which can be incorporated into our networks and enhance numerical stability.

• For further flexibility, we also design an equivariant layer, InvarLayer, which serves as a drop-in replacement for convolutional networks of various architectures.

• Our framework for constructing affine equivariant networks is also applicable to its continuous subgroups. We implement equivariant networks for three non-Euclidean groups: the scale group, the rotation-scale group and the affine group. Extensive experiments demonstrate the outstanding performance of our framework. Particularly, we achieve a 3.37% improvement in accuracy compared with affConv [37] on the public affNIST<sup>1</sup> dataset under the out-of-distribution setting.<sup>2</sup>

## 2. Related works

Currently there are two mainstream methods for constructing group equivariant networks. One approach stems from Cohen and Welling [3], which treats feature maps as functions defined on a group. Some works extend this approach to subgroups of Euclidean groups on various domains, such as rotation on the 2D plane [1, 24, 31, 32], rotation over the 3D space [11, 12, 68, 69], symmetries on spheres [5, 9, 10] and surfaces [6, 7]. Besides, with some proper approximations, some works utilize this approach to handle non-compact groups, such as the scale group [48, 54, 67, 70, 79] and Lie groups [16, 37, 39]. The other approach follows the steerable CNNs framework [4, 62–64], which views feature maps as vector fields. This approach has also been further applied to subgroups of Euclidean groups on the 2D plane [20, 58, 59, 71, 76], the 3D space [17, 63], and spheres [13, 65]. In addition, similar to the first approach, this approach has also been extended to the scale group [19, 41, 53] and the rotation-scale group [18, 56, 75].

Besides the above approaches, some works utilize PDOs with learnable coefficients to design equivariant neural networks on 2D plane [25, 27, 50]. Besides, PDOs can also be applied to spheres [28, 51], volumetric data [52] and surfaces [66]. Differential invariants, as a specialized form of partial differential operators, hold a distinctive role in the field of image processing [22, 40, 45, 57, 60]. The equivariant method of moving frames offers an elegant tool to derive differential invariants of a given group [15, 43, 44]. Wang et al. [60] provide a practical and simplified approach for deriving relative affine differential invariants. Theoretical links reveal that differential invariants are closely intertwined with symmetric PDEs [42]. Building upon this connection, Liu et al. [33, 34] design a shift and rotationally equivariant system comprised of learnable PDEs with linear combinations of fundamental differential invariants. Subsequently, some works apply learnable PDEs to feature learning and extensive vision tasks [14, 78]. Some works further develop the approach and create equivariant networks on Euclidean groups [47, 49]. Additionally, some researchers draw inspiration from PDEs to design deep convolutional networks [35, 36].

## 3. Theoretical framework

In this section, we propose a new framework based on differential invariants to achieve equivariance of the affine

group. We also describe some extensions of the framework and how they can be implemented.

## 3.1. Basic concepts and notations

To explicitly present the proposed method and theoretical derivation in the following, we first give a preliminary introduction to concepts involved and notations used.

Inputs and intermediate feature maps of neural networks can be modeled as vector functions defined on a continuous domain, $e . g .$ . the 2D plane for image data. Each layer of the network thereby can be regarded as an operator. In this paper, we study $\mathcal { F } = \{ \mathbf { u } | \mathbf { u } : X  \mathbb { R } ^ { n } \}$ as the set of smooth functions defined on $X \overset { \cdot } { = } \mathbb { R } ^ { 2 }$ , which are constant outside a compact set. Given a group G acting on $X ,$ , it naturally induces a group action on , i.e. $( g \cdot \mathbf { u } ) ( \mathbf { x } ) = \mathbf { u } ( g ^ { - 1 } \cdot \mathbf { x } )$ where $g \in G , \mathbf { x } \in X , \mathbf { u } \in \mathcal { F }$

Equivariance indicates that the output of a mapping transforms in accordance with transformation of the input.

Definition 1 Let G be a group acting on function sets $\mathcal { F }$ and ${ \mathcal { F } } ^ { \prime } .$ . An operator $\Psi : \mathcal { F }  \mathcal { F } ^ { \prime }$ is said to be equivariant with respect to G, $i f \Psi [ g \cdot \mathbf { u } ] = g \cdot \Psi [ \mathbf { u } ] , \forall g \in G , \mathbf { u } \in \mathcal { F }$

Transitivity is an important property of equivariance. As a result, when equivariant operators are composed together, they still possess equivariance.

The concept of invariants is crucial and widely applied in various fields. Invariants extract some symmetric information and remain constant on the orbits of group actions. Here we give the definition of invariants below.

Definition $2 \ L e t \ G$ be a group acting on X, and ${ \mathcal { F } } =$ $\{ \mathbf { u } | \mathbf { u } : X \ \to \ \mathbb { R } ^ { n } \}$ be a function set defined on X. An  invariant $o f G$ is a map $\mathcal { T } : \boldsymbol { X } \times \mathcal { F }  \mathbb { R }$ such that $\forall \mathbf { u } \in { \mathcal { F } } , \mathbf { x } \in X , g \in G ,$

$$
{ \mathcal { T } } ( g \cdot \mathbf { x } , g \cdot \mathbf { u } ) = { \mathcal { T } } ( \mathbf { x } , \mathbf { u } ) .\tag{1}
$$

We call $\pmb { \mathcal { T } } \triangleq ( \mathbb { Z } _ { 1 } , . . . , \mathbb { Z } _ { k } ) ^ { \intercal }$ a k-dimensional invariant of $G ,$ $i f T _ { 1 } , . . . , T _ { k }$ are invariants $o f G$

Invariants under operation of postcomposition maintain the property of invariance, which can be formulated as follows:

Proposition 3 Let $\pmb { \mathscr { T } } : X \times \mathscr { F }  \mathbb { R } ^ { k }$ be a k-dimensional invariant, and $\mathbf { h } : \mathbb { R } ^ { k }  \mathbb { R } ^ { k ^ { \prime } }$ be a k0-dimensional vector function. Then h is a k0-dimensional invariant.

Intuitively, invariants and equivariant operators somehow both imply symmetry of group G. In fact, we can construct an equivariant operator with an invariant.

Proposition 4 Let $\pmb { \mathscr { T } } : X \times \mathscr { F }  \mathbb { R } ^ { k }$ be a k-dimensional invariant ofG, where $\mathcal { F } = \{ \mathbf { u } | \mathbf { u } : X  \mathbb { R } ^ { n } \}$ . View $\pmb { \mathcal { Z } } ( \cdot , \mathbf { u } )$  as a k-dimensional function in $\mathcal { F } ^ { \prime } = \{ \mathbf { v } | \mathbf { v } : X  \mathbb { R } ^ { k } \}$ and define an operator $\hat { \boldsymbol { \mathcal { Z } } } : \mathcal { F }  \mathcal { F } ^ { \prime }$  such that

$$
{ \hat { \pmb { \mathscr { L } } } } [ { \mathbf { u } } ] \triangleq { \pmb { \mathscr { L } } } ( \cdot , { \mathbf { u } } ) .\tag{2}
$$

Then $\hat { \boldsymbol { \tau } }$ is equivariant.

Proof. $\forall \mathbf { u } \in { \mathcal { F } } , g \in G , x \in X$ , we have

$$
\begin{array} { r l } & { \hat { \mathbf { \mathcal { Z } } } [ g \cdot \mathbf { u } ] ( x ) = \mathbf { \mathcal { Z } } ( x , g \cdot \mathbf { u } ) } \\ & { \quad \quad \quad = \mathbf { \mathcal { Z } } ( g ^ { - 1 } \cdot x , \mathbf { u } ) } \\ & { \quad \quad \quad = \hat { \mathbf { \mathcal { Z } } } [ \mathbf { u } ] ( g ^ { - 1 } \cdot x ) } \\ & { \quad \quad \quad = ( g \cdot \hat { \mathbf { \mathcal { Z } } } [ \mathbf { u } ] ) ( x ) . } \end{array}
$$

Therefore, ${ \hat { \pmb { \mathscr { T } } } } [ g \cdot \mathbf { u } ] = g \cdot { \hat { \pmb { \mathscr { T } } } } [ \mathbf { u } ]$

It is worth noting that the equivariant operator $\hat { \boldsymbol { \tau } }$ composed with a function h remains equivariant. Specifically, the operator $\mathbf { u } \mapsto$ h $\circ \hat { \mathcal { Z } } [ \mathbf { u } ]$ is actually equivalent to $\mathbf { u } \mapsto \hat { \pmb { \mathscr { T } } } _ { \mathbf { h } } [ \mathbf { u } ]$ where $\pmb { \mathcal { T } } _ { \mathbf { h } } \triangleq \mathbf { h } \circ \pmb { \mathcal { T } }$ is still an invariant according to Proposition 3.

As a special type of invariants, a differential invariant is a quantity involving the derivatives of functions that remains unchanged under the prolongation of group actions.

Definition 5 Let f be a smooth function and ${ \mathcal { T } } ( \mathbf { x } , \mathbf { u } ) \ { \stackrel { \Delta } { = } }$ $f ( \mathbf { x } , \mathbf { u } ( \mathbf { x } ) , \nabla \mathbf { u } ( \mathbf { x } ) , . . . , \nabla ^ { d } \mathbf { u } ( \mathbf { x } ) ) )$ . If  is an invariant, we call  a d-th order differential invariant.

As we always require translation equivariance by default, it is sufficient to consider differential invariants in the form $\begin{array} { r } { \mathcal { T } ( \mathbf { x } , \mathbf { u } ) \ \triangleq \ f ( \mathbf { u } ( \mathbf { x } ) , \nabla \mathbf { u } ( \mathbf { x } ) , . . . , \nabla ^ { d } \mathbf { u } ( \mathbf { x } ) ) ) } \end{array}$ , omitting the term x [34, 61]. According to the differential invariant theory [42], there are finite independent differential invariants up to the d-th order such that any d-th order differential invariant can be expressed by these differential invariants. We call them fundamental differential invariants.

In this paper, we focus on the affine group, which is ubiquitous in computer vision. The affine group consists of translation and invertible linear transformations. Denote the affine group as G, and any element $g \in G$ can be represented as $g = ( \mathbf { A } , \mathbf { b } )$ , where $\mathbf { A } \in \mathbb { R } ^ { 2 \times 2 }$ is invertible and $\mathbf { b } \in \mathbb { R } ^ { 2 }$ . Then $g \in G$ acts on $\mathbb { R } ^ { 2 }$ via the following way: $g \cdot \mathbf { x } = \mathbf { A } \mathbf { x } + \mathbf { b } , \forall \mathbf { x } \in \mathbb { R } ^ { 2 }$

## 3.2. From symmetric PDE to equivariant network

Inspired by learnable PDEs, we model the process of feature extraction as the evolution process governed by PDEs [14, 33–36, 78]. If the utilized PDE exhibits symmetry, the resultant feature extraction process will inherently possess equivariance [14, 33, 34, 78].

Let $\tilde { \mathcal { F } } = \{ \tilde { \mathbf { u } } | \tilde { \mathbf { u } } : [ 0 , T ] \times \mathbb { R } ^ { 2 }  \mathbb { R } ^ { n } \}$ be a set of smooth  functions involving a temporal variable $t \in [ 0 , T ]$ and a spatial variable $\textbf { x } \in \ \mathbb { R } ^ { 2 }$ . We focus on high-dimensional evolutionary PDEs in the following form:

$$
\frac { \partial { \tilde { \mathbf { u } } } } { \partial { t } } = \mathbf { F } \left( t , \mathbf { x } , \tilde { \mathbf { u } } , \nabla _ { \mathbf { x } } \tilde { \mathbf { u } } , . . . , \nabla _ { \mathbf { x } } ^ { d } \tilde { \mathbf { u } } \right) ,\tag{3}
$$

where $\mathbf { F }$ is a smooth function. We can view $\mathbf { u } ^ { ( t ) } \triangleq \tilde { \mathbf { u } } ( t , \cdot )$ as a function in $\mathcal { F } = \{ \mathbf { u } | \mathbf { u } : \mathbb { R } ^ { 2 }  \mathbb { R } ^ { n } \}$ , and consider the group action of G on $\tilde { \mathbf { u } } \in \tilde { \mathcal { F } }$ following the same way of the group action on $\mathbf { u } ^ { ( t ) } \in \mathcal { F }$ , i.e. $( g \cdot \tilde { \mathbf { u } } ) ( t , \mathbf { x } ) = \tilde { \mathbf { u } } ( t , g ^ { - 1 } \cdot \mathbf { x } )$ For a given symmetry group G, a PDE in the form (3) is called G-symmetric as long as if u˜ is a solution, then $g \cdot { \tilde { \mathbf { u } } }$ is also a solution, for any $g \in G$

![](images/cda0c6ddf130699c41c111dc11813e4642fee802693ffebb89a500a1e34e3f7c.jpg)  
Figure 2. Each iteration of the evolutionary PDE can be viewed as a layer of the network.

According to the differential invariant theory [42], the PDE (3) is G-symmetric if and only if the right side of (3) is a function of differential invariants. Additionally, any differential invariant can be expressed as a function of fundamental differential invariants. Therefore, any G-symmetric PDE in the form (3) can be written as:

$$
\frac { \partial \tilde { \mathbf { u } } } { \partial t } ( t , \mathbf { x } ) = \mathbf { H } \left( t , \mathcal { T } _ { 1 } ( \mathbf { x } , \mathbf { u } ^ { ( t ) } ) , . . . , \mathcal { T } _ { k } ( \mathbf { x } , \mathbf { u } ^ { ( t ) } ) \right) ,\tag{4}
$$

where H is a smooth function and $\mathcal { T } _ { i } ( i = 1 , 2 , . . . , k )$ form a complete set of fundamental differential invariants. Denote $\pmb { \mathcal { Z } } _ { F D I }$ as the concatenation of fundamental differential invariants, i.e $\mathbf { \nabla } _ { \cdot } \mathcal { I } _ { F D I } \triangleq ( \mathbb { Z } _ { 1 } , . . . , \mathbb { Z } _ { k } ) ^ { \intercal }$ . We can present (4) in a more compact form:

$$
\frac { \partial \tilde { \mathbf { u } } } { \partial t } = \mathbf { H } ^ { ( t ) } \circ \hat { \pmb { \mathcal { T } } } _ { F D I } [ \mathbf { u } ^ { ( t ) } ] ,\tag{5}
$$

where $\mathbf { H } ^ { ( t ) } \triangleq \mathbf { H } ( t , \cdot )$ is a smooth function indexed by t with input dimension k and output dimension n, and the definition of the operator $\hat { \boldsymbol { \mathcal { I } } } _ { F D I }$ is given in (2).

Consider a PDE system consisting of a G-symmetric PDE in the form (5) with an initial condition,

$$
\left\{ \begin{array} { l l } { \frac { \partial \tilde { \mathbf { u } } } { \partial t } = \mathbf { H } ^ { ( t ) } \circ \hat { \pmb { \mathcal { T } } } _ { F D I } [ \mathbf { u } ^ { ( t ) } ] , } \\ { \tilde { \mathbf { u } } ( 0 , \mathbf { x } ) = \mathbf { u } _ { 0 } ( \mathbf { x } ) . } \end{array} \right.\tag{6}
$$

We approximate the temporal derivative by forward difference to discretize the PDE and formally solve the PDE system (6) by iteration. Let $0 = t _ { 0 } < t _ { 1 } < . . . < t _ { N } = T$ be a partition of the interval $[ 0 , T ]$ , and the forward scheme is shown as follows:

$$
\mathbf { u } ^ { ( t _ { 0 } ) } = \mathbf { u } _ { 0 } ,\tag{7}
$$

$$
\mathbf { u } ^ { ( t _ { i + 1 } ) } = \mathbf { u } ^ { ( t _ { i } ) } + \Delta t _ { i } \cdot \mathbf { H } ^ { ( t _ { i } ) } \circ \hat { \mathbf { \mathcal { T } } } _ { F D I } [ \mathbf { u } ^ { ( t _ { i } ) } ] ,\tag{8}
$$

where $\Delta t _ { i } \triangleq t _ { i + 1 } - t _ { i } , \mathbf { u } ^ { ( t _ { i } ) } \triangleq \tilde { \mathbf { u } } ( t _ { i } , \cdot )$ . As is well known, neural networks have universal approximation capabilities. Theoretically, if we choose $\mathbf { H } ^ { ( t ) }$ to be a neural network, we can represent any differential invariant. In practice, we introduce a series of parameterized multilayer perceptrons (MLPs), $\{ \mathbf { h } _ { \theta _ { i } } , 0 \le i \le N - 1 \}$ , whose input dimension matches $\mathcal { L } _ { F D I }$ and output dimension matches $\mathbf { u } _ { 0 }$ . Consequently, we have the iterative process:

$$
\mathbf { u } ^ { ( t _ { i + 1 } ) } = \mathbf { u } ^ { ( t _ { i } ) } + \Delta t _ { i } \cdot \mathbf { h } _ { \theta _ { i } } \circ \hat { \pmb { { \mathcal { T } } } } _ { F D I } [ \mathbf { u } ^ { ( t _ { i } ) } ] .\tag{9}
$$

We regard each iteration as an operator $\Psi _ { i } : \mathbf { u } ^ { ( t _ { i } ) } \mapsto \mathbf { u } ^ { ( t _ { i + 1 } ) }$ (see Figure 2), which is equivariant. Note that equivariance of $\Psi _ { i }$ does not rely on the existence and uniqueness of the solution to the original PDE system (6). Furthermore, if we replace $\mathcal { T } _ { F D I }$ with general invariants, the operator $\Psi _ { i }$ is still equivariant.

Utilizing transitivity of equivariance, we stack these equivariant operators together to get a feed-forward deep equivariant network, i.e. $\Psi \triangleq \Psi _ { N - 1 } \circ \cdots \circ \Psi _ { 1 } \circ \Psi _ { 0 }$ . The number of layers corresponds to the number N of iterations, and the number of channels corresponds to the dimension of u˜ in the PDE. The network takes $\mathbf { u } ^ { ( t _ { 0 } ) } = \mathbf { u } _ { 0 }$ as inputs, and produces $\mathbf { u } ^ { ( t _ { N } ) }$ as the output features. Inference of the network aligns with the evolution process of the PDE. Furthermore, the network naturally incorporates the skip connection structure [23], which is renowned for its advantageous impact on network optimization. Inspired by PDEs based on invariants, we call the network InvarPDE-Net.

## 3.3. SupNorm normalized differential invariants

The basic version of InvarPDE-Net provides an approach to create equivariant networks without discretizing or sampling groups. However, for the affine group, its fundamental differential invariants are in the form of fractional polynomials, such as $\frac { u _ { x x } u _ { y y } - u _ { x y } ^ { 2 } } { u _ { y } ^ { 2 } u _ { x x } - 2 u _ { x } u _ { y } u _ { x y } + u _ { x } ^ { 2 } u _ { y y } }$ , potentially leading to the division-by-zero issue in practice. For instance, when a region of an image has uniform color, the denominator approaches or equals zero.

We notice that differential invariants of the affine group can be expressed by polynomial relative differential invariants. Building upon the observation, we propose a technique to construct a new type of affine invariants by normalizing polynomial relative differential invariants with a special norm, which not only avoid the division-by-zero issue but also exhibit better expressive power than classical differential invariants. To start with, we give a definition of polynomial relative differential invariants.

Definition 6 Let G be the affine group acting on $X = \mathbb { R } ^ { 2 }$ $\mathcal { F } = \{ \mathbf { u } | \mathbf { u } : X  \mathbb { R } ^ { n } \}$ be the set of smooth functions, w $: G \to \mathbb { R } ^ { + }$ be a positive multiplier, and P be a m-degree homogeneous polynomial. Define $\mathcal { I } : X \times \mathcal { F } $ R as follows $\mathcal { I } ( \mathbf { x } , \mathbf { u } ) \triangleq P ( \mathbf { u } ( \mathbf { x } ) , \nabla \mathbf { u } ( \mathbf { x } ) , . . . , \nabla ^ { d } \mathbf { u } ( \mathbf { x } ) ) )$ ). We call $\mathcal { I }$ a d-th order (polynomial) relative differential invariant of G with weight w and degree m, if u $\mathbf { \xi } \in { \mathcal { F } } , \mathbf { x } \in X , g \in$ $G ,$ we have

$$
\mathcal { I } ( g \cdot \mathbf { x } , g \cdot \mathbf { u } ) = w ( g ) \mathcal { I } ( \mathbf { x } , \mathbf { u } ) .\tag{10}
$$

<table><tr><td>Relative Differential Invariants</td><td>Weight</td><td>Degree</td></tr><tr><td>u</td><td>1</td><td>1</td></tr><tr><td> $u _ { x x } u _ { y y } - u _ { x y } ^ { 2 }$ </td><td> $1 / ( \operatorname* { d e t } \mathbf { A } ) ^ { 2 }$ </td><td>2</td></tr><tr><td> $u _ { y } ^ { 2 } u _ { x x } - 2 u _ { x } \overset { \vartriangle } { u } _ { y } u _ { x y } + u _ { x } ^ { 2 } u _ { y y }$ </td><td> $1 / ( \operatorname* { d e t } \mathbf { A } ) ^ { 2 }$ </td><td>3</td></tr></table>

Table 1. We present relative differential invariants of the affine group for scalar functions up to order 2. Note that any element g in the affine group can be represented as $g = ( \mathbf { A } , \mathbf { b } )$ .

Low-order relative differential invariants of the affine group for scalar functions on $\mathbb { R } ^ { 2 }$ are shown in Table 1. Unless the weight $w \equiv 1$ , a relative differential invariant is generally not an invariant. Actually, the result of dividing two relative differential invariants with the same weight is a differential invariant. But fractional polynomials may suffer from the division-by-zero issue as mentioned before.

Next, we present a technique to construct invariants based on relative differential invariants via normalization.

Theorem 7 Let G be the affine group acting on $X = \mathbb { R } ^ { 2 }$ Let $\mathcal { F } = \{ \mathbf { u } | \mathbf { u } : X  \mathbb { R } ^ { n } \}$ and $\mathcal { F } ^ { \prime } = \{ \mathbf { v } | \mathbf { v } : X  \mathbb { R } ^ { k } \}$  be sets of smooth functions on $X ,$  , which are constant outside a compact set. Define a norm on ${ \mathcal { F } } ^ { \prime }$ called SupNorm, $\begin{array} { r } { \| \mathbf { v } \| _ { \mathrm { s u p } } \triangleq \operatorname* { s u p } _ { \mathbf { x } \in X } \| \mathbf { v } ( \mathbf { x } ) \| _ { \infty } . } \end{array}$ . Given a collection of relative differential invariants $o f G$ with weight w, denoted as $\mathcal { T } _ { i } : X \times \mathcal { F }  \mathbb { R } ( i = 1 , 2 , . . . , k )$ . Define $\mathcal { I } : X \times \mathcal { F }  \mathbb { R } ^ { k }$ as $\mathcal { T } \triangleq ( \mathcal { I } _ { 1 } , . . . , \mathcal { I } _ { k } ) ^ { \intercal }$ , and $\mathcal { I } ( \cdot , { \mathbf { u } } )$ can be viewed as an element in ${ \mathcal { F } } ^ { \prime }$ . Define $\pmb { \mathcal { T } } : X \times \pmb { \mathcal { F } }  \mathbb { R } ^ { k }$ as follows:

$$
\mathcal { T } ( \mathbf { x } , \mathbf { u } ) \triangleq \frac { 1 } { \| \mathcal { I } ( \cdot , \mathbf { u } ) \| _ { \mathrm { s u p } } } \cdot \mathcal { T } ( \mathbf { x } , \mathbf { u } ) .\tag{11}
$$

Then is a k-dimensional invariant ofG.

The key to the proof of Theorem 7 is that for any $g \in G ,$ we have $\| \mathcal { I } ( \cdot , g \cdot { \bf u } ) \| _ { \mathrm { s u p } } = w ( g ) \| \mathcal { I } ( \cdot , { \bf u } ) \| _ { \mathrm { s u p } } . \mathrm { ~ A ~ }$ detailed proof is provided in Supplementary Material. We call the invariant constructed in (11) a SupNorm normalized differential invariant (SNDI). As the invariant involves global spatial information of derivatives, it is no longer a classical differential invariant. It may contain information beyond fundamental differential invariants.

To construct SNDIs, we start from a collection of polynomial relative differential invariants. Although the selection of relative differential invariants does not affect invariance, a recommended practice is to encompass those that sufficiently represent fundamental differential invariants, thus capturing adequate information. Next, we can normalize each relative differential invariant individually, as Theorem 7 also holds when $k = 1$ . Alternatively, we can normalize all relative differential invariants with the same weight and the same degree together, which preserves more information between relative differential invariants. An additional benefit is that SNDIs derived in this way exhibit illumination invariance, i.e. $\mathcal { T } ( \mathbf { x } , c \cdot \mathbf { u } ) = \mathcal { T } ( \mathbf { x } , \mathbf { u } ) , \forall c > 0$

Here is an example to have a glimpse of the advantage in expressive power of SNDIs compared with that of classical differential invariants. Assuming u a smooth scalar function on $\begin{array} { r } { \mathbb { R } ^ { 2 } , \frac { u _ { x x } u _ { y y } - u _ { x y } ^ { 2 } } { u _ { y } ^ { 2 } u _ { x x } - 2 u _ { x } u _ { y } u _ { x y } + u _ { x } ^ { 2 } u _ { y y } } } \end{array}$ is the only fundamental differential invariant of the affine group up to second order apart from the trivial one u itself. Through the newly proposed method of normalization, we can obtain two SNDIs $\begin{array} { r } { \frac { u _ { x x } u _ { y y } - u _ { x y } ^ { 2 } } { \left\| u _ { x x } u _ { y y } - u _ { x y } ^ { 2 } \right\| _ { \mathrm { s u p } } } \mathrm { a n d } \frac { u _ { y } ^ { 2 } u _ { x x } - 2 u _ { x } u _ { y } u _ { x y } + u _ { x } ^ { 2 } u _ { y y } } { \left\| u _ { y } ^ { 2 } u _ { x x } - 2 u _ { x } u _ { y } u _ { x y } + u _ { x } ^ { 2 } u _ { y y } \right\| _ { \mathrm { s u p } } } } \end{array}$ . It is not hard to find that we can express the fundamental differential invariant as the quotient of two SNDIs up to a constant multiple, but not vice versa. From another perspective, we need to discretize functions by sampling on grid points in implementation. Given k polynomial relative differential invariants, each one can be viewed as an $M \times M$ matrix. Obtaining differential invariants through division would lead to the loss of at least $M ^ { 2 }$ degrees of freedom, while normalization only sacrifices at most k degrees of freedom.

In summary, SNDIs not only avoid the division-by-zero issue but also exhibit better expressive power than classical differential invariants. Given a collection of polynomial relative differential invariants, we construct SNDIs via normalization, and concatenate them together, resulting in a higher dimensional invariant $\pmb { \mathcal { T } } _ { S N D I }$ With theoretical guarantee of invariance, we can directly employ $\pmb { \mathcal { T } } _ { S N D I }$ to replace fundamental differential invariants $\mathcal { T } _ { F D I }$ in (9). Thus, each layer of InvarPDE-Net is adjusted to:

$$
\mathbf { u } ^ { ( t _ { i + 1 } ) } = \mathbf { u } ^ { ( t _ { i } ) } + \Delta t _ { i } \cdot \mathbf { h } _ { \theta _ { i } } \circ \hat { \pmb { { \mathcal { T } } } } _ { S N D I } [ \mathbf { u } ^ { ( t _ { i } ) } ] .\tag{12}
$$

## 3.4. Extensions of network architecture

The equivariant network InvarPDE-net derived from a symmetric PDE requires the dimension of features, namely the number of channels, to be consistent across each layer. This is not the case for the majority of conventional networks. Hence, we generalize the network to accommodate varying channel numbers while maintaining equivariance.

Note that we can stack several PDEs of the same dimension sequentially, with the output of one PDE serving as the input of the subsequent one. Furthermore, when dealing with PDEs of different dimensions, we can linearly combine the output channels of one PDE to match the number of channels in the subsequent PDE. Since linear combinations of invariants remain invariants, the process does not affect equivariance. This extension allows us to create an equivariant network composed of multiple PDEs with varying channel numbers. We name the network InvarPDEs-Net (see Figure 1), including InvarPDE-Net as a special case.

In addition, we aim to design an equivariant layer that can be directly integrated into convolutional networks of various architectures by replacing convolutional layers. Such an equivariant layer will offer enhanced flexibility in its applications. A key aspect lies in the ability to freely specify input and output channel numbers, similar to a convolutional layer. By observing the iterative process in (12), we can adjust the output dimension of $\mathbf { h } _ { \theta _ { i } }$ , and directly employ $\mathbf { h } _ { \theta _ { i } } \circ \hat { \pmb { { \mathcal { T } } } } _ { S N D I } [ \mathbf { u } ^ { ( t _ { i } ) } ]$ as the output of this layer, which is still equivariant. Given input and output channel numbers, $C _ { 1 }$ and $C _ { 2 } .$ , we formulate the equivariant layer as follows:

![](images/d80d62f3b21efa5cecdb43162ad62d5cab1c79f2854c3342f48d22b72d7deb94.jpg)  
Figure 3. InvarLayer is an equivariant layer extracted and adapted from the iterative process of a symmetric PDE, which allows for free specification of input and output channel numbers.

$$
\mathbf { u } _ { o u t } = \mathbf { h } _ { \theta } \circ \hat { \pmb { { \mathcal { T } } } } _ { S N D I } [ \mathbf { u } _ { i n } ] ,\tag{13}
$$

where $\pmb { \mathcal { T } } _ { S N D I }$ is a k-dimensional SNDI, and $\mathbf { h } _ { \theta }$ is an MLP with input dimension k and output dimension $C _ { 2 }$ . We name the equivariant layer InvarLayer (see Figure 3). It has a similar structure to the PDE iteration process (see Figure 2) but without the skip connection, allowing different input and output channel numbers.

## 3.5. Implementation

We establish a theoretical foundation in the continuous setting. When it comes to implementation, in the context of processing image data, discretization on 2D grids becomes necessary. We employ Gaussian derivatives to estimate derivatives by applying derivatives of a Gaussian kernel $[ 2 5 , 2 7 ]$ . For example, $\begin{array} { r } { \bar { f } _ { x } ( x _ { 0 } ) \approx \sum _ { n = 1 } ^ { N } \partial _ { x } G ( x _ { n } ; \sigma ) f ( x _ { n } + } \end{array}$ $x _ { 0 } )$ , where $G ( x ; \sigma )$ is a Gaussian kernel with standard deviation   centered around 0, and $x _ { n }$ are grid points around 0. In the case of 2D grid points, it can be implemented using convolutions with specific kernels.

It is important to highlight that common network components are compatible with our approach. Proposition 3 guarantees that invariants under the operation of postcomposition maintain their invariance property. This means BatchNorm [26], pointwise nonlinearities, $1 \times 1$ Convolution, and DropOut [55] can all be seamlessly integrated into our models without compromising equivariance. Pooling can also be incorporated into the models, though it introduces equivariance error to some extent. Specially, when using global pooling, we obtain invariant features.

In the following, we will discuss the input and ultimate output in InvarPDEs-Net, with a specific focus on image classification tasks. Currently, we simply replicate the image data along the channel dimension multiple times until the given number of channels is reached, which serves as the input of the network, i.e. $\mathbf { u } ^ { ( t _ { 0 } ) } = \mathbf { u } _ { 0 }$ . For the final output of equivariant features of the network $\mathbf { u } ^ { ( t _ { N } ) }$ , we perform spatial global pooling to extract a set of invariant features, matching the number of channels. Subsequently, we apply two fully connected layers to acquire the ultimate classification result.

As for MLPs used for combining invariants in our networks, we apply two layer perceptrons in practice. Since MLPs operate on the vector $\pmb { \mathcal { T } } ( \mathbf { x } , \mathbf { u } )$ for each point x and share weights spatially, they can be effectively implemented using 1 1 convolutions with the ReLU activation function. Likewise, connections between PDEs of different dimensions in InvarPDEs-Net can also be realized using $1 \times 1$ convolutions. As for the computation of SupNorm in constructing SNDIs, it can be easily implemented by applying global Max-Pooling over the channels corresponding to relative differential invariants that are normalized together.

## 3.6. Discussion

Unlike existing methods for designing equivariant networks, our framework does not apply discretization or sampling to the group. The number of channels is independent of the dimension of the group. When the group is larger, the number of fundamental differential invariants is bounded by the number of derivatives, and the same holds true for polynomial relative differential invariants. Therefore, the model size does not increase as the group becomes larger. That is why our framework can handle affine equivariance.

Moreover, our framework can be extended to continuous subgroups of the affine group. Common examples include the scale group, the shearing group, the rotation group, the rotation-scale group and the equi-affine group. To construct equivariant networks for these groups, we simply compute corresponding differential invariants and incorporate them into InvarPDE-Net. If the differential invariants involve fractions, the normalization technique is also applicable. The network structures, InvarPDEs-Net and InvarLayer, are compatible with these groups, and the implementation process remains the same. Therefore, it is a unified framework for the affine group and its continuous subgroups.

## 4. Experiments

For empirical validation, we implement InvarPDEs-Net and InvarLayer for three non-Euclidean groups: the scale group, the rotation-scale group, and the affine group. We conduct classification experiments on image datasets with different group transformations, and refrain from using data augmentation to emphasize the innate equivariance of networks.

## 4.1. Scale equivariance

Following previous works on scale equivariance [29, 41, 53, 79], we conduct experiments on datasets with scale variations, specifically Scale-MNIST and Scale-Fashion. We build Scale-MNIST and Scale-Fashion by rescaling the images of the MNIST [30] dataset and the Fashion-MNIST [72] dataset with the scaling factor randomly selected from [0.3, 1]. Then we reshape them back to the original size $2 8 \times 2 8$ by zero paddings. For both datasets, we use 10k samples for training and 50k for testing. In line with prior works [41, 53, 79], we integrate InvarLayer into a CNN with three convolution layers and two fully connected layers, and ensure that both InvarPDEs-Net and InvarLayer have fewer than 500k trainable parameters. For more details on the models and experiments, please refer to Supplementary Material.

<table><tr><td>Models</td><td>Scale-MNIST</td><td>Scale-Fashion</td></tr><tr><td>SiCNN [74]</td><td> $9 7 . 5 3 \pm 0 . 1 2$ </td><td> $8 5 . 3 2 \pm 0 . 2 2$ </td></tr><tr><td>SI-ConvNet [29]</td><td> $9 7 . 5 6 \pm 0 . 1 3$ </td><td> $8 5 . 1 6 \pm 0 . 1 4$ </td></tr><tr><td>SEVF [38] DSS [70]</td><td> $9 7 . 2 8 \pm 0 . 1 6$ </td><td> $8 4 . 7 3 \pm 0 . 1 1$ </td></tr><tr><td>SS-CNN [19]</td><td> $9 7 . 3 4 \pm 0 . 1 3$   $9 7 . 6 8 \pm 0 . 1 5$ </td><td> $8 4 . 5 0 \pm 0 . 5 1$   $8 5 . 3 9 \pm 0 . 3 2$ </td></tr><tr><td>SESN [53]</td><td> $9 7 . 9 2 \pm 0 . 0 9$ </td><td> $8 5 . 9 3 \pm 0 . 2 8$ </td></tr><tr><td>ScDCFNet [79]</td><td></td><td></td></tr><tr><td>SE-CNN [41]</td><td> $9 7 . 9 1 \pm 0 . 0 8$ </td><td> $8 6 . 1 9 \pm 0 . 1 5$ </td></tr><tr><td></td><td>97.16</td><td>87.48</td></tr><tr><td>InvarPDEs-Net (Ours)</td><td> ${ \bf 9 8 . 3 0 \pm 0 . 0 6 }$ </td><td> ${ \bf 8 9 . 6 2 \pm \ 0 . 2 6 }$ </td></tr><tr><td>InvarLayer (Ours)</td><td> $9 7 . 7 5 \pm 0 . 0 5$ </td><td> $8 9 . 5 0 \pm 0 . 1 5$ </td></tr></table>

Table 2. Test accuracy (%) on Scale-MNIST and Scale-Fashion. All models have approximately 500k trainable parameters.

Experiments are repeated for six times using datasets <sup>generated</sup> <sup>with</sup> <sup>independent</sup> <sup>seeds.</sup> <sup>We</sup> <sup>report</sup> <sup>the</sup> <sup>mean</sup> ± std of the test accuracy of our models in Table 2. The results of SE-CNN on both datasets and SESN on Scale-MNIST come from the original papers [41, 53], and the others come from [79] under the same settings. On Scale-MNIST, Invar-Layer achieves comparable results with other models and InvarPDEs-Net delivers the best performance. On Scale-Fashion, InvarPDEs-Net and InvarLayer outperform other models significantly.

## 4.2. Rotation-Scale equivariance

Gao et al. [18] first presented a rotation-scale equivariant network, RST-CNN. Following [18], we generate datasets RS-MNIST and RS-Fashion for evalutation. With the same procedure, we apply rotation (uniformly in [0, 2⇡]) and rescaling (uniformly in [0.3, 1]) to the images of MNIST and Fashion-MNIST, and zero-pad them back to the original size followed by upsizing images to $5 6 \times 5 6 .$ . For both datasets, we use 5k samples for training and 50k for testing. Consistent with [18], we integrate InvarLayer into a CNN with three convolution layers and two fully connected layers, and keep the number of trainable parameters below 500k for both InvarPDEs-Net and InvarLayer.

The mean std of the test accuracy over six independent trials are reported in Table 3. Compared models include RST-CNN [18] and other models that are equivariant to either rotation (SFCNN [64] and RDCF [2]) or scaling (SEVF [38], SESN [53], and ScDCFNet [79]). The results of these models are obtained from [18] under the same settings. On RS-MNIST, InvarPDEs-Net significantly outperforms other models and InvarLayer exhibits comparable results with RST-CNN. On RS-Fashion, InvarPDEs-Net remains the top-performing model, while InvarLayer delivers relatively modest results. With minor adjustments of hyperparameters, InvarLayer lifts the accuracy to 93.40% on RS-MNIST and to 76.08% on RS-Fashion. More details about models and experiments are provided in Supplementary Material.

<table><tr><td>Models</td><td>RS-MNIST</td><td>RS-Fashion</td></tr><tr><td>SFCNN [64]</td><td> $8 9 . 6 9 \pm 0 . 4 0$ </td><td> $7 5 . 8 0 \pm 0 . 1 1$ </td></tr><tr><td>RDCF [2]</td><td> $9 0 . 4 6 \pm 0 . 3 3$ </td><td> $7 3 . 9 6 \pm 0 . 1 9$ </td></tr><tr><td>SEVF [38]</td><td> $9 0 . 2 9 \pm 0 . 3 7$ </td><td> $7 1 . 0 3 \pm 0 . 3 1$ </td></tr><tr><td>SESN [53]</td><td> $9 0 . 1 9 \pm 0 . 3 9$ </td><td> $7 2 . 1 9 \pm 0 . 0 5$ </td></tr><tr><td>ScDCFNet [79]</td><td> $9 0 . 4 0 \pm 0 . 0 9$ </td><td> $7 2 . 2 4 \pm 0 . 2 3$ </td></tr><tr><td>RST-CNN [18]</td><td> $9 3 . 1 9 \pm 0 . 2 9$ </td><td> $7 8 . 6 4 \pm 0 . 6 0$ </td></tr><tr><td>InvarPDEs-Net (Ours)</td><td> ${ \bf 9 5 . 8 0 \pm 0 . 0 9 }$ </td><td> $\mathbf { 7 9 . 4 8 \pm 0 . 3 1 }$ </td></tr><tr><td>InvarLayer (Ours)</td><td> $9 3 . 1 5 \pm 0 . 2 1$ </td><td> $7 4 . 5 1 \pm 0 . 7 1$ </td></tr></table>

Table 3. Test accuracy (%) on RS-MNIST and RS-Fashion. All models have approximately 500k trainable parameters. RST-CNN is a rotation-scale equivariant network, while other compared models are only equivariant to rotation (SFCNN and RDCF) or scaling (SEVF, SESN, and ScDCFNet).

## 4.3. Affine equivariance

<table><tr><td>Models</td><td>Accuracy</td><td>Parameters</td></tr><tr><td>CapsNet [46]</td><td>79</td><td>8.1M</td></tr><tr><td>GE CapsNet [31]</td><td>89.10</td><td>235K</td></tr><tr><td>affine CapsNet [21]</td><td>93.21</td><td></td></tr><tr><td>RU CapsNet [8]</td><td>97.69</td><td>&gt; 580K</td></tr><tr><td>affConv [37]</td><td> $9 5 . 0 8 \pm 0 . 3 1$ </td><td>373K</td></tr><tr><td>InvarPDEs-Net (Ours) InvarLayer (Ours)</td><td> $9 5 . 7 2 \pm 0 . 1 2$   ${ \bf 9 8 . 4 5 \pm 0 . 1 5 }$ </td><td>340K 365K</td></tr></table>

Table 4. Test accuracy (%) on affNIST after training on MNIST. The first four models are Capsule Networks that demonstrate robustness to affine transformations but admit few rigorous mathematical guarantees, while affConv is an affine equivariant network.

As for affine equivariance, the main counterpart we compare with is the affine equivariant model, affConv [37]. Following [37], we evaluate our models on the public dataset affNIST under the out-of-distribution setting. Specifically, we train our models on 50k non-transformed MNIST images (padded to 40  40) and test them on 320k affine-<sup>perturbed</sup> <sup>MNIST</sup> <sup>(affNIST)</sup> <sup>images</sup> <sup>with</sup> <sup>size</sup> <sup>40</sup> ⇥ <sup>40.</sup> <sup>As</sup> mentioned before, it is impractical to apply affConv to deep networks, while InvarLayer overcomes the limitation. We use the structure of ResNet-32 for InvarLayer. For a fair comparison, we ensure that InvarPDEs-Net and InvarLayer both have fewer parameters than affConv (373k). Additional details can be found in Supplementary Material.

We present the mean  std of test accuracy over six training runs with different random seeds in Table 4. Besides affConv, we also list the results under the same setup from some Capsule Networks [8, 21, 31, 46], which may lack rigorous theoretical guarantees of invariance. Although RU CapsNet performs better than affConv, which could not be well understood according to [37], our InvarLayer beats it by a margin of 0.76%. Moreover, our InvarPDEs-net also outperforms affConv. Additional results of the conventional setting, training on affNIST and testing on affNIST, can be found in Supplementary Material.

## 5. Conclusion

In this paper, we propose a new framework to achieve affine equivariance, a long-standing challenge in the field of equivariant networks. Within our framework, we construct a PDE-inspired equivariant network, InvarPDEs-Net, which showcases strong performance across extensive experiments. Furthermore, for more flexibility, we introduce an equivariant layer, InvarLayer, which can serve as a dropin replacement for convolutional networks of various architectures. When combined with a ResNet structure, Invar-Layer retains state-of-the-art results on the affNIST dataset. While the performance of InvarLayer exhibits some variability in certain setups, we recognize its immense potential. We believe that further refinement of the layer design based on our paradigm will elevate its capabilities to a higher level.

Our framework is quite promising and merits further extension. It is known that differential invariants exist for Lie groups satisfying certain regular conditions [42]. We concentrate on the affine group and make differential invariants applicable through the normalization technique, which is also suitable for its subgroups. How to adapt differential invariants of more general Lie groups into equivariant networks remains a future research. Additionally, besides the 2D planes considered in our work, it is worthwhile to study the extension of our framework to other manifolds, such as spheres and 3D spaces. Moreover, while our experiments involve image classification tasks, applications to a broader range of tasks in real world can be further explored.

## Acknowledgment

Z. Lin was supported by National Key R&D Program of China (2022ZD0160300), the NSF China (No. 62276004), and the major key project of PCL, China (No. PCL2021A12).

## References

[1] Erik J Bekkers, Maxime W Lafarge, Mitko Veta, Koen AJ Eppenhof, Josien PW Pluim, and Remco Duits. Rototranslation covariant convolutional networks for medical image analysis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 2018. 3

[2] Xiuyuan Cheng, Qiang Qiu, Robert Calderbank, and Guillermo Sapiro. Rotdcf: Decomposition of convolutional filters for rotation-equivariant deep networks. arXiv preprint arXiv:1805.06846, 2018. 7, 8

[3] Taco Cohen and Max Welling. Group equivariant convolutional networks. In International Conference on Machine Learning, pages 2990–2999. PMLR, 2016. 1, 3

[4] Taco S Cohen and Max Welling. Steerable CNNs. In International Conference on Learning Representations, 2016. 1, 3

[5] Taco S Cohen, Mario Geiger, Jonas Kohler, and Max¨ Welling. Spherical CNNs. In International Conference on Learning Representations, 2018. 3

[6] Taco S Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant convolutional networks and the icosahedral CNN. In International Conference on Machine Learning, 2019. 3

[7] Pim De Haan, Maurice Weiler, Taco S Cohen, and Max Welling. Gauge equivariant mesh CNNs: Anisotropic convolutions on geometric graphs. In International Conference on Learning Representations, 2021. 3

[8] Fabio De Sousa Ribeiro, Georgios Leontidis, and Stefanos Kollias. Introducing routing uncertainty in capsule networks. Advances in Neural Information Processing Systems, 33: 6490–6502, 2020. 8

[9] Michael Defferrard, Martino Milani, Fr¨ ed´ erick Gusset, and´ Nathanael Perraudin. Deepsphere: a graph-based spherical¨ CNN. In International Conference on Learning Representations, 2019. 3

[10] Carlos Esteves, Christine Allen-Blanchette, Ameesh Makadia, and Kostas Daniilidis. Learning SO(3) equivariant representations with spherical CNNs. In Proceedings ofthe European Conference on Computer Vision, 2018. 3

[11] Carlos Esteves, Avneesh Sud, Zhengyi Luo, Kostas Daniilidis, and Ameesh Makadia. Cross-domain 3D equivariant image embeddings. In International Conference on Machine Learning, 2019. 1, 3

[12] Carlos Esteves, Yinshuang Xu, Christine Allen-Blanchette, and Kostas Daniilidis. Equivariant multi-view networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1568–1577, 2019. 1, 3

[13] Carlos Esteves, Ameesh Makadia, and Kostas Daniilidis. Spin-weighted spherical CNNs. Advances in Neural Information Processing Systems, 33:8614–8625, 2020. 3

[14] Cong Fang, Zhenyu Zhao, Pan Zhou, and Zhouchen Lin. Feature learning via partial differential equation with applications to face recognition. Pattern Recognition, 69:14–25, 2017. 2, 3, 4

[15] Mark Fels and Peter J Olver. Moving coframes: II. regularization and theoretical foundations. Acta Applicandae Mathematica, 55:127–208, 1999. 3

[16] Marc Finzi, Samuel Stanton, Pavel Izmailov, and Andrew Gordon Wilson. Generalizing convolutional neural networks for equivariance to Lie groups on arbitrary continuous data. In International Conference on Machine Learning, pages 3165–3176. PMLR, 2020. 1, 3

[17] Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. SE(3)-transformers: 3D roto-translation equivariant attention networks. Advances in Neural Information Processing Systems, 33:1970–1981, 2020. 3

[18] L Gao, G Lin, and W Zhu. Deformation robust roto-scaletranslation equivariant CNNs. Transactions on Machine Learning Research, 2022. 3, 7, 8

[19] Rohan Ghosh and Anupam K Gupta. Scale steerable filters for locally scale-invariant convolutional neural networks. arXiv preprint arXiv:1906.03861, 2019. 3, 7

[20] Simon Graham, David Epstein, and Nasir Rajpoot. Dense steerable filter CNNs for exploiting rotational symmetry in histology images. IEEE Transactions on Medical Imaging, 2020. 3

[21] Jindong Gu and Volker Tresp. Improving the robustness of capsule networks to image affine transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7285–7293, 2020. 8

[22] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge University Press, 2003. 3

[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. 5

[24] Lingshen He, Yuxuan Chen, Yiming Dong, Yisen Wang, Zhouchen Lin, et al. Efficient equivariant network. Advances in Neural Information Processing Systems, 34:5290–5302, 2021. 3

[25] Lingshen He, Yuxuan Chen, Zhengyang Shen, Yibo Yang, and Zhouchen Lin. Neural ePDOs: Spatially adaptive equivariant partial differential operator based networks. In International Conference on Learning Representations, 2022. 1, 3, 6

[26] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448–456. pmlr, 2015. 6

[27] Erik Jenner and Maurice Weiler. Steerable partial differential operators for equivariant neural networks. In International Conference on Learning Representations, 2021. 1, 3, 6

[28] Chiyu Max Jiang, Jingwei Huang, Karthik Kashinath, Philip Marcus, Matthias Niessner, et al. Spherical CNNs on unstructured grids. In International Conference on Learning Representations, 2018. 3

[29] Angjoo Kanazawa, Abhishek Sharma, and David Jacobs. Locally scale-invariant convolutional neural networks. arXiv preprint arXiv:1412.5104, 2014. 7

[30] Yann LeCun. The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/, 1998. 7

[31] Jan Eric Lenssen, Matthias Fey, and Pascal Libuschewski. Group equivariant capsule networks. Advances in Neural Information Processing Systems, 31, 2018. 3, 8

[32] Junying Li, Zichen Yang, Haifeng Liu, and Deng Cai. Deep rotation equivariant network. Neurocomputing, 2018. 3

[33] Risheng Liu, Zhouchen Lin, Wei Zhang, and Zhixun Su. Learning PDEs for image restoration via optimal control. In Proceedings of the European Conference on Computer Vision, pages 115–128. Springer, 2010. 1, 2, 3, 4

[34] Risheng Liu, Zhouchen Lin, Wei Zhang, Kewei Tang, and Zhixun Su. Toward designing intelligent PDEs for computer vision: an optimal control approach. Image and Vision Computing, 31(1):43–56, 2013. 1, 2, 3, 4

[35] Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. PDE-net: Learning PDEs from data. In International Conference on Machine Learning, pages 3208–3216. PMLR, 2018. 3

[36] Zichao Long, Yiping Lu, and Bin Dong. PDE-net 2.0: Learning PDEs from data with a numeric-symbolic hybrid deep network. Journal of Computational Physics, 399:108925, 2019. 3, 4

[37] Lachlan E MacDonald, Sameera Ramasinghe, and Simon Lucey. Enabling equivariance for arbitrary Lie groups. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8183–8192, 2022. 1, 2, 3, 8

[38] Diego Marcos, Benjamin Kellenberger, Sylvain Lobry, and Devis Tuia. Scale equivariance in CNNs with vector fields. arXiv preprint arXiv:1807.11783, 2018. 7, 8

[39] Mircea Mironenco and Patrick Forre. Lie group decomposi-´ tions for equivariant neural networks. In International Conference on Learning Representations, 2024. 1, 3

[40] Joseph L Mundy and Andrew Zisserman. Geometric invariance in computer vision. MIT Press, 1992. 3

[41] Hanieh Naderi, Leili Goli, and Shohreh Kasaei. Scale equivariant CNNs with scale steerable filters. In 2020 International Conference on Machine Vision and Image Processing (MVIP), pages 1–5. IEEE, 2020. 3, 7

[42] Peter J Olver. Applications of Lie groups to differential equations. Springer Science & Business Media, 1993. 2, 3, 4, 8

[43] Peter J Olver. Moving frames. Journal of Symbolic Computation, 36(3-4):501–512, 2003. 3

[44] Peter J Olver. Modern developments in the theory and applications of moving frames. London Math. Soc. Impact150 Stories, 1:14–50, 2015. 3

[45] Peter J Olver, Guillermo Sapiro, and Allen Tannenbaum. Affine invariant detection: edge maps, anisotropic diffusion, and active contours. Acta Applicandae Mathematica, 59:45– 77, 1999. 3

[46] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. Advances in Neural Information Processing Systems, 30, 2017. 8

[47] Mateus Sangalli, Samy Blusseau, Santiago Velasco-Forero, and Jesus Angulo. Differential invariants for SE(2)- ´ equivariant networks. In 2022 IEEE International Conference on Image Processing, pages 2216–2220. IEEE, 2022. 2, 3

[48] Mateus Sangalli, Samy Blusseau, Santiago Velasco-Forero, and Jesus Angulo. Scale equivariant U-Net. In 33rd British Machine Vision Conference, 2022. 3

[49] Mateus Sangalli, Samy Blusseau, Santiago Velasco-Forero, and Jesus Angulo. Moving frame net: SE(3)-equivariant network for volumes. In NeurIPS Workshop on Symmetry and Geometry in Neural Representations, pages 81–97. PMLR, 2023. 2, 3

[50] Zhengyang Shen, Lingshen He, Zhouchen Lin, and Jinwen Ma. PDO-eConvs: Partial differential operator based equivariant convolutions. In International Conference on Machine Learning, pages 8697–8706. PMLR, 2020. 1, 3

[51] Zhengyang Shen, Tiancheng Shen, Zhouchen Lin, and Jinwen Ma. PDO-eS2CNNs: Partial differential operator based equivariant spherical CNNs. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 9585– 9593, 2021. 3

[52] Zhengyang Shen, Tao Hong, Qi She, Jinwen Ma, and Zhouchen Lin. PDO-s3DCNNs: Partial differential operator based steerable 3D CNNs. In International Conference on Machine Learning, pages 19827–19846. PMLR, 2022. 3

[53] Ivan Sosnovik, Michał Szmaja, and Arnold Smeulders. Scale-equivariant steerable networks. In International Conference on Learning Representations, 2019. 3, 7, 8

[54] Ivan Sosnovik, Artem Moskalev, and Arnold Smeulders. How to transform kernels for scale-convolutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1092–1097, 2021. 3

[55] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014. 6

[56] Zikai Sun and Thierry Blu. Empowering networks with scale and rotation equivariance using a similarity convolution. In International Conference on Learning Representations, 2022. 3

[57] Stanley L Tuznik, Peter J Olver, and Allen Tannenbaum. Affine differential invariants for invariant feature point detection. arXiv preprint arXiv:1803.01669, 2018. 3

[58] Dian Wang, Robin Walters, Xupeng Zhu, and Robert Platt. Equivariant q learning in spatial action spaces. In Conference on Robot Learning, pages 1713–1723. PMLR, 2022. 3

[59] Rui Wang, Robin Walters, and Rose Yu. Incorporating symmetry into deep dynamics models for improved generalization. In International Conference on Learning Representations, 2020. 3

[60] Yuanbin Wang, Bin Zhang, and Tianshun Yao. Projective invariants of co-moments of 2D images. Pattern Recognition, 43(10):3233–3242, 2010. 3

[61] Yuanbin Wang, Xingwei Wang, Bin Zhang, et al. Affine differential invariants of functions on the plane. Journal of Applied Mathematics, 2013, 2013. 4

[62] Maurice Weiler and Gabriele Cesa. General E(2)-equivariant steerable CNNs. Advances in Neural Information Processing Systems, 32, 2019. 1, 3

[63] Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco S Cohen. 3D steerable CNNs: Learning rotationally equivariant features in volumetric data. Advances in Neural Information Processing Systems, 31, 2018. 1, 3

[64] Maurice Weiler, Fred A Hamprecht, and Martin Storath. Learning steerable filters for rotation equivariant CNNs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 849–858, 2018. 1, 3, 7, 8

[65] Ruben Wiersma, Elmar Eisemann, and Klaus Hildebrandt. CNNs on surfaces using rotation-equivariant features. ACM Transactions on Graphics, 39(4):92–1, 2020. 3

[66] Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, and Klaus Hildebrandt. DeltaConv: anisotropic operators for geometric deep learning on point clouds. ACM Transactions on Graphics, 41(4):1–10, 2022. 3

[67] Thomas Wimmer, Vladimir Golkov, Hoai Nam Dang, Moritz Zaiss, Andreas Maier, and Daniel Cremers. Scaleequivariant deep learning for 3D data. arXiv preprint arXiv:2304.05864, 2023. 3

[68] Marysia Winkels and Taco S Cohen. Pulmonary nodule detection in CT scans with equivariant CNNs. Medical Image Analysis, 2019. 1, 3

[69] Daniel Worrall and Gabriel Brostow. Cubenet: Equivariance to 3D rotation and translation. In Proceedings of the European Conference on Computer Vision, pages 567–584, 2018. 1, 3

[70] Daniel Worrall and Max Welling. Deep scale-spaces: Equivariance over scale. Advances in Neural Information Processing Systems, 32, 2019. 3, 7

[71] Daniel E Worrall, Stephan J Garbin, Daniyar Turmukhambetov, and Gabriel J Brostow. Harmonic networks: Deep translation and rotation equivariance. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. 3

[72] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. 7

[73] Wenju Xu, Guanghui Wang, Alan Sullivan, and Ziming Zhang. Towards learning affine-invariant representations via data-efficient CNNs. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 904–913, 2020. 3

[74] Yichong Xu, Tianjun Xiao, Jiaxing Zhang, Kuiyuan Yang, and Zheng Zhang. Scale-invariant convolutional neural networks. arXiv preprint arXiv:1411.6369, 2014. 7

[75] Yilong Yang, Srinandan Dasmahapatra, and Sasan Mahmoodi. Rotation-scale equivariant steerable filters. In Medical Imaging with Deep Learning, 2023. 3

[76] Linfeng Zhao, Xupeng Zhu, Lingzhi Kong, Robin Walters, and Lawson LS Wong. Integrating symmetry into differentiable planning with steerable convolutions. In International Conference on Learning Representations, 2022. 3

[77] Yunhan Zhao, Ye Tian, Charless Fowlkes, Wei Shen, and Alan Yuille. Resisting large data variations via introspective transformation network. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3080–3089, 2020. 3

[78] Zhenyu Zhao, Zhouchen Lin, and Yi Wu. A fast alternating time-splitting approach for learning partial differential equations. Neurocomputing, 185:171–182, 2016. 2, 3, 4

[79] Wei Zhu, Qiang Qiu, Robert Calderbank, Guillermo Sapiro, and Xiuyuan Cheng. Scaling-translation-equivariant networks with decomposed convolutional filters. The Journal of Machine Learning Research, 23(1):2958–3002, 2022. 3, 7, 8