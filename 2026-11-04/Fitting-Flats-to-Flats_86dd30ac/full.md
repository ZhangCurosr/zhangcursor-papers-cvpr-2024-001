# Fitting Flats to Flats

Gabriel Dogadov Ugo Finnendahl Marc Alexa TU Berlin, Computer Graphics Group

cg.tu-berlin.de

## Abstract

Affine subspaces of Euclidean spaces are also referred to as flats. A standard task in computer vision, or more generally in engineering and applied sciences, is fitting a flat to a set ofpoints, which is commonly solved using the PCA. We generalize this technique to enable fitting a flat to a set of other flats, possibly of varying dimensions, based on representing the flats as squared distance fields. Compared to previous approaches such as Riemannian centers of mass in the manifold of affine Grassmannians, our approach is conceptually much simpler and computationally more efficient, yet offers desirable properties such as respecting symmetries and being equivariant to rigid transformations, leading to more intuitive and useful results in practice. We demonstrate these claims in a number of synthetic experiments and a multi-view reconstruction task of line-like objects.

## 1. Introduction

Affine subspaces of Euclidean spaces, also called flats, are the basic entities in affine geometry, and are foundational building blocks for computations in computer vision, computer graphics, and generally all engineering disciplines. To motivate this work with a concrete example, notice that a point observed in a calibrated camera gives rise to a line in 3-space. Observing the same point in several cameras allows recovering its position by fitting a point to the collection of lines. Minimizing the squared distances to the lines only requires solving a linear system. Now imagine the features observed in the images are lines, giving rise to planes in each camera view. We would like to fit a line to the set of observed planes. Surprisingly, despite the problem appearing almost identical, there is no established ’standard’ solution to this problem.

Conversely, we often have to work with samples (observations) in some Euclidean space. The PCA [29] is arguably the standard way to fit a line or plane to the observation if they are points in space – but what do we do if the observations are lines or planes?

Approaching these problems in a systematic way commonly leads to Grassmannians, the manifold of linear subspaces. Similar to how homogeneous coordinates of Euclidean space lead to a representation of projective space, Grassmannians can be projectivized and then represent flats. The first and likely best-known example are Plucker¨ coordinates for affine lines in $\mathbb { R } ^ { 3 }$ [5, 16, 27]. One may argue that the resulting spaces are not strictly representing flats, as they also contain ideal elements, i.e., flats ’at infinity’. This is rectified in the explicit construction of a Grassmannian of affine subspaces [23]. While the manifold of the former construction is represented by the Klein quadric (and its generalizations), the latter is described by orthogonality conditions, a so-called Stiefel manifold (we review several representations of flats in Sec. 2). It has been argued [22] that computation on Stiefel manifolds is more convenient and better established [1, 2, 8]. Still, the mathematical sophistication and complexity of necessary computations for something as mundane as the mean of a set of affine lines or planes in 3D, let alone fitting flats to flats of different dimensions, is baffling when one compares it to the simplicity of fitting a point or fitting to points.

In this work, we provide a simple framework, based on representing flats as squared distance fields (see Sec. 2 for details on the representation and the relation to other representations). As we derive in Sec. 4, it allows fitting a flat of desired dimension k to a given set of flats of arbitrary and possibly varying dimension. Importantly, we show that it is equivariant under rigid transformations, the isometries of affine space. Unlike Plucker coordinates and related repre-¨ sentations, the representation is unoriented, which together with equivariance immediately implies that both angles as well as distances are bisected by least-squares fits to two flats. To our knowledge, this is the first method with these very natural properties. In addition, the necessary computations are similar to fitting flats to points with the PCA, with the most complex operation being the eigendecomposition of a symmetric PSD matrix. An interpretation of the procedure as projected means in ambient space leads to extensions to $L _ { p }$ means, enabling more robust fitting in the presence of outliers using the $L _ { 1 }$ -norm.

We evaluate the properties of our approach in comparison to Riemannian means in the space of affine Grassmannians on synthetic examples in low dimension (Sec. 5). We verify the predicted properties of both classes of methods. In addition, we demonstrate the method at the example of reconstructing line-like objects from multiple views.

The basic approach leaves much room for applying more advanced statistical methods, some of which we discuss in Sec. 6.

## 2. Background: Representations of flats

In the following, we assume the ambient space is $\mathbb { R } ^ { d }$ (unless otherwise noted). A k-flat is a k-dimensional affine subspace of the ambient space. In contrast, a k-plane is a linear subspace. Every k-plane is a k-flat, but a k-flat is a k-plane only if it passes through the origin. For $d = 3 ,$ flats can be points $~ ( k ~ = ~ 0 )$ , lines $\mathbf { \Phi } ( k \mathbf { \theta } ) = \mathbf { \Phi } 1 )$ , or planes $( k = 2 )$ . A (d  1)-flat in $\mathbb { R } ^ { d }$ is called hyperplane. The set of all k-planes in $\mathbb { R } ^ { d }$ is the Grassmannian of linear subspaces $\operatorname { G r } ( k , d )$ , a smooth compact manifold of dimension $k ( d - k )$ Similarly, the set of all k-flats is called the Grassmannian of affine subspaces or short the affine Grassmannian Graf(k, d) [19], a smooth $( k + 1 ) ( d - k )$ dimensional manifold. While the Grassmannian, representations, and computational methods are well established, the affine Grassmannian is a more recent development [22, 23], but created interest for applications such as image classification [30] or LiDAR registration [24].

The orthogonal complement of a k-plane $\mathcal { P } \subseteq \mathbb { R } ^ { d }$ is a linear subspace denoted $\mathcal { \overline { { P ^ { \perp } } } }$ and contains all vectors that are orthogonal to $\mathcal { P }$ . The dimension of $\mathcal { P } ^ { \perp }$ is the co-dimension $\bar { k } = d - k$ of .

In the following, we describe several representations of flats, i.e., elements of Graf $( k , d )$ , needed for the ensuing discussion of existing and new techniques for least squares fitting flats to flats.

Parameter form A matrix $\mathbf { A } \in \mathbb { R } ^ { d \times k }$ with full column rank represents a linear k-space in $\mathbb { R } ^ { d }$ . A flat $\mathcal { F }$ can be represented by additionally specifying a displacement b $\in$ $\mathbb { R } ^ { \dot { d } }$

$$
{ \mathcal { F } } = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } : \mathbf { x } = \mathbf { A } \mathbf { y } + \mathbf { b } , \mathbf { y } \in \mathbb { R } ^ { k } \right\} .\tag{1}
$$

We ask that the columns of A are orthonormal and that $\mathbf { A } ^ { \mathsf { T } } \mathbf { b } = \mathbf { 0 }$ . This implies that b is the vector to the point closest to the origin and is unique. Under this condition, the representation $\mathbf { y }$ in Eq. (1) has been referred to as orthogonal affine coordinates [22, 23]. We may arbitrarily change the basis of these coordinates with an orthogonal transformation $\mathbf { O } \in \mathbb { R } ^ { k \times k }$ without affecting the flat. So $( \mathbf { A O } , \mathbf { b } )$ $\mathbf { O } ^ { \mathsf { T } } \mathbf { O } = \mathbf { I }$ describes the set of equivalent representations of $\mathcal { F }$

Stiefel and Grassmann-Plucker coordinates¨ The affine Grassmannian in parameter form can be embedded into a (standard) Grassmannian one dimension higher, similar to Euclidean points being represented as lines in homogeneous coordinates: given a k-flat $( \mathbf { A } , \mathbf { b } )$ , it is treated as a $k + 1 -$ plane in $\mathbb { R } ^ { d + \bar { 1 } }$ spanned by $( \mathring { \mathbf { A } } ^ { \top } , \mathbf { 0 } ) ^ { \top }$ and $( \mathbf { b } ^ { \mathsf { T } } , 1 ) ^ { \mathsf { T } }$ . Normalizing the last basis vector leads to the (homogeneous) Stiefel coordinates [11, 22] for a k-flat ${ \mathcal { F } } \mathrm { i }$

$$
\mathbf { Y } = \left[ \mathbf { A } \quad \mathbf { b } / { \sqrt { 1 + \| \mathbf { b } \| ^ { 2 } } } \right] .\tag{2}
$$

This representation of a flat still admits orthogonal transformations that map span(Y) into span(Y). To reduce the degrees of freedom, one may represent the parallelotope spanned by the basis of this linear space as the exterior product of the basis vectors. This construction is known as the Plucker embedding¨ [27]. Intuitively, one may think of the representation as the signed (hyper)areas of the shadows of the parallelotope onto all k-planes spanned by the axes of a fixed coordinate system. These (hyper)areas are independent of the choice of the vectors $\mathbf { A } ^ { \prime }$ spanning the parallelotope, with the only degree of freedom being the total signed volume of the parallelotope. This suggests that the resulting representation consisting of $\binom { d } { k }$ elements is unique up to scale for a given flat. In general, $\binom { d } { k } > ( k + 1 ) ( d - k )$ , so only a subset of the coordinates corresponds to flats. This subset is described as the intersection of quadratic surfaces, the Grassmann-Plucker relations¨ . The particular case of affine lines (k=1) in $( d { = } 3 ) { \mathrm { - s p a c e } }$ of this representation is well known as Plucker coordinates¨ <sup>1</sup>.

Normal form We may also represent the flat $\mathcal { F }$ as the i<sub>ntersect</sub>i<sub>on o</sub>f ¯k h<sub>yperp</sub>l<sub>anes, eac</sub>h d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> $\mathbf { n } _ { i } ^ { \mathsf { T } } \mathbf { x } = c _ { i }$ Writing the normal vectors ${ \bf n } _ { i }$ as rows of a matrix $\textbf { N } \in$ $\mathbb { R } ^ { \bar { k } \times d }$ and the offsets as a vector $\mathbf { c } \in \mathbb { R } ^ { \bar { k } }$ , the flat is represented as:

$$
\mathcal { F } = \left. \mathbf { x } \in \mathbb { R } ^ { d } : \mathbf { N x } = \mathbf { c } \right. .\tag{3}
$$

Note that the row space of N is the orthogonal complement of the column space of A in $\mathbb { R } ^ { d }$ . As before, we ask that the basis is chosen orthonormal. Under this condition, the vector c contains the signed distances of the hyperplanes to the origin (see Fig. 1 (left) for an illustration). Note that equality in Eq. (3) is preserved for any change of basis $\mathbf { O } \in \bar { \mathbb { R } ^ { k \times \bar { k } } }$ and that the rows of ON remain orthonormal if O is orthogonal. Thus, in normal form, the set (ON, Oc) describes the same flat ${ \mathcal F } .$

![](images/aff7b3cbff7b530e3350986566dc927d0b07bb0465464cd50fca5dd8183c9f3c.jpg)  
Figure 1. Left: The squared distance between a point (black) and a line (red) corresponds to the sum of squared distances to two orthogonal hyperplanes (blue and green) whose intersection is the line. Right: The connection between the squared distance field q of a line in $\mathbb { R } ^ { 2 }$ , its parametric representation $( \mathbf { a } _ { 1 } , \mathbf { b } )$ and normal representation $\mathbf { \Psi } ( \mathbf { n } _ { 1 } , c _ { 1 } )$ .

Squared distance function The normal representation of $\mathcal { F }$ yields the distances to the hyperplanes for any $\mathbf { x } \in \mathbb { R } ^ { d }$ as $\mathbf { N x } - \mathbf { c }$ . We can get the squared distance by computing the the squared norm of this vector:

$$
\begin{array} { r } { d _ { \mathcal { F } } ^ { 2 } ( \mathbf { x } ) = ( \mathbf { N } \mathbf { x } - \mathbf { c } ) ^ { \mathsf { T } } ( \mathbf { N } \mathbf { x } - \mathbf { c } ) = \mathbf { x } ^ { \mathsf { T } } \mathbf { N } ^ { \mathsf { T } } \mathbf { N } \mathbf { x } - 2 \mathbf { c } ^ { \mathsf { T } } \mathbf { N } \mathbf { x } + \mathbf { c } ^ { \mathsf { T } } \mathbf { c } . } \end{array}\tag{4}
$$

Note that the squared distance field is unaffected by starting from a different orthogonal normal frame ON and corresponding distance vector Oc: the resulting distances are ${ \bf O } ( { \bf N x - c } )$ and the inner product $\mathbf { O } ^ { \mathsf { T } } \mathbf { O } = \bar { \mathbf { I } }$ cancels. This means the symmetric positive semi-definite (PSD) matrix $\mathbf { Q } = \mathbf { N } ^ { \mathsf { T } } \mathbf { N }$ , the vector $\mathbf { r } = - \mathbf { N } ^ { \mathsf { T } } \mathbf { c }$ and the scalar $s = \mathbf { c } ^ { \mathsf { T } } \mathbf { c }$ uniquely describe a k-flat as

$$
\mathcal { F } = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } : d _ { \mathcal { F } } ^ { 2 } ( \mathbf { x } ) = \mathbf { x } ^ { \mathsf { T } } \mathbf { Q } \mathbf { x } + 2 \mathbf { r } ^ { \mathsf { T } } \mathbf { x } + s = 0 \right\}\tag{5}
$$

Any non-zero scalar multiple of the triple $( \mathbf { Q } , \mathbf { r } , s )$ of the squared distance field has the same zero-set, so we may interpret this representation as a homogeneous coordinate for k-flats. Similar to Plucker-Grassmann coordinates, only a¨ subset of symmetric matrices $\mathbf { Q } ,$ vectors r, and scalars s correspond to k-flats. Note that Q contains the bases encoded by A and N as eigenspaces corresponding to the eigenvalues 0 and 1. So Q is characterized by the spectrum consisting of the eigenvalues 0 with multiplicity k and 1 with multiplicity <sup>¯</sup>k.

The vector r has been constructed $\mathbf { a s } - \mathbf { N } ^ { \mathsf { T } } \mathbf { c } .$ , so it is a linear combination of the normals and has to be an eigenvector of Q with eigenvalue 1.

Since Q is PSD, the zero set is equivalent to the local minima of the squared distance field and could be computed by setting the gradient $2 \mathbf { Q } \mathbf { x } + 2 \mathbf { r }$ to zero. This allows recovering s from $( \mathbf { Q } , \mathbf { r } )$ by asking that the squared distances evaluate to zero for the minima, yielding

$$
s = \mathbf { r } ^ { \mathsf { T } } \mathbf { Q } ^ { + } \mathbf { r } = \mathbf { r } ^ { \mathsf { T } } \mathbf { Q } \mathbf { r } = \mathbf { r } ^ { \mathsf { T } } \mathbf { r } ,\tag{6}
$$

where we have exploited that $\mathbf { Q }$ is its own pesudo-inverse under the assumptions on its spectrum and that r is an eigenvector with eigenvalue one. Since s is redundant, we can represent flats by the pair $( \mathbf { Q } , \mathbf { r } )$ . We summarize its relation to the standard parameter form $( \mathbf { A } , \mathbf { b } )$ below:

$$
( \mathbf { Q } , \mathbf { r } ) = ( \mathbf { I } - \mathbf { A } \mathbf { A } ^ { \mathsf { T } } , - \mathbf { b } ) \qquad ( \mathbf { A } , \mathbf { b } ) = ( \mathrm { k e r } ( \mathbf { Q } ) , - \mathbf { r } ) .\tag{7}
$$

Fig. 1 (right) illustrates the connection to both standard and normal forms.

## 3. Related work: Riemannian centers of mass

Recall that the Grassmannian is a smooth compact manifold. Let $\{ \mathcal { P } _ { i } \}$ represent k-planes, then least-squares fitting a k-plane to this data implies

$$
m = \underset { \mathcal { P } \in \mathrm { G r } ( k , d ) } { \arg \operatorname* { m i n } } \sum _ { i } d ^ { 2 } ( \mathcal { P } , \mathcal { P } _ { i } ) ,\tag{8}
$$

where d is a metric. In fact, this is a way of computing a mean in Riemannian manifolds and has been aptly termed Riemannian center of mass [17], now often referred to as Karcher mean. Note that by embedding $\mathrm { G r a f f } ( k , d )$ into $\operatorname { G r } ( k + 1 , d + 1 )$ , we can use the same approach for fitting a k-flat to given k-flats. In the following, we will first explain the metric we have found to be commonly used in our context, and then how to extend the basic ideas to linear subspaces of varying dimensions.

Principal angles and metric The concept of angles between two lines (in the plane or in space), two planes or a line and a plane in space can be generalized to flats of any dimension. Given a k-plane represented by ${ \bf A } _ { 1 }$ and an lplane represented by ${ \bf A } _ { 2 }$ and assuming $k \leq l ,$ , the principal angles are a set of k mutual angles $0 \leq \theta _ { 1 } \leq . . . \leq \theta _ { k } \leq \frac \pi 2$ that are defined recursively as:

$$
\theta _ { i } : = \operatorname* { m i n } \left\{ \cos ^ { - 1 } \left( \frac { | \mathbf { u } ^ { \mathsf { T } } \mathbf { v } | } { \| \mathbf { u } \| \| \mathbf { v } \| } \right) \begin{array} { c c } { \left| \mathbf { u } \in \mathrm { s p a n } ( \mathbf { A } _ { 1 } ) , \right.} & { \mathbf { u } ^ { \mathsf { T } } \mathbf { u } _ { j } = 0 , } \\ { \mathbf { v } \in \mathrm { s p a n } ( \mathbf { A } _ { 2 } ) , } & { \mathbf { v } ^ { \mathsf { T } } \mathbf { v } _ { j } = 0 , } \\ { \forall j \in \{ 1 , . . . , i - 1 \} } & \end{array}  \right\} .\tag{9}
$$

The vectors $\mathbf { \Pi } ( \mathbf { u } _ { i } , \mathbf { v } _ { i } )$ forming the principal angles are the principal vectors. The principal angles and vectors can be computed via a (reduced) Singular Value Decomposition (SVD) [4]. When decomposing $\mathbf { A } _ { 1 } ^ { \mathsf { T } } \mathbf { A } _ { 2 } \mathbf { \Psi } = \mathbf { \Psi } \mathbf { U } \mathbf { \Sigma } \mathbf { V } ^ { \mathsf { T } }$ with $\Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { k } )$ , the principal vectors are the columns of $\mathbf { A } _ { 1 } \mathbf { U }$ and $\mathbf { A } _ { 2 } \mathbf { V }$ , respectively, and the corresponding principal angles are given by $\theta _ { i } = \cos ^ { - 1 } ( \sigma _ { i } )$ A commonly used metric on ${ \mathrm { G r } } ( k , d )$ constructed from the principal angles is $\textstyle \left( \sum _ { i = 1 } ^ { k } \theta _ { i } ^ { 2 } \right) ^ { 1 / 2 }$

While one can compute principal angles between flats in the very same way, the angles lack information about the displacement between the flats if they are not intersecting (e.g., consider two skew lines in 3-space). The distance due to rotation of the linear subspaces and translation of the points closest to the origin are automatically reconciled by using the Stiefel coordinates introduced above. The resulting principal angles in this embedding have been referred to as affine principal angles [22]. Moreover, optimization as necessary here for computing the mean is well-understood in Stiefel manifolds [2], and we detail the computation of means next.

Computing the mean in Stiefel coordinates Let flats ${ \mathcal { F } } _ { i }$ be given and represented in Stiefel coordinates $\mathbf { Y } _ { i }$ . For Stiefel coordinate Y, the gradient of the sum of squared distances in Eq. (8) is given by [18]

$$
- \sum _ { i = 1 } ^ { m } \exp _ { \mathbf { Y } } ^ { - 1 } ( { \mathbf { Y } } _ { i } ) ,\tag{10}
$$

with $\exp _ { \mathbf { Y } } ^ { - 1 } ( \mathbf { X } )$ denoting the derivative of the geodesic that connects Y and X. As the affine principal angles, similar to principal angles for planes as in Eq. (9), are also defined for flats of different dimensions, geodesic distances between flats of different dimensions and their gradient can be computed in a similar manner as described in [22, 36]. The gradient can be exploited to minimize the sum of squared geodesic distances in an iterative minimization scheme such as gradient descent or Newton’s method on Stiefel manifolds [1, 2, 8]. This essentially requires reorthogonalization of Y after each descent step.

In our implementation, we used a variable step size gradient descent scheme but avoided computing the Hessian. For further information on the gradient computation, our adaptation to flats of different dimensions, and the orthogonalization scheme we found to yield the best results, we refer to the supplementary material.

## 4. Method

Given a set of flats <sub>i</sub> , we want to fit a flat $\hat { \mathcal { F } }$ with fixed dimension k in the least-squares sense. We start by revisiting the case of fitting a point, i.e., the case $k = 0$ , and then extend the approach to $k > 0$

Fitting a point The sum of squared distances to flats from an arbitrary point x can be written as the sum of the squared distance fields to the flats:

11)

$$
\begin{array} { l } { { \displaystyle { \boldsymbol \sigma } ^ { 2 } ( { \bf x } ) = \sum _ { i } d _ { { \mathcal F } _ { i } } ^ { 2 } ( { \bf x } ^ { * } ) } \ ~ ( \mathbf { x } ^ { * } ) } \\ { { \displaystyle ~ = { \bf x } ^ { \mathsf { T } } \left( \sum _ { i } \mathbf { Q } _ { i } \right) { \bf x } + 2 \left( \sum _ { i } \mathbf { r } _ { i } \right) { \bf x } + \sum _ { i } s _ { i } } \ ~ ( \mathbf { \Sigma } } \\ { { \displaystyle ~ = { \bf x } ^ { \mathsf { T } } \mathbf { Q } ^ { * } { \bf x } + 2 { \bf r } ^ { * } { \bf x } + s ^ { * } } . } \end{array}\tag{12}
$$

Algorithm 1: Closest Flat (Mean-SDF)   
Input : Flats (of arbitrary dimension) as squared   
distance fields $( \mathbf { Q } _ { 1 } , \mathbf { r } _ { 1 } ) , \ldots , ( \mathbf { Q } _ { m } , \mathbf { r } _ { m } )$   
Output: k-Flat in parameter form $( \mathbf { A } , \mathbf { b } )$   
begin   
$\mathbf { Q } ^ { * }  \textstyle \sum _ { i = 1 } ^ { m } \mathbf { Q } _ { i }$   
$\mathbf { r } ^ { * } \gets \sum _ { i = 1 } ^ { m } \mathbf { r } _ { i }$   
$\mathbf { U D U } ^ { \mathsf { T } } \gets \mathbf { Q } ^ { * } / /$ Eigendecomposition,   
see Eq. 14 for conventions   
$\mathbf { A }  [ \mathbf { u } _ { 1 } , \dotsc , \mathbf { u } _ { k } ] / /$ Eq. 15   
$\mathbf { Q ^ { * } } ^ { + }  \mathbf { U } \mathbf { D } ^ { + } \mathbf { U } ^ { \top }$   
b $\mathbf { \Omega } )  ( \mathbf { I } - \mathbf { A } \mathbf { A } ^ { \mathsf { T } } ) \mathbf { Q } ^ { * ^ { + } } \mathbf { r } ^ { * }$   
end

The minimal value is attained at a critical point and can be found by setting the gradient to zero, yielding the necessary condition $\mathbf { Q } ^ { * } \hat { \mathbf { x } } = \mathbf { r } ^ { * }$ . If this sum of squared distances has a unique minimum, the linear system has a unique solution.

Notice that simply adding the representations of the flats is quite convenient, particularly in applications where flats are being added dynamically: this is commonly exploited in computer graphics for modeling with piecewise planar surfaces. By associating squared distances with the planar polygons, it is easy to measure changes to the shape relative to the original shape, most prominently used in surface simplification [10]. Conversely, the same concept can be used for fitting planes to sample points [38].

Fitting discretely sampled flats Now let us consider the case $k = 1$ , an affine line. Naturally, one would ask that the sum of squared distances over all points on the line are minimized. This leads to an indefinite integral that takes on a finite value only in degenerate cases. It is illuminating, however, to start with a finite number m of samples of the line: represent the line in parameter form as ya+b with $\left\| \mathbf { a } \right\| = 1$ and sample it in parametric locations $\{ y _ { j } \}$ . We assume that these locations are mean-unbiased, i.e. $\textstyle \sum _ { j } y _ { j } = 0$ . The sum of squared distances summed up over all samples is

$$
\sum y _ { j } ^ { 2 } \mathbf { a } ^ { \mathsf { T } } \mathbf { Q } ^ { * } \mathbf { a } + m \mathbf { b } ^ { \mathsf { T } } \mathbf { Q } ^ { * } \mathbf { b } + 2 m \mathbf { b } ^ { \mathsf { T } } \mathbf { r } ^ { * } + m s ^ { * } ,\tag{13}
$$

where we have already removed all terms with the factor $\textstyle \sum _ { j } y _ { j } = 0$ . We notice that we can independently optimize for a and b. For a we are looking to minimize $c \mathbf { a } ^ { \mathsf { T } } \mathbf { Q } \mathbf { a }$ with positive $c = \textstyle \sum _ { j } y _ { j } ^ { 2 }$ subject to $\left\| \mathbf { a } \right\| = 1$ . This means a is the eigenvector of $\breve { \mathbf { Q } } ^ { \ast }$ corresponding to the smallest eigenvalue. The positive constant c has no consequences. For b we find that it has to satisfy $\mathbf { Q } ^ { * } \mathbf { b } = \mathbf { r } ^ { * }$ (regardless of the sampling $\{ y _ { j } \} )$ . The resulting pair $( \mathbf { a } , \mathbf { b } )$ describes the desired line, but is not yet in standard form, because b is not necessarily orthogonal to a. This can be achieved by replacing b with $\mathbf { b } - ( \mathbf { a } ^ { \mathsf { T } } \mathbf { b } ) \mathbf { a }$

It is straightforward to extend this analysis to k-flats for $k > 1$ . We find that the basis A has to be chosen as the eigenvectors corresponding to the smallest k eigenvalues of $\mathbf { Q } ^ { * }$ . To make this concrete, let

$$
\begin{array} { r l } & { \mathbf { Q } ^ { * } = \mathbf { U D U } ^ { \mathsf { T } } , \quad \mathbf { U } ^ { \mathsf { T } } \mathbf { U } = \mathbf { I } , } \\ & { \quad \mathbf { D } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } ) , \quad 0 \leq \lambda _ { 0 } \leq \ldots \leq \lambda _ { d - 1 } } \end{array}\tag{14}
$$

then A consists of the first k columns of $\mathbf { U } \mathbf { : }$

$$
\mathbf { U } = [ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { d } ] \quad \Longrightarrow \quad \mathbf { A } = [ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { k } ]\tag{15}
$$

The condition $\mathbf { Q } ^ { * } \mathbf { b } = \mathbf { r } ^ { * }$ is independent of k and bringing it into standard parameter form leads to

$$
\mathbf { b } = ( \mathbf { I } - \mathbf { A } \mathbf { A } ^ { \mathsf { T } } ) \mathbf { Q } ^ { * ^ { + } } \mathbf { r } ^ { * } .\tag{16}
$$

Note that this solution is independent of the number of samples and their distribution. In fact, it extends to any weighting of the point samples as long as the weighted mean is still zero. With appropriate weighting, we could make sure that the constant c remains finite for infinite sampling, including a dense sampling of the line. This means the choice for $( \mathbf { A } , \mathbf { b } )$ described in Eqs. (15) and (16) solve the fitting problem for flats in the least-squares sense. Alg. 1 summarizes the described procedure in pseudocode.

Complexity Assuming we start from flats given in parameter form $( \mathbf { A } _ { i } , \mathbf { b } _ { i } )$ building the matrices $\mathbf { Q } _ { i } = \mathbf { I } - \mathbf { A } \mathbf { A } ^ { \mathsf { T } }$ is $\mathcal { O } ( k d ^ { 2 } )$ for each flat. Summing up $\mathbf { Q } ^ { * }$ and $\mathbf { r } ^ { * }$ is linear in the number of flats and the number of coefficients in the matrices and vectors. The most complex operation is the necessary eigendecomposition of $\mathbf { Q } ^ { * }$ , which is $\mathcal { O } ( d ^ { 3 } )$ We implement this using known closed-form solutions for $d = 2 , 3 [ 7 ]$ and a symmetric QR algorithm [12] for $d > 3$ Overall, the complexity of fitting a k-flat in this way to a set of arbitrary flats is identical to the complexity of fitting flats to points using the PCA. This appears to be quite natural, yet it is remarkable that existing solutions (to our knowledge) are significantly more involved and much slower in practice (see also Sec. 5).

Uniqueness and degeneracies All steps in the above fitting procedure are uniquely determined, except for selecting the eigenvectors A corresponding to the smallest k eigenvalues of $\mathbf { Q } ^ { * }$ . This step assumes that $\lambda _ { k }$ is strictly smaller than $\lambda _ { k + 1 }$

The dependence of the spectrum of the sum of matrices on the spectra of the summands is generally quite involved [9, 20], but it is worth pointing out some special cases in our context: (1) If all input flats are points, then $\mathbf { Q } ^ { * }$ also represents a point, i.e., all its eigenvalues are identical. So while our approach works for arbitrary k-flats, it fails for the special case $k = 0$ for all flats (which is exactly what the PCA solves). (2) Two orthogonal lines in $\mathbb { R } ^ { 2 }$ , three mutually orthogonal planes in $\mathbb { R } ^ { 3 }$ , or more generally d mutually orthogonal $d - 1$ flats in $\mathbb { R } ^ { d }$ result in an isotropic squared distance field, so all eigenvalues of $\mathbf { Q } ^ { * }$ are identical. (3) Similar statements can be made for subspaces, leading to parts of the spectrum being isotropic. For example, two non-intersecting lines in $\mathbb { R } ^ { 3 }$ with orthogonal directions uniquely define a plane, but every line in the plane spanned by the directions is an equally good fit.

Equivariance It seems very natural to ask that least squares fitting is isometry invariant. The isometries of affine spaces are rigid transformations and reflections. This means we expect that if the input is rigidly transformed and reflected, the fitted flat undergoes the same transformation – it is equivariant to these transformations.

This property is not difficult to show for the procedure above. Instead of $\mathbf { x } ,$ we plug in the transformed point $\mathbf { R } \mathbf { x } + \mathbf { t } ,$ , where R is an orthogonal transformation in $\mathbb { R } ^ { d }$ and $\mathbf { t } \in \mathbb { R } ^ { d }$ a translation. What we find is that in the sum of squared distances Eq. (13), the term $c \mathbf { a } ^ { \mathsf { T } } \mathbf { Q } ^ { * } \mathbf { a }$ transforms into $\mathbf { \bar { \rho } } _ { c \mathbf { a } } \mathsf { T } \mathbf { R } ^ { \mathsf { T } } \mathbf { Q } ^ { \ast } \mathbf { R } \mathbf { a } .$ , indicating that the direction vector a has to be rotated by $\mathbf { R } ,$ as expected. For the terms involving b we get $( \mathbf { b } + \mathbf { t } ) ^ { \mathsf { T } } \mathbf { Q } ^ { * } ( \mathbf { b } + \mathbf { \bar { t } } ) + 2 ( \mathbf { b } + \mathbf { t } ) ^ { \mathsf { T } } \mathbf { r } ^ { * }$ and setting the gradient w.r.t. b to zero yields $\mathbf { Q } ^ { * } ( \mathbf { b } + \mathbf { t } ) = - \mathbf { r } ^ { * }$ , again, as expected. Note that the standard form for b as computed in Eq. (16) is not necessarily translated by t, as the projection onto A is removed.

It seems worth pointing out that, as natural as this property may appear, the Riemannian center on the affine Grassmannian varies with the translation of the coordinate system. We demonstrate this with examples in the supplementary material.

Elementary properties At least for two k-flats, we can formulate what we expect for their mean: if the two flats intersect, the mean should be the bisector; and if they are parallel, the mean should be parallel and have the same distance to both of them.

In fact, both properties directly follow from the fact that the computations are symmetric in the inputs and equivariant to translations and reflections. To see this, translate the flats so that they are symmetric w.r.t. the origin. Then reflection at the origin has no effect on the input, so the output has to respect the symmetry as well.

Interpretation as projected mean Consider the fitting as starting from a set of flats represented by the pairs $( \mathbf { Q } _ { i } , \mathbf { r } _ { i } )$ and generating the mean in this form as $( \hat { \mathbf { Q } } , \hat { \mathbf { r } } )$ . In this setting, we may interpret the procedure as: first, compute the mean in the ambient space of symmetric PSD matrices and Euclidean vectors and, second, project onto the manifold of matrices and vectors representing k-flats.

Clearly, $( \mathbf { Q } ^ { * } , \mathbf { r } ^ { * } )$ arise from the set of $\mathbf { Q } _ { i } , \mathbf { r } _ { i }$ as a mean (modulo irrelevant scale factors). Given $\mathbf { Q } ^ { * }$ , we constructed A based on the eigenvector corresponding to the smallest eigenvectors. This suggests that the normal space N contains the eigenvectors corresponding to the largest eigenvectors. Or, in other words, the fitted flat is represented by $\hat { \mathbf { Q } }$ with the same eigenspace as $\mathbf { Q } ^ { * }$ , but the $k$ smallest eigenvalues mapped to zero and the remaining eigenvalues mapped to one. Recall the decomposition of $\mathbf { Q } ^ { * }$ in Eq. (14), then we get

$$
\begin{array} { r } { \hat { \mathbf { Q } } = \mathbf { U } \mathrm { d i a g } ( \underbrace { 0 , \dots , 0 } _ { k \mathrm { t i m e s } } , 1 , \dots , 1 ) \mathbf { U } ^ { \mathsf { T } } . } \end{array}\tag{17}
$$

And ˆr is related to b in Eq. (16) simply by ${ \hat { \mathbf { r } } } = - \mathbf { b } =$ $- \hat { \mathbf { Q } } \mathbf { Q } ^ { * ^ { + } } \mathbf { r } ^ { * }$

We want to show that this mapping is an orthogonal projection. For the PSD matrix, we claim that any unitarily invariant norm in the space of symmetric PSD matrices is minimized. To see this, consider minimizing $\| \mathbf { Q } ^ { * } - \mathbf { X } \|$ among symmetric matrices $\mathbf { X }$ having the desired spectrum $( \mathbf { 0 } _ { k } ^ { \mathsf { T } } , \bar { \mathbf { 1 } } _ { k } ^ { \mathsf { T } } )$ . If X has this spectrum then so does $\mathbf { X } ^ { \prime } = \mathbf { \bar { U } X U } ^ { \mathsf { T } }$ so we get the equivalent minimization problem

$$
\operatorname { a r g m i n } _ { \boldsymbol { \lambda } ( \mathbf { X } ^ { \prime } ) = ( \mathbf { 0 } _ { k } ^ { \intercal } , \mathbf { 1 } _ { \overline { { k } } } ^ { \intercal } ) } \Vert \mathbf { D } - \mathbf { X } ^ { \prime } \Vert , \quad \mathbf { D } = \mathbf { U } ^ { \intercal } \mathbf { Q } ^ { * } \mathbf { U } .\tag{18}
$$

Mirsky [28] has shown that for any matrix the norm of the difference is not smaller than the norm difference of the sorted singular values. For symmetric PSD matrices, the singular values are the eigenvalues. So we have

$$
\begin{array} { r } { \| \mathbf { D } - \mathbf { X } ^ { \prime } \| \geq \| \mathbf { D } - \mathrm { d i a g } ( \mathbf { 0 } _ { k } ^ { \mathsf { T } } , \mathbf { 1 } _ { \overline { { k } } } ^ { \mathsf { T } } ) \| , } \end{array}\tag{19}
$$

showing that the best solution for $\mathbf { X } ^ { \prime }$ is the diagonal matrix, and we find $\mathbf { X } = \mathbf { U } ^ { \mathsf { T } } \mathbf { X } ^ { \prime } \mathbf { U }$ as constructed.

For ˆr, we see that it minimizes $\| \hat { \mathbf { r } } + \hat { \mathbf { Q } } \mathbf { Q } ^ { * ^ { + } } \mathbf { r } ^ { * } \|$

Means in other norms The interpretation of the leastsquares fitting as first computing the $L _ { \mathrm { { 2 } } } { \mathrm { - m e a n } }$ in the space of matrix and vector coefficients of the representation $( \mathbf { Q } , \mathbf { r } )$ and then projecting back onto the manifold of $k \mathrm { - }$ flats suggest a simple approximation of other means: we can compute any $L _ { p }$ mean in the space of the matrix and vector computations and then project. This reduces the problem of computing $L _ { p }$ means to the well-understood problem of doing so in Euclidean spaces.

We particularly consider the case of the $L _ { \mathrm { 1 } } \mathrm { - n o r m } , \mathrm { o r }$ geometric median, as it is known to be robust against outliers. For this, we use Weiszfeld’s algorithm [34], which is essentially an iterative re-weighted least squares method. As we show in Sec. 5, it indeed shows robustness to outliers, and is not only significantly simpler to implement than Riemannian $L _ { p }$ centers of mass [3], but also preserves the equivariance properties that are missing from the methods based on Stiefel coordinates.

## 5. Experiments

We demonstrate in the following experiments that our method is both faster and yields more useful results compared to optimization in the affine Grassmannian [22], and provide a brief experiment in a realistic application scenario for reconstructing line-like objects from more than two views. All results reported are based on implementations in C++ using the Eigen library [15] for numerical linear algebra. Running times were gathered on a computer with an Intel i5-13600K CPU and 32GB RAM.

Data generation We randomly generate a target k-flat ${ \mathcal { F } } ^ { * }$ in standard form $( \mathbf { A } ^ { * } , \mathbf { b } ^ { * } )$ . Then, we generate m flats of dimension l by sampling $n \geq d$ random points $\mathbf { x } _ { i } ~ =$ ${ \bf b } ^ { * } + { \bf A } ^ { * } { \bf y } _ { i } + \xi _ { i }$ in a sample interval of $\mathbf { y } _ { i } \in [ - 1 0 , 1 0 ] ^ { k }$ and displacing them with additive, zero-mean Gaussian noise $\xi _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ . Subsequently, PCA is used to fit an l-flat to the n sampled points. The noise on the sample points is intended to simulate measurement and calibration errors from the real world. This procedure is repeated until m flats $\{ \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { m } \}$ are generated as observations to reconstruct ${ \mathcal { F } } ^ { * }$

Iterative approach on the Grassmannian As a comparison to our method, we optimize the objective in Eq. (8) using gradient descent on the Stiefel manifold. We use a combination of the Frobenius norm of the gradient, the difference between two consecutive iterates, and the number of iterations as the stopping criteria.

Our method We use our method from Eq. (15) and Eq. (16) in two variations. In the first one, we compute $\mathbf { Q } ^ { * }$ and $\mathbf { r } ^ { * }$ as the arithmetic mean of the respective matrices $\left\{ \mathbf { Q } _ { i } \right\}$ and $\{ { \bf { r } } _ { i } \}$ (Mean-SDF). In the second one, we compute $\mathbf { Q } ^ { * }$ and $\mathbf { r } ^ { * }$ as the median of the respective matrices using Weiszfeld’s algorithm, where we use the Frobenius norm for computing $\mathbf { Q } ^ { * }$ and the Euclidean norm for $\mathbf { r } ^ { * }$ (Median-SDF). Although the projection is identical to the previous case, with this method, we may get more stable results in the presence of outliers.

Metrics To compare the reconstructed flat ${ \mathcal { F } } ^ { \prime }$ to the original ${ \mathcal { F } } * .$ , we use their principal angle(s) $\theta _ { 1 } , \ldots , \theta _ { k }$ and their least-squares distance [14]

$$
d _ { \operatorname* { m i n } } ( \mathcal { F } ^ { * } , \mathcal { F } ^ { \prime } ) = \operatorname* { m i n } _ { \mathbf { x } ^ { * } \in \mathcal { F } ^ { * } , \mathbf { x } ^ { \prime } \in \mathcal { F } ^ { \prime } } \| \mathbf { x } ^ { * } - \mathbf { x } ^ { \prime } \|\tag{20}
$$

as separate metrics.

Efficiency Fig. 2 shows the average running time of our method and the method from [22] over ten trials on a log scale. For each dimension of the ambient space $d \in$ $\{ 4 , \ldots , 2 0 \}$ , three different experiments were conducted: (1) reconstructing a line from other lines $( k _ { \mathrm { i n } } = k _ { \mathrm { o u t } } = 1 )$ (2) reconstructing a line from hyperplanes $( k _ { \mathrm { i n } } ~ = ~ d - 1$ $k _ { \mathrm { o u t } } = 1 )$ , and lastly (3) reconstructing a hyperplane from hyperplanes $( k _ { \mathrm { i n } } = k _ { \mathrm { o u t } } = d - 1 )$ . In every case, we chose the number of input flats as $m = d .$ A limit of 200 iterations was set for the iterative optimization. It is to be expected that our method is significantly faster than the iterative computation of the Riemannian center, and our results clearly show that this is the case. The running time of our method is similar for all cases irrespective of the dimension of the input and output flats, which is, to some degree, expected, as our method mainly involves the diagonalization of a d d matrix whose dimensionality only depends on the dimension of the ambient space. We note that the dimension of the ambient space d is likely to be less pronounced for a larger number of samples $m ,$ as then the assembly of $\mathbf { Q } ^ { * } , \mathbf { r } ^ { * }$ might dominate the computation.

![](images/1e01dedb2dc72428f3bc1f77c8972fc25472d65999dccf4177712a8f2c3bcb39.jpg)  
Figure 2. Running time comparison between our methods and the iterative optimization on the affine Grassmannian (Graff) for different dimensions of flats and ambient space but constant input size m = 20 and noise level σ = 0.2.

![](images/b5ee4867c225b5f2efad45afc988c3d70061da01807ffe6f18b2eccc21e5a303.jpg)  
Figure 3. Comparison between the mean and the median estimation of $\mathbf { Q } ^ { \ast }$ and r<sup>∗</sup> using our method. A line (red) that is the intersection of four planes (yellow) is reconstructed while having a fifth plane as an outlier (purple). On the left, only the displacement of the outlier is off. On the right, both orientation and displacement are off. The Mean-SDF method yields the blue line, while the green line is the output of the Median-SDF method.

Accuracy Tab. 1 shows distances of a reconstructed affine line from planes in $\mathbb { R } ^ { 3 }$ . While we are aware that, in this particular scenario, other techniques could have been used, it still serves to illustrate the difference between our method and computations in the affine Grassmannian, which we have found to be consistent across dimensions of flats and ambient dimension.

Table 1. The minimum distance $d _ { \mathrm { m i n } }$ and (principal) angle θ between a target line $\mathcal { F } ^ { * }$ and its reconstruction $\mathcal { F } ^ { \prime }$ in $\mathbb { R } ^ { 3 }$ from m = 20 planes, averaged over ten trials. Top: varying noise levels σ with no outliers. Bottom: varying percentage of outliers with constant noise level $\sigma = 0 . 2$ . The best result for each condition is highlighted in bold, the second best is underlined.
<table><tr><td rowspan="2"></td><td colspan="2">Graff</td><td colspan="2">Mean-SDF</td><td colspan="2">Median-SDF</td></tr><tr><td> $\downarrow \ : d _ { \mathrm { m i n } }$ </td><td>↓θ</td><td> $\downarrow \ : d _ { \mathrm { m i n } }$ </td><td>↓θ</td><td> $\downarrow \ : d _ { \mathrm { m i n } }$ </td><td>↓θ</td></tr><tr><td>Noise σ</td><td colspan="6"></td></tr><tr><td>0.5</td><td>0.2174</td><td>0.0402</td><td>0.1292</td><td>0.0381</td><td>0.1192</td><td>0.0434</td></tr><tr><td>1.0</td><td>0.3411</td><td>0.1112</td><td>0.2417</td><td>0.0913</td><td>0.1870</td><td>0.0817</td></tr><tr><td>1.5</td><td>0.3953</td><td>0.3240</td><td>0.3297</td><td>0.1357</td><td>0.2697</td><td>0.1252</td></tr><tr><td>2.0</td><td>0.4167</td><td>0.3914</td><td>0.4843</td><td>0.1045</td><td>0.2854</td><td>0.1095</td></tr><tr><td>2.5</td><td>0.5946</td><td>0.4817</td><td>0.4599</td><td>0.1374</td><td>0.3200</td><td>0.1339</td></tr><tr><td colspan="7">Outlier %</td></tr><tr><td>0.0</td><td>0.2325</td><td>0.0212</td><td>0.0227</td><td>0.0190</td><td>0.0929</td><td>0.0177</td></tr><tr><td>0.1</td><td>0.1801</td><td>0.0242</td><td>2.6684</td><td>0.0438</td><td>0.1274</td><td>0.0423</td></tr><tr><td>0.2</td><td>0.1892</td><td>0.0189</td><td>6.6070</td><td>0.0696</td><td>0.0919</td><td>0.0518</td></tr><tr><td>0.3</td><td>0.2944</td><td>0.0387</td><td>8.2614</td><td>0.0951</td><td>0.1650</td><td>0.0822</td></tr><tr><td>0.4</td><td>0.2891</td><td>0.0325</td><td>12.4167</td><td>0.1880</td><td>0.2195</td><td>0.1764</td></tr><tr><td>0.5</td><td>0.3917</td><td>0.0568</td><td>15.0596</td><td>0.2536</td><td>0.3857</td><td>0.2232</td></tr></table>

The upper half shows results for experiments with only Gaussian noise added. For the remaining rows in Tab. 1, we have added outliers to the data (see Fig. 3). It is evident that our Mean-SDF variant is not robust to outliers, especially in terms of displacement, in which the results appear to be worse than the iterative method. Yet the Median-SDF variant recovers from this loss of robustness, especially in terms of the translation distance, even with a high proportion of outliers, and provides the best overall results. The difference between our two variants is visualized in Fig. 3.

Application: Archery A possible application is line reconstruction in multiple-view geometry. Imagine a line-like object being extracted in more than 2 registered cameras. For each camera, the line in screen space creates an instance of a plane in world-space, containing the line in 3-space. These planes are expected to intersect in the common line, but due to noise in calibration and registration of the camera as well as in the feature extraction in the discrete images this will not be the case. The standard approach to this problem is to represent the planes as a matrix M by stacking the normal representation $\mathbf { N } _ { i }$ (see Sec. 2) in rows and apply the SVD to find the best rank-2 approximation with respect to the Frobenius norm. The kernel of the resulting rank-reduced matrix represents the line. We note that the properties of this approach in the presence of noise are not clear, because the normal representation is not unique and, as a consequence, it is not clear what is being minimized. In our method, it is clear that the least-squares distance of the reconstructed line to the planes is being minimized, as we showed that this happens for any sampling of the liner.

![](images/47ed12052e51f631bd624c83773c6d943ff52ddb7e719566fc7129b21543d1c4.jpg)  
Figure 4. Multiple views of an archery target. Given the camera parameters, the re-projected arrows form planes in world space that do not intersect due to detection and calibration errors. Our method can be used to find a good representative. The original arrow is colored in red, given a bad calibrated camera our result is colored in blue and the rank minimization approach [16] is colored in green.

We compare the two approaches using a synthetic archery example, where we try to reconstruct the arrows in 3D space. A scene is rendered from four different perspectives and the cameras are calibrated with moderate noise being introduced. We then ’shoot’ random arrows onto the target and detect the features in the image spaces of the cameras. As expected, the results for the rankminimization [16] depends on the choice of representation of the planes. Fig. 4 shows the results for a typical case, comparing to our results. While our method naturally yields consistent reconstruction very close to ground truth, rankminimization is significantly off in some cases.

## 6. Discussion

The method for least-squared fitting of flats to flats based on squared distance functions, as simple as it is, exhibits various nice properties that are not present in other, seemingly more principled approaches such as the manifold of affine Grassmannians. We are unaware of any construction that offers the desired equivariance to the isometries of affine space for means or fitting procedures of flats in arbitrary dimensions (without shifting the input to compensate for the missing equivariance to translations [24]). We briefly comment below on a possible similar construction using Grassmann-Plucker coordinates, and then men-¨ tion some further use cases and investigations.

Interpolation in Grassmann-Plucker coordinates¨ Since Plucker coordinates are unique up to scale, one might¨ want to try, similar to the interpretation of our approach, computing the mean of the coefficients and then projecting back onto the space described by the Grassmann-Plucker¨ relations.

Taking the mean of the coefficient requires selecting an appropriate scale. In general, this might be difficult [21] and typically introduces bias for noisy data. In our context, it is not difficult to see that taking the mean leads to consistent results if the basis taken on the flat is orthonormal, i.e., if the coefficients are constructed from the Stiefel coordinates as described in Sec. 2. This method still suffers from several drawbacks: (1) Grassmann-Plucker coor-¨ dinates represent oriented spaces [31]. This means the line constructed from point b and direction a is not the same as the one constructed from b and a. Taking the mean depends on the choice of orientation. (2) The ’projection’ onto coefficients satisfying the Grassmann-Plucker relations is,¨ in general, significantly more involved than computing the eigendecomposition [13]. (3) The Grassmann-Plucker co-¨ ordinates naturally arise as antisymmetric rank-k tensors. It is unclear (to us) how one could perform computations involving different dimensions, although admittedly Schubert varieties may be used [26], similar to the construction for affine Grassmannians.

Outlook The PCA fitting flats to points may be considered the simplest statistical analysis of a set of point-like observations. This can be extended to fitting several flats such as in clustering methods or, more generally, describing data using mixture models [6, 32, 33, 37]. All of these methods may be generalized to work with flats as the observed samples, using the methods we have introduced here.

As described, our method can be re-interpreted as taking the mean in ambient space and projecting, and in this way used similar to methods in Euclidean spaces. This suggests other methods for improving robustness to outliers could be used, as well as considering the L<sub>∞</sub>-norm, whose minimization yields the center of the smallest enclosing ball [35]. This approach would be considerably simpler than computing geodesic disks enclosing a set of flats [25].

## Acknowledgements

This work was funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (Grant agreement No. 101055448, ERC Advanced Grand EMERGE).

## References

[1] Pierre-Antoine Absil, Robert Mahony, and Rodolphe Sepulchre. Riemannian geometry of grassmann manifolds with a view on algorithmic computation. Acta Applicandae Mathematica, 80:199–220, 2004. 1, 4

[2] Pierre-Antoine Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds. Princeton University Press, 2008. 1, 4

[3] Bijan Afsari. Riemannian lp center of mass: existence, uniqueness, and convexity. Proceedings of the American Mathematical Society, 139(2):655–673, 2011. 6

[4] Ake Bj<sup>˚</sup> orck and Gene H Golub. Numerical methods for¨ computing angles between linear subspaces. Mathematics ofComputation, 27(123):579–594, 1973. 3

[5] James F. Blinn. A homogeneous formulation for lines in 3 space. SIGGRAPH Comput. Graph., 11(2):237–241, jul 1977. 1

[6] Paul S Bradley and Olvi L Mangasarian. K-plane clustering. Journal ofGlobal optimization, 16:23–32, 2000. 8

[7] Charles-Alban Deledalle, Loic Denis, Sonia Tabti, and Florence Tupin. Closed-form expressions of the eigen decomposition of $2 \times 2$ and 3 x 3 hermitian matrices. Technical report, 2017. 5

[8] Alan Edelman, Tomas A Arias, and Steven T Smith. The ge-´ ometry of algorithms with orthogonality constraints. SIAM journal on Matrix Analysis and Applications, 20(2):303– 353, 1998. 1, 4

[9] William Fulton. Eigenvalues of sums of hermitian matrices. Seminaire Bourbaki´ , 40:255–269, 1998. 5

[10] Michael Garland and Paul S Heckbert. Surface simplification using quadric error metrics. In SIGGRAPH ’97, pages 209– 216, 1997. 4

[11] Israel M. Gelfand, Mikhail M. Kapranov, and Andrei V. Zelevinsky. Discriminants, Resultants, and Multidimensional Determinants. Birkhauser Boston, 1994.¨ 2

[12] Gene H Golub and Charles F Van Loan. Matrix computations. JHU press, 2013. 5

[13] P. Griffiths and J. Harris. Principles ofAlgebraic Geometry. Wiley Classics Library. Wiley, 2014. 8

[14] Jurgen Gross and G¨ otz Trenkler. On the least squares dis-¨ tance between affine subspaces. Linear Algebra and its Applications, 237-238:269–276, 1996. 6

[15] Gael Guennebaud, Beno ¨ ˆıt Jacob, et al. Eigen v3. http://eigen.tuxfamily.org, 2010. 6

[16] Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, USA, 2 edition, 2003. 1, 8

[17] Hermann Karcher. Riemannian center of mass and mollifier smoothing. Communications on pure and applied mathematics, 30(5):509–541, 1977. 3

[18] Hermann Karcher. Riemannian center of mass and so called karcher mean, 2014. 4

[19] Daniel A Klain and Gian-Carlo Rota. Introduction to geometric probability. Cambridge University Press, 1997. 2

[20] Allen Knutson and Terence Tao. Honeycombs and sums of hermitian matrices. Notices Amer. Math. Soc, 48(2), 2001. 5

[21] Vincent Lesueur and Vincent Nozick. Least square for grassmann-cayley agelbra in homogeneous coordinates. In Fay Huang and Akihiro Sugimoto, editors, Image and

Video Technology – PSIVT 2013 Workshops, pages 133–144, Berlin, Heidelberg, 2014. Springer Berlin Heidelberg. 8

[22] Lek-Heng Lim, Ken Sze-Wai Wong, and Ke Ye. Numerical algorithms on the affine grassmannian. SIAM Journal on Matrix Analysis and Applications, 40(2):371–393, 2019. 1, 2, 4, 6

[23] Lek-Heng Lim, Ken Sze-Wai Wong, and Ke Ye. The grassmannian of affine subspaces. Foundations ofComputational Mathematics, 21(2):537–574, 2021. 1, 2

[24] Parker C. Lusk, Devarth Parikh, and Jonathan P. How. Graffmatch: Global matching of 3d lines and planes for wide baseline lidar registration. IEEE Robotics and Automation Letters, 8(2):632–639, 2023. 2, 8

[25] Tim Marrinan, P-A Absil, and Nicolas Gillis. On a minimum enclosing ball of a collection of linear subspaces. Linear Algebra and its Applications, 625:248–278, 2021. 8

[26] Ezra Miller and Bernd Sturmfels. Matrix schubert varieties. In Combinatorial Commutative Algebra, pages 289– 310. Springer New York, New York, NY, 2005. 8

[27] Ezra Miller and Bernd Sturmfels. Plucker coordinates.¨ In Combinatorial Commutative Algebra, pages 273–288. Springer New York, New York, NY, 2005. 1, 2

[28] L. Mirsky. Symmetric gauge functions and unitarily invariant norms. The Quarterly Journal ofMathematics, 11(1):50–59, 01 1960. 6

[29] Karl Pearson. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559–572, 1901. 1

[30] Krishan Sharma and Renu Rameshan. Image set classification using a distance-based kernel over affine grassmann manifold. IEEE Transactions on Neural Networks and Learning Systems, 32(3):1082–1095, 2021. 2

[31] Jorge Stolfi. Oriented projective geometry: a frameworkfor geometric computations. PhD thesis, Stanford University, 1995. 8

[32] Paul Tseng. Nearest q-flat to m points. Journal of Optimization Theory and Applications, 105:249–252, 2000. 8

[33] Rene Vidal, Yi Ma, and Shankar Sastry. Generalized principal component analysis (gpca). IEEE Trans. Pattern Anal. Mach. Intell., 27(12):1945–1959, 2005. 8

[34] Endre Weiszfeld and Frank Plastria. On the point for which the sum of the distances to n given points is minimum. Annals ofOperations Research, 167(1):7–41, 2009. 6

[35] Emo Welzl. Smallest enclosing disks (balls and ellipsoids). In New Results and New Trends in Computer Science: Graz, Austria, June 20–21, 1991 Proceedings, pages 359–370. Springer, 2005. 8

[36] Ke Ye and Lek-Heng Lim. Schubert varieties and distances between subspaces of different dimensions. SIAM Journal on Matrix Analysis andApplications, 37(3):1176–1197, 2016. 4

[37] Teng Zhang, Arthur Szlam, and Gilad Lerman. Median kflats for hybrid linear modeling with many outliers. In 12th International Conference on Computer Vision Workshops, ICCV Workshops, pages 234–241. IEEE, 2009. 8

[38] Tong Zhao, Laurent Buse, David Cohen-Steiner, Tamy´ Boubekeur, Jean-Marc Thiery, and Pierre Alliez. Variational shape reconstruction via quadric error metrics. In ACM SIG-GRAPH 2023 Conference Proc., pages 1–10, 2023. 4