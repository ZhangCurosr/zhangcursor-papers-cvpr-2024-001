# Continuous Pose for Monocular Cameras in Neural Implicit Representation

Qi Ma<sup>1,2</sup> Danda Pani Paudel<sup>2</sup> Ajad Chhatkuli<sup>1</sup> Luc Van Gool<sup>1,2</sup> <sup>1</sup>Computer Vision Lab, ETH Zurich <sup>2</sup>INSAIT, Sofia University

## Abstract

In this paper, we showcase the effectiveness ofoptimizing monocular camera poses as a continuous function of time. The camera poses are represented using an implicit neural function which maps the given time to the corresponding camera pose. The mapped camera poses are then usedfor the downstream tasks wherejoint camera pose optimization is also required. While doing so, the network parameters – that implicitly represent camera poses – are optimized. We exploit the proposed method in four diverse experimental settings, namely, (1) NeRFfrom noisy poses; (2) NeRFfrom asynchronous Events; (3) Visual Simultaneous Localization and Mapping (vSLAM); and (4) vSLAM with IMUs. In all four settings, the proposed method performs significantly better than the compared baselines and the state-of-the-art methods. Additionally, using the assumption of continuous motion, changes in pose may actually live in a manifold that has lower than 6 degrees offreedom (DOF) is realized. We call this low DOF motion representation as the intrinsic motion and use the approach in vSLAM settings, showing impressive camera tracking performance. We release our code at: https://github.com/qimaqi/Continuous-Pose-in-NeRF.

## 1. Introduction

The concept of motion, the change of position and orientation of an object in its surroundings, is fundamentally continuous in nature. This continuity is evident in the ways we achieve, perceive and measure motion, with velocity and acceleration being the most common measures for both linear and angular motion. This idea of continuity is also true for the 3D poses of navigating cameras. Often the camera motion needs to be estimated from its measurements – also known as the camera localization problem. In most common settings, the inputs are RGB-only frames, depth frames, asynchronous event streams, or a combination thereof. In some cases, these measurements are augmented by Inertial Measurement Unit (IMU) outputs, which measure a change in pose directly. In all those settings, the camera motion is estimated via some optimization technique that searches $S E ( 3 )$ pose parameters. While doing so, existing techniques choose to optimize a discrete set of SE(3) parameters, ignoring the inter-frame continuity of camera poses. This choice can be primarily attributed to the otherwise difficulty in optimization.

While handling high-frequency IMUs or asynchronous events in common practice, pose optimization at every measurement time is avoided, for computational reasons. Instead, the measurements between two arbitrarily chosen keyframes are accumulated before utilizing them. Then the poses are optimized only for those keyframes. We argue that this raises three major concerns: (i) inaccurate accumulation of intermediate measurements; (ii) loss of fine-grained motion details; (iii) lack of the continuous motion prior.

In order to address these concerns, we represent and optimize the pose of a moving camera as a continuous function of time. Unlike classical state estimation method [2, 12, 35] which models continuous pose with Gaussian Process or B-spline, our neural pose function can be easily optimized jointly with other task-specific implicit neural representation (INR) [30, 32, 39]. More precisely, for translation $\textsf { v } \in \ \mathbb { R } ^ { 3 }$ and rotation $\mathsf { R } ( \mathsf { q } ) \ \in \ S O ( 3 )$ parameterized by quaternions $\mathsf { q } \in \mathbb { R } ^ { 4 }$ with $| | \mathsf { q } | | = 1$ , the continuous pose of the monocular camera is given by,

$$
[ \mathsf { q } ; \mathsf { v } ] = f _ { \theta } ( t ) ,\tag{1}
$$

where $f _ { \theta } ( . )$ is the contineous neural function parameterized by θ that maps the time $t \in \mathbb { R }$ to the pose in $S E ( 3 )$ . While being simple, this representation has numerous benefits including ease of optimization and its cosmopolitan applicability. Some example applications are illustrated in Figure 1. In the following, we further discuss how our simple approach addresses the previously raised concerns.

No error due to measurement accumulation: High frequency or asynchronous measurements can be utilized directly without accumulation, integral, or rounding. We infer the camera pose precisely at the measurement time. For example, in the case of an event camera, each asynchronous event’s pose is inferred precisely at the event time. Similarly, in the case of IMUs, no motion integration before supervision is required. These abilities protect our approach against error injection due to any form of accumulation.

![](images/1b67630ac099c59a21efce65ed783cfefc0ed26fe6282d7afcdb61fd5e23ce8a.jpg)  
Figure 1. We showcase the benefits of optimizing the poses as a continuous function of time in diverse settings. We conduct exhaustive experiments on (a) rectifying inaccurate poses in RGB-only settings; (b) utilizing the asynchronous stream of events, (c) performing vSLAM in RGB-D camera settings; (d) integrating high-frequency IMUs in vSLAM. All experiments use neural functions for both camera poses and scene representations. Additionaly we exploit low dof motion representation in intrinsic motion frame $T _ { I }$

Fine-grained motion details: By virtue of the continuous representation, temporally fine-grained details of the pose can be captured. This is particularly interesting with high-frequency IMUs or asynchronous event cameras. Our approach allows for the recovery of the pose at the very moment of measurement, which otherwise often is an ill-posed problem and could only be interpolated with an assumed smoothness and order.

Continuous motion prior: The inductive bias of continuous monocular camera motion is meaningfully injected by the proposed method. This resulted in very encouraging results in our experiments. In particular, while denoising the inaccurate camera poses and during the vSLAM experiments, the benefits were evident under the standard settings of BARF [27] and NICE-SLAM [63], respectively. It is important to note that our representation offers first- and second-order derivatives via auto-differentiation of the neural network. Consequently, quantities such as velocity and acceleration do not require additional care. Thus the fusion of IMU measurements is natural and straightforward.

In addition to the above, we further show the utility of the neural pose in order to optimize the continuous pose by decomposing each change in pose into a slowly changing reference and a low DOF motion. We define this as the intrinsic motion. In our experiments we observed that our continuous pose representation improves the tracking performance significantly in the vSLAM tasks. This can be primarily attributed to the reasons mentioned above, which serve to facilitate the optimization process. Inspired by the fact that actual motion always possesses a lower degree of freedom, we define the intrinsic motion frame as a coordinate system that can express the camera motion with the lowest dimensional manifold. For example: Rotational motion around a fixed axis can be expressed in the coordinate system aligned with the rotational axis with only one degree of freedom. A natural observation is that the relative motion with respect to intrinsic motion is usually sparse, moreover, the continuous motion tends to share the same intrinsic motion frame which can be well modeled as a continuous function of time. By exploiting it we decompose the camera relative motion with a low-dimensional intrinsic motion $[ \mathsf { R } _ { I } , \mathsf { v } _ { I } ]$ and the rigid transformation from camera frame to the intrinsic motion frame $[ \mathsf { R } _ { o } , \mathsf { v } _ { o } ]$ as follows:

$$
[ \mathsf { R } , \mathsf { v } ] = [ \mathsf { R } _ { o } , \mathsf { v } _ { o } ] [ \mathsf { R } _ { I } , \mathsf { v } _ { I } ] ,\tag{2}
$$

Our major contributions can be summarized as follows:

• We propose a simple yet effective way to represent the monocular camera motion via a neural function of time that can be optimized efficiently together with implicit neural representations.

• We demonstrate the utility of the proposed representation in four diverse applications with different camera setups, including IMUs and moving event cameras.

• Through exhaustive experiments, we demonstrate clear benefits of the proposed representation over the existing alternatives and classical method. These benefits include ease of optimization, widespread use for different camera and sensor types, and notable performance gain with no additional effort.

• We further improve the full 6-DOF pose of monocular camera by exploiting the sparsity of the intrinsic motion, which fits neatly into the proposed framework of continuous neural pose. The final pose thus obtained shows remarkable improvement over the conventional baselines.

## 2. Related work

## 2.1. Camera Poses in NERF

NERF [32] consists ofjoint optimization of the surface density and the rendered color given the images with known camera rays. Consequently NERF models are highly sensitive to camera pose errors [8, 27, 29, 55, 58, 59]. Recently several works have tackled the pose error by jointly optimizing poses with the radiance field. [6, 7, 19, 27] optimizes camera poses in bundle adjustment fashion in order to solve the same issue. While these methods use the smooth pose prior, the poses are still optimized as discrete variables. On the other spectrum [55] optimizes noisy poses for sparse camera views with the radiance fields opting for a different class of applications. [3] enforces the inter-frame consistency by incorporating monocular depth prior.

## 2.2. Camera Poses with IMUs

The inertial measurement unit (IMU) serves as a sceneindependent sensor that is the ideal complement to cameras in order to achieve robustness in low texture, high speed, and HDR scenarios. Fusing visual information and IMU tightly [48] to estimate pose as discrete states is proposed first by MSCKF [34] (an extended Kalman Filter (EKF)), [25] further improves it with keyframes and bundle adjustment. [11, 17, 41, 56] improve in robustness compared to feature matching by using the direct photometric error. [5] propose fast and accurate IMU initialization based on MAP estimation. Recent research has also focused on integrating IMU and visual priors with neural network, e.g., the camera pose is implicitly used for image deblurring [37] or video stabilization [49]. [14] proposes neural inertial localization with IMU alone for indoor scenes.

## 2.3. Camera Poses in Dense SLAM

Visual SLAM [10, 23, 43] is a key 3D vision application where an agent camera is localized simultaneously while building the map using visual information. We again focus on methods in the context of radiance fields [1, 44, 53, 63, 64]. IMAP [53] is a recent seminal work which works on RGBD images to optimize an implicit scene representation with a single MLP network. It optimizes the camera pose while representing them as discrete sets of parameters for the keyframes. NICE-SLAM [63] improves on it by using 3D voxel features along with corresponding 2D image features thus providing a better scene representation. Indeed most approaches [26, 44, 47, 64] focus on improving the scene representation for better localization and mapping or with RGB-only input.

## 2.4. Camera Poses in Event Cameras

Unlike standard frame-based camera imaging, event cameras provide image signals as asynchronous events in microsecond intervals [22]. Thus, it forms the perfect use case for a continuous time representation of camera poses. Similar to NERF-less SLAM [10], this is traditionally done using variations of Kalman Filter with motion models [22, 33]. A recent work [62] represents camera tracking as a function of time but uses a Levenberg-Marquardt optimization directly on the sets of poses without intermediate representation. Recently, there have been efforts to use eventbased radiance fields in the neural network [18, 24, 28, 46]. However, camera pose optimization as a function of time is still not fully explored in the radiance field literature with events.

## 2.5. Continuous Pose representation

While discrete-time representations are commonly employed in Simultaneous Localization and Mapping (SLAM) tasks, they face challenges when integrating data from highfrequency sensors like Inertial Measurement Units (IMUs) and asynchronous events. [12] address this issue by proposing representing the continuous-time state using temporal basis functions such as B-spline basis.[2] model the continuous state using Gaussian processes, defining continuoustime priors through covariance functions.[40] leverage cumulative cubic B-splines to mitigate rolling-shutter artifacts. Notably, spline-based continuous-time trajectory representations have found application in laser-based SLAM methods [21, 38].

## 3. Time-to-Pose Mapping Network

## 3.1. Architecture of the Proposed PoseNet

In order to learn time-to-pose mapping, we use 8-layer MLP parameterized by $f ( \theta _ { p } )$ with ReLU activation functions and 256-dimensional hidden units, which we refer to as posenetwork (PoseNet). PoseNet first embeds the time variable into high-dimensional space using sinusoidal harmonic functions [32]. The outputs of this network are [v, q]: translation vector $ \textsf { v } \in \mathbb { R } ^ { 3 }$ and the rotation represented by a quaternion $\mathsf { q } \in \mathbb { R } ^ { 4 }$ . Finally, we use the tanh activation in the last layer to map output to the range [−1, 1], and normalize it as a unit quaternion. We study different embedding dimensions and architectures in the context of NeRF from the inaccurate pose, which is reported in Tables 3. The best-performing embedding and architecture, in these experiments, are then used for the other applications. Additional information concerning network size and computational details is provided in the supplementary material.

## 3.2. Implementation Variances across Applications

The simplicity of PoseNet allows us to use it in diverse applications in a plug-and-play manner. In all applications that we report in the following sections, we optimize the PoseNet parameters $\theta _ { p }$ as a surrogate of the direct pose optimization. We denote the network parameters for the INR of the scene as $\theta _ { s }$ . In NeRF from inaccurate poses, the objective is to minimize the radiance field loss [27, 32] given N images and corresponding timestamp $t _ { i }$ for image i:

$$
\operatorname* { m i n } _ { \theta _ { s } , \theta _ { p } } \sum _ { i = 1 } ^ { N } \lVert \mathcal { T } _ { i } - g \left( \theta _ { s } , f ( \theta _ { p } , t _ { i } ) \right) \rVert .\tag{3}
$$

$g ( \theta _ { s } , T _ { i } )$ represents the mapping from the camera pose $T _ { i }$ to the RGB value, including ray composition and radiance field model. In case of NeRF with asynchronous events [46], N refers to the number of the sampled events. Note that in both cases, we output the predicted transformation and compose it with the initial pose: $\begin{array} { r l } { T _ { i } } & { { } = } \end{array}$ $T _ { i n i t _ { i } } \circ T _ { r e f i n e _ { i } }$ The refined transformation is obtained as $T _ { r e f i n e _ { i } } \stackrel { \cdot } { = } P ( f ( \theta _ { p } , t _ { i } ) )$ , $P ( . )$ being the vector to rigid transformation conversion operator.

In the task of Dense-SLAM tracking, for each tracking iteration we optimize PoseNet with the following objective:

$$
\operatorname* { m i n } _ { \theta _ { p } } \sum _ { i = 1 } ^ { M } ( \mathcal { L } _ { g } ( D _ { i } , P ( f ( \theta _ { p } , t _ { i } ) ) ) + \lambda _ { p } \mathcal { L } _ { p } ( I _ { i } , P ( f ( \theta _ { p } , t _ { i } ) ) ) ) .\tag{4}
$$

We use the same geometric loss $\mathcal { L } _ { g }$ and photometric loss ${ \mathcal { L } } _ { p }$ as in NICE-SLAM [63]. $D _ { i } , I _ { i }$ represent depth and RGB measurements for M sampled pixels respectively, obtained via volume rendering.

## 3.3. Intrinsic Motion Frame

Within the neural dense SLAM application, we additionally introduce intrinsic motion frame in order to improve tracking within a low-dimensional manifold. This is accomplished through motion decomposition and enforcing minimal DOF. More specifically, we use two PoseNet $f _ { o } ( \theta _ { p _ { o } } ) , ~ f _ { I } ( \theta _ { p _ { I } } )$ in order to model the intrinsic motion $T _ { o } \doteq [ \mathsf { R } _ { o } , \mathsf { v } _ { o } ] , \mathsf { \bar { T } } _ { I } = [ \mathsf { R } _ { I } , \mathsf { v } _ { I } ]$ , such that $T = T _ { o } \circ T _ { I }$ . Here, $T _ { o }$ is the transformation to the intrinsic frame or in short, intrinsic transform. $T _ { I }$ then denotes the intrinsic motion.

Therefore we can rewrite Eq (4) as:

$$
\begin{array} { l } { \displaystyle \underset { \theta _ { p _ { o } } , \theta _ { p _ { I } } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { M } ( \mathcal { L } _ { g } ( D _ { j } , f _ { o } ( \theta _ { p _ { o } } , t _ { i } ) \circ f _ { I } ( \theta _ { p _ { I } } , t _ { i } ) ) } \\ { \displaystyle + \mathcal { L } _ { p } \left( I _ { j } , f _ { o } ( \theta _ { p _ { o } } , t _ { i } ) \circ f _ { I } ( \theta _ { p _ { I } } , t _ { i } ) \right) } \\ { \displaystyle + \mathcal { L } _ { d o f } \left( f _ { I } ( \theta _ { p _ { I } } , t _ { i } ) \right) + \mathcal { L } _ { o } ( f _ { o } ( \theta _ { p _ { o } } , t _ { i } ) ) . } \end{array}\tag{5}
$$

Note that the operator P should be included for absolute correctness in the function compositions in Eq (5). The DOF loss $\mathcal { L } _ { d o f }$ is computed as follows:

• Step1: Obtain $[ \mathsf { R } _ { I } , \mathsf { v } _ { I } ]$ from intrinsic motion PoseNet $f _ { I }$ • Step2: Convert rotation matrix $\mathsf { R } _ { I }$ to Euler angles $\alpha _ { I } \in$ $\mathbb { R } ^ { 3 }$ , normalize with angle of view $\gamma , \hat { \alpha _ { I } } = 2 \alpha _ { I } / \gamma$

• Step3: Normalize translation vector with $\hat { v _ { I } } = v _ { I } / \| v _ { I } \|$

• Step4: DOF Loss $\mathcal { L } _ { d o f } = \| [ \hat { \alpha _ { I } } , \hat { v _ { I } } ] \| _ { 0 }$

We relax the $\ell _ { 0 }$ norm to $\ell _ { 1 }$ norm for optimization. In steps 2 and 3, normalization also serves to balance translation and rotation components during optimization. We employ view angle normalization with the assumption that the angle between two relative views in vSLAM tasks is always smaller than half of the viewing angle. To handle the cases where unconstrained intrinsic motion tends move to infinity in cases of small rotation, we introduce an additional L1 regularization term for the translation $\mathcal { L } _ { o } = \vert v _ { o } \vert$

## 3.4. IMU fusion

Up to our knowledge we are the first to integrate IMU data in NeRF + SLAM setting. The IMU fusion is straightforward in PoseNet taking advantage of the autodifferentiation of the neural network. We propose two different IMU fusion methods with details as follows:

Loose coupling. Given 3-axis angular velocity measurement from gyroscope $\hat { \omega } = ( \hat { \omega } _ { x } , \hat { \omega } _ { y } , \hat { \omega } _ { z } )$ we get the time step from frequency $\textstyle \triangle t = { \frac { 1 } { f } }$ . We can express the rotation angle to be $\triangle t { \| \hat { \boldsymbol { \omega } } \| }$ around axis $\frac { \hat { \omega } } { | | \hat { \omega } | | } \left[ 5 0 \right]$ . This instantaneous rotation from the local sensor between previous and current timestamp can be represented as follows:

$$
\mathsf { q } _ { \triangle } = \mathsf { q } \left( \triangle t \| \hat { \omega } \| , \frac { \hat { \omega } } { \| \hat { \omega } \| } \right) .\tag{6}
$$

By continuously integrating the measurements we can get the rotation estimation at time $t _ { i }$ with respect to $t _ { j - 1 }$ from gyroscope: $\mathsf { q ^ { \prime } } _ { t _ { i } } = \mathsf { q } ^ { ( t _ { j - 1 } ) } \mathsf { q } _ { \triangle }$ . We add $\ell _ { 1 }$ loss $\mathcal { L } _ { l o o s e } =$ $| q _ { t _ { i } } - q _ { t _ { i } } ^ { \prime } |$ into Eq 4, where $q _ { g _ { \mathcal { I } } }$ integrate all gyroscope measurements from timestamps $t _ { j - 1 }$ to $t _ { j }$

Tight coupling. However, simply integrating IMU information leads to large drift and noise over time. As an immediate consequence of our continuous pose representation over time, we can directly fuse the angular velocity using the quaternion derivative [50]:

$$
\dot { \mathsf { q } } = \frac { 1 } { 2 } \Omega ( \hat { \omega } ) .\tag{7}
$$

![](images/2a8b0e528dbcaba03876cab96c93a02ce9b04f28ddc8ec228000058643da8c65.jpg)

Figure 2. Patch Reconstruction Color-coded patch correspond to Fig 3.Note that patch 2D rigid motion exhibits continuity over time (left to right)
<table><tr><td rowspan="2">Method</td><td colspan="3">Cat</td></tr><tr><td>CE(pixel) ↓</td><td>PSNR ↑</td><td>SR↑</td></tr><tr><td rowspan="3">BARF[2] B-spline Ours</td><td>13.55</td><td>27.61</td><td>30%</td></tr><tr><td>35.14</td><td>21.95</td><td>0%</td></tr><tr><td>0.01</td><td>37.00</td><td>100%</td></tr><tr><td>1</td><td colspan="3">Girl</td></tr><tr><td rowspan="3">BARF[2] B-Spline Ours</td><td>29.94</td><td>22.09</td><td>15%</td></tr><tr><td>39.08</td><td>19.42</td><td>10%</td></tr><tr><td>6.92</td><td>32.40</td><td>95%</td></tr></table>

![](images/091ad2740bd4d2263008f2c3b7d3d8f7a2631fba69459b3a93c49700ff3336f7.jpg)  
Table 1. Image alignment experiment. Results of average 20 sampled 2D rigid motion, CE refer to Corner Error and SR refer to successful rate.  
(a) BARF[27]

(b) B-Spline  
![](images/42c5031d65ac039d98c1225c9bcaecdc06408d59c9d38418f1f62a9c46060427.jpg)

(c) Ours  
![](images/f261d032e94b558b846b047c86bd305b2c63db36b26c402a9a7355bc6fcf7298.jpg)  
(d) GT  
Figure 3. Qualitative results of 2D planar Alignment. We report the results of planar image alignment. Given input as ground truth (d) shown in Fig. 2, the goal is to find the 2D rigid transformation for each patch and optimize the entire neural image. Our method optimizes for accurate alignment and highfidelity image reconstruction, while baselines fail due to local minima.

## 4.1. NeRF from Inaccurate Poses

## 4. Experiments

![](images/a30d407fcfb7c8d2980d0bb1a8a011543ab6456b0a1c371fc5b02b0b5d46593a.jpg)

Thus we can supervise PoseNet by constraining the jacobian with the measured angular velocity. We use $\ell _ { 1 }$ loss as $\begin{array} { r } { \mathcal { L } _ { t i g h t } = | \dot { \mathsf { q } } - \frac { 1 } { 2 } \Omega ( \hat { \omega } ) } \end{array}$ | and jointly optimize it with the tracking target function in Eq 4.

We validate the effectiveness of our proposed method through 2D planar image alignment experiments and 3D scene experiments similar to BARF [27]. During this process, BARF refines a discrete set of inaccurate camera poses while our method leverages the continuous pose information and is therefore less prone to local minima.

It is noteworthy that, in the aforementioned equation, our PoseNet outputs pose with respect to the body frame rather than the camera frame. Further details regarding the coordinate change can be found in the supplementary materials, along with an explanation of how acceleration is utilized.

## 4.1.1 Planar Image Alignment (2D)

We choose the same images as [7, 27] as shown in Fig 3. To obtain a continuous rigid transformation, we initially randomly sample 10 data points from $T \in S E ( 2 )$ , we then interpolate a cubic spline along each dimension. Finally, we interpolate on 7 uniformly spaced points at previous time instants. As a result, the rigid transformation demonstrates temporal correlation, as illustrated in Fig 2. The initialized pose is identity with respect to center crop.

Experimental settings. We compare our method against BARF [27] and BaRF with B-spline. For the latter, we introduce continuity by resetting the learned $T \in S E { ( 2 ) }$ for every 100 steps using B-spline interpolation. The learning rate is 1e − 3 for translation and $2 e - 4$ for rotation. For the B-Spline method we report with 5 knots placed time-wise uniformly with degree = 3.

Results. The results are visualized in Fig 2,3. The align ment performance of BARF suffers from local minima, resulting in sub-optimal performance. Experiments are deemed successful if the corner error is below 1 pixel. Although some patches correctly learn the transformation, they do not effectively contribute to neighbouring patches. Merely introducing B-spline directly does not work, as it can over-smooth or under-smooth the poses, whereas our proposed method successfully captures all rigid transformations resulting in high-fidelity neural image. Furthermore as demonstrated in Table 1 our method performs consistently well across 20 different trajectories.

![](images/75c92ac0f5d5ec3ef80b6505dea65074687a9b5eb8a5b2e463423a19cfc9a0ce.jpg)

<table><tr><td rowspan="2">Scene</td><td colspan="2">Rotation ↓</td><td colspan="2">Translation ↓</td><td colspan="2">PSNR ↑</td><td colspan="2">SSIM ↑</td><td colspan="2">LPIPS↓</td></tr><tr><td>BARF</td><td>ours</td><td>BARF</td><td>ours</td><td>BARF</td><td>ours</td><td>BARF</td><td>ours</td><td>BARF</td><td>ours</td></tr><tr><td>Fern</td><td>0.199</td><td>0.181</td><td>0.196</td><td>0.181</td><td>21.01</td><td>21.08</td><td>0.62</td><td>0.63</td><td>0.33</td><td>0.31</td></tr><tr><td>Fern/2</td><td>0.344</td><td>0.331</td><td>0.195</td><td>0.173</td><td>19.72</td><td>19.74</td><td>0.53</td><td>0.53</td><td>0.33</td><td>0.32</td></tr><tr><td>Fern/4</td><td>0.289</td><td>0.264</td><td>0.212</td><td>0.215</td><td>19.65</td><td>21.33</td><td>0.54</td><td>0.63</td><td>0.33</td><td>0.32</td></tr><tr><td>Fortress</td><td>0.444</td><td>0.360</td><td>0.369</td><td>0.283</td><td>23.17</td><td>22.17</td><td>0.48</td><td>0.41</td><td>0.12</td><td>0.12</td></tr><tr><td>Fortress/2 *</td><td>6.507</td><td>0.574</td><td>3.545</td><td>0.418</td><td>14.86</td><td>20.00</td><td>0.35</td><td>0.33</td><td>0.40</td><td>0.17</td></tr><tr><td>Fortress/4</td><td>0.607</td><td>0.630</td><td>0.583</td><td>0.629</td><td>20.71</td><td>20.72</td><td>0.38</td><td>0.40</td><td>0.17</td><td>0.20</td></tr><tr><td>Orchids</td><td>0.719</td><td>0.645</td><td>0.390</td><td>0.364</td><td>13.22</td><td>14.46</td><td>0.17</td><td>0.24</td><td>0.35</td><td>0.30</td></tr><tr><td>Orchids/2</td><td>0.809</td><td>0.730</td><td>0.387</td><td>0.375</td><td>12.60</td><td>13.50</td><td>0.15</td><td>0.19</td><td>0.37</td><td>0.35</td></tr><tr><td>Orchids/4 *</td><td>92.176</td><td>0.865</td><td>46.772</td><td>0.388</td><td>11.07</td><td>12.64</td><td>0.18</td><td>0.16</td><td>0.97</td><td>0.49</td></tr><tr><td>Room</td><td>0.288</td><td>0.106</td><td>0.245</td><td>0.101</td><td>21.78</td><td>25.32</td><td>0.79</td><td>0.88</td><td>0.14</td><td>0.10</td></tr><tr><td>Room/2</td><td>0.329</td><td>0.274</td><td>0.284</td><td>0.172</td><td>21.20</td><td>21.27</td><td>0.77</td><td>0.78</td><td>0.13</td><td>0.13</td></tr><tr><td>Room/4*</td><td>118.58</td><td>0.403</td><td>76.14</td><td>0.550</td><td>11,00</td><td>22.76</td><td>0.42</td><td>0.80</td><td>0.89</td><td>0.17</td></tr><tr><td>Average</td><td>18.44</td><td>0.446</td><td>10.777</td><td>0.320</td><td>17.50</td><td>19.58</td><td>0.448</td><td>0.498</td><td>0.378</td><td>0.248</td></tr><tr><td></td><td>(0.448)</td><td>(0.391)</td><td>(0.318)</td><td>(0.276)</td><td>(19.23)</td><td>(19.95)</td><td>(0.492)</td><td>(0.520)</td><td>(0.252)</td><td>(0.238)</td></tr></table>

Figure 4. We introduce continuous errors on the camera trajectories and perform pose refinement in the NeRF setting. (a) Initial pose error; (b) results obtained using BARF [27] that uses a discrete set of poses; (c) results obtained using our continuous pose representation.  
Table 2. Real data with unknown pose. Our PoseNet compared to BARF [27] for the real dataset, simulating different camera moving speeds. Whenever BARF diverges and provides very inaccurate results, we consider them failures and denote them as \*. The average across all experiments is provided for all (and averaged only when BARF succeeds). In addition to the 12/12 (Ours) vs. 9/12 (BARF) success rate, PoseNet performs better than BARF also in cases when BARF succeeds.

<table><tr><td>Method</td><td>RE↓</td><td>TE↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>LE, C</td><td>13.62</td><td>48.05</td><td>9.79</td><td>0.59</td><td>0.56</td></tr><tr><td>Sinusoidal(2), C</td><td>3.70</td><td>15.84</td><td>14.05</td><td>0.66</td><td>0.20</td></tr><tr><td>Sinusoidal(5), C,</td><td>0.07</td><td>0.32</td><td>27.25</td><td>0.91</td><td>0.05</td></tr><tr><td>Sinusoida(10), C</td><td>0.18</td><td>0.88</td><td>24.88</td><td>0.88</td><td>0.06</td></tr><tr><td>Sinusoidal(2), D</td><td>2.86</td><td>10.53</td><td>15.97</td><td>0.69</td><td>0.14</td></tr><tr><td>Sinusoidal(5), D</td><td>0.07</td><td>0.28</td><td>27.30</td><td>0.92</td><td>0.06</td></tr><tr><td>Sinusoidal(10), D</td><td>0.07</td><td>0.28</td><td>27.20</td><td>0.91</td><td>0.05</td></tr><tr><td>Sigmoid, D</td><td>14.31</td><td>37.21</td><td>11.28</td><td>0.67</td><td>0.55</td></tr><tr><td>Sinusoidal(10) c2f, D</td><td>14.74</td><td>49.07</td><td>9.78</td><td>0.60</td><td>0.60</td></tr></table>

Table 3. Ablation study. We investigate the effectiveness of our PoseNet with diverse architecture. LE refers to linear encoder and C, D refer to coupled and decoupled representations. RE, TE refer to rotational and translation error. The best and second-best results are in bold and underlined.

## 4.1.2 Synthetic and Real NeRF (3D)

We explore the more challenging problem of learning 3D Neural Radiance Field with inaccurate poses. For the synthetic data, we render Lego [32] with a circular movement as shown in Figure 4. The simulated camera orbits the Lego model in the xy-plane, moving up and down at a constant speed in the z-direction.

Experimental settings. Similar to the 2D experiment, we introduce temporal correlation between neighboring SE(3) disturbances with interpolation. We use spherical linear interpolation for rotation. The introduced error corresponds to $5 5 ^ { \circ }$ in rotation and 110% in translation. For real data, we use the Fern, Fortress, Orchids, and Room datasets in LLFF [31], since these sequences allow us to perform experiments with varying numbers of images, thus simulating fast-moving cameras. Unlike in the synthetic case, we do not use any pose initialization in the real data experiments. Following [27] we report the MSE distance and rotational angle after alignment using Procrustes analysis for registration evaluation and PSNR, SSIM [57] and LPIPS [61] to evaluate view synthesis quality.

Results. We report our experimental results in Table 3 and Table 2, for synthetic and real data, respectively. In Table 2, the proposed continuous pose representation clearly offers better results than the discrete BARF. Ablation experiments further illustrate that the rotation and translation decoupled representation, i.e., two MLPs instead of one, performs better, offering the best results with the embedding frequency bands $F = 5 .$ . In real data with completely unknown camera poses, PoseNet performs significantly better than BARF. These results are reported in Table 2, where dataset/n refers to $1 / n ^ { t h }$ fraction of uniformly downsampled cases. It can be seen that PoseNet successfully handles all three failure cases of BARF. This is particularly evident when only sparse images are available. At the same time, even when BARF succeeds, PoseNet performs significantly better than BARF. More results and experiments using B-Spline can be checked in supplementary material.

<table><tr><td rowspan="2">Num</td><td colspan="2">TE↓</td><td colspan="2">PSNR ↑</td><td colspan="2">SSIM↑</td><td colspan="2">LPIPS ↓</td></tr><tr><td>without</td><td>ours</td><td>without</td><td>ours</td><td>without</td><td>ours</td><td>without</td><td>ours</td></tr><tr><td colspan="9">Chair</td></tr><tr><td>20</td><td>3.66</td><td>1.74</td><td>26.36</td><td>26.58</td><td>0.89</td><td>0.91</td><td>0.19</td><td>0.15</td></tr><tr><td>10</td><td>15.63</td><td>3.38</td><td>22.48</td><td>25.02</td><td>0.81</td><td>0.86</td><td>0.34</td><td>0.18</td></tr><tr><td>6</td><td>59.31</td><td>22.68</td><td>21.45</td><td>22.06</td><td>0.70</td><td>0.80</td><td>0.57</td><td>0.34</td></tr><tr><td colspan="9">Hotdog</td></tr><tr><td>20</td><td>3.66</td><td>2.42</td><td>23.59</td><td>25.64</td><td>0.90</td><td>0.92</td><td>0.14</td><td>0.10</td></tr><tr><td>10</td><td>15.63</td><td>4.87</td><td>21.85</td><td>23.15</td><td>0.85</td><td>0.87</td><td>0.20</td><td>0.16</td></tr><tr><td>6</td><td>59.31</td><td>6.70</td><td>21.06</td><td>22.03</td><td>0.79</td><td>0.85</td><td>0.34</td><td>0.18</td></tr></table>

Table 4. Interpolation error experiments. We improve the EventNeRF [46] using PoseNet. A small number of sparsely known poses are used to estimate the poses in between. PoseNet improves EventNeRF significantly in all six experimental setups.  
![](images/84d5e5c283fa953918fe13875020a14a2560095cbad85d5f63d22bf522d1a0e7.jpg)  
Figure 5. With and without calibration experiments. We investigate the effectiveness of our method under different deviations from the actual rotational axis. Our method can successfully reposition the object back to the center without additional calibration.

## 4.2. Continuous Pose for Asynchronous Events

By virtue of continuous pose representation, handling asynchronous event streams acquired by event cameras becomes natural. Hence, we use our PoseNet to learn the radiance field-based 3D scene representation from only colour event streams. This experimental setup is similar to recent work EventNeRF [46]. Note that EventNeRF accumulates asynchronous events to high-frequency synchronous event frames. The poses of each of those event frames are then assumed to be known. We argue that these assumptions limit the potential of the event cameras which come from their asynchronous nature. Therefore, we query for the pose of every event precisely at their trigger times. We conduct two experiments to address two practical issues of using events in EventNeRF setup using both synthetic and real datasets.

## 4.2.1 Unknown continuous pose for single event

Events are triggered asynchronously, and in practice where there is no precisely measured control available such as with a turntable [46] or a motorized linear slider [42], event pose can only be interpolated from measured discrete poses (from Vicon or Colmap [15]). However, this introduces interpolation errors.

Experimental settings. For synthetic data, we use chair and hotdog sequences from [46]. The events are simulated using the model in [45]. While EventNeRF performs interpolation, our method jointly learns intermediate poses as a continuous function of time.

Results. In Table 4, we reveal that integration of our PoseNet significantly enhances the overall performance with a notable reduction in translation errors and better scene reconstruction. More visual results can be found in supplementary material.

## 4.2.2 Unknown calibration in practice

EventNeRF [46] uses turntable to achieve stable and consistent object rotation speed. This setup also requires the actual rotational axis. Therefore, an additional checkerboardbased calibration technique, to estimate the axis offset, is also proposed in [46].

Experimental setting. For real cases, we use sewing machine datasets, which hold difficulties in reconstructing thin structures, view-dependent effects, and colored texts.

Results. We show that when PoseNet is used, additional calibration may not be required. The qualitative results of these experiments are shown in Figure 5. We demonstrate that when some offset is introduced, the 3D object deviates from the center for EventNeRF, while our method can reduce artifacts, learn the offset angle, and reposition the object back to the image center.

## 4.3. Visual SLAM with Depth and IMUs

While the previously discussed tasks are offline, vSLAM is an online method with different considerations. In this application we approach the problem as incremental SLAM. For each incoming frame, our objective is to determine its transformation with respect to the last frame $T _ { r e l a t i v e }$ . Similar to NICE-SLAM [63] we maintain a list of all optimal relative poses. It is trivial to solve the forgetting issue by retraining our PoseNet with such a list.

Experimental settings. We report the tracking results of our method compared with the standard NICE-SLAM. We report results of intrinsic motion on Replica [51], Scannet [9] and TUM-RGBD[52]. Note that during tracking we assume intrinsic motion reference slowly changes over time and only optimize $f _ { o }$ for every keyframe, with a frequency set to 10 for our experiments. In EUROC dataset [4] we follow the same pre-processing step as [13] and use nearest interpolation to get dense depth map. In order to evaluate the trajectory quality we report the ATE-RMSE [cm] of all sequences. More details regarding convergence rate and run-time can be found in the supplementary material.

<table><tr><td>Method</td><td>Rm 0</td><td>Rm 1</td><td>Rm 2</td><td>Off 0</td><td>Off 1</td><td>Off 2</td><td>Off 3</td><td>Off 4</td><td>Avg</td></tr><tr><td>Vox-Fusion* [60]</td><td>1.37 0</td><td>4.70</td><td>1.47</td><td>8.48</td><td>2.04</td><td>2.58</td><td>1.11</td><td>2.94</td><td>3.09</td></tr><tr><td>ESLAM[20]</td><td>0.71 0</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td>0.63</td></tr><tr><td>NICE-SLAM[63]</td><td>0.97 0</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>1.06</td></tr><tr><td>Ours</td><td>0.53</td><td>0.45</td><td>0.84</td><td>0.54</td><td>0.33</td><td>0.48</td><td>0.66</td><td>0.51</td><td>0.54</td></tr><tr><td>Ours(world)</td><td>0.62</td><td>0.52</td><td>0.91</td><td>0.60</td><td>0.62</td><td>0.36</td><td>0.54</td><td>0.72</td><td>0.58</td></tr><tr><td>Ours(rand)</td><td>35.84</td><td>9.29</td><td>34.67</td><td>N/A</td><td>9.69</td><td>26.92</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Ours (intrinsic)</td><td>0.53</td><td>0.47</td><td>0.81</td><td>0.35</td><td>0.24</td><td>0.43</td><td>0.64</td><td>0.50</td><td>0.49</td></tr></table>

Table 5. Tracking performance on Replica [51]. By integrating our method into the tracking branch of NICE-SLAM, we observe significant improvements. We investigate the impact of varying reference coordinates on tracking. It is evident that our proposed low DOF motion further improves the tracking performance.
<table><tr><td>Method</td><td>0000</td><td>0059</td><td>0106</td><td>0181</td><td>0207</td><td>Avg</td></tr><tr><td>DI-Fusion [16]</td><td>62.99</td><td>128.00</td><td>18.50</td><td>87.88</td><td>100.19</td><td>78.89</td></tr><tr><td>Vox-Fusion* [60]</td><td>68.84</td><td>24.18</td><td>8.41</td><td>23.30</td><td>9.41</td><td>26.90</td></tr><tr><td>NICE-SLAM[63]</td><td>12.00</td><td>14.00</td><td>7.90</td><td>13.40</td><td>6.20</td><td>10.70</td></tr><tr><td>Ours</td><td>10.98</td><td>11.98</td><td>7.10</td><td>13.50</td><td>5.76</td><td>9.86</td></tr><tr><td>Ours(intrinsic)</td><td>11.21</td><td>8.78</td><td>7.57</td><td>12.21</td><td>4.87</td><td>8.93</td></tr></table>

Table 6. Tracking performance on ScanNet [9]. Our approach yields consistently better results than the baseline. Note that the gain of utilizing intrinsic motion is relatively small, possibly attributed to the challenges posed by the noisy ground truth poses.

Results. We report all tracking results using ATE-RMSE [cm]. The numbers for the baselines are taken from [47] except EUROC. We showcase the effectiveness of our method for tracking across all scenes in Table 5, 6, 7. We observe significant improvements in both relatively easy and challenging scenarios.

Moreover as illustrated in Table 5, we underscore the importance of defining the coordinate system for relative pose optimization. The tracking is unstable and difficult when fixed on world origin or random coordinates. Figure 6 further demonstrates that, through our estimation of intrinsic motion and its transformation with PoseNet, we attain pose within a low-dimensional manifold, resulting in a substantial enhancement of tracking performance.

Finally, we validate the effectiveness of our IMU-Fusion method. While baseline methods fail in the face of large illumination changes and noisy depth, our approach maintains robust tracking and achieves accuracy comparable to state-of-the-art sparse feature-based tracking methods.

## 5. Conclusion

We proposed a simple yet effective technique for optimizing camera pose as a continuous function of time. The benefits of this approach were illustrated through several experiments of diverse applications, namely NeRF from the inaccurate pose, NeRF using Event Cameras, and visual SLAM with Depth and IMUs. We also studied different designs of the time-to-pose mapping continuous function, leading us to prefer a decoupled architecture. Furthermore, we justified the ease of using the proposed PoseNet in a plug-and-play manner. We first propose IMU-Fusion in NeRF-SLAM and analyze the advantage of adopting intrinsic motion frame for camera tracking tasks. Clear advantages in terms of performance were also observed in all settings, thanks to the continuous motion prior.

<table><tr><td>Method</td><td>fr1/desk</td><td>fr2/xyz</td><td>fr3/office</td><td>Avg</td></tr><tr><td>DI-Fusion [16]</td><td>4.4</td><td>2.0</td><td>5.8</td><td>4.1</td></tr><tr><td>Vox-Fusion* [60]</td><td>3.52</td><td>1.49</td><td>26.01</td><td>10.34</td></tr><tr><td>NICE-SLAM[63]</td><td>4.26</td><td>31.73</td><td>3.87</td><td>13.28</td></tr><tr><td>Ours</td><td>2.97</td><td>7.38</td><td>3.76</td><td>4.70</td></tr><tr><td>Ours(intrinsic)</td><td>2.72</td><td>1.98</td><td>2.74</td><td>2.48</td></tr></table>

Table 7. Tracking performance on TUM-RGBD [52] Our method consistently outperforms NICE-SLAM and other dense neural RGBD methods. The effectiveness of intrinsic motion is also demonstrated for reducing the tracking error significantly.

![](images/204cd1944e92e11326458cc5769c0255c36451f30ba90ff86c72389179288da7.jpg)

Figure 6. DOF Comparison on Replica room 1 dataset, we report that DOF of actual motion drop 41% from 5.22 to 3.08, demonstrating the sparsity of intrinsic motion.
<table><tr><td>Method</td><td>v101</td><td>v102</td><td>v103</td><td>v201</td><td>v202</td><td>v203</td><td>Avg</td></tr><tr><td>VINS-MONO[41]</td><td>7.9</td><td>11.0</td><td>18.0</td><td>8.0</td><td>16.0</td><td>27.0</td><td>14.6</td></tr><tr><td>ORB-SLAM [36]</td><td>1.5</td><td>2.0</td><td>N/A</td><td>2.1</td><td>1.8</td><td>N/A</td><td>N/A</td></tr><tr><td>DROID-SLAM [54]</td><td>3.7</td><td>1.2</td><td>2.0</td><td>1.7</td><td>1.3</td><td>1.4</td><td>2.2</td></tr><tr><td>NICE-SLAM[63]</td><td>2.58</td><td>N/A</td><td>5.66</td><td>6.56</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Ours(loose)</td><td>2.20</td><td>6.74</td><td>5.04</td><td>4.52</td><td>3.87</td><td>19.06</td><td>6.77</td></tr><tr><td>Ours(tight)</td><td>1.98</td><td>6.09</td><td>5.55</td><td>4.99</td><td>3.03</td><td>15.34</td><td>6.16</td></tr></table>

Table 8. Tracking performance on EUROC [4]. Our IMUfusion improves tracking with lower error and robustness, outperforming NICE-SLAM. We report results with sparse tracking method for reference. Despite the gap, our method narrows differences with state-of-the-art sparse tracking.

Acknowledgements: Research is partially funded by VIVO Collaboration Project and the Ministry of Education and Science of Bulgaria (support for INSAIT, part of the Bulgarian National Roadmap for Research Infrastructure).

## References

[1] Dejan Azinovic, Ricardo Martin-Brualla, Dan B. Goldman, Matthias Nießner, and Justus Thies. Neural RGB-D surface reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 6280–6291. IEEE, 2022. 3

[2] Tim D Barfoot, Chi Hay Tong, and Simo Sarkk ¨ a. Batch¨ continuous-time trajectory estimation as exactly sparse gaussian process regression. In Robotics: Science and Systems, pages 1–10. Citeseer, 2014. 1, 3, 5

[3] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4160–4169, 2023. 3

[4] Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Sammy Omari, Markus W Achtelik, and Roland Siegwart. The euroc micro aerial vehicle datasets. The International Journal ofRobotics Research, 35 (10):1157–1163, 2016. 7, 8

[5] Carlos Campos, Richard Elvira, Juan J Gomez Rodr´ ´ıguez, Jose MM Montiel, and Juan D Tard´ os. Orb-slam3: An accu-´ rate open-source library for visual, visual–inertial, and multimap slam. IEEE Transactions on Robotics, 37(6):1874– 1890, 2021. 3

[6] Yu Chen and Gim Hee Lee. Dbarf: Deep bundle-adjusting generalizable neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24–34, 2023. 3

[7] Yue Chen, Xingyu Chen, Xuan Wang, Qi Zhang, Yu Guo, Ying Shan, and Fei Wang. Local-to-global registration for bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8264–8273, 2023. 3, 5

[8] Shin-Fang Chng, Sameera Ramasinghe, Jamie Sherrah, and Simon Lucey. GARF: gaussian activated radiance fields for high fidelity reconstruction and pose estimation. CoRR, abs/2204.05735, 2022. 3

[9] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 7, 8

[10] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. Monoslam: Real-time single camera slam. IEEE transactions on pattern analysis and machine intelligence, 29(6):1052–1067, 2007. 3

[11] Christian Forster, Zichao Zhang, Michael Gassner, Manuel Werlberger, and Davide Scaramuzza. Svo: Semidirect visual odometry for monocular and multicamera systems. IEEE Transactions on Robotics, 33(2):249–265, 2016. 3

[12] Paul Furgale, Timothy D Barfoot, and Gabe Sibley. Continuous-time batch estimation using temporal basis functions. In 2012 IEEE International Conference on Robotics and Automation, pages 2088–2095. IEEE, 2012. 1, 3

[13] Ariel Gordon, Hanhan Li, Rico Jonschkowski, and Anelia Angelova. Depth from videos in the wild: Unsupervised

monocular depth learning from unknown cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8977–8986, 2019. 7

[14] Sachini Herath, David Caruso, Chen Liu, Yufan Chen, and Yasutaka Furukawa. Neural inertial localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6604–6613, 2022. 3

[15] Javier Hidalgo-Carrio, Guillermo Gallego, and Davide´ Scaramuzza. Event-aided direct sparse odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5781–5790, 2022. 7

[16] Jiahui Huang, Shi-Sheng Huang, Haoxuan Song, and Shi-Min Hu. Di-fusion: Online implicit 3d reconstruction with deep priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8932– 8941, 2021. 8

[17] Weibo Huang and Hong Liu. Online initialization and automatic camera-imu extrinsic calibration for monocular visualinertial slam. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 5182–5189. IEEE, 2018. 3

[18] Inwoo Hwang, Junho Kim, and Young Min Kim. Ev-nerf: Event based neural radiance field. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 837–847, 2023. 3

[19] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, and Jaesik Park. Selfcalibrating neural radiance fields, 2021. 3

[20] Mohammad Mahdi Johari, Camilla Carta, and Franc¸ois Fleuret. Eslam: Efficient dense slam system based on hybrid representation of signed distance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17408–17419, 2023. 8

[21] Lukas Kaul, Robert Zlot, and Michael Bosse. Continuoustime three-dimensional mapping for micro aerial vehicles with a passively actuated rotating laser scanner. Journal of Field Robotics, 33(1):103–132, 2016. 3

[22] Hanme Kim, Stefan Leutenegger, and Andrew J Davison. Real-time 3d reconstruction and 6-dof tracking with an event camera. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VI 14, pages 349–364. Springer, 2016. 3

[23] Georg Klein and David Murray. Parallel tracking and mapping on a camera phone. In 2009 8th IEEE International Symposium on Mixed and Augmented Reality, pages 83–86. IEEE, 2009. 3

[24] Simon Klenk, Lukas Koestler, Davide Scaramuzza, and Daniel Cremers. E-nerf: Neural radiance fields from a moving event camera. IEEE Robotics and Automation Letters, 2023. 3

[25] Stefan Leutenegger, Paul Furgale, Vincent Rabaud, Margarita Chli, Kurt Konolige, and Roland Siegwart. Keyframebased visual-inertial slam using nonlinear optimization. Proceedings of Robotis Science and Systems (RSS) 2013, 2013. 3

[26] Heng Li, Xiaodong Gu, Weihao Yuan, Luwei Yang, Zilong Dong, and Ping Tan. Dense rgb slam with neural implicit maps. arXiv preprint arXiv:2301.08930, 2023. 3

[27] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5741–5751, 2021. 2, 3, 4, 5, 6

[28] Qi Ma, Danda Pani Paudel, Ajad Chhatkuli, and Luc Van Gool. Deformable neural radiance fields using rgb and event cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3590–3600, 2023. 3

[29] Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, and Jingyi Yu. Gnerf: Gan-based neural radiance field without posed camera. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 6331– 6341. IEEE, 2021. 3

[30] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In CVPR, 2019. 1

[31] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 2019. 6

[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021. 1, 3, 4, 6

[33] Michael Milford, Hanme Kim, Stefan Leutenegger, and Andrew Davison. Towards visual slam with event-based cameras. In The problem of mobile sensors workshop in conjunction with RSS, 2015. 3

[34] Anastasios I Mourikis and Stergios I Roumeliotis. A multistate constraint kalman filter for vision-aided inertial navigation. In Proceedings 2007 IEEE international conference on robotics and automation, pages 3565–3572. IEEE, 2007. 3

[35] Elias Mueggler, Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. Continuous-time visual-inertial odometry for event cameras. IEEE Transactions on Robotics, 34(6): 1425–1440, 2018. 1

[36] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147–1163, 2015. 8

[37] Janne Mustaniemi, Juho Kannala, Simo Sarkk¨ a, Jiri Matas,¨ and Janne Heikkila. Gyroscope-aided motion deblurring with deep networks. In 2019 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 1914–1922. IEEE, 2019. 3

[38] Andreas Nuchter, Michael Bleier, Johannes Schauer, and Pe-¨ ter Janotta. Improving google’s cartographer 3d mapping by continuous-time slam. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 42:543–549, 2017. 3

[39] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 165–174, 2019. 1

[40] Alonso Patron-Perez, Steven Lovegrove, and Gabe Sibley. A spline-based trajectory representation for sensor fusion and rolling shutter cameras. International Journal of Computer Vision, 113(3):208–219, 2015. 3

[41] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4):1004–1020, 2018. 3, 8

[42] Henri Rebecq, Guillermo Gallego, and Davide Scaramuzza. Emvs: Event-based multi-view stereo. 2016. 7

[43] A Richard, RA Newcombe, L Steven, et al. Dense tracking and mapping in real-time. In Proceedings of IEEE International Conference on Computer Vision, Barcelona, page 2327, 2011. 3

[44] Antoni Rosinol, John J Leonard, and Luca Carlone. Nerfslam: Real-time dense monocular slam with neural radiance fields. arXiv preprint arXiv:2210.13641, 2022. 3

[45] Viktor Rudnev, Vladislav Golyanik, Jiayi Wang, Hans-Peter Seidel, Franziska Mueller, Mohamed Elgharib, and Christian Theobalt. Eventhands: Real-time neural 3d hand pose estimation from an event stream. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12385–12395, 2021. 7

[46] Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, and Vladislav Golyanik. Eventnerf: Neural radiance fields from a single colour event camera. arXiv preprint arXiv:2206.11896, 2022. 3, 4, 7

[47] Erik Sandstrom, Yue Li, Luc Van Gool, and Martin R Os-¨ wald. Point-slam: Dense neural point cloud-based slam. arXiv preprint arXiv:2304.04278, 2023. 3, 8

[48] Davide Scaramuzza and Zichao Zhang. Visual-inertial odometry of aerial robots. arXiv preprint arXiv:1906.03289, 2019. 3

[49] Zhenmei Shi, Fuhao Shi, Wei-Sheng Lai, Chia-Kai Liang, and Yingyu Liang. Deep online fused video stabilization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1250–1258, 2022. 3

[50] Joan Sola. Quaternion kinematics for the error-state kalman filter. arXiv preprint arXiv:1711.02508, 2017. 4

[51] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019. 7, 8

[52] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A benchmark for the evaluation of rgb-d slam systems. In Proc. ofthe International Conference on Intelligent Robot Systems (IROS), 2012. 7, 8

[53] Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J. Davison. imap: Implicit mapping and positioning in real-time. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 6209–6218. IEEE, 2021. 3

[54] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569, 2021. 8

[55] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. arXiv preprint arXiv:2211.11738, 2022. 3

[56] Vladyslav Usenko, Nikolaus Demmel, David Schubert, Jorg¨ Stuckler, and Daniel Cremers. Visual-inertial mapping with¨ non-linear factor recovery. IEEE Robotics and Automation Letters, 5(2):422–429, 2019. 3

[57] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6

[58] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. NeRF−−: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 3

[59] Yitong Xia, Hao Tang, Radu Timofte, and Luc Van Gool. Sinerf: Sinusoidal neural radiance fields for joint pose estimation and scene reconstruction. CoRR, abs/2210.04553, 2022. 3

[60] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian Liu, and Guofeng Zhang. Vox-fusion: Dense tracking and mapping with voxel-based neural implicit representation. In 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 499–507. IEEE, 2022. 8

[61] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6

[62] Yi Zhou, Guillermo Gallego, and Shaojie Shen. Event-based stereo visual odometry. IEEE Transactions on Robotics, 37 (5):1433–1450, 2021. 3

[63] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12786–12796, 2022. 2, 3, 4, 7, 8

[64] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui, Martin R Oswald, Andreas Geiger, and Marc Pollefeys. Nicer-slam: Neural implicit scene encoding for rgb slam. arXiv preprint arXiv:2302.03594, 2023. 3