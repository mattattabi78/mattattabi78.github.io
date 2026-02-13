---
layout: post
title: "Feature Abstraction: exploring feature with PCA in Diffusion model"
paper: "Exploring diffusion model's feature with PCA matric"
category: project
summary: Simple Feature Manipulation with PCA in feature space 
---

3학년 2학기 Explainable AI 과목의 기말 과제로 디퓨전 모델의 feature space에서 PCA 연산이 갖는 의의를 탐구하고 이를 활용하여 서로 다른 두 개념을 임의로 섞는 comcept mixture를 해보았다. PCA 자체는 인공지능 수학 수업시간에 배웠지만, 실제로 내가 알고있는 디퓨전 모델에 적용시키고 결과를 볼 때 쾌감이 있었다. 본 프로젝트는 학습 없이도 단순 선형 변환을 통해 디퓨전 모델의 feature들을 의미적으로 정렬할 수 있음을 시사하며 feature space의 각 차원의 분산량에 따라 semantic feature와 visual feature가 어느 정도 구분될 가능성을 암시한다.

기말 기간동안 재미있게 프로젝트를 진행했으며 결과적으로 해당 수업에서 1등을 할 수 있었다 v^^v.


<img src="/assets/images/diffusion_PCA/image8.png" class="img-medium" alt="Poster">

## Abstract
Recently, there has been an increasing number of attempts to directly manipulate or analyze the feature space of diffusion models. Such efforts are akin to uncovering the hidden rules within the denoising process of these models. In this study, building on this spirit, systematically observed the changes that occur during reconstruction by performing low-rank approximations along the principal component directions of the diffusion model feature space, a process we refer to as Abstraction. This experimental results show that a small number of leading principal components are sufficient to preserve the core semantic structure of images, while lower-ranked components mainly contribute to fine textures and noise removal. Furthermore, through Abstracting Mixture experiments, I demonstrated that replacing top principal components allows for the effective blending of semantics from different concepts. This indicates that PCA-based low-rank analysis can be applied not only for structural exploration but also for image editing and semantic feature control. Overall, this study presents a method for visually analyzing and understanding the structure and semantic roles of the feature space in diffusion models using relatively simple linear tools.



## Introduction
### Diffusion Model
Diffusion models are generative models that learn a gradual stochastic reconstruction process to generate images from a data distribution. In the forward process, Gaussian noise is progressively added to a real image $$x_0 \in \mathbb{R}^{H \times W \times C}$$, producing a latent variable $$x_t$$ at timestep $$t$$. This process is defined as

$$
    q(x_t \mid x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1 - \alpha_t) I)
$$



In the reverse process, a neural network $$p_\theta(x_{t-1} \mid x_t)$$ is trained to progressively reconstruct structured images from noise. During generation, the model starts from random noise $$x_T \sim \mathcal{N}(0, I)$$ and samples the final image $$x_0$$ by following the learned reverse process.

In this context, the feature space of a diffusion model should not be understood as the pixel space itself, but rather as a high-dimensional latent representation space formed as the input $$x_t$$ propagates through the network at each timestep $$t$$. Formally, let as $$f_\theta^l : \mathbb{R}^{H \times W \times d_{l-1}} \rightarrow \mathbb{R}^{H \times W \times d_l}$$ denote the activation of the $$l$$-th layer of the network. The feature representation at timestep $$t$$ is then given by $$h_t^l = f_\theta^l(x_t, t)$$.

This feature space evolves continuously over time: early diffusion steps primarily encode global low-frequency structures, while later steps progressively incorporate fine-grained high-frequency visual attributes. Consequently, the diffusion generation process can be interpreted not merely as noise removal, but as a stochastic traversal over a structured feature manifold along the temporal axis.

### Principal Component Analysis(PCA)

Principal Component Analysis **PCA** is a linear transformation technique that projects high-dimensional feature spaces into lower-dimensional subspaces while preserving maximal variance. Given a collection of $$l$$-th layer's reshaped features $$h_t^l \in \mathbb{R}^{H * W \times d_l}$$, I construct a feature matrix $$H = [h_{t,1}^l, \dots, h_{t,N}^l]^\top \in \mathbb{R}^{N * H * W \times d_l}$$. PCA identifies principal component vectors $${v_1, \dots, v_k}$$ via eigendecomposition of the covariance matrix $$\Sigma = \frac{1}{N} H^\top H$$.



Each principal component $$v_i$$ corresponds to a direction of maximal variance in the feature space, enabling dimensionality reduction to $$k \ll d_l$$ dimensions. In diffusion models, PCA facilitates the analysis of which feature directions at a given timestep $$t$$ dominate key visual attributes such as global structure, color, and texture, and further enables the exploration of concept-specific latent subspaces.

## Methods

### PCA Analysis of Diffusion Feature Dynamics


Generate a large-scale dataset using a diffusion model. Due to computational constraints, I sample 1,000 images from 100 text prompts using Stable Diffusion 3.5. Let $$X = {X_0, X_1, \dots, X_T}$$ denote the collection of intermediate feature maps recorded during the reverse diffusion process, where

$$
    X_t = {h_{t,1}, h_{t,2}, \dots, h_{t,N}} \subset \mathbb{R}^{d_l}
$$


Here, $$h_{t,i} = f_\theta^l(x_t^{(i)}, t)$$ represents the activation of the $$l$$-th layer at diffusion step $$t$$ for the $i$-th sample, $$N$$ is the number of samples, $$d_l$$ is the channel dimensionality of layer $$l$$, and $$x_t^{(i)}$$ is the noisy input at step $$t$$.

Perform PCA independently at each diffusion step, as the noise distribution and geometric structure of the reverse process vary across timesteps. Let $$H_t \in \mathbb{R}^{N \times d_l}$$ be the matrix obtained by stacking features in $$X_t$$. The covariance matrix and its eigendecomposition are given by for $$j = 1, \dots, d_l$$.

$$
    \Sigma_t = \frac{1}{N} H_t^\top H_t, \\
    \quad \Sigma_t v_{t,j} = \lambda_{t,j} v_{t,j}, 
$$



Denote by $$V_t^{(J)} = [v_{t,1}, \dots, v_{t,J}] \in \mathbb{R}^{d_l \times J}  \ \ \ \ \ \ (J \le d_l)$$ the subspace spanned by the top-$$J$$ principal components at step $$t$$. Feature representations are then projected onto this subspace for $$t = 0, \dots, T$$

$$
\begin{aligned}
\tilde{h}_{t,i}^{(J)} &= V_t^{(J)} V_t^{(J)\top} h_{t,i}, \\
\tilde{H}_t^{(J)} &= H_t V_t^{(J)} V_t^{(J)\top}
\end{aligned}
$$




By reconstructing features using different subsets of principal components and comparing the resulting images with the originals, I analyze the functional roles of dominant and residual components throughout the diffusion process.





## Experiments

### Abstracting to single step $$t$$

<img src="/assets/images/diffusion_PCA/poster.png" class="img-medium" alt="Figure 1">

**FIGURE1**: Feature abstraction. Reconstruction results obtained by applying low-rank approximation to various images using different types of principal components at step 0.

By performing a low-rank approximation, in specific diffusion step along $$J$$ selected principal component directions, we can visually inspect the role of each principal component $$v_{t,j}$$.

For most images, reducing $$J$$ leads to a progressive loss of fine details, leaving only the skeletal structure of grayscale images. This behavior can be interpreted as an **"Abstraction"** process. These results indicate that, at early diffusion steps, principal components associated with larger eigenvalues predominantly capture abstract semantic features of the image.

A notable observation is that, as indicated in image below, even when projecting the features at single diffusion steps onto the top 300 principal components—chosen based on the point where the eigenvalue spectrum exhibits the most rapid decay—there is no significant degradation in the reconstructed images. This suggests that during the denoising process, the model effectively operates within a very low-dimensional semantic subspace of the high-dimensional feature space at each step.

This interpretation is consistent with prior studies and can be regarded as an empirical, example-level demonstration of phenomena that were previously established through rigorous mathematical analysis using Riemannian geometry. Furthermore, this behavior aligns closely with the effective rank values computed at early diffusion steps, as shown in below.

<img src="/assets/images/diffusion_PCA/image3.png" class="img-small" alt="Figure 2.1">

**FIGURE2.1**: A log-scaled plot showing the variation of eigenvalues corresponding to principal components at each denoising step.


<img src="/assets/images/diffusion_PCA/image5.png" class="img-small" alt="Figure 2.2">

**FIGURE2.2**: Change in the effective rank of the eigenvector matrix across timesteps.


### Abstracting to whole step $$T$$

<img src="/assets/images/diffusion_PCA/image1.png" class="img-medium" alt="Figure 3">

**FIGURE3**: This figure shows the reconstructed images obtained by projecting the features onto the top $$J$$ principal components at every step of the denoising process. As $$J$$ decreases, the images become increasingly noisy.

It can be expanded and enables the analysis of the continuous contribution of principal components by applying a low-rank approximation along $$J$$ selected principal directions at every diffusion step. This analysis assumes that the principal subspaces at consecutive steps $$t$$ and $$t'$$ are aligned. I verify this assumption by measuring the Frobenius-overlap-based subspace similarity between PCA-induced channel subspaces:

$$
    S(t, t') = \frac{1}{J} \left| \left(U_J^{(t)}\right)^\top \left(U_J^{(t')}\right) \right|_F^2 
$$


Across all consecutive steps, the subspace similarity consistently exceeds 0.9 according to above equation, indicating that the principal subspaces evolve smoothly over diffusion time. Together with the effective rank curve, this shows that the denoising process follows a low-dimensional manifold embedded in the high-dimensional channel feature space.

Above figure shows reconstructions obtained by projecting features at each step onto the top $$J$$ principal components. Increasing $$J$$ monotonically improves reconstruction fidelity, while beyond a certain threshold additional components primarily refine visual quality rather than introduce new semantic content. This indicates that semantic information is highly concentrated in a small number of principal components.



### Role of the 0th Principal Component

<img src="/assets/images/diffusion_PCA/image2.png" class="img-medium" alt="Figure 4">

**FIGURE4**: Reconstruction with the 0th principal component removed during the denoising process.


As shown in figure above, removing only the 0th principal component at all steps produces abstracted reconstructions characterized by simplified global structure and preserved local textures, suggesting that this component is strongly associated with global shape. However, analogous experiments on other individual components do not yield semantically isolated effects, implying that semantic features are not encoded in single principal directions. Instead, semantic content emerges from combinations of multiple principal components.





## Abstracted Feature Mixture

<img src="/assets/images/diffusion_PCA/image6.png" class="img-medium" alt="Figure 5">

This paper demonstrates an **Abstracting mixture**, which allows the blending of two concepts with entirely different textures and forms using the abstraction process. At every step of the original image generation, features from both the original image and the injecting concept are projected into the channel-wise PCA space $$\mathbb{R}^{d_l}$$. Within this projected space, the top $$J$$ principal component values of the original image are replaced with those of the injecting concept, enabling a simple yet effective mixing of the two concepts. Notably, the longer the injection process continues, the more the resulting image resembles the injecting concept in terms of structure, color, and texture. Consequently, at low $$J$$, the global structure of the injecting concept is reflected, while at higher $J$, its color and texture are also preserved.

For instance, by replacing the top $$J$$ principal components of features in a generation trajectory for ‘rose’ with those used to generate ‘tree’, one can effectively blend the abstracted semantics of the two concepts. This shows several examples of images generated using this Abstracting mixture. Interestingly, visual features such as color, which are typically considered low-level, emerge even from relatively low-ranked principal components, highlighting a notable finding about the hierarchical organization of semantic and visual features within the diffusion model’s feature space.


## Conclusion

This study systematically investigates the feature space of diffusion models, using PCA-based analysis and low-rank approximation during the generation process. This experiments show that a small number of top principal components preserve most of the semantic structure of generated images, while lower-ranked components primarily refine details or contribute to noise removal. This indicates that diffusion models operate along a relatively low-dimensional semantic manifold at each step, rather than exploring the full high-dimensional space with example based methods. 

Furthermore, Abstracting mixture experiments demonstrate that simply replacing top principal components can effectively mix semantics from different concepts, highlighting the potential of PCA-based low-rank analysis for practical applications such as image editing and semantic feature manipulation. The observation that visual attributes like color and texture emerge from low-dimensional components provides insight into the hierarchical and semantically concentrated organization of diffusion features. Overall, this work leverages simple linear tools to visualize and interpret the structural and semantic roles of diffusion model features, offering a foundation for future research in semantic control, feature editing.
