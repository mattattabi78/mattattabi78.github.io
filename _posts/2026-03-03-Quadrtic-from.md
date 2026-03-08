---
layout: post
title: "Quadratic form"
category: math
summary: funfun Quadratic from
---

# Quadratic form(이차형식)

$$x^{\top}Ax$$을 인공지능 수학 시간에 처음 봤었던 이후로 이 간단학 식에 매료되었었다. 단순히 두 벡터의 내적 사이에 어떤 행렬을 끼워넣은 형태는 겉보기에는 간단하지만 본능적으로 그 안에 깊은 의미가 있을 것이란 생각이 들었다. 따라서 오늘은 이 이차형식에 대해 파보고자 한다.

쉽게 생각하면, 이차형식의 $$A$$는 공간을 늘리고 압축하는 렌즈 역할이다. 즉, $$x^{\top}Ax$$는 더이상 단순 유클리드 거리가 아니라 **$$A$$로 측정하는 거리가 되는 것**이다. 

*“Quadratic form occupy a central place in various branches of mathematics, including number theory, linear algebra, group theory, differential geometry, differential topology, Lie theory and statistics”* - WIKIPEDIA

## 1. Definition

Quadratic form is scalar function that appears as a quadratic for vectors where $$x \in \mathbb R^n$$, $$A \in \mathbb R^{n \times n}$$ and $$Q(x)$$ is scalar. 

$$
Q(x) = x^{\top}Ax
$$

For example, given $$x$$ and $$A$$ as,

$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix},\ \ A = \begin{bmatrix} a & b \\ b & c \end{bmatrix} \\ 
\\ 
$$

then 2-variable quadratic equation is just like below.

$$
Q(x) = x^\top A x= \begin{bmatrix} x_1 & x_2 \end{bmatrix}  \begin{bmatrix} a & b \\ b & c \end{bmatrix}  \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}= a x_1^2 + 2 b x_1 x_2 + c x_2^2
$$

## 2. Geometric meaning

Quadratic form define curves or surface in space. Most easily, 

$$
x^⊤x=x_1^2+x_2^2=k
$$

is a **circle** when $$A$$ is diagonal matrix, while $$x^{\top}Ax = k$$. 

For an **ellipse** where $$A$$ is not a diagonal matrix, it determine…

- Diagonal entries → scaling of space
- Off-diagonal entries → correlation / rotation

## 3. Usage

### 3-1. Eigen Decomposition intuition

Let $$A = U\Lambda U^{\top}$$ where $$U$$ is orthogonal matrix and $$\Lambda$$ is diagonal matrix, then,

$$
x^{\top}Ax = y^{\top} \Lambda y, \ \ \mathrm{where} \ \ y = U^{\top}x
$$

This means…

- The coordinate system from $$x$$ is rotated along the $$U$$
- Each squared term is scaled by the corresponding eigenvalue $$\lambda_i$$

Thus, the quadratic form weights different direction differently.

### 3-2. Regularization Metric and Natural Gradient

L2 regularization can be generalized like below.

$$
\mid \mid w \mid \mid = w^{\top} w \ \ \rightarrow \ \ w^{\top}Aw
$$

This regularization suppress specific direction.

General gradient descent “$$w_{t+1} = w_t - \eta \nabla L(W)$$” updated weights under the propose the parameter space follows euclidean metric. But in many cases, actual parameter spaces are not in euclidean, to solve this mathematic gap, Natural gradient use Fisher information matrix.

$$
w_{t+1} = w_t - \eta F^{-1} \nabla L(w)
$$

In natural gradient, the definition of gradient is like below.

$$
\underset{dw}{\mathrm{max}} \nabla L^{\top}dw \ \ \ \mathrm{s.t.} \ \ dw^{\top}Gdw = \epsilon^2
$$

From the Lagrangian,

$$
\mathcal J = \nabla L^{\top}dw - \lambda(dw^{\top}Gdw - \epsilon^2)
$$

$$
\frac{\partial \mathcal{J}}{\partial dw}=\nabla L - 2\lambda G dw \\ \nabla L - 2\lambda G dw = 0 \\ G dw = \frac{1}{2\lambda} \nabla L \\ dw = \frac{1}{2\lambda} G^{-1} \nabla L \\ dw \propto G^{-1} \nabla L
$$

즉, 주어진 거리 제약 하에 L을 가장 많이 증가시키는 방향을 라그랑주 방법으로 찾은 것이다. 라그랑주 방법에 대한 자세한 설명은 다음에 하도록한다.

### 3-3. Metric Tensor

From paper - “Not All Classes Stand on Same Embeddings: Calibrating a Semantic Distance with Metric Tensor (CVPR 2025)”

Metric tensor is a matrix representing the intrinsic geometry on a multi-dimensional manifold. It addresses the curvature of space for difference measurement in the non-Euclidean space. 

The shortest distance between two vectors in Euclidean space is..

$$
s^2 = \bold{u}^{\top}\bold{u}, \ \ \ \mathrm{where} \ \ \bold{u} = \bold{p} - \bold{q}
$$

And this equation can be generalized in d-dimensional manifold space as follows:

$$
s^2 = \bold{u}\bold{M}\bold{u}^{\top}, \ \ \ \mathrm{where} \ \bold{u} \in \mathbb R^{1 \times d}
$$

The authors of the paper noted that the shortest distance between two points in a high-dimensional manifold can be derived through linear transformation by the metric tensor $$\bold{M}$$. Consequently, by estimating the metric tensor of the manifold spcae, we further boost the existing consistency training effect.

<img src="/assets/images/Qfrom/image1.png" class="img-medium" alt="Figure 1">


### 4. Citation

[https://en.wikipedia.org/wiki/Quadratic_form](https://en.wikipedia.org/wiki/Quadratic_form)

[https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Not_All_Classes_Stand_on_Same_Embeddings_Calibrating_a_Semantic_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Not_All_Classes_Stand_on_Same_Embeddings_Calibrating_a_Semantic_CVPR_2024_paper.pdf)
