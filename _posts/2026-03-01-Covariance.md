---
layout: post
title: "Covariance & Covariance Matrix"
category: math
---

그동안 배웠던, 지나가면서 봤던, 혹은, 관심이 가는 수학 개념들을 차례대로 정리해보려고 한다. 첫 번째는 공분산 행렬과 그 친구들이다. 실제로 쓰고 있는 논문이나 진행 중인 프로젝트에 쓰일 일은 적지만, 새로운 개념을 이해하고 정리하면서 많은 재미를 느끼는 것 같다.

# Covariance & Covariance Matrix (공분산 & 공분산 행렬)

“Covariance is measure of the joint variability of two random variables. The sign of the covariance shows the tendency in the linear relationship between the variables” - WIKIPEDIA 

## Definition of Covariance

For two jointly distributed real-valued random variables $$X$$ and $$Y$$ with finite second moments $$( \mathbb E [X^2] < ∞ )$$, the covariance is defined as the expected value of the product of their deviations from their individual expected values.

$$
\mathrm{cov}(X, Y) = \mathbb E [(X-\mathbb E[X])(Y - \mathbb E[Y]) ]
$$

 It also denoted by $$\sigma_{XY}$$ or $$\sigma (X,Y)$$.

And by using linearity property of expectations the definition of covariance can be simplified like below.

$$
\mathrm{cov}(X, Y) = \mathbb E[(X - \mathbb E[X]) (Y - \mathbb E[Y])] \\ = \mathbb E[(XY - X\mathbb E[Y] - \mathbb E[X]Y + \mathbb E [X] \mathbb E[Y])] \\ = \mathbb E[XY] - \mathbb E[X]\mathbb E[Y]
$$

Covariance is positive when variables tend to show similar behavior and negative when variable then to show opposite behavior. 

## Meanings of Covariance

The magnitude of the covariance is the geometric mean of the variance that are shared for the two random variables.

<img src="/assets/images/Covariance/image1.png" class="img-small" alt="Figure 1">

A larger magnitude means to variables more strongly depend on each other, smaller magnitude means to variables more weakly depend on each other and magnitude close to 0 means there is no dependency between two random variables. 

And we can say the variance of $$X$$ is special case of covariance. $$\mathrm{Var}(X) = \mathrm{Cov}(X, X)$$

## About Covariance Matrix

For a given vector $$x \in \mathbb R^d$$, covariance matrix can be denoted as…

$$
\Sigma = \mathbb E [(x-\mu)(x-\mu)^T] = 
\begin{bmatrix}
\mathrm{Var}(x_1) & \mathrm{Cov}(x_1, x_2) & \cdots \\
\mathrm{Cov}(x_2, x_1) & \mathrm{Var}(x_2) & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} \in \mathbb R^{d \times d}
$$

This matrix describes the spread in each dimension(diagonal elements) and linear relationships between dimensions(non-diagonal elements).  And Covariance matrix also can means **centered second moment matrix**. The second moment matrix of $$x$$ is $$\mathbb E [xx^T]$$ and cov matrix $$\Sigma$$ is $$\mathbb E [xx^T] - \mu \mu^T$$. These attributes is fundamental in multivariate statistics, Multivariate Guassian distributions and feature analysis.


**Geometrical Understand of Covariance Matrix**

In geometrically, covariance matrix define **ellipsoid structure** of data. Consider the quadratic form $$v^T\Sigma ^-1 v = 1$$ because symmetric positive-definite matrices stretch space differently in different directions. 

- Eigenvector of it → direction of spread
- Eigenvalue of it → magnitude of correspond eigenvector


**Attribute of Covariance Matrix**

And this matrix is also always **Positive semi-definite**. Thus $$\Sigma$$ satisfies,

$$
v^T \Sigma v \ge 0 \ \ \mathrm {for \ all \ vectors} \ v
$$

Which semantically means ‘the spread of data in any direction is never negative’. Since $$\Sigma$$ is **PSD**, ****all eigenvalues $$\lambda_i \ge 0$$.

The determinant $$\mathrm{det}(\Sigma)$$ represents the squared volume of the ellipsoid.

- Large determinant → data spread widely
- Small determinant → data is concentrated

## Usage

Now, let see the usage of the covariance.

### Principal Component Analysis (PCA)

PCA can be interpreted as eigendecomposition of data’s covariance matrix. By mean centered data $$x_c$$, since $$\Sigma$$ is symmetric,

$$
\Sigma = Q \Lambda Q^T
$$

where $$Q$$ means orthogonal matrix (rotation) and $$\Lambda$$ means diagonal matrix of eigenvalues which geometrically means 1. rotate into eigenvectors basis and scale along each axis by eigenvalues. So $$\Sigma$$ defines a rotation + anisotropic scaling of space.

The purpose of PCA and be expressed by,

$$
\underset{||v||=1} {\mathrm{max}}\mathrm{Var}(v^Tx)
$$

which means finding most spread axis of data.

More about PCA will be introduced in feature post.

### Multivariate Gaussian Distributions

A multivariate Gaussian Distribution is a probability distribution over vectors $$x \in \mathbb R ^k$$, it is writtens as $$x \sim \mathcal N(\mu, \Sigma)$$ with k-dimensional mean vector $$\mu = (\mathbb E[X_1], … , \mathbb E[X_k])$$.

<img src="/assets/images/Covariance/image2.png" class="img-small" alt="Figure 2">

And the density function $$p(x)$$ is defined by below with **quadratic form** of covariance

$$
p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2}
(x-\mu)^T
\Sigma^{-1}
(x-\mu)
\right)
$$

where $$\Sigma$$ determine shape, orientation and axis length of distribution.

Specifically, this ellipsoid of contour $$( (x-\mu)^T \Sigma^{-1} (x-\mu) = c )$$, is formed by linear transformation of covariance matrix.

Let the equation of circle as $$z^Tz = c$$ and $$z$$ is output of linear transformation of $$x$$ as $$A^{-1}x$$.

$$
(A^{-1}x)^T(A^{-1}x) = x^T(A^{-T}A^{-1})x = c
$$

Thus,

$$
A^{-T}A^{-1} = (AA^T)^{-1}
$$

And let $$\Sigma = AA^T$$ so, $$x^T \Sigma^{-1}x = c$$. Where actually $$A$$ is square root matrix of $$\Sigma$$. (remeber $$\Sigma$$ is PSD so it always can be fractorized.)

### Mahalanobis Distance

It is distance adjusted by covariance. 

$$
d(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}
$$

Mahalanobis distance used in anomaly detection, clustering etc.

Euclidean distance ignored each dimension’s scale and correlation between variables. To adjust these relation between variables, mahalanobis distance reflect data’s own ellipse structure by matrix multiplication of inverse covariance matrix.

### Kalman Filter

Kalman Filter is an recursive algorithm that estimate data under assumption that the data follows linear and gaussian distribution. It estimate joint distribution of present variation based on past one. It assume two model which called prediction model and measurement model.

The detail will be treated in future…

**Citation** 
[https://en.wikipedia.org/wiki/Covariance](https://en.wikipedia.org/wiki/Covariance)

[https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

[https://thekalmanfilter.com/covariance-matrix-explained/](https://thekalmanfilter.com/covariance-matrix-explained/)

[https://dhpark1212.tistory.com/entry/다변량-가우시안-분포Multivariate-Gaussian-Distribution](https://dhpark1212.tistory.com/entry/%EB%8B%A4%EB%B3%80%EB%9F%89-%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%B6%84%ED%8F%ACMultivariate-Gaussian-Distribution)
