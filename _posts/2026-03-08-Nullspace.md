---
layout: post
title: "Null Space"
category: math
summary: funfun Null space
---

# Null space and free Unknown (영공간과 자유변수)

Kernel which known as ‘null space’ is a set of vectors that satisfy $$A\textbf x = 0$$. 

## 1. Definition

For given a linear map $$L: V → W$$ between two vector space $$V$$ and $$W$$, the kernel $$L$$ is the vector space of all elements $$\textbf v$$ of $$V$$ such that $$L(\textbf v) = \textbf 0$$, which mean the set of solution of homogeneous or more symbolically:

$$
\mathrm{ker}(L) = {\textbf v \in V \mid L(\textbf v) = 0} = L^{-1}(\textbf 0)
$$

or

$$
\mathrm{Null}(A) = \{ \textbf{x} \in \mathbb R^n \mid A \textbf x = \textbf 0 \}
$$

## 2. Geometrical Understand

### 2-1. Subspace and Level set

The kernel of $$L$$ is always a linear subspace of the $$V$$ ($$L \subset V$$). And by the null space’s definition, $$\mathrm{Null}(A)$$ is a set of directions which exactly vanish when through linear transformation by $$A$$.


<img src="/assets/images/Null/image1.png" class="img-small" alt="Figure 1">

At this point, the null space directions are exactly the directions that move along a level set, meaning the output value does not change.

### 2-2. Null space and Left null space


<img src="/assets/images/Null/image2.png" class="img-small" alt="Figure 2">

Again, Null space can be represented by the vanishing space of linear transformation $$A$$. Thus, for $$A \in \mathbb R^n$$…

$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = n
$$

Thus, we can understand null space of A as linear subspace of A which always contain origin point.

And row space of $$A$$ and null space of $$A$$ are always orthogonal.

$$
\mathrm{Null}(A) \ \ ⊥ \ \ \mathrm{Row}(A)
$$

In contrast, left null space denoted by, 

$$
\mathrm{Null}(A^{\top}) = \{ \textbf{x} \in \mathbb R^n \mid A^{\top} \textbf x = \textbf 0 \}  \ \ \ \ \ \  \mathrm{or} \ \ \ \ \ \textbf x^{\top}A = \textbf{0}
$$

These vectors are in the output space (column space), while the vectors in null space are in input space (row space).

즉, 행렬 $$A$$와 임의의 벡터 $$\textbf x, \textbf y$$를 다음과 같이 이해할 때,

$$
A \textbf{x} =\begin{bmatrix} \textbf r_1^{\top} \textbf{x}  \\ \textbf r_2 ^{\top} \textbf{x}  \\ \vdots \\  \textbf r_n^{\top} \textbf{x}  \end{bmatrix}, \ \ \ A \in \mathbb R^{n\times m}
$$

null space에 속하는 벡터들은 모든 행 방향 $$r$$과 수직인 방향이다. 다시 말해 행렬 $$A$$의 어느 행도 $$\textbf x$$를 측정할 수 없음을 뜻한다. 반대로 left null space에 속하는 벡터들은 선형 변환 A의 열 방향 $$c$$와 수직인 방향이다. 다시 말해 행렬 $$A$$의 어느 열도 $$\textbf y$$를 나타낼 수 없음을 뜻한다.

### 2-3. The solution space

The solution of linear equation $$A\textbf x = b$$ is alway form of…

$$
\mathrm{solution} = \textbf x + \mathrm{Null}(A)
$$

On the other hand, the left null space of $$A$$ determine the existence of solution $$A\textbf x = \textbf b$$ . For $$A$$ to represent $$\textbf b$$, $$\textbf b$$ should be orthogonal with $$\mathrm{Null}(A^{\top})$$.

## 3. Usage & Deep understand

### 3-1. In SVD…

Let the singular value decomposition as…

$$
A = U\Sigma V^{\top}
$$

$$\mathrm{Null}(A)$$  is subspace of $$V$$’s column vectors which correspond to singular value $$0$$ in $$\Sigma$$. This concept extends to data compression: even when singular values are not exactly zero but near-zero, the corresponding vectors represent relatively insignificant aspects of the data.

At the same time, we can also easily examine the left null space in addition to the null space: the columns of $$U$$ corresponding to zero singular values in $$\Sigma$$ form the directions of the left null space.

Moreover not only $$0$$, small values in $$\Sigma_r$$, correspond to direction that carry very little information, the subspace corresponding to near-zero singular values can be treated as ‘effectively null’ for practical purpose.

### 3-2. Null space and model editing

The null space also can be interpreted as **the set of direction that matrix $A$ cannot discriminate**’, which means, ‘**Moving parameters to the null space direction is not changes the output of the model**’.

In paper “ALPHAEDIT: NULL-SPACE CONSTRAINED KNOWLEDGE EDITING FOR LANGUAGE MODELS”, Authors suggest model’s knowledge editing by embedding new knowledge to null space of weight matrix $$W \in \mathbb R^{\mathrm{Patch} \times \mathrm{Channel}}$$.

In most case, because of dimension reduction by deep learning model’s layers, $$\mathrm{Patch} \le \mathrm{Channel}$$. Thus, there is large null space in $$W$$.

More specifically, in the parameter space of a deep learning model, the null space corresponds to directions of the parameter perturbations that do not change the model’s output. Let $$f_{\theta}(x)$$ denote the model output with parameters $$\theta$$. For a small perturbation 
$$\delta \theta$$,

$$
f_{\theta + \delta \theta}(x) \approx f_{\theta}(x) + J_{\theta}(x) \delta \theta
$$

where $$J_{\theta}(x)$$ is the Jacobian with respect to the parameters. If $$J_{\theta}(x) \delta \theta = 0$$, then the perturbation lies in the null space of the jacobian and does not affect the model output.

## 4. Citation

[https://en.wikipedia.org/wiki/Kernel_(linear_algebra)](https://en.wikipedia.org/wiki/Kernel_(linear_algebra))

[https://diffrentedcon.tistory.com/27](https://diffrentedcon.tistory.com/27)

[https://arxiv.org/pdf/2410.02355](https://arxiv.org/pdf/2410.02355)

[https://mbernste.github.io/posts/matrixspaces/](https://mbernste.github.io/posts/matrixspaces/)
