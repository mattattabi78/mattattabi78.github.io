---
layout: post
title: Various Diffusion Model(DDPM, DDIM, LDM)
paper: "Denoising Diffusion Probabilistic Models(2020), Denosing Diffusion Implicit Models(2020), High-Resolution Image Synthesis with Latent Diffusion Models(2021)"
category: generative-ai
summary: Summary of DDPM, DDIM, LDM paper!
---



# Diffusion model

방학 동안 Diffusion model에 대해 공부했었다. 현재 쓰고 있는 논문이 디퓨전 모델과 깊은 관련이 있기 때문에 교수님께서 읽어보라고 추천해주셨다. 나중에 기회가 되면 DiT나 Rectified Flow Model(RFM)까지 읽어볼 예정이다.

사실 읽으면서 많은 어려움을 느꼈고 아직까지 완벽히 이해했다는 느낌이 들지 않는다. 연구를 진행하면서도 계속해서 모르는 부분들이 나오는 것 같다. 참 신기하다.

# Denoising Diffusion Probabilistic Models

[paper link](https://arxiv.org/pdf/2006.11239)

## Abstract

DDPM(Denoising Diffusion Probabilistic Model) is probabilistic model which sampling images in processing noising and denoising images according to time $$t$$.

## What is diffusion Model?

Diffusion model is **Latent variable model** which can be represented by equation below.

$$
p_θ(x_0):=∫p_θ(x_{0:T})dx_{1:T}
$$

In this equation, finding $$\theta$$ that make probability $$p_θ(x_0)$$ max is target.

- $$p_\theta(x_0)$$: probability distribution of target data.
- $$x_{1:T}$$: latent variables

**The meaning of integral**
Let probability distribution $$p(x, y)$$. You can get distribution of single $$x$$, by calculating integral of $$p$$ as $$y$$. $$p(x)=∫p(x,y)dy$$

$$p(x, y)$$ means probability that $x$ and $$y$$ occur simultaneously. Thus, integral with respect to $$y$$ means probability distribution of $$x$$ regardless of $$y$$.


And $$p_θ(x_0)$$ can be written like below.

$$
p_θ(x_{0:T}):=p(x_T)\underset{t=1}{\overset{T}∏}p_θ(x_{t−1}∣x_t),\ \ \ \ p_θ(x_{t−1}∣x_t):=N(x_{t−1};μ_θ(x_t,t),Σ_θ(x_t,t))
$$

$$
p_θ(x_{0:T})=p(x_T)⋅p_θ(x_{T−1}∣x_T)⋅p_θ(x_{T−2}∣x_{T−1})⋯p_θ(x_0∣x_1)
$$

At contrast, $$q(x_{1:T}∣x_0)$$ that ‘adding noise process’ also represented by equation below 

$$
q(x_{1:T}∣x_0):=\underset{t=1}{\overset{T}{∏}}q(x_t∣x_{t−1}),\ \ \ q(x_t∣x_{t−1}):= \mathcal{N}(x_t;\sqrt{1−β}_tx_t−1,\ β_tI)
$$

 $$p_θ(x_0)$$ called **Reverse Process**  and  $$q(x_{1:T}∣x_0)$$ called **Forward Process** and both equation defined as Markov chain(Each time step $$t$$ only related with $$t-1$$). This means each time step influenced by only previous step so, in sampling step, we can compute simple chain structure. 

And as see, only reverse process has learnable parameter $$\theta$$.

## Forward Process and Backward(Reverse) Process

Evidently, DDPM has two Processes each represents noising and denoising images

<img src="/assets/images/Diffusion model/1.png" class="img-medium" alt="Figure 1">

### Forward Process

<img src="/assets/images/Diffusion model/2.png" class="img-medium" alt="Figure 2">

In forward Process, adding gaussian noise gradually from time $$t$$. This process can be represented by below. Briefly say,  $$x_t$$ is sampling value from normal distribution which have mean $$x_{t-1}$$, variance $$\beta_tI$$.

$$
q(x_t∣x_{t−1})=\mathcal{N}(x_t;\ \sqrt{1−β_t}⋅x_{t−1},β_tI)
$$

$$
\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s) \ \ \ \, \ \ \ \alpha_t = 1 - \beta_t
$$

- mean: $$\mu = \sqrt{1-\beta_t}⋅x_{t-1}$$
- variance: $$\sigma^2 = \beta_tI$$
- $$\beta_t$$: variance schedule. noise strength(0~0.02). It can be trainable
- $$\alpha_t$$: signal retention factor

**How $$\beta$$(variance of normal distribution) is related with noise strength?**

$$x_t ∼ \mathcal{N}(\mu, \sigma^2)$$ means $$x=μ+σ⋅ϵ,\ \ \ ϵ∼N(0,I)$$. Thus increase of $$\beta$$ means adding more noise.

**What it the meaning of “Adding noise”?**

In general DDPM has forward step $$x_t=\sqrt{1−β_t}⋅x_{t−1}+β_t⋅ϵ$$. And let image format follow $$x_{t-1} \in \mathbb R^{3×512×512}$$. And noise $$\epsilon$$ also follow same format $$ϵ∈R^{3×512×512},ϵ_{i,j,k}∼\mathcal N(0,1)$$. Thus, adding $$\epsilon$$ refer to adding a random float value to each R, G and B channel.

But in forward process, model learn noting. Researcher ignore the fact that the forward process variances $$\beta_t$$ are learnable by reparameterization and instead fix them to constants, thus, $$L_T$$ is a constant during training and can be ignored.

순방향 과정에서는 그저 가우시안 분포로부터 노이즈를 받아 주입하는 과정이 전부이다. 쉽게 말해, 순방향 과정은 역방향 과정 때 노이즈를 제거하는 방법을 학습하기 위한 데이터를 만드는 과정이라 볼 수 있다.

### Backward(reverse) Process

In Backward Process, we start from the totally random noise  $$x_T \sim \mathcal{N}(0, I)$$ to clean image $$x_0$$.

Below equation is formula of backward process. Each step sampled from normal distribution with predicted $$\mu_\theta$$ and $$\Sigma_\theta$$ form U-net in each step.

$$
  p_θ(x_{t−1}∣x_t)= \mathcal{N}(x_{t−1};μ_θ(x_t,t),Σ_θ(x_t,t))
$$

What the U-net does is predict $$\mu_\theta$$ or $$\epsilon_\theta$$ from input $$x_t$$ and $$t$$. Generally $$\epsilon_\theta$$ is predicted rather than $$\mu_\theta$$.

### **How to predict $$x_t$$ from  $$\epsilon$$?**

From time step $$t$$ in forward process, predicted image can be represented by below.

$$
  x_t=\sqrt{\bar{α_t}}x_0+\sqrt{1−\bar{α}_t}⋅ϵ, \ \ \ ϵ∼\mathcal{N}(0,I)
$$

So, follow above equation, approximation of original image $$\hat{x_0}$$ can be represented like below.

$$
  \hat{x}_0 = \frac{1}{\sqrt{\bar{α}_t}} (x_t−\sqrt{1−\bar{α}_t}⋅ϵ_θ(x_t,t))
$$

It is hard to predict $$\mu_\theta$$ directly, so we need technical detour. So we predict $$\epsilon_\theta$$. In backward process, learnable parameters are $$\epsilon_\theta(x_t, t)$$ shape of $$[B, 3, 512, 512]$$. 

### **How is the Markov chain assumption applied to the forward and backward processes in DDPM with closed-form? (why it is so simple?)**

DDPM’s processes are defined by Markov chain property. So we can decompose into each step with conditional probability. And we use Gaussian(normal) distribution for estimate noise $$\epsilon$$ which directly related to generating target images. Thanks to linearity of Gaussian distribution, Markov chain property can stay in closed form. Combining these two mathematical properties, DDPM becomes tractable and efficient generative model.

## Sampling

 From forward and backward process, we know what $$\epsilon_\theta(x_t, t)$$ looks like in each time step $$t$$. But the noise values just predict from single image in training steps. Thus, for sampling wide-variation image in specific target, we need the value of $$\mu_\theta$$ and $$\Sigma_\theta$$ of gaussian distribution of noise $$\epsilon_\theta$$ that makes our target images.

**Formula of calculate $$\mu_\theta(x_t, t)$$**

from below,

$$
  \hat{x}_0 = \frac{1}{\sqrt{\bar{α}_t}} (x_t−\sqrt{1−\bar{α}_t}⋅ϵ_θ(x_t,t))
$$

$$
  μ_θ(x_t,t)=\frac{\sqrt{\bar{α}_{t−1}}⋅\hat{x}_0 + \sqrt{1−\bar{α}_{t−1}}⋅ϵ_θ(x_t,t)}{\sqrt{1−\bar{α}_t}}
$$

we can calculate like above.

And $$\Sigma_\theta$$ of most diffusion model are not predicted. It just managed by $$\beta_t$$ with schedular.

## Training

Diffusion model is latent variable model, it is hard to directly calculate exact mean and variance of target distribution, thus, training is driven by optimizing the usual **variational bound(ELBO)** on **negative log likelihood**.

$$
  E[−\log {p_θ}(x_0)]\ ≤\ E_q[−\log{\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)}}]
$$

- $$E[−\log {p_θ}(x_0)]$$: expected value of negative log likelihood of sampling target images.
- $$E_q[−\log{\frac{pθ(x_{0:T})}{q(x_{1:T}∣x_0)}}]$$: ELBO of $E[−\log {p_θ}(x_0)]$.

$$
  \mathbb E_q \Biggr [−\log{\frac{p_θ(x_{0:T})}{q(x_{1:T}∣x_0)}} \Biggr ]=\mathbb E_q \Biggr[−\log{p(x_T)} -\underset{t ≥ 1}{∑} \log \frac{p_θ(x_{t−1}∣x_t)}{q(x_t∣x_{t−1})} \Biggr ]:= \mathcal{L}
$$

And ELBO formula can be rewritten like above.

**more about ELBO**
    
$$\log p_θ(x_0)=\log∫p_θ(x_{0:T})dx_{1:T}$$ is hard to compute. So we need to take a detour to estimate that value. ELBO is one of the estimation scheme, makes computable lower bound and maximizes the bound. Put simply,  original distribution is too complicated, so estimate original distribution with simple, computable distribution(linear combination of gaussian distribution). Once we obtain this approximation, we can compute similarity between the approximated distribution and the model’s distribution with KL-divergence.
    

## What actual value model predicts?

Pure diffusion model is trained to predict $$\mu_\theta(x_t, t)$$ and $$\epsilon_\theta(x_t, t)$$. Aspect of stability and effortless, model generally predict $$\epsilon_\theta(x_t, t) \in \mathbb R^{H \times W \times 3}$$.

### Objective function

By doing this, final objective function is $$MSE$$ loss of $\epsilon$.

$$
  L_{\mathrm{simple}}(θ) := E_{t,x_0 ,ϵ} \Bigr[|| ϵ−ϵ_θ(\sqrt{\bar{\alpha_t}} x_0 + \sqrt{1−\bar{\alpha}_t}ϵ,t)||^2 \Bigr]
$$

Additionally, researcher set down-weights in small $$t$$ that very small amounts of noise, to focus on more difficult denoising tasks at larger $$t$$ terms.

### Meaning of $\theta$

In diffusion model’s training, given noise map from forward process( $$x_t= \sqrt{\bar{α}_t} x_0 + \sqrt{1−\bar{α}_t} \epsilon$$ ) thus, $$\epsilon_\theta$$ that model predict is noise map per coordinate of whole images. And in here, $$\theta$$ as weight of networks, like CNN filter. 

---

---

# Denoising Diffusion Implicit Models

[paper link](https://arxiv.org/pdf/2010.02502)

DDPMs have achieved high quality image generation, yet they require simulating a Markov chain for many steps in order to produce a sample. DDIM used same structure with DDPM but using non-Markov process. 

## Problem of DDPM

From a variational perspective, a large T allows the reverse process to be close to a Gaussian, so that the generative process modeled select large t like  $$T=1000$$. But to reconstruct $$x_0$$ model have to process sequentially which means hard to parallelize.

## Non-Markov process

Actually, Loss function $$L_\gamma$$ depend on the marginals probability likes $$q(x_t|x_0)$$, not directly on the joint $$q(x_{1:T}|x_0)$$. That means model do not need to consider every before time steps when generate image in $$t$$. 

Let consider a family $$\mathcal Q$$ of inference distribution, indexed by real vector $$\sigma \in \mathbb R^T_{\ge 0}$$

- $$\mathcal Q$$: A defined family of inference distribution adjusted by $$\sigma$$. Given real data $$x_0$$, inference distribution is distribution of middle steps $$x_1, x_2, …, x_T$$ that generated with some noise which represented by $$q(x_{1:T}|x_0)$$ in DDPM.
- $$\sigma = (\sigma_1, \sigma_2, ...,\sigma_T)$$ is hyper parameter vector set of how much noise to add in forward process per time step. Bigger $$\sigma$$ add more noise.

Now, we can rewrite inference distribution like below.

$$
  q_σ(x_1, x_2, ..., x_T |x_0) := q_σ(x_T | x_0) \overset{T}{\underset{t=2}{\prod}} q_σ(x_{t−1}|x_t, x_0) 
$$

$$

  \mathrm{where}  \ q_σ(x_T |x_0) = \mathcal N (\sqrt{α_T} x_0, (1 − α_T )I) \ \mathrm{and \ for \ all} \ \ t > 1,

$$

$$
  q_σ(x_{t−1}|x_t, x_0) = \mathcal{N} \Biggr ( \sqrt{α_{t−1}}x_0 + \sqrt{1 − α_{t−1} − σ^2_t} ·
  \frac{x_t − \sqrt{α_t}x_0}{\sqrt{1 − α_t}} , σ^2_t I \Biggr)
$$

### Mean function

Mean function is needed to order to ensure that $$q_\sigma(x_t | x_0) = \mathcal N(\sqrt{\alpha_t}x_0, (1-\alpha_t)I)$$ for all $$t$$. This means all $$t$$ in forward process, distributions of $$x_t$$ have to same. And again, this means each distribution per $$t$$ have to share same equation that represented like above. Mean function align whole distribution of each time step in non-Markovian forward process which $$x_t$$ depends on $$x_{t-1}$$

### Role of $$\sigma$$

$$\sigma$$ regulates stochasticity of forward process. if $$\sigma = 0$$, note that $$\sigma^2_t$$ is variance of distribution of adding noise. So, bigger $$\sigma$$ means high variance of noise distribution.

---

---

# LDM

High-Resolution Image Synthesis with Latent Diffusion Models

[paper link](https://arxiv.org/pdf/2112.10752)

Diffusion model are strong but still require amount of high computational resource to synthesis. To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, new method is needed that reduces the computational complexity for both training and sampling. (DDIM give faster inferencing but not in training.)

<img src="/assets/images/Diffusion model/3.png" class="img-medium" alt="Figure 3">

## Departure to Latent Space (1)

Researchers analysis diffusion model with tho step.  

**Perceptual compression :** At the beginning of training, model remove high-frequency details(e.g. hair, thin noise sharp edge etc) but still learns little semantic variations like whole shape or color of image.

**Semantic compression:** In semantic compression, model learned each concept and relation between every object in dataset.

<img src="/assets/images/Diffusion model/5.png" class="img-medium" alt="Figure 5">

And researcher address condensed training only **semantic compression** period with latent space without training whole pixel with end-to-end.

## Method

First at all, they train auto encoder that provide a lower dimensional representational space which is perceptually equivalent to data space. With lower dimensional representational space provide lower complexity of computation, and once trained auto encoder can adapted in any task of various diffusion model. Mapping image data to latent space means efficient connecting with text encoding and other domain. According to above, they propose Latent diffusion model that separated compression from generative learning phase.

### Perceptual Image Compression(3.1)

Auto encoder for LDM based on combination of perceptual loss and patch-based adversarial objective function.

Given $$x \in \mathbb R^{H \times W \times 3}$$ in RGB space, encoder $$E$$ encode $$x$$ to latent representation $$z = E(x)$$, decoder $$D$$ reconstruct $$\tilde{x} = D(x) = D(E(x))$$ where $$z \in \mathbb R^{h \times w \times c}$$

In order to avoid arbitrarily hight-variance latent spaces, they use two different kind of regularizations which called $$KL-reg$$ and $$VQ-reg$$.

### Generative Modeling of Latent Representations

With pair of encoder and decoder, we can access to efficient, low-dimensional latent space. Also, we can utilize **image-specific inductive biases**  which contain U-Net primarily from 2D convolutional layers and focusing the objective on the perceptually most relevant bit using the reweighted bound. And objective function of LDM are represented by below.

$$
  L_{LDM}:= \mathbb E_{\mathcal E(x),\ ϵ∼N(0,1),\ t} \biggr[ ∥ϵ−ϵ_θ(z_t,t)||^2_2 \biggr ]
$$

- $$\mathcal{E}(x)$$: encoded image $$x$$
- $$\epsilon ∼ N(0, 1)$$ random noise adding image from $$t$$.

### Pair of Encoder and Decoder

But this method only focused on semantic compression. Now, decoder $$\mathcal D$$ generate high-resolution and realistic image that pre-trained pair of encoder and decoder.

Below is Architecture of encoder and decoder

- **Encoder $$\mathcal E$$**
    - CNN-based downsampling network
    - Compresses a high-resolution image into a reduced latent representation of size(For example, ***stable diffusion 1.5*** generate $$512*512$$ image from latent space which dimension $$64*64$$.)
    - Outputs mean and log-variance vectors at the final layer, followed by the reparameterization trick.
- **Decoder $$\mathcal D$$**
    - CNN-based upsampling network
    - Gradually restores the resolution starting from the latent representation
    - Incorporates residual blocks and self-attention blocks to enhance texture reconstruction quality

### Conditioning Mechanism

To turn DMs into more flexible conditional image generators, researchers added **cross-attention mechanism.** They introduced $$\tau_\theta(y) \in \mathbb R^{M \times d_\tau}$$ which domain specific encoder that projects $$y$$ to an intermediate representation. It is mapped to the intermediate layer of the U-net via cross-attention layer. As implemented $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax} \bigr( \frac{QK^T}{\sqrt{d}} \bigr) \cdot V$$, cross-attention mechanism can be represented by below.

$$
Q= W^{(i)}_Q · \varphi _i(z_t), \ \ K = W^{(i)}_K ·τ_θ(y), \ \ V = W^{(i)}_V · τ_θ(y)
$$

- $$\varphi_i(z_t)$$: flatten latent vector of intermediate representation from U-net. $$\varphi_i(z_t) \in \mathbb R^{N \times d^i_\epsilon}$$ , where $N$ is number of spatial and $$d^i_\epsilon$$ is channel dimension of each block.
- $$\tau_\theta(y)$$: embedding output from encoder $$\tau_\theta$$ with input $$y$$. $$\tau_\theta(y) \in \mathbb R^{M \times d_\tau}$$, where $$M$$ is number of tokens(num of input sequence) and $$d_\tau$$ dimension of conditional embedding.
- $$W^{(i)}_V \in \mathbb R^{d \times d^i_\epsilon} \ \ \mathrm{and} \ \ W^{(i)}_Q ,\   W^{(i)}_K \in \mathbb R^{d \times d_\tau}$$
    
<img src="/assets/images/Diffusion model/6.png" class="img-medium" alt="Figure 6">


conditioned LDMs either via concatenation or by more general cross-attention mechanism

---

---
