---
layout: post
title: "Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment"
paper: "Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment(NeurIPS 2025) - KAIST"
category: computer-vision
summary: paper reivew of chain of zoom
link: https://arxiv.org/pdf/2505.18600
---

## 간단 리뷰
나의 관심 분야와는 거리가 있는 super-resolution에 관한 논문이지만 해결 방법이 매우 독창적이어서 한 번 읽어보았다. VLM을 이용한 간단한 아이디어만으로도 기존의 방법보다 효율적으로
작동할 수 있다는게 신기했다.

![image1.png](/assets/images/CoZ/1.png)

They propose Chain of Zoom(CoZ) in Single image super resolution(SISR), which repeatedly reuse a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolution without training.

They solve fundamental question: How can we effectively utilize super resolution models to  explore much higher resolutions than they trained? SR models has limitation with computational prohibitive due to memory and time constraints.

## What is autoregression chain?
    
Autoregression chain is modeling framework where a sequence is generated on element(token) at a time and each new one is conditioned on all previous generated.

**Probabilistic view**

$$
p(x_1, x_2, \dots, x_T) = \prod_{T}^{t=1}p(x_t | x_1, \dots x_{t-1})
$$

It express joint distribution  as a chain of conditional probabilities.
    

## Methods

![image2.png](/assets/images/CoZ/2.png)

CoZ use intermediated scale-states modeling to low resolution image and high resolution images with tractable components of scale level autoregressive frameworks. In CoZ process VLM model supply description of zoomed images.


### Why CoZ need VLM in SR process?
    
In extreme super resolution task, each pixel information level is too low that can not be semantic meanings. So, VLM supply description of images to help SR model to keep in semantic track of process. 

In here, just vision model is hard to convey explicit semantic information of target images.
    

## Intermediate Scale State Modeling

Paper propose to bridge the gap between a target HR image $$x_H \in \mathbb R^{d_i}$$ and inout LR image $$x_L \in \mathbb R^{d_0}$$ with intermediate Scale state $$x_i \in \mathbb R^{d_i}$$.

To reduce hallucinations caused by incorrect text guidance across scale, they used non-Markov multi-scale aware. $$x_{i-1}$$ and $$x_{i-2}$$ are input when generating prompt with VLM.

$$
P(x_0, x_1, \dots, x_{n}) = p(x_0, x_1) \prod_{i=2}^{n}p(x_i | x_{i-1}, x_{i-2})
$$

$$
p(x_i | x_{i-1}, x_{i-2}) = ∫p(x_i | x_{i-1}, x_{i-2}, c_i)p(x_i)| x_{i-1}, x_{i-2}dc_i
$$

Thus, CoZ’s Objective function is expressed as below(maximize), and they used parameterized models $\theta$ and $\phi$

$$
\mathcal L = \log p(x_0) + \sum^n_{i=1} \log p(x_i | x_{i-1}, x_{i-2}, c_i) + \sum^n_{i=1} \log p(c_i | x_{i-1}, x_{i-2})
$$

## Training Multi-Scale-Aware Prompt Extraction using RL

![image3.png](/assets/images/CoZ/3.png)

To prevent high-frequency hallucinations, they fine-tune the prompt extraction VLM so that its textual guidance aligns with human aesthetic and semantic preferences.

### Limitation

1. Repeated application for extreme magnification can cause error accumulation over iterations.
2. CoZ is effective at super-resolving image details but it can miss global constraits.
3. Architecture that include a VLM can limit reasoning over super-resolution.
