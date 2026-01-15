---
layout: post
title: Fairness in Diffusion Models
paper: "Bias and Fairness in Diffusion Models"
link: https://arxiv.org/abs/xxxx.xxxxx
---

## Problem Setting

We analyze bias amplification in diffusion-based generative models.

## Key Equation

$$
p_\theta(x_0) = \int p_\theta(x_0 \mid x_T) p(x_T) \, dx_T
$$

## Method

```python
def fairness_metric(x, y):
    return (x.mean() - y.mean()).abs()
