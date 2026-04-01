# ML_AI_imperial_capstone_project
This repo contains all the codes associated with my capstone project, completed during the ML/AI Pcert course at Imperial college London
# BBO Capstone Project README Draft

## 1. Project overview

This project tackles a **black-box optimisation (BBO)** problem: the objective function is unknown, can only be queried at selected input points, and returns a performance value that must be used to guide future queries. The overall goal is to learn from a limited set of expensive evaluations and choose new inputs that are likely to perform well.

This is highly relevant in real-world machine learning because many important optimisation problems are black-box in nature. Hyperparameter tuning, engineering design, scientific simulation and experimental optimisation often involve functions with no closed-form expression, costly evaluations and limited query budgets. In such settings, it is not enough to search randomly; the optimiser must use previous information efficiently.

For me, this project is useful because it brings together probabilistic modelling, optimisation and decision-making under uncertainty. These are central ideas in data science and machine learning roles, particularly where model-based optimisation, experimentation or high-value decision support are important.

## 2. Inputs and outputs

The model receives previously evaluated input-output pairs for each hidden function. The inputs are numerical vectors in a bounded search space, and the outputs are scalar performance values returned by the black-box function.

In practice, the workflow is:

- input: a set of previously queried points  
  $$
  X = \{x_1, \dots, x_n\}
  $$
- input: the corresponding observed outputs  
  $$
  y = \{y_1, \dots, y_n\}
  $$
- output: a proposed next input point  
  $$
  x_{\text{next}}
  $$
  at which to query the black-box function

At this stage, the optimiser does not know the analytic form of the function, its derivatives or its smoothness in advance. It must infer useful structure from the sampled data alone.

## 3. Challenge objectives

The objective is to identify input locations that **maximise** the hidden function while using a limited number of queries. The core difficulty is that each new query is valuable, so the method must make informed decisions rather than relying on brute-force search.

The main constraints are:

- the function is unknown and only observable through sampled evaluations
- the number of queries is limited
- the quality of future proposals depends strongly on how well the current surrogate model captures the function
- exploration and exploitation must be balanced carefully

This means the challenge is not just to find a good point, but to do so efficiently under uncertainty.

## 4. Technical approach

My current approach is based on **Gaussian process regression (GPR)** as a surrogate model for the unknown function. The Gaussian process is attractive because it provides both:

- a predicted mean response at unseen points
- a predictive uncertainty, which is essential for choosing where to sample next

A key development in my approach was moving away from manually sweeping kernel noise assumptions. Instead, I now fit a Gaussian process whose hyperparameters are learned automatically by maximising the **log marginal likelihood**. This allows the model to learn quantities such as the kernel length scale and noise level directly from the observed data, making the surrogate more adaptive and principled.

More specifically, if the Gaussian process kernel depends on hyperparameters $\theta$, these are chosen by maximising the log marginal likelihood

$$
\log p(y \mid X, \theta)
=
-\frac{1}{2} y^\top K_\theta^{-1} y
-\frac{1}{2} \log |K_\theta|
-\frac{n}{2} \log(2\pi),
$$

where $K_\theta$ is the covariance matrix determined by the kernel and its hyperparameters.

For an RBF-based Gaussian process model, the kernel has the form

$$
k(x, x')
=
\sigma_f^2
\exp\!\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right)
+
\sigma_n^2 \delta_{x,x'},
$$

where:

- $\ell$ is the kernel length scale
- $\sigma_f^2$ is the signal variance
- $\sigma_n^2$ is the noise variance

To select candidate points, I use three standard Bayesian optimisation acquisition functions:

### Upper Confidence Bound (UCB)

$$
\mathrm{UCB}(x) = \mu(x) + \beta \sigma(x),
$$

which rewards both high predicted performance and high uncertainty.

### Expected Improvement (EI)

$$
\mathrm{EI}(x)
=
(\mu(x)-f_{\text{best}}-\xi)\Phi(z)
+
\sigma(x)\phi(z),
$$

with

$$
z = \frac{\mu(x)-f_{\text{best}}-\xi}{\sigma(x)},
$$

which targets points expected to improve on the current best observation.

### Probability of Improvement (PI)

$$
\mathrm{PI}(x)
=
\Phi\!\left(
\frac{\mu(x)-f_{\text{best}}-\xi}{\sigma(x)}
\right),
$$

which measures the probability of outperforming the current best observation.

These acquisition functions represent different exploration-exploitation trade-offs. UCB can be made more exploratory by increasing its $\beta$ parameter, while EI and PI can be adjusted through their $\xi$ threshold. My current plan is to rely mainly on the learned-parameter GPR model and tune these acquisition parameters to control how aggressively the search explores.

I also began exploring **PCA** and **t-SNE** as tools to visualise how the search moves through the input space, although I am treating those as diagnostic tools rather than core decision rules for now. Overall, my approach is to use a probabilistic surrogate model, let it learn its own hyperparameters from data, and use acquisition functions to make informed sequential decisions under uncertainty.