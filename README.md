# Bayesian Optimisation for Black-Box Functions

This repository contains the code, analysis, and supporting material for my capstone project for the **Machine Learning and Artificial Intelligence Professional Certificate at Imperial College London**.

The project investigates how **Bayesian optimisation** can be used to maximise an unknown, expensive-to-evaluate objective function under a limited query budget.

## Project overview

In black-box optimisation, the objective function is not available in closed form. It can only be evaluated by selecting an input and observing the corresponding scalar output. Gradients are unavailable, and each evaluation may be costly.

The central task is therefore to use all previously observed data to choose the next evaluation point efficiently.

This setting appears in applications such as:

- hyperparameter tuning;
- scientific simulation;
- engineering design;
- experimental optimisation; and
- other sequential decision-making problems with expensive evaluations.

The project combines:

- probabilistic surrogate modelling;
- optimisation under uncertainty;
- acquisition-function design;
- geometric analysis of the observed inputs; and
- model diagnostics and visualisation.

## Problem formulation

For each hidden objective function, the optimiser receives a set of observed input-output pairs,

$$
\mathcal{D}_n
=
\left\{
(\mathbf{x}_i,y_i)
\right\}_{i=1}^{n},
$$

where

$$
\mathbf{x}_i \in \mathcal{X} \subseteq \mathbb{R}^{d}
$$

is an evaluated input and

$$
y_i = f(\mathbf{x}_i) + \varepsilon_i
$$

is the corresponding observed output.

The aim is to choose a new query point,

$$
\mathbf{x}_{n+1},
$$

that is likely to improve on the best value observed so far while using the remaining evaluation budget effectively.

This requires balancing two competing objectives:

- **Exploration:** evaluate uncertain or sparsely sampled regions to learn more about the objective.
- **Exploitation:** evaluate regions that already appear likely to produce high objective values.

## Methodology

### 1. Input preprocessing

Each input dimension is scaled to a comparable range before fitting the surrogate model. For a bounded input coordinate \(x_j\),

$$
x_j^{\mathrm{scaled}}
=
\frac{x_j-x_j^{\min}}
{x_j^{\max}-x_j^{\min}},
\qquad
j=1,\ldots,d.
$$

This transformation improves numerical conditioning and prevents dimensions with larger numerical ranges from dominating the kernel distance calculation.

### 2. Gaussian process surrogate

The unknown objective is modelled using **Gaussian process regression (GPR)**,

$$
f(\mathbf{x})
\sim
\mathcal{GP}
\left(
m(\mathbf{x}),
k(\mathbf{x},\mathbf{x}')
\right).
$$

The current implementation uses an RBF covariance function with an additive white-noise term,

$$
k(\mathbf{x},\mathbf{x}')
=
\sigma_f^2
\exp
\left(
-\frac{\lVert\mathbf{x}-\mathbf{x}'\rVert^2}
{2\ell^2}
\right)
+
\sigma_n^2
\delta_{\mathbf{x},\mathbf{x}'},
$$

where:

- \(\sigma_f^2\) is the signal variance;
- \(\ell\) is the kernel length scale;
- \(\sigma_n^2\) is the observation-noise variance; and
- \(\delta_{\mathbf{x},\mathbf{x}'}\) is the Kronecker delta.

In `scikit-learn`, this is represented by a kernel of the form:

```python
ConstantKernel() * RBF() + WhiteKernel()
```

The Gaussian process provides both:

- a posterior mean \(\mu(\mathbf{x})\), representing the predicted objective value; and
- a posterior standard deviation \(\sigma(\mathbf{x})\), representing predictive uncertainty.

These quantities allow the optimiser to account for both expected performance and uncertainty when selecting the next point.

### 3. Hyperparameter learning

The kernel hyperparameters are learned from the observed data by maximising the **log marginal likelihood**.

For kernel hyperparameters \(\theta\), define the covariance matrix

$$
K_{\theta}
=
\left[
k_{\theta}(\mathbf{x}_i,\mathbf{x}_j)
\right]_{i,j=1}^{n}.
$$

The log marginal likelihood is

$$
\log p(\mathbf{y}\mid X,\theta)
=
-\frac{1}{2}
\mathbf{y}^{\mathsf{T}}
K_{\theta}^{-1}
\mathbf{y}
-\frac{1}{2}
\log\lvert K_{\theta}\rvert
-\frac{n}{2}
\log(2\pi).
$$

This objective balances:

- agreement with the observed data;
- model complexity; and
- the probabilistic consistency of the surrogate model.

The optimisation is performed internally by `GaussianProcessRegressor.fit(...)`. Multiple optimiser restarts are used to reduce sensitivity to poor local optima.

### 4. Posterior prediction

At a set of candidate inputs \(X_*\), the Gaussian process produces the posterior mean

$$
\mu(X_*)
=
K(X_*,X)
K(X,X)^{-1}
\mathbf{y},
$$

and posterior covariance

$$
\Sigma(X_*)
=
K(X_*,X_*)
-
K(X_*,X)
K(X,X)^{-1}
K(X,X_*).
$$

The predictive uncertainty used by the acquisition functions is

$$
\sigma(\mathbf{x})
=
\sqrt{
\operatorname{Var}
\left[
f(\mathbf{x})
\mid
X,\mathbf{y}
\right]
}.
$$

In practice, the implementation uses the numerically stable linear-algebra routines provided by `scikit-learn` rather than forming matrix inverses explicitly.

## Acquisition functions

The current workflow compares three standard acquisition functions.

### Upper Confidence Bound

For a maximisation problem,

$$
a_{\mathrm{UCB}}(\mathbf{x})
=
\mu(\mathbf{x})
+
\kappa\sigma(\mathbf{x}),
$$

where \(\kappa>0\) controls the strength of exploration.

A larger value of \(\kappa\) places more emphasis on uncertain regions, while a smaller value favours exploitation.

### Expected Improvement

Let

$$
f_{\mathrm{best}}
=
\max_{1\leq i\leq n} y_i
$$

be the best observed objective value. Define

$$
Z(\mathbf{x})
=
\frac{
\mu(\mathbf{x})
-
f_{\mathrm{best}}
-
\xi
}{
\sigma(\mathbf{x})
},
$$

where \(\xi\geq 0\) controls the desired improvement margin.

The expected improvement is

$$
a_{\mathrm{EI}}(\mathbf{x})
=
\left(
\mu(\mathbf{x})
-
f_{\mathrm{best}}
-
\xi
\right)
\Phi\!\left(Z(\mathbf{x})\right)
+
\sigma(\mathbf{x})
\phi\!\left(Z(\mathbf{x})\right),
$$

where \(\Phi\) and \(\phi\) are the standard normal cumulative distribution function and probability density function, respectively.

When \(\sigma(\mathbf{x})=0\), the expected improvement is set to zero.

### Probability of Improvement

Using the same definition of \(Z(\mathbf{x})\),

$$
a_{\mathrm{PI}}(\mathbf{x})
=
\Phi\!\left(Z(\mathbf{x})\right).
$$

Probability of Improvement is intuitive, but it does not account for the magnitude of a possible improvement. For that reason, Expected Improvement is a stronger candidate for the primary acquisition rule, while UCB and PI remain useful comparison baselines.

## PCA and SVM candidate filtering

An optional candidate-filtering stage is used alongside the Gaussian process.

The scaled input matrix is projected onto its leading principal components,

$$
X_{\mathrm{PCA}}
=
\widetilde{X}W,
$$

where \(\widetilde{X}\) is the centred input matrix and the columns of \(W\) contain the leading principal directions.

The first two principal components are used to:

1. visualise the geometry of the observed inputs;
2. label a subset of observations as relatively high-performing; and
3. fit an RBF-kernel support vector classifier in the reduced space.

The classifier provides a coarse estimate of regions that appear promising. Candidate points can then be sampled from these regions, mapped back to the original input space, and scored using the Gaussian-process acquisition functions.

This stage is a heuristic candidate-generation mechanism rather than a replacement for the Gaussian process. Because a two-dimensional PCA projection may discard information relevant to the objective, its effect should be evaluated against a simpler GP-only baseline.

## Optimisation workflow

For each hidden objective function, the current workflow is:

1. Load the observed input-output pairs.
2. Scale the input coordinates.
3. Fit a Gaussian process with learnable kernel hyperparameters.
4. Estimate the signal variance, length scale, and noise level by maximising the log marginal likelihood.
5. Generate a set of candidate inputs.
6. Evaluate the Gaussian-process posterior mean and uncertainty at each candidate.
7. Optionally use PCA and SVM classification to restrict candidate generation to a coarse promising region.
8. Score the candidates using UCB, Expected Improvement, and Probability of Improvement.
9. Select the candidate that maximises the chosen acquisition function.
10. Submit the selected point as the next query.

## Libraries

The main Python libraries used in the project are:

- **NumPy** for numerical array operations;
- **pandas** for data organisation and manipulation;
- **scikit-learn** for Gaussian process regression, PCA, scaling, and SVM classification; and
- **Matplotlib** for visualisation and diagnostic plots.

These libraries are appropriate for the relatively small datasets and interpretable modelling workflow used in this project.

## Current strengths

The current approach:

- uses a probabilistic surrogate rather than a purely heuristic search strategy;
- incorporates predictive uncertainty directly into the optimisation process;
- learns Gaussian-process hyperparameters from the observed data;
- supports multiple acquisition strategies;
- includes an optional geometric candidate-filtering stage; and
- remains interpretable through posterior, acquisition, and PCA diagnostics.

## Limitations

The current implementation also has several limitations:

- an isotropic RBF kernel assumes the same characteristic length scale in every input dimension;
- the surrogate may be sensitive to extreme output ranges or outliers;
- random candidate generation may leave parts of the search space poorly covered;
- the PCA/SVM filter may remove useful regions if the projection or classification threshold is misleading;
- comparing several acquisition functions without a fixed selection rule can make the optimisation policy inconsistent; and
- performance has not yet been fully established through systematic ablation studies.

## Planned improvements

Planned extensions include:

- applying adaptive output transformations, such as standardisation, `arcsinh`, or signed-log transforms;
- replacing the isotropic RBF kernel with an automatic relevance determination kernel using one length scale per input dimension;
- comparing RBF and Matérn kernels;
- adopting Expected Improvement as the primary acquisition function;
- adapting \(\kappa\) or \(\xi\) as the query budget is consumed;
- enforcing a minimum distance from previously evaluated points;
- replacing uniform random candidates with Sobol sequences or Latin hypercube sampling;
- optimising the acquisition function using multi-start local optimisation after an initial global screening stage;
- using adaptive or probabilistic labels in the PCA/SVM stage;
- testing the PCA/SVM filter against a GP-only baseline;
- evaluating hybrid acquisition rules such as

$$
a_{\mathrm{hybrid}}(\mathbf{x})
=
a_{\mathrm{EI}}(\mathbf{x})
\,p(\text{promising}\mid\mathbf{x});
$$

- tracking best-so-far performance, selected-point uncertainty, candidate diversity, and distance from previous observations; and
- adding reproducible experiments with fixed random seeds and clearly recorded optimisation settings.

## Project status

This project is under active development. Current work focuses on refining the surrogate model, improving acquisition optimisation, evaluating the value of the PCA/SVM filtering stage, and making the overall workflow more reproducible.
