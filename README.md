# ML_AI_imperial_capstone_project

This repository contains the code, analysis, and supporting materials for my capstone project completed as part of the Machine Learning and Artificial Intelligence Professional Certificate at Imperial College London.

## 1. Project overview

This project tackles a **black-box optimisation (BBO)** problem. The objective function is unknown, can only be evaluated by querying selected input points, and returns a scalar output that must be used to guide future decisions. The goal is to use a limited number of expensive evaluations to identify high-performing inputs as efficiently as possible.

Black-box optimisation is important in many practical settings, including hyperparameter tuning, scientific simulation, engineering design, and experimental optimisation. In these problems, the function often has no closed-form expression, gradients are unavailable, and each evaluation may be expensive. As a result, the optimiser must make careful use of all previously collected information.

This project is useful to me because it combines several important ideas from machine learning and data science:

- probabilistic surrogate modelling
- optimisation under uncertainty
- sequential decision-making
- model diagnostics and visualisation

## 2. Inputs and outputs

For each hidden function, the optimiser receives previously evaluated input-output pairs.

If the observed inputs are written as

$$
X = \{\mathbf{x}_1,\dots,\mathbf{x}_n\},
$$

and the corresponding observed outputs as

$$
\mathbf{y} = (y_1,\dots,y_n)^\top,
$$

then the task is to use this information to propose a new query point

$$
\mathbf{x}_{\mathrm{next}}
$$

that is likely to improve on the best result seen so far.

The function itself is never observed directly. Its structure must be inferred from the sampled data alone.

## 3. Challenge objective

The aim is to identify input locations that **maximise** the hidden objective while working under a limited query budget.

This creates a standard exploration-exploitation trade-off:

- **exploration**: sample uncertain or less-visited regions to learn more about the function
- **exploitation**: sample regions that already appear promising

A good black-box optimisation strategy must balance these two goals efficiently.

## 4. Current technical approach

My current method is based on **Gaussian process regression (GPR)** as a surrogate model for the unknown objective function.

A Gaussian process is well suited to black-box optimisation because it provides, for each candidate point \(\mathbf{x}\),

- a **posterior mean** \(\mu(\mathbf{x})\), representing the predicted objective value
- a **posterior standard deviation** \(\sigma(\mathbf{x})\), representing predictive uncertainty

This makes it possible to use both predicted performance and uncertainty when selecting the next query.

### 4.1 Input preprocessing

The input coordinates are scaled before fitting the Gaussian process so that dimensions are treated more comparably during modelling. If \(\mathbf{x} = (x_1,\dots,x_d)\), the scaled input coordinates are written as

$$
x_j^{\mathrm{scaled}}
=
\frac{x_j - x_j^{\min}}{x_j^{\max} - x_j^{\min}},
\qquad j=1,\dots,d.
$$

This improves numerical stability and helps prevent one coordinate from dominating purely because of scale.

### 4.2 Gaussian process surrogate model

The latent objective is modelled as

$$
f(\mathbf{x}) \sim \mathcal{GP}\!\left(0,\,k(\mathbf{x},\mathbf{x}')\right).
$$

The kernel currently used is an RBF-based kernel with a noise term,

$$
k(\mathbf{x},\mathbf{x}')
=
\sigma_f^2
\exp\!\left(
-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\ell^2}
\right)
+
\sigma_n^2 \delta_{\mathbf{x},\mathbf{x}'},
$$

where:

- \(\sigma_f^2\) is the signal variance
- \(\ell\) is the kernel length scale
- \(\sigma_n^2\) is the observation noise variance
- \(\delta_{\mathbf{x},\mathbf{x}'}\) is the Kronecker delta

In implementation terms, this corresponds to a kernel of the form

$$
\text{ConstantKernel} \times \text{RBF} + \text{WhiteKernel}.
$$

### 4.3 Hyperparameter learning via log marginal likelihood

An important refinement in my current workflow is that I no longer manually sweep over assumed noise levels. Instead, the Gaussian process learns its own hyperparameters directly from the data by maximising the **log marginal likelihood**.

If the kernel depends on hyperparameters \(\theta\), then the covariance matrix over the observed inputs is

$$
K_\theta(X,X)
=
\bigl[k(\mathbf{x}_i,\mathbf{x}_j)\bigr]_{i,j=1}^n.
$$

The hyperparameters are chosen by maximising

$$
\log p(\mathbf{y}\mid X,\theta)
=
-\frac{1}{2}\mathbf{y}^\top K_\theta^{-1}\mathbf{y}
-\frac{1}{2}\log|K_\theta|
-\frac{n}{2}\log(2\pi).
$$

This objective balances:

- fit to the observed data
- model complexity
- probabilistic consistency of the GP model

In practice, this optimisation is carried out internally by `GaussianProcessRegressor.fit(...)`, with multiple optimiser restarts to reduce the chance of a poor local optimum.

### 4.4 Posterior prediction

Once fitted, the GP provides a posterior mean and posterior covariance at candidate inputs \(X_*\):

$$
\mu(X_*)
=
K(X_*,X)\,K(X,X)^{-1}\mathbf{y},
$$

$$
\Sigma(X_*)
=
K(X_*,X_*)
-
K(X_*,X)\,K(X,X)^{-1}K(X,X_*).
$$

The uncertainty used in the acquisition functions is the posterior standard deviation

$$
\sigma(\mathbf{x})
=
\sqrt{\operatorname{Var}(f(\mathbf{x})\mid X,\mathbf{y})}.
$$

## 5. Acquisition functions

To choose candidate next points, I evaluate three standard Bayesian optimisation acquisition functions.

### 5.1 Upper Confidence Bound (UCB)

$$
a_{\mathrm{UCB}}(\mathbf{x})
=
\mu(\mathbf{x}) + \beta\,\sigma(\mathbf{x}),
$$

where \(\beta > 0\) controls the exploration-exploitation trade-off.

### 5.2 Expected Improvement (EI)

Let

$$
f_{\mathrm{best}} = \max_{1\le i\le n} y_i.
$$

Then define

$$
Z(\mathbf{x})
=
\frac{\mu(\mathbf{x}) - f_{\mathrm{best}} - \xi}{\sigma(\mathbf{x})}.
$$

The expected improvement is

$$
a_{\mathrm{EI}}(\mathbf{x})
=
\bigl(\mu(\mathbf{x}) - f_{\mathrm{best}} - \xi\bigr)\Phi(Z(\mathbf{x}))
+
\sigma(\mathbf{x})\phi(Z(\mathbf{x})),
$$

where \(\Phi\) and \(\phi\) are the standard normal CDF and PDF.

### 5.3 Probability of Improvement (PI)

Using the same definition of \(Z(\mathbf{x})\),

$$
a_{\mathrm{PI}}(\mathbf{x})
=
\Phi(Z(\mathbf{x})).
$$

These acquisition functions encode different trade-offs between exploitation and exploration. In practice, I inspect all three, though a cleaner long-term strategy is likely to choose one main acquisition rule and keep the others as comparisons.

## 6. Additional structure learning: PCA and SVM filtering

Alongside the GP, I also analyse the geometry of the observed input cloud using **principal component analysis (PCA)**.

If \(X\) is the scaled input matrix, PCA projects it into a lower-dimensional representation,

$$
X_{\mathrm{PCA}} = XW,
$$

where the columns of \(W\) are the leading principal directions.

I currently use the first two principal components to visualise the sampled region and to define a coarse promising-region model. Observed outputs are thresholded to label higher-performing points, and an RBF-kernel SVM is then fitted in the 2D PCA space. This gives a simple classifier for regions that appear more promising based on the current observations.

The purpose of this step is not to replace the Gaussian process, but to add additional structure to candidate generation by restricting attention to regions that appear potentially useful.

## 7. Workflow summary

For each hidden function, the current workflow is:

1. Load the observed input-output pairs.
2. Scale the input coordinates.
3. Fit a Gaussian process with learnable kernel hyperparameters.
4. Learn the kernel length scale, signal variance, and noise level by maximising the log marginal likelihood.
5. Evaluate the GP posterior mean and uncertainty on a candidate set.
6. Use PCA to visualise the observed input geometry.
7. Threshold outputs and fit an SVM in PCA space to identify a coarse promising region.
8. Sample candidates from that region and map them back to the original input space.
9. Evaluate UCB, EI, and PI on candidate points to propose the next query.

## 8. Libraries and tools used

The main Python libraries used in this project are:

- **NumPy** for numerical array operations
- **pandas** for organising and handling data
- **scikit-learn** for Gaussian process regression, PCA, and SVM models
- **Matplotlib** for plotting surrogate predictions, uncertainty, PCA projections, and diagnostics

These libraries were appropriate because the project focuses on relatively small datasets, interpretable modelling, and fast iteration rather than large-scale deep learning training.

## 9. Current strengths of the method

The main strengths of the current approach are:

- it uses a **probabilistic surrogate model** rather than relying on purely heuristic search
- it incorporates **uncertainty estimates** directly into the optimisation process
- it avoids manual noise sweeps by learning GP hyperparameters from data
- it adds a **geometric structure-learning step** through PCA and SVM filtering
- it remains relatively interpretable and easy to inspect visually

## 10. Possible improvements and future work

There are several directions that could improve the current workflow.

- Use **adaptive output transformations** depending on the objective, for example raw outputs, standardisation, or `arcsinh` / signed-log transforms for extreme dynamic ranges.
- Replace the isotropic RBF kernel with an **ARD kernel**, allowing a separate length scale in each input dimension.
- Compare **RBF and Matérn kernels** to test whether a less smooth prior improves surrogate performance.
- Use **Expected Improvement (EI)** as the main acquisition function, while keeping UCB and PI as comparison baselines.
- Make the exploration parameters **adaptive across rounds**, for example reducing \(\beta\) or \(\xi\) as more data is collected.
- Add a **diversity penalty** or minimum-distance rule to avoid proposing points too close to existing observations.
- Replace purely uniform random candidate generation with **Sobol sequences** or **Latin hypercube sampling** for better coverage of the search space.
- Optimise the acquisition function more directly, for example using **multi-start local optimisation** after an initial Monte Carlo screening step.
- Improve the PCA/SVM stage by using **adaptive thresholds** or probabilistic success labels rather than a fixed percentile cutoff.
- Run **ablation studies** to test whether the PCA/SVM filtering stage genuinely improves performance compared with a simpler GP-only BO baseline.
- Combine the classifier and acquisition function into a **hybrid acquisition rule**, for example
  $$
  a_{\mathrm{hybrid}}(\mathbf{x}) = \mathrm{EI}(\mathbf{x})\,p(\text{promising}\mid \mathbf{x}),
  $$
  so that search is guided by both GP uncertainty and a learned promising-region score.
- Track more diagnostics, such as best value so far, uncertainty at selected points, and distance to previous samples, to better understand optimisation behaviour across rounds.

## 11. Repository documentation plan

To make the reasoning behind the project clear to peers, facilitators, and future employers, I plan to document the workflow in several layers:

- a high-level explanation in this README
- clear comments and markdown explanations in notebooks
- visual diagnostics showing how the GP and acquisition functions behave
- references to the main Bayesian optimisation ideas that motivated the design

## 12. Ongoing development

This project is still evolving. I am continuing to refine the surrogate model, investigate acquisition behaviour, and explore whether ideas from more advanced Bayesian optimisation methods can be integrated into the current pipeline while keeping the method interpretable and reproducible.