.. _confounders:

Accounting for confounding factors
==========================================================
We detect common confounding factors using probabilistic principal component
analysis modeled by gaussian process latent variable models (GPLVM)
[Lawrence2004]_. This probabilistic approach to detect low
dimensional significant features can be interpreted as detecting
common confounding factors in time series experiments by applying
GPLVM in advance to two-sample tests of [Stegle2010]_ on the
whole dataset. Two-sample tests on Gaussian Processes decide
differential expression based on the bayes factor of marginal probabilities
for control and treatment being modeled by one common or two separate
underlying function(s). As GPLVM is based on Gaussian Processes it
provides a covariance structure of confounders in the dataset. We take
this covariance structure between features to build up a two-sample
Gaussian Process model taking confounding factors throughout the
dataset into account.  

.. [Lawrence2004]   Neil Lawrence, `Gaussian process latent variable models for visualisation of high dimensional data`, Advances in neural information processing systems, `2004`
.. [Stegle2010]        Stegle, Oliver and Denby, Katherine J and Cooke, Emma J and Wild, David L and Ghahramani, Zoubin and Borgwardt, Karsten M, `A robust Bayesian two-sample test for detecting intervals of differential gene expression in microarray time series`, Journal of Computational Biology, `2010`
