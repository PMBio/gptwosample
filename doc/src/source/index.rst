.. GPTwoSample documentation master file, created by
   sphinx-quickstart on Fri Oct 29 11:59:50 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPTwoSample
+++++++++++++++++++++++

gptwosample.py is a tool to run two-sample tests on time series
differential gene expression experiments. It can either be called form
the command line or using the interactive Python Modules. Here, we
will explore usage from the command line, for detailed description,
refers to the other sections. 

Command line Tool
======================
``gptwosample`` is designed to compare two gene expression time series
experiments, including the possibility for several replicates in each
experiment. The test fits latent functions to both time series,
comparing the assumption that all data originated from a single latent
process (`common model fit`) or a two distinct separate time series
(`individual model fit`). See [Stegle2010]_ for details. The Raw gene expression data can
be supplied via simple CSV files, one for each experiments. The data
format is flexible, permits missing values and non-synchronized time
points. Fur full details please see :ref:`dataformat`.

.. _install:

Installing the package
------------------------
To install ``gptwosample`` run::

 pip install gptwosample 

or run::

 python setup.py install

from ``gptwosample`` directory if you downloaded the source.

This will install a script ``gptwosample`` into you python bin. In
some cases this bin is not in $PATH and must be included extra.

Try printing the full help of the script using::

 gptwosample --help

restart your unix shell if it is not yet registered.

To run optional package tests before installing run::

 python setup test

Example usage
----------------------------
Once the data has been prepared, GPTwoSample can be executed from the
unix command line.

General command line parameters of interest include::

 --help
 -v

Also, to create plots of the fitted functions, which creates verbose plots illustrating the fit for every tested gene::

 -p

For example, to run the basic ``gptwosample`` model on the tutorial
datasets provided alongside the package including verbose output and
plots, run::

 gptwosample -v -p -t -o ./examplerun/ examples/ToyCondition1.csv examples/ToyCondition2.csv

This stores results in ``./examplerun/``. Quantitative readouts
summarizing the differential expression stores ares provided in
"results.csv" (see :ref:`results` for format). Plots in will be saved
in a subfolder ``./examplerun/plots/``.


Further Details
----------------------------------
.. toctree::
  usage
  confounders
  timeshift
  results
  tutorial

.. [Lawrence2004]   Neil Lawrence, `Gaussian process latent variable models for visualisation of high dimensional data`, Advances in neural information processing systems, `2004`
.. [Stegle2010]        Stegle, Oliver and Denby, Katherine J and Cooke, Emma J and Wild, David L and Ghahramani, Zoubin and Borgwardt, Karsten M, `A robust Bayesian two-sample test for detecting intervals of differential gene expression in microarray time series`, Journal of Computational Biology, `2010`

Developer
============
.. toctree::   
   GPTwoSample <base>
   Generate beautiful plots with gptwosample.plot <plot>
   Data handling for GPTwoSample taksks <data>

.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

