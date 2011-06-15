#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


__description__ = """Python package for Gaussian process regression in python 

========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions"""

setup(name='gptwosample',
      version = '0.0.7',
      description = __description__,
      author = 'Oliver Stegle, Max Zwießele',
      #packages = find_packages("./"),
      packages=['gptwosample'],
      package_dir={'gptwosample': './'},
      exclude=['gptwosample/cmd_line_tool'],
      install_requires = ['numpy','scipy']
      )
