#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools

__description__ = """
Python package for time series comparison by Gaussian Processes
===============================================================


"""

standard_params = dict(name='gptwosample',
      version = '0.0.7',
      description = __description__,
      author = 'Oliver Stegle, Max Zwießele')

setuptools.setup(
    install_requires = ['numpy','scipy'],
    packages=['gptwosample'],
    package_dir={'gptwosample': './'},
    
    **standard_params
    )
