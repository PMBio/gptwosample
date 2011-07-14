#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools,os

__description__ = """
Python package for time series comparison by Gaussian Processes
===============================================================


"""

def find_example_files():
    for (p,d,f) in os.walk('./examples'):
	examples = [os.path.join(p,file) for file in f if os.path.splitext(file)[-1] in ['.csv','.sh']]
    return examples

standard_params = dict(name='gptwosample',
      version = '0.0.7',
      description = __description__,
      author = 'Oliver Stegle, Max Zwießele')

setuptools.setup(
    install_requires = ['numpy','scipy'],
    packages=setuptools.find_packages('./'),#['gptwosample','examples'],
    package_data={'examples':['*.csv','*.py']},
    data_files=[('examples',find_example_files())],
    **standard_params
    )
