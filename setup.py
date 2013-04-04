#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools,os

__description__ = """
Python package for time series comparison by Gaussian Processes
===============================================================


"""

def get_recursive_data_files(path):
    out = []
    for (p,d,files) in os.walk(path):
        files = [os.path.join(p,f) for f in files]
        out.append((p,files))
    return out

standard_params = dict(name='gptwosample',
      version = '0.1.7a',
      description = __description__,
      author = 'Max Zwießele, Oliver Stegle')

setuptools.setup(
    install_requires = ['numpy','scipy','pygp'],
    packages = setuptools.find_packages('./'),#['gptwosample','examples'],
    package_data ={'gptwosample.examples':['*.csv','*.sh']},
    data_files = [('',['README'])] + get_recursive_data_files('./doc') + [('tests/',['*.py'])],
    test_suite='tests',
    **standard_params
    )
