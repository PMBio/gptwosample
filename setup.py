#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools,os

__description__ = """
Python package for time series comparison by Gaussian Processes
===============================================================


"""

def get_recursive_data_files(name,path):
    out = []
    for (p,d,files) in os.walk(path):
        files = [os.path.join(p,f) for f in files]
        out.append((os.path.join(name,p),files))
    return out

standard_params = dict(name='gptwosample',
      version = '0.0.7',
      description = __description__,
      author = 'Oliver Stegle, Max Zwießele')

setuptools.setup(
    install_requires = ['numpy','scipy'],
    packages = setuptools.find_packages('./'),#['gptwosample','examples'],
    package_data ={'gptwosample.examples':['*.csv','*.sh']},
    data_files = get_recursive_data_files('doc','./doc/doc'),
    **standard_params
    )
