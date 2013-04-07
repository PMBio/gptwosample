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
      author = 'Max Zwießele, Oliver Stegle',
      author_email='ibinbei@gmail.com',
      url='http://')

setuptools.setup(
    install_requires = ['numpy','scipy','pygp >=1.1.0'],
    packages = setuptools.find_packages('./'),#['gptwosample','examples'],
    package_data ={'gptwosample.examples':['*.csv','*.sh']},
    data_files = [('',['README'])] + get_recursive_data_files('./doc'),
        #[('tests/',['*.py'])],
    entry_points={
        'console_scripts': [
            'gptwosample=gptwosample.__main__:main',
            ]
        },
    test_suite='gptwosample.tests',
    **standard_params
    )
