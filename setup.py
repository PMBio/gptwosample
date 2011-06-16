#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools
#import cx_Freeze

__description__ = """Python package for Gaussian process regression in python 

========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions"""

standard_params = dict(name='gptwosample',
      version = '0.0.7',
      description = __description__,
      author = 'Oliver Stegle, Max Zwießele')
#packages = find_packages("./"),

setuptools.setup(
    exclude=['gptwosample/cmd_line_tool'],
    install_requires = ['numpy','scipy'],
    #scripts=['examples/*.py'],
    packages=['gptwosample'],
    package_dir={'gptwosample': './'},
    package_data={'gptwosample': ['./doc/*.html','./doc/*.js','./doc/*.html',
				  './examples/*.py', './examples/*.csv']},
    **standard_params
    )

# cx_Freeze.setup(
#     executables=[cx_Freeze.Executable('cmd_src/GPTwoSample.py', 
#                                       #initScript='gptwosample/__init__.py', base='gptwosample', 
#                                       #path='./gptwosample', 
#                                       #targetDir=None, 
#                                       targetName="GPTwoSample",# includes=None, 
#                                       excludes='gptwosample/cmd_line_tool', packages=["gptwosample"], 
#                                       #replacePaths=None, compress=None, copyDependentFiles=None, appendScriptToExe=None, appendScriptToLibrary=None, icon=None, namespacePackages=None, shortcutName=None, shortcutDir=None)],
#                                       )],
#     **standard_params
#     )


