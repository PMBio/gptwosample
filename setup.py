#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools, os

with open(os.path.join("README"), 'r') as r:
    README_TEXT = r.read()
    

__description__ = """
    Differential expression detection between gene expression time series experiments with confounder correction and timehift detection.
    """

def get_recursive_data_files(path):
    out = []
    for (p, d, files) in os.walk(path):
        files = [os.path.join(p, f) for f in files]
        out.append((p, files))
    return out

standard_params = dict(name='gptwosample',
      version='0.1.19',
      description=__description__,
      long_description=README_TEXT,
      author='Max Zwießele, Oliver Stegle',
      author_email='ibinbei@gmail.com',
      url='https://www.assembla.com/code/gptwosample/git/nodes',
      license='Apache License v2.0')

reqs = ['numpy >=1.7.1', 'scipy >=0.12.0', 'pygp >=1.1.08', 'matplotlib >=1.2']
data_files = get_recursive_data_files('doc') + [('', ["LICENSE"])]

setuptools.setup(
    install_requires=reqs,
    requires=map(lambda x: x.split(" ")[0], reqs),
    packages=setuptools.find_packages(os.path.curdir), # ['gptwosample','examples'],
    package_data={'gptwosample.examples':['*.csv', '*.sh', '*.txt']},
    #data_files=data_files,
    # [('tests/',['*.py'])],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'gptwosample=gptwosample.__main__:main',
            ]
        },
    test_suite='test',
    classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
        
          ],
    **standard_params
    )
