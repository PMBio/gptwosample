#!/usr/bin/python
# -*- coding: utf-8 -*-
import setuptools, os

with open("README", 'r') as r:
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
      version='0.1.11',
      description=__description__,
      long_description=README_TEXT,
      author='Max Zwießele, Oliver Stegle',
      author_email='ibinbei@gmail.com',
      url='https://www.assembla.com/code/gptwosample/git/nodes',
      license='Apache v2.0')

reqs = ['scipy', 'pygp >=1.1.07', 'matplotlib >=1.2']
setuptools.setup(
    install_requires=reqs,
    requires=map(lambda x: x.split(" ")[0], reqs),
    packages=setuptools.find_packages('./'),  # ['gptwosample','examples'],
    package_data={'gptwosample.examples':['*.csv', '*.sh']},
    data_files=get_recursive_data_files('doc'),  # [('',['README'])] +
    # [('tests/',['*.py'])],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'gptwosample=gptwosample.__main__:main',
            ]
        },
    test_suite='gptwosample.tests',
    **standard_params
    )
