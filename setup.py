#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup


__description__ = """Package for using GPTwoSample
=============================

This module allows the user to compare two timelines with respect to diffferential expression.

It compares two timeseries against each other, depicting whether these two
timeseries were more likely drawn from the same function, or from
different ones. This prediction is defined by which covariance function :py:class:`pygp.covar` you use."""

setup(name='gptwosample',
      #namespace_packages=['covar'],
      version = '0.0.7',
      description = __description__,
      #summary = __description__.split("\n")[0],
      #platform = "Linux/MaxOSX/Windows"
      #author = 'Oliver Stegle, Max Zwießele',
      #author_email = 'email_not_yet@support.ed',
      #url = 'no.url.given'
      #install_requires = ['numpy','scipy']
      )
