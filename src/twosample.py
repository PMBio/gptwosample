"""
Different implementations for using GPTwoSample
===============================================

Some implementations of GPTwoSample, to detect differential expression between two timelines.
"""

import sys
sys.path.append("./../")

from GPTwoSample.src import GPTwoSample

from pygp import gpr as GPR, composite as COMP

import scipy as SP

class GPTwoSampleMLII(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to wach other.

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.

    """
    def __init__(self, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        GPTwoSample.__init__(self, *args, **kwargs)


    def _init_twosample_model(self, covar):
        gpr1 = GPR.GP(covar)
        gpr2 = GPR.GP(covar)
        individual_model = COMP.GroupGP([gpr1,gpr2])
        common_model = GPR.GP(covar)
        # set models for this GPTwoSample Test
        self._models = {'individual':individual_model,'common':common_model}
