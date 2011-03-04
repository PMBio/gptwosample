"""
Different implementations for using GPTwoSample
===============================================

Some implementations of GPTwoSample, to detect differential expression between two timelines.
"""


from pygp.gp import GP
from pygp.gp.composite import GroupGP

from gptwosample.twosample import GPTwoSample

class GPTwoSampleMLII(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to wach other.

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.

    """
    def __init__(self, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        super(GPTwoSampleMLII, self).__init__(*args, **kwargs)


    def _init_twosample_model(self, covar):
        gpr1 = GP(covar)
        gpr2 = GP(covar)
        individual_model = GroupGP([gpr1,gpr2])
        common_model = GP(covar)
        # set models for this GPTwoSample Test
        self._models = {'individual':individual_model,'common':common_model}
