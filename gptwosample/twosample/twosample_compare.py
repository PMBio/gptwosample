"""
Different implementations for using GPTwoSample
===============================================

Some implementations of GPTwoSample, to detect differential expression between two timelines.

@author: Max Zwiessele, Oliver Stegle
"""


from pygp.gp import GP
from pygp.gp.composite import GroupGP

from gptwosample.twosample import GPTwoSample
from gptwosample.data.data_base import individual_id, common_id

class GPTwoSampleMLII(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to each other.

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.
    
    """
    def __init__(self, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        super(GPTwoSampleMLII, self).__init__(*args, **kwargs)


    def _init_twosample_model(self, covar, **kwargs):
        gpr1 = GP(covar)
        gpr2 = GP(covar)
        individual_model = GroupGP([gpr1,gpr2])
        common_model = GP(covar)
        # set models for this GPTwoSample Test
        self._models = {individual_id:individual_model,common_id:common_model}
        
class GPTimeShift(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to one another, inlcuding timeshifts in replicates, respectively

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.
    
    Note that this model will need one covariance function for each model, respectively!
    """
    def __init__(self, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        super(GPTimeShift, self).__init__(*args, **kwargs)


    def _init_twosample_model(self, covar, **kwargs):
        gpr1 = GP(covar[0])
        gpr2 = GP(covar[1])
        individual_model = GroupGP([gpr1,gpr2])
        common_model = GP(covar[2])
        # set models for this GPTwoSample Test
        self._models = {individual_id:individual_model,common_id:common_model}

