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

class GPTwoSample_share_covariance(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to each other.

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.
    
    """
    def __init__(self, covar, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        super(GPTwoSample_share_covariance, self).__init__(*args, **kwargs)
        gpr1 = GP(covar)
        gpr2 = GP(covar)
        individual_model = GroupGP([gpr1,gpr2])
        common_model = GP(covar)
        self.covar = covar
        # set models for this GPTwoSample Test
        self._models = {individual_id:individual_model,common_id:common_model}
        
class GPTwoSample_individual_covariance(GPTwoSample):
    """
    This class provides comparison of two Timeline Groups to one another, inlcuding timeshifts in replicates, respectively

    see :py:class:`GPTwoSample.src.GPTwoSample` for detailed description of provided methods.
    
    Note that this model will need one covariance function for each model, respectively!
    """
    def __init__(self, covar_individual_1, covar_individual_2, covar_common, *args, **kwargs):
        """
        see :py:class:`GPTwoSample.src.GPTwoSample`
        """
        super(GPTwoSample_individual_covariance, self).__init__(*args, **kwargs)
        gpr1 = GP(covar_individual_1)
        gpr2 = GP(covar_individual_2)
        individual_model = GroupGP([gpr1,gpr2])
        common_model = GP(covar_common)
        self.covar_individual_1 = covar_individual_1
        self.covar_individual_2 = covar_individual_2
        self.covar_common = covar_common
        # set models for this GPTwoSample Test
        self._models = {individual_id:individual_model,common_id:common_model}
    