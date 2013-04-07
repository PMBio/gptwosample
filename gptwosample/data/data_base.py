'''
Data Structure Module
=====================

This Module is for easy access to data structures gptwosample works with.

Created on Mar 18, 2011

@author: Max Zwiessele
'''
replicate_indices_id = 'replicate'
individual_id = 'individual model fit'
common_id = 'common model fit'
input_id = 'in'
output_id = 'out'

#class ModelStructure():
#    """
#    Object holding model data.
#
#    **Fields**:
#    individual: :py.class:`gptwosample.data.data_base.DataStructure`
#        DataStructure holding data for the individual model
#    common: :py.class:`gptwosample.data.data_base.DataStructure`
#        DataStructure holding data for the common model
#    """
#    individual = None
#    common = None
#
#class DataStructure():
#    """
#    Object holding data comprising two time series.
#
#    **Fields**:
#
#    """

def get_training_data_structure(x1, x2, y1, y2):
    """
    Get the structure for training data, given two inputs x1 and x2
    with corresponding outputs y1 and y2. Make sure, that replicates have
    to be tiled one after the other for proper resampling of data!
    """
    return {input_id:{'group_1':x1, 'group_2':x2},
            output_id:{'group_1':y1, 'group_2':y2}}

def get_model_structure(individual=None, common=None):
    """
    Returns the valid structure for model dictionaries, used in gptwosample.
    Make sure to use this method if you want to use the model structure in this package!
    """
    if common is not None and individual is not None:
        return {individual_id:individual, common_id:common}
    elif individual is not None:
        return {individual_id:individual, common_id:individual}
    else:  # if common is not None: # or both are None:
        return {individual_id:common, common_id:common}

def has_model_structure(structure):
    """
    Returns the valid structure for model dictionaries, used in gptwosample.
    Make sure to use this method if you want to use the model structure in this package!
    """
    if isinstance(structure, dict):
        ret = True
        for name in get_model_structure().keys():
            ret &= structure.has_key(name)
        return ret
    return False

class DataStructureError(TypeError):
    """
    Thrown, if DataStructure given does not fit.
    Training data training_data has following structure::

                {input_id : {'group 1':[double] ... 'group n':[double]},
                 output_id : {'group 1':[double] ... 'group n':[double]}}
    """
    def __init__(self, *args, **kwargs):
        super(DataStructureError, self).__init__(*args, **kwargs)
