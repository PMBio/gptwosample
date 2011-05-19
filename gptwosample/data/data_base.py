'''
Created on Mar 18, 2011

@author: maxz

This Class is for easy access to data structures gptwosample works with.
'''

replicate_indices_id = 'rep'
individual_id = 'ind'
common_id = 'com'
input_id = 'in'
output_id = 'out'

def get_training_data_structure(x1,x2,y1,y2):
    """
    Get the structure for training data, given two inputs x1 and x2 
    with corresponding outputs y1 and y2. Make shure, that replicates have
    to be tiled one after the other for proper resamping of data!
    """
    return {input_id:{'group_1':x1, 'group_2':x2},
            output_id:{'group_1':y1, 'group_2':y2}}
    
def get_model_structure(individual=None, common=None):
    """
    Returns the valid structure for model dictionaries, used in gptwosample.
    Make sure to use this method if you want to use the model structure in this package!
    """
    return {individual_id:individual, common_id:common}

class DataStructureError(TypeError):
    """
    Thrown, if DataStructure given does not fit.
    Training data training_data has following structure::

                {input_id : {'group 1':[double] ... 'group n':[double]},
                 output_id : {'group 1':[double] ... 'group n':[double]}}
    """
    def __init__(self, *args, **kwargs):
        super(DataStructureError, self).__init__(*args, **kwargs)