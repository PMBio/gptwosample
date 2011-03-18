'''
Created on Mar 18, 2011

@author: maxz
'''


def get_training_data_structure(x1,x2,y1,y2):
    """
    Get the structure for training data, given two inputs x1 and x2 
    with corresponding outputs y1 and y2.
    """
    return {'input':{'group_1':x1, 'group_2':x2},
            'output':{'group_1':y1, 'group_2':y2}}
    
    
class DataStructureError(TypeError):
    """
    Thrown, if DataStructure given does not fit.
    Training data training_data has following structure::

                {'input' : {'group 1':[double] ... 'group n':[double]},
                 'output' : {'group 1':[double] ... 'group n':[double]}}
    """
    def __init__(self, *args, **kwargs):
        super(DataStructureError, self).__init__(*args, **kwargs)