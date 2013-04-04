import os, cPickle

def getWarwickData():
    """ Get the warwick data """
    # Data of warwick.pickle (depends on which system were running)

    data_path = './'
    data_file = os.path.join(data_path,'data_warwick.pickle')
    data_file_f = open(data_file,'rb')

    data = cPickle.load(data_file_f)

    return data

