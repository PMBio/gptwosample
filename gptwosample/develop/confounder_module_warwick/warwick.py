'''
Created on Feb 26, 2013

@author: Max
'''
from gptwosample.data.dataIO import get_data_from_csv
import os
import pickle
import numpy
import sys
from gptwosample.confounder.confounder_model import Confounder_Model
from multiprocessing.process import Process
import time

_usage = """usage: python warwick.py root-dir [warwick_control-file warwick_treatment-file] [regplvm|relikelihood]
warwick_control-file and warwick_treatment-file have to be given only in first run - data will be pickled"""

try:
    root = sys.argv[1]
    if not os.path.exists(root):
        raise IOError("Missing root directory")
except:
    print _usage
    sys.exit(0)
    

def finished(s, process=None):
    if process is not None:
        while p.is_alive():
            p.terminate()
            p.join(1)
    sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '\n')
    sys.stdout.flush()
def start_mill(s):
    mill_symb = {0:'-',1:'\\',2:"|",3:'/'}
    def mill():
        i=-1
        while True:
            i = (i+1)%4
            sys.stdout.flush()
            sys.stdout.write("{}{}\r".format(s,mill_symb[i]))
            time.sleep(.3)
    p = Process(target=mill)
    p.start()
    return p

s = "loading data..."
sys.stdout.write(s)
data_file_path = os.path.join(root,"./data.pickle") 
if not os.path.exists(data_file_path):
    cond1 = get_data_from_csv(sys.argv[2])#os.path.join(root,'warwick_control.csv'))
    cond2 = get_data_from_csv(sys.argv[3])#os.path.join(root,'warwick_treatment.csv'))
    
    T1 = numpy.array(cond1.pop("input"))[:, None]
    T2 = numpy.array(cond2.pop("input"))[:, None]
    
    Y1 = numpy.array(cond1.values()).T.swapaxes(0, 1)
    Y2 = numpy.array(cond2.values()).T.swapaxes(0, 1)
    Y = numpy.array([Y1, Y2])
    
    n, r, t, d = Y.shape    

    T1 = numpy.tile(T1, r).T
    T2 = numpy.tile(T2, r).T
    
    T = numpy.array([T1, T2])

    gene_names = cond1.keys()
    
    del T1, T2, Y1, Y2, cond1, cond2
    
    assert T.shape == Y.shape[:3]
    assert gene_names == cond2.keys()
    
    data_file = open(data_file_path, 'w')
    pickle.dump([T, Y, gene_names], data_file)
else:
    data_file = open(data_file_path, 'r')
    T, Y, gene_names = pickle.load(data_file)
    n, r, t, d = Y.shape    

data_file.close()
finished(s)

s = "setting up gplvm module..."
sys.stdout.write(s)
conf_model = Confounder_Model(T, Y, components=4)
finished(s)

lvm_hyperparams_file_name = os.path.join(root,'lvm_hyperparams.npy')
if not os.path.exists(lvm_hyperparams_file_name) or "regplvm" in sys.argv:
    s = 'learning confounder matrix... '
    p = start_mill(s)
    conf_model.learn_confounder_matrix()
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'w')
    pickle.dump(conf_model._lvm_hyperparams, lvm_hyperparams_file)
    finished(s, process=p)
else:
    s = "loading confounder matrix..."
    print s,
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'r')
    conf_model._init_conf_matrix(pickle.load(lvm_hyperparams_file))
    finished(s)
lvm_hyperparams_file.close()

likelihoods_file_name = os.path.join(root,'likelihoods.npy')
hyperparams_file_name = os.path.join(root,'hyperparams.npy')
if not os.path.exists(likelihoods_file_name) or "relikelihood" in sys.argv:
    s = "predicting model likelihoods... "
    p = start_mill(s)
    likelihoods = conf_model.predict_model_likelihoods(messages=False)
    hyperparams = conf_model._hyperparameters
    likelihoods_file = open(likelihoods_file_name, 'w')
    hyperparams_file = open(hyperparams_file_name, 'w')
    pickle.dump(likelihoods_file, likelihoods)
    pickle.dump(hyperparams_file, hyperparams)
    finished(s,p)
else:
    s = "loading model likelihoods... "
    likelihoods_file = open(likelihoods_file_name, 'r')
    hyperparams_file = open(hyperparams_file_name, 'r')
    conf_model._likelihoods = pickle.load(likelihoods_file)
    conf_model._hyperparameters = pickle.load(hyperparams_file)
    finished(s)
likelihoods_file.close()
hyperparams_file.close()
