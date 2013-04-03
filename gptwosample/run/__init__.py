import sys
import numpy
from gptwosample.data.dataIO import get_data_from_csv
import os

__PREFIX__ = "GPTwoSample"
message = lambda s:"{}:{}".format(__PREFIX__,s)

def started(s):
    sys.stdout.write(message(s))
    sys.stdout.flush()
    sys.stdout.write("                  \r")
    
def finished(s):
    try:
        sys.stdout.write(message(s) + " " + '\033[92m' + u"\u2713" + '\033[0m' + '                              \n')
    except:
        sys.stdout.write(message(s) + " done                            \n")
    sys.stdout.flush()

def get_header_for_covar(CovFun, CovFunInd=None):
    if CovFunInd is None:
        CovFunInd = CovFun
    ret = map(lambda x:"Common " + x, CovFun.get_hyperparameter_names())
    ret.extend(map(lambda x:"Individual " + x, \
                   CovFunInd.get_hyperparameter_names()))
    return ret

def twosampledata(cond1, cond2):
    T1 = numpy.array(cond1.pop("input"))[:, None]
    T2 = numpy.array(cond2.pop("input"))[:, None]
    
    Y1 = numpy.array(cond1.values()).T.swapaxes(0, 1)
    Y2 = numpy.array(cond2.values()).T.swapaxes(0, 1)
    Y = numpy.array([Y1, Y2])
    
    _, r, _, _ = Y.shape
    
    T1 = numpy.tile(T1, r).T
    T2 = numpy.tile(T2, r).T

    T = numpy.array([T1, T2])
    
    gene_names = cond1.keys()
    
    assert T.shape == Y.shape[:3]
    assert gene_names == cond2.keys()
    
    del T1, T2, Y1, Y2, cond1, cond2
    
    Y -= Y.mean(1).mean(1)[:, None, None, :]
    return T,Y, gene_names

def loaddata(cond1file, cond2file, verbose=0):
    s = "loading data..."
    started(s)
    if verbose: sys.stdout.write(os.linesep)
    cond1 = get_data_from_csv(cond1file, verbose=verbose)
    cond2 = get_data_from_csv(cond2file, verbose=verbose)
    T,Y,gene_names = twosampledata(cond1, cond2)
    finished(s)
    return T,Y,gene_names