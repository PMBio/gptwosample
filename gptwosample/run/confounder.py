'''
Created on Mar 14, 2013

@author: Max
'''
from gptwosample.data.dataIO import get_data_from_csv
import sys
import numpy
import os
from gptwosample.confounder.confounder import ConfounderTwoSample
import csv
from gptwosample.data.data_base import common_id, individual_id
import itertools

def run_confounder_twosample(cond1file, cond2file, outdir, plot=True):    
    s = "loading data..."
    started(s)
    sys.stdout.write(os.linesep)
    cond1 = get_data_from_csv(cond1file)
    cond2 = get_data_from_csv(cond2file)
    T,Y,gene_names = twosampleconfounderdata(cond1, cond2)
    finished(s)
    
    s = "setting up gplvm..."
    started(s)
    confoundertwosample = ConfounderTwoSample(T,Y)
    finished(s)
    
    s = "learning confounders..."
    started(s)
    restarts = 10
    for r in range(restarts):
        try:
            gtol=1./(10**(12-r))
            #sys.stdout.write(os.linesep)
            #sys.stdout.flush()
            confoundertwosample.learn_confounder_matrix(messages=False, gradient_tolerance=gtol)
            break
        except:
            started("{} restart {}".format(s,r+1))
            pass
    else:
        raise Exception("no confounders found after {} iterations".format(restarts))
    finished(s)
    
    s = "predicting likelihoods..."
    started(s)
    confoundertwosample.predict_likelihoods(message=s)
    finished(s)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'results.csv'),'w') as resultfile:
        csvout = csv.writer(resultfile)
        s = 'writing out to {}...'.format(resultfile.name)
        started(s)
        line = ["Gene Name", "Bayes Factor"]
        covars = confoundertwosample.get_covars()
        covarcommon = covars[common_id]
        covarindividual = covars[individual_id][0]
        line.extend(get_header_for_covar(covarcommon, covarindividual))
        csvout.writerow(line)
        
        for name, bayes, hyp in itertools.izip(gene_names,
                                   confoundertwosample.bayes_factors(), 
                                   confoundertwosample.get_learned_hyperparameters()):
            common = covarcommon.get_reparametrized_theta(hyp[common_id]['covar'])
            individual = covarindividual.get_reparametrized_theta(hyp[individual_id]['covar'])
            line = [name, bayes]
            line.extend(common)
            line.extend(individual)
            csvout.writerow(line)
        finished(s)
    
    if plot:
        mi = T[0,0,:].min()
        ma = T[0,0,:].max()
        s = "predicting means and variances"
        started(s)
        confoundertwosample.predict_means_variances(numpy.linspace(mi,ma,100), message=s)
        #finished(s)
        
        s = "plotting..."
        started(s)
        import pylab
        pylab.ion()
        pylab.figure()
        plotdir = os.path.join(outdir, "plots")
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        for i,name,_ in itertools.izip(itertools.count(), gene_names, confoundertwosample.plot()):
            started("{0} {1:%.3}".format(name, float(i+1)/len(gene_names)))
            try:
                pylab.savefig(os.path.join(plotdir, "{}.pdf".format(name)))
            except:
                pylab.savefig(os.path.join(plotdir, "{}".format(name)))
        finished(s)
    
def twosampleconfounderdata(cond1, cond2):
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
    
    assert T.shape == Y.shape[:3]
    assert gene_names == cond2.keys()
    
    del T1, T2, Y1, Y2, cond1, cond2
    
    Y -= Y.mean(1).mean(1)[:, None, None, :]
    return T,Y, gene_names

def started(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    sys.stdout.write("                  \r")
    
def finished(s):
    try:
        sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '            \n')
    except:
        sys.stdout.write(s + " done             \n")
    sys.stdout.flush()

def get_header_for_covar(CovFun, CovFunInd=None):
    if CovFunInd is None:
        CovFunInd = CovFun
    ret = map(lambda x:"Common " + x, CovFun.get_hyperparameter_names())
    ret.extend(map(lambda x:"Individual " + x, \
                   CovFunInd.get_hyperparameter_names()))
    return ret

     
if __name__ == '__main__':
    run_confounder_twosample("../examples/gsample1.csv", "../examples/gsample2.csv", "./test", True)
    