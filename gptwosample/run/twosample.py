'''
Created on Mar 14, 2013

@author: Max
'''
import numpy
import os
import csv
from gptwosample.data.data_base import common_id, individual_id
import itertools
from gptwosample.run import started, finished, get_header_for_covar, message

def run_twosample(twosample, gene_names, outdir):    
    s = "predicting likelihoods..."
    #started(s)
    twosample.predict_likelihoods(twosample.T, twosample.Y, message=message(s))
    #finished(s)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'results.csv'),'w') as resultfile:
        csvout = csv.writer(resultfile)
        s = 'writing out to {}...'.format(resultfile.name)
        started(s)
        line = ["Gene Name", "Bayes Factor"]
        covars = twosample.get_covars()
        covarcommon = covars[common_id]
        covarindividual = covars[individual_id][0]
        line.extend(get_header_for_covar(covarcommon, covarindividual))
        csvout.writerow(line)
        
        for name, bayes, hyp in itertools.izip(gene_names,
                                   twosample.bayes_factors(), 
                                   twosample.get_learned_hyperparameters()):
            common = covarcommon.get_reparametrized_theta(hyp[common_id]['covar'])
            individual = covarindividual.get_reparametrized_theta(hyp[individual_id]['covar'])
            line = [name, bayes]
            line.extend(common)
            line.extend(individual)
            csvout.writerow(line)
            resultfile.flush()
        finished(s)
    return twosample
    
if __name__ == '__main__':
    run_twosample("../examples/gsample1.csv", "../examples/gsample2.csv", "./test", True)
    