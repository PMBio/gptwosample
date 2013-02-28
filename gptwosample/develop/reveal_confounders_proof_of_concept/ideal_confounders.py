'''
Created on Feb 20, 2013

@author: Max
'''
from gptwosample.develop.reveal_confounders_proof_of_concept.simple_confounders import get_priors,\
    run_gptwosample_on_data, write_back_data
import logging
import pickle
import os
import numpy
from pygp.covar import se, noise
from pygp.covar.fixed import FixedCF
from pygp.covar.combinators import SumCF
import sys
import csv
import pylab
from gptwosample.data.data_analysis import plot_roc_curve
import ipdb
from gptwosample.twosample.twosample_base import GPTwoSample_individual_covariance
import itertools


def run_demo(cond1_file, cond2_file, components = 4, root='.'):
    logging.basicConfig(level=logging.INFO)

    ######################################
    #            LOAD DATA               #
    ######################################
    data_file = open(os.path.join(root,"toy_data.pickle"), "r")
    (Y, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
     n_replicates, gene_length, T) = pickle.load(data_file)
    data_file.close; del data_file
    
    conf_file_name = os.path.join(os.path.join(root,"toy_data_sim_conf.pickle"))
    if not os.path.exists(conf_file_name):
        print "Did not run simple_confounder.py before"
        exit
    else:
        conf_file = open(conf_file_name, 'r')
        simulated_confounders, X_sim = pickle.load(conf_file)
    conf_file.close()
    K_sim = numpy.dot(X_sim, X_sim.T)
    
    # Get variances right:
    Y = Y-Y.mean(0)
    Y = Y/Y.std(0)
    #simulated_confounders = simulated_confounders/simulated_confounders.std()
    Y_confounded = Y+simulated_confounders
    
    # get Y values for all genes    
    Y_dict = dict([[name, {'confounded':Y_confounded[i],
                           #'raw':Y[i]
                           }] for i,name in enumerate(gene_names)])
    
    # from now on Y matrices are transposed:
    Y = Y.T
    Y_confounded = Y_confounded.T
    
    # X = X_sim
    dim = 1    
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    
#    priors_normal = get_priors(dim, confounders = False)
    priors_conf = get_priors(dim, confounders = True)

#    covar_normal = SumCF((SECF, noiseCF))
    covar_conf_common = SumCF((SECF, FixedCF(K_sim), noiseCF))

    r1t = T1.shape[0] * n_replicates_1
    covar_conf_r1 =  SumCF((SECF, FixedCF(K_sim[:r1t, :r1t]), noiseCF))
    covar_conf_r2 =  SumCF((SECF, FixedCF(K_sim[r1t:, r1t:]), noiseCF))

    T1 = numpy.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = numpy.tile(T2, n_replicates_2).reshape(-1, 1)
    
    plots_out_dir = os.path.join(root,"plots/")
    results_out_dir = os.path.join(root,"results/")
    if not os.path.exists(results_out_dir):
        os.mkdir(results_out_dir)
    if not os.path.exists(plots_out_dir):
        os.mkdir(plots_out_dir)

    gt_file_name = "../../examples/ground_truth_random_genes.csv"
    out_conf_file_name = os.path.join(results_out_dir,"ideal.csv")
    
    if "retwosample" in sys.argv or not os.path.exists(out_conf_file_name):
        # get csv files to write to
        out_conf_file = open(out_conf_file_name,'wb')
        out_conf = csv.writer(out_conf_file)
        
        first_line = ["gene name","bayes factor"]
        first_line.extend(map(lambda x:"Common: "+x,covar_conf_common.get_hyperparameter_names()))
        first_line.extend(map(lambda x:"Individual: "+x,covar_conf_r1.get_hyperparameter_names()))
        out_conf.writerow(first_line)
        
        # get ground truth genes for comparison:
        gt_names = []
        gt_file = open(gt_file_name,'r')
        for [name,_] in csv.reader(gt_file):
            gt_names.append(name)
        gt_file.close()
        
        current = itertools.count()
        lgt_names = len(gt_names)
        
        #loop through genes
        for gene_name in gt_names:
            if gene_name is "input":
                continue
            gene_name = gene_name.upper()
            if gene_name in Y_dict.keys():
                sys.stdout.flush()
                sys.stdout.write('processing {0:s} {1:.3%}             \r'.format(gene_name, float(current.next())/lgt_names))
                
                twosample_object_conf = GPTwoSample_individual_covariance(covar_conf_r1, covar_conf_r2, covar_conf_common, priors=priors_conf)
                run_gptwosample_on_data(twosample_object_conf, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                        Y_dict[gene_name]['confounded'][:len(T1)],
                                        Y_dict[gene_name]['confounded'][len(T1):], 
                                        gene_name,os.path.join(plots_out_dir,gene_name+"_ideal"))
                write_back_data(twosample_object_conf, gene_name, out_conf)
                out_conf_file.flush()
            
        out_conf_file.close()
    if "plot_roc" in sys.argv:
        conf_plot, conf_auc = plot_roc_curve(out_conf_file_name, gt_file_name, label="ideal")
        pylab.legend()
    
if __name__ == '__main__':
    run_demo(cond1_file='./../../examples/warwick_control.csv', cond2_file='../../examples/warwick_treatment.csv', root=sys.argv[1])
