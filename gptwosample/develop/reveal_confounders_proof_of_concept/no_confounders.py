'''
Created on Sep 14, 2011

@author: maxz
'''
from gptwosample.data.data_analysis import plot_roc_curve
from gptwosample.develop.reveal_confounders_proof_of_concept.simple_confounders import \
    get_priors, run_gptwosample_on_data, write_back_data
from pygp.covar import se, noise
from pygp.covar.combinators import SumCF
import cPickle as pickle
import csv
import logging
import os
import pylab
import scipy as SP
import sys
from gptwosample.twosample.twosample_base import GPTwoSample_share_covariance
import itertools

# Private variables:
__debug = 1


def run_demo(cond1_file, cond2_file, components = 4, root='.'):
    logging.basicConfig(level=logging.INFO)

    ######################################
    #            LOAD DATA               #
    ######################################

    (Y, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
     n_replicates, gene_length, T) = pickle.load(open(os.path.join(root,"toy_data.pickle"), "r"))
    print "finished loading data"
        
    # Get variances right:
    Y = Y-Y.mean(1)[:,None]
    Y = Y/Y.std(1)[:,None]
    #simulated_confounders = simulated_confounders-simulated_confounders.mean(1)[:,None]
    #simulated_confounders = simulated_confounders/simulated_confounders.std(1)[:,None]
    
    # from now on Y matrices are transposed:
    Y = Y.T
    
    # get Y values for all genes        
    Y_dict = dict([[name, {
                           #'subtracted': Y_subtracted[:,i],
                           #'confounded':Y_confounded[:,i],
                           'raw':Y[:,i]
                           }] for i,name in enumerate(gene_names)])
    
    # X = X_sim
    dim = 1    
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    
    priors_normal = get_priors(dim, confounders = False)
    
    covar_normal = SumCF((SECF, noiseCF))
    # covar_conf = SumCF((ProductCF((SECF,confounderCF), n_dimensions=components+1), noiseCF))
    # covar_conf = SumCF((ProductCF((SECF,confounderCF)), noiseCF))
    
    T1 = SP.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2, n_replicates_2).reshape(-1, 1)
    
    plots_out_dir = os.path.join(root,"plots/")
    results_out_dir = os.path.join(root,"results/")
    if not os.path.exists(results_out_dir):
        os.mkdir(results_out_dir)
    if not os.path.exists(plots_out_dir):
        os.mkdir(plots_out_dir)

    gt_file_name = "../../examples/ground_truth_random_genes.csv"
    out_normal_file_name = os.path.join(results_out_dir,"raw.csv")
    
    if "retwosample" in sys.argv or not os.path.exists(out_normal_file_name):
        # get csv files to write to
        out_normal_file = open(out_normal_file_name,'wb')
        out_normal = csv.writer(out_normal_file)
        
        first_line = ["gene name","bayes factor"]
        first_line.extend(map(lambda x:"Common: "+x,covar_normal.get_hyperparameter_names()))
        first_line.extend(map(lambda x:"Individual: "+x,covar_normal.get_hyperparameter_names()))
        out_normal.writerow(first_line)
        
        # get ground truth genes for comparison:
        gt_names = []
        gt_file = open(gt_file_name,'r')
        for [name,val] in csv.reader(gt_file):
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

                twosample_object_normal = GPTwoSample_share_covariance(covar_normal, priors=priors_normal)
                run_gptwosample_on_data(twosample_object_normal, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                        Y_dict[gene_name]['raw'][:len(T1)],
                                        Y_dict[gene_name]['raw'][len(T1):], 
                                        gene_name,os.path.join(plots_out_dir,gene_name+"_raw"))
                write_back_data(twosample_object_normal, gene_name, out_normal)
            
        out_normal_file.close()
      
    if "plot_roc" in sys.argv:
        normal_plot, normal_auc = plot_roc_curve(out_normal_file_name, gt_file_name, label="raw")
        pylab.legend()
    
if __name__ == '__main__':
    run_demo(cond1_file='./../../examples/warwick_control.csv', cond2_file='../../examples/warwick_treatment.csv', root=sys.argv[1])
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
