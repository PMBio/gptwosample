'''
Created on Sep 14, 2011

@author: maxz
'''
from gptwosample.data.dataIO import get_data_from_csv
import scipy as SP
import pygp.gp.gplvm as gplvm
from pygp.covar.combinators import ProductCF, ShiftCF, SumCF
from pygp.covar.linear import LinearCFISO
from pygp.covar.se import SqexpCFARD
from pygp import likelihood as lik, covar
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar import se, noise, gradcheck, combinators
from pygp.priors import lnpriors
from pygp.covar.fixed import FixedCF
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
import os
import csv
import cPickle as pickle
from gptwosample.twosample.twosample_compare import *
from gptwosample.plot.plot_basic import plot_results
from numpy.linalg.linalg import cholesky
import pdb
import logging
import pylab

def run_demo(cond1_file, cond2_file, components = 4):
    logging.basicConfig(level=logging.INFO)

    ######################################
    #            LOAD DATA               #
    ######################################
    if not os.path.exists("toy_data.pickle"):
	read_files_and_pickle(cond1_file, cond2_file)

    (Y1, Y2, Y, cond1, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
     n_replicates, gene_length, T) = pickle.load(open("toy_data.pickle", "r"))
    
    simulated_confounders, X_sim = sample_confounders_linear(components, gene_names, n_replicates, gene_length)
    Y_confounded = Y+simulated_confounders
    # get Y values for all genes    
    Y_dict = dict([[name, {'confounded':Y_confounded[i],'raw':Y[i]}] for i,name in enumerate(cond1.keys())])
    # from now on Y matrices are transposed:
    Y = Y.T
    Y_confounded = Y_confounded.T

    X = run_gplvm(Y_confounded, T, Y2, components, only_pca = True)
    # X = X_sim
    dim = 1    
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    confounderCF = FixedCF(X)   


    priors_normal = get_priors(dim, confounders = False)
    priors_conf = get_priors(dim, confounders = True)

    covar_normal = SumCF((SECF, noiseCF))
    # covar_conf = SumCF((ProductCF((SECF,confounderCF), n_dimensions=components+1), noiseCF))
    # covar_conf = SumCF((ProductCF((SECF,confounderCF)), noiseCF))
    covar_conf = SumCF((SumCF((SECF,confounderCF)), noiseCF))    

    still_to_go = len(gene_names)
    T1 = SP.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2, n_replicates_2).reshape(-1, 1)

    # get ground truth genes for comparison:
    gt_names = []
    for [name,val] in csv.reader(open("../examples/ground_truth_balanced_set_of_100.csv",'r')):
        gt_names.append(name)
    
    #loop through genes
    for gene_name in gt_names:
        if gene_name is "input":
            continue
	gene_name = gene_name.upper()
        print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)

	twosample_object = GPTwoSample_share_covariance(covar_conf, priors=priors_conf)
        run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                Y_dict[gene_name]['confounded'][:len(T1)],
                                Y_dict[gene_name]['confounded'][len(T1):], 
                                gene_name)

	twosample_object = GPTwoSample_share_covariance(covar_normal, priors=priors_normal)
        run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                Y_dict[gene_name]['confounded'][:len(T1)],
                                Y_dict[gene_name]['confounded'][len(T1):], 
                                gene_name)

        still_to_go -= 1
	pdb.set_trace()
	
def write_back_data(twosample_object, gene_name, csv_out):
    line = [gene_name, twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = SP.exp(common)
    individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = SP.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)

def run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, Y0, Y1, gene_name):
    #create data structure for GPTwwoSample:
    #note; there is no need for the time points to be aligned for all replicates
    #creates score and time local predictions
    twosample_object.set_data_by_xy_data(T1, T2, Y0.reshape(-1, 1), Y1.reshape(-1, 1))
    twosample_object.predict_model_likelihoods()
    twosample_object.predict_mean_variance(Tpredict)

    pylab.figure()
    plot_results(twosample_object,
		 title='%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
		 shift=None,
		 draw_arrows=1)
    pylab.xlim(T1.min(), T1.max())
    
def sample_confounders_from_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T):
        # or draw from a GP:
    NRT = n_replicates * gene_length 
    X = SP.concatenate((T.copy().T,SP.randn(NRT,components).T)).T
    sigma = 1e-6
    Y_conf = SP.array([SP.dot(cholesky(lvm_covariance.K(hyperparams['covar'], X) + sigma * SP.eye(NRT)), SP.randn(NRT, 1)).flatten() for i in range(len(gene_names))])
    return Y_conf.T
    
def sample_confounders_linear(components, gene_names, n_replicates, gene_length):
    NRT = n_replicates * gene_length 
    X = SP.random.randn(NRT,components)
    W = SP.random.randn(components, len(gene_names))*0.5
    Y_conf = SP.dot(X, W)
    return Y_conf.T, X

def get_priors(dim, confounders):
    covar_priors_common = []
    covar_priors_individual = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])

    if confounders:
	covar_priors_common.append([lnpriors.lnuniformpdf, [0, 0]])
	covar_priors_individual.append([lnpriors.lnuniformpdf, [0, 0]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .5]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .5]])
    return get_model_structure({'covar':SP.array(covar_priors_individual)})

def read_files_and_pickle(cond1_file, cond2_file):
    #1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    # get replicate stuff organized
    gene_names = sorted(cond1.keys()) 
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1 + n_replicates_2
    gene_length = len(T1)
    T = SP.tile(T1, n_replicates).reshape(-1, 1)

    # data merging and stuff
    Y1 = SP.array(cond1.values())
    Y2 = SP.array(cond2.values())
    Y = SP.concatenate((Y1,Y2),1).reshape(Y1.shape[0],-1)
    
    
    dump_file = open("toy_data.pickle", "w")
    pickle.dump((Y1, Y2, Y, cond1, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
		 n_replicates, gene_length, T), dump_file, -1)
    dump_file.close()
    
def run_gplvm(Y_confounded, T, Y2, components = 4, only_pca = True):
    # lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]), 
    #                             LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 5))),
    # 			       n_dimensions=components + 1)
    lvm_covariance = SumCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]), 
			    LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 5))),
			   n_dimensions=components + 1)
    hyperparams = {'covar': SP.log([1, 1, 1.2])}
    linear_cf = LinearCFISO(n_dimensions=components)
    mu_cf = FixedCF(SP.ones([Y_confounded.shape[0],Y_confounded.shape[0]]))
    lvm_covariance = combinators.SumCF((mu_cf, linear_cf))
    hyperparams = {'covar': SP.log([1, 1])}


    # Simulate linear Kernel by PCA estimation:
    X_pca = gplvm.PCA(Y_confounded, components)[0]

    if only_pca:
	return X_pca
    
    # Get X right:
    X0 = SP.concatenate((T.copy(), X_pca.copy()), axis=1) 
    # optimize X?
    hyperparams['x'] = X_pca.copy()
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    
    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 5), covar_func=lvm_covariance, likelihood=likelihood, x=X0, y=Y_confounded)
    bounds = {}
    bounds['lik'] = SP.array([[-5., 5.]]*Y2.shape[1])


    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, hyperparams, gradcheck=False)
    
    # Adjust Confounders for proper kernel usage
    X = opt_hyperparams_comm['x']# * SP.exp(opt_hyperparams_comm['covar'][2]) 

    return X

if __name__ == '__main__':
    run_demo(cond1_file='./../examples/warwick_control.csv', cond2_file='../examples/warwick_treatment.csv')
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
