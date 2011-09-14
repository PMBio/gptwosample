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
from pygp import likelihood as lik
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar import se, noise
from pygp.priors import lnpriors
from pygp.covar.fixed import FixedCF
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
import os
import csv
from gptwosample.twosample.twosample_compare import GPTwoSample_individual_covariance

def run_demo(cond1_file, cond2_file):
    #1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    gene_names = sorted(cond1.keys()) 
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1 + n_replicates_2
    gene_length = len(T1)
    
    components = 4
    
    Y1 = SP.array(cond1.values()).reshape(T1.shape[0]*n_replicates_1, -1)
    Y2 = SP.array(cond2.values()).reshape(T2.shape[0]*n_replicates_2, -1)

    # Simulate linear Kernel by PCA estimation:
    Y_comm = SP.concatenate((Y1, Y2))
    X_pca = gplvm.PCA(Y_comm, components)[0]
    
    lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]), LinearCFISO(n_dimensions=components, dimensio_indices=xrange(1, 5))), n_dimensions=components + 1)
    hyperparams = {'covar': SP.log([1, 1, 1.2])}

    T = SP.tile(T1, n_replicates).reshape(-1, 1)
    # Get X right:
    X0 = SP.concatenate((T.copy(), X_pca.copy()), axis=1)
    # optimize X?
    # hyperparams['x'] = X_pca.copy()
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    
    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 5), covar_func=lvm_covariance, likelihood=likelihood, x=X0, y=Y_comm)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5., 5.]]*Y2.shape[1])
    
    # Filter for scalar factor
    Ifilter = {'covar':SP.array([1, 1, 1]), 'lik':SP.ones(1), 'x':SP.array([1, 1, 1, 1, 1, 1])}
    
    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, hyperparams, gradcheck=True)
    
    import pdb;pdb.set_trace()
    
    # Adjust Confounders for proper kernel usage
    X_conf_comm = X_pca * SP.exp(opt_hyperparams_comm['covar'][2])    
    X_len = X_conf_comm.shape[0]
    X_conf_1 = X_conf_comm[:X_len / 2]
    X_conf_2 = X_conf_comm[X_len / 2:]
    X_conf_1 = SP.dot(X_conf_1, X_conf_1.T)
    X_conf_2 = SP.dot(X_conf_2, X_conf_2.T)    
    X_conf_comm = SP.dot(X_conf_comm, X_conf_comm.T) \


    #hyperparamters
    dim = 1
    replicate_indices_1 = []
    for rep in SP.arange(n_replicates_1):
        replicate_indices_1.extend(SP.repeat(rep, gene_length))
    replicate_indices_1 = SP.array(replicate_indices_1)
    replicate_indices_2 = []
    for rep in SP.arange(n_replicates_2):
        replicate_indices_2.extend(SP.repeat(rep, gene_length))
    replicate_indices_2 = SP.array(replicate_indices_2)
    
    SECF = se.SqexpCFARD(dim)
    #noiseCF = noise.NoiseReplicateCF(replicate_indices)
    noiseCF = noise.NoiseCFISO()
    
    shiftCFInd1 = ShiftCF(SECF, replicate_indices_1)
    shiftCFInd2 = ShiftCF(SECF, replicate_indices_2)
    shiftCFCom = ShiftCF(SECF, SP.concatenate((replicate_indices_1, replicate_indices_2 + n_replicates_1)))

    CovFun = SumCF((SECF, noiseCF))
    
    covar_priors_common = []
    covar_priors_individual = []
    covar_priors = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp, [1, 2]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [1, 2]])
    covar_priors.append([lnpriors.lnGammaExp, [1, 2]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, 1]])
        covar_priors.append([lnpriors.lnGammaExp, [1, 1]])
    #shift
    for i in range(n_replicates):
        covar_priors_common.append([lnpriors.lnGauss, [0, .5]])
    for i in range(n_replicates_1):
        covar_priors_individual.append([lnpriors.lnGauss, [0, .5]])
        
    covar_priors_common.append([lnpriors.lnuniformpdf, [0, 0]])
    covar_priors_individual.append([lnpriors.lnuniformpdf, [0, 0]])
    covar_priors.append([lnpriors.lnuniformpdf, [0, 0]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, 1]])
        covar_priors.append([lnpriors.lnGammaExp, [1, 1]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})
    #Ifilter = {'covar': SP.ones(n_replicates+3)}
    covar = [SumCF((SumCF((shiftCFInd1, FixedCF(X_conf_1))), noiseCF)),
             SumCF((SumCF((shiftCFInd2, FixedCF(X_conf_2))), noiseCF)),
             SumCF((SumCF((shiftCFCom,
                                                   FixedCF(X_conf_comm))),
                                noiseCF))]
    
    csv_out_file = open(os.path.join('out', "result.csv"), 'wb')
    csv_out = csv.writer(csv_out_file)
    header = ["Gene", "Bayes Factor"]
    
    header.extend(map(lambda x:'Common ' + x, covar[2].get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, covar[0].get_hyperparameter_names()))
    csv_out.writerow(header)
    
    twosample_object = GPTwoSample_individual_covariance(covar, priors=priors)

    still_to_go = len(gene_names)
    T1 = SP.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2, n_replicates_2).reshape(-1, 1)
    #loop through genes
    for gene_name in gene_names:
        if gene_name is "input":
            continue

        #expression levels: replicates x #time points
        Y0 = cond1[gene_name]
        Y1 = cond2[gene_name]
        
        #create data structure for GPTwwoSample:
        #note; there is no need for the time points to be aligned for all replicates
        #creates score and time local predictions
        twosample_object.set_data_by_xy_data(T1, T2,
                                             Y0.reshape(-1, 1),
                                             Y1.reshape(-1, 1))
        print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
        twosample_object.predict_model_likelihoods()
        twosample_object.predict_mean_variance(Tpredict)

        line = [gene_name, twosample_object.bayes_factor()]
        common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
        individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
        timeshift_index = SP.array(SP.ones_like(common), dtype='bool')
        timeshift_index[dim + 1:dim + 1 + n_replicates_1 + n_replicates_2] = 0
        common[timeshift_index] = SP.exp(common[timeshift_index])
        timeshift_index = SP.array(SP.ones_like(individual), dtype='bool')
        timeshift_index[dim + 1:dim + 1 + n_replicates_1] = 0
        individual[timeshift_index] = SP.exp(individual[timeshift_index])
        line.extend(common)
        line.extend(individual)
        csv_out.writerow(line)
        

if __name__ == '__main__':
    run_demo(cond1_file='./../examples/warwick_control_ground_truth.csv', cond2_file='../examples/warwick_treatment_ground_truth.csv')
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
