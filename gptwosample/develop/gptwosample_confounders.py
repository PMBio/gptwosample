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
from gptwosample.twosample.twosample_compare import GPTwoSample_individual_covariance
from numpy.linalg.linalg import cholesky
import pdb
import logging

def run_demo(cond1_file, cond2_file):
    logging.basicConfig(level=logging.INFO)

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
    
    # how many confounders to learn?
    components = 4
    
    Y1 = SP.array(cond1.values())
    Y2 = SP.array(cond2.values())

    lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]), 
                                LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 5))), n_dimensions=components + 1)
    hyperparams = {'covar': SP.log([1, 1, 1.2])}

    # get Y values for all genes
    Y = SP.concatenate((Y1,Y2),1).reshape(Y1.shape[0],-1)
    simulated_confounders, W = sample_confounders_linear(components, gene_names, n_replicates, gene_length)
    Y_confounded = Y+simulated_confounders
    
    Y_dict = dict([[name, {'confounded':Y_confounded[i],'raw':Y[i]}] for i,name in enumerate(cond1.keys())])

    # from now on Y matrices are transposed:
    Y = Y.T
    Y_confounded = Y_confounded.T

    # Simulate linear Kernel by PCA estimation:
    X_pca = gplvm.PCA(Y_confounded, components)[0]
    
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
    
    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, hyperparams, gradcheck=True)

    # Here comes the analysis: >>>>
    pdb.set_trace()
    # Note: simulated confounders is W*X

    mean_predicted_confounders = g.predict(opt_hyperparams_comm, g.x) # 
    Y_wx=SP.tile(mean_predicted_confounders[0].reshape(-1,1),simulated_confounders.shape[0]) # >>> Here might be the mistake <<<
    Y2 = Y_confounded-Y_wx

    SP.mean((Y-Y2)**2)
    SP.mean((Y_confounded-Y)**2)
    SP.mean((Y_wx - simulated_confounders)**2)
    
    # end of analysis <<<<
    
    # Adjust Confounders for proper kernel usage
    learned_confounders = opt_hyperparams_comm['x'] * SP.exp(opt_hyperparams_comm['covar'][2])    

    #hyperparamters
    dim = 1
    
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    confounderCF = covar.fixed.FixedCF(learned_confounders)
    
    covar_priors_common = []
    covar_priors_individual = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])
        
    covar_priors_common.append([lnpriors.lnuniformpdf, [0, 0]])
    covar_priors_individual.append([lnpriors.lnuniformpdf, [0, 0]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .5]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .5]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})

    covar = SumCF((ProductCF((SECF,confounderCF)),noiseCF))
    
    out_path = "simulated_confounders"
    out_file = "%i_confounders"%(components)
    num = 1
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    while os.path.exists(os.path.join(out_path,out_file)):
        out_file = out_file.replace("_%i.csv"%num,".csv")
        num+=1
        out_file_raw = out_file.replace(".csv","_raw_%i.csv"%num)
        out_file = out_file.replace(".csv","_%i.csv"%num)
    
    csv_out_file_confounded = open(os.path.join(out_path, out_file), 'wb')
    csv_out_confounded = csv.writer(csv_out_file_confounded)

    csv_out_file_raw = open(os.path.join(out_path, out_file_raw), 'wb')
    csv_out_raw = csv.writer(csv_out_file_raw)

    header = ["Gene", "Bayes Factor"]
    
    header.extend(map(lambda x:'Common ' + x, covar.get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, covar.get_hyperparameter_names()))

    csv_out_confounded.writerow(header)
    csv_out_raw.writerow(header)
    
    twosample_object = GPTwoSample_individual_covariance(covar, priors=priors)

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
        print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
        run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                Y_dict[name]['confounded'][:len(T1)],
                                Y_dict[name]['confounded'][len(T1):], 
                                name, dim)
        write_back_data(twosample_object, gene_name, csv_out_confounded)
        
        run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, 
                                Y_dict[name]['raw'][:len(T1)],
                                Y_dict[name]['raw'][len(T1):], 
                                name, dim)
        write_back_data(twosample_object, gene_name, csv_out_raw)

        still_to_go -= 1

def write_back_data(twosample_object, gene_name, csv_out):
    line = [gene_name, twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = SP.exp(common)
    individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = SP.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)

def run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, Y0, Y1):
    #create data structure for GPTwwoSample:
    #note; there is no need for the time points to be aligned for all replicates
    #creates score and time local predictions
    twosample_object.set_data_by_xy_data(T1, T2, Y0.reshape(-1, 1), Y1.reshape(-1, 1))
    twosample_object.predict_model_likelihoods()
    twosample_object.predict_mean_variance(Tpredict)
    
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
    return Y_conf.T, W

if __name__ == '__main__':
    run_demo(cond1_file='./../examples/warwick_control.csv', cond2_file='../examples/warwick_treatment.csv')
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
