'''
Created on Sep 14, 2011

@author: maxz
'''
from gptwosample import plot
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id, get_training_data_structure
from gptwosample.twosample.twosample_compare import GPTwoSample_share_covariance,\
    GPTwoSample_individual_covariance
from numpy.linalg.linalg import cholesky
from pygp import likelihood as lik, covar
from pygp.covar import se, noise, fixed
from pygp.covar.combinators import ProductCF, SumCF
from pygp.covar.linear import LinearCFISO
from pygp.covar.se import SqexpCFARD
from pygp.optimize.optimize_base import opt_hyper
from pygp.priors import lnpriors
import cPickle
import csv
import logging
import os
import pdb
import pygp.gp.gplvm as gplvm
import pylab
import scipy as SP
import sys
import gptwosample_confounders_standard_prediction
sys.path.append('./../../')


def get_out_path(confounder_model, confounder_learning_model, components):
    return 'sampledfrom-%s learnedby-%s conf-%i' % (confounder_model, confounder_learning_model, components)

def run_demo(cond1_file, cond2_file, fraction=0.1, confounder_model='linear', confounder_learning_model= 'linear', prediction_model='reconstruct', components=4):
    """run demo script with condition file on random fraction of all data"""
    
    print "Sampled from: %s. Learned by: %s. Predicted by: %s" % (confounder_model, confounder_learning_model, prediction_model)
    
    logging.basicConfig(level=logging.INFO)
    # how many confounders to learn?
    components = 4

    Y_dict = get_data(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, components)

    import pdb;pdb.set_trace()
    
    #hyperparamters
    dim = 1
    
    CovFun_pca, CovFun_gplvm = get_gptwosample_covariance_function(Y_dict, prediction_model, dim)    
    
    out_path = get_out_path(confounder_model, confounder_learning_model, components)
    out_file_PCA = "%s-%i_confounders_PCA.csv" % (prediction_model,components)
    out_file_GPLVM = "%s-%i_confounders_GPLVM.csv" % (prediction_model,components)    
    
    csv_out_PCA = prepare_csv_out(CovFun_pca, out_path, out_file_PCA)
    csv_out_GPLVM = prepare_csv_out(CovFun_gplvm, out_path, out_file_GPLVM)

    if not os.path.exists(os.path.join(out_path, "plots")):
        os.mkdir(os.path.join(out_path, "plots"))
        
    priors = get_gptwosample_priors(dim, prediction_model)
    
    twosample_object_pca = GPTwoSample_individual_covariance(CovFun_pca, priors=priors)
    twosample_object_gplvm = GPTwoSample_individual_covariance(CovFun_gplvm, priors=priors)

    T1 = Y_dict['T'][Y_dict['condition'] == 0]
    T2 = Y_dict['T'][Y_dict['condition'] == 1]

    Tpredict = get_model_structure(SP.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 96)[:, SP.newaxis],SP.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 2*96)[:, SP.newaxis]);

    # get ground truth genes for comparison:
    gt_names = {}
    for [name, val] in csv.reader(open("../examples/ground_truth_random_genes.csv", 'r')):
#    for [name, val] in csv.reader(open("../examples/ground_truth_balanced_set_of_100.csv", 'r')):
        gt_names[name.upper()] = val
    
    still_to_go = int(fraction*(len(gt_names)))
    
    #loop through genes
#    for gene_name in ["CATMA1A24060", "CATMA1A49990"]:
    for gene_name in SP.random.permutation(gt_names.keys())[:still_to_go]:
#        try:
            gene_name = gene_name
            gene_name_hit = Y_dict['gene_names'] == gene_name
            if gene_name is "input":
                continue
            if not gene_name_hit.any():
                gene_name = gene_name.upper()
                gene_name_hit = Y_dict['gene_names'] == gene_name
                if not gene_name_hit.any():
                    print "%s not in random set"%(gene_name)
                    still_to_go -= 1
                    continue

            print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
            gene_index = SP.where(gene_name_hit)[0][0]
            
            run_gptwosample_on_data(twosample_object_pca, Tpredict, T1, T2,
                                    get_gptwosample_data_for_model(prediction_model, "PCA", 0, Y_dict, gene_index),
                                    get_gptwosample_data_for_model(prediction_model, "PCA", 1, Y_dict, gene_index))
            write_back_data(twosample_object_pca, gene_name, csv_out_PCA)
            plot_and_save_figure(T1, twosample_object_pca, gene_name, savename=os.path.join(out_path, "plots", "%s_%s-PCA.png" % (gene_name, prediction_model)))
            
            run_gptwosample_on_data(twosample_object_gplvm, Tpredict, T1, T2,
                                    get_gptwosample_data_for_model(prediction_model, "GPLVM", 0, Y_dict, gene_index),
                                    get_gptwosample_data_for_model(prediction_model, "GPLVM", 1, Y_dict, gene_index))
            write_back_data(twosample_object_gplvm, gene_name, csv_out_GPLVM)
            plot_and_save_figure(T1, twosample_object_gplvm, gene_name, savename=os.path.join(out_path, "plots", "%s_%s-GPLVM.png" % (gene_name, prediction_model)))
    
            still_to_go -= 1
#        except:
#            
#            still_to_go -= 1

def get_gptwosample_data_for_model(prediction_model, learning_model, condition, Y_dict, gene_index):
    if prediction_model == 'reconstruct':
        return Y_dict['Y_reconstruct_'+learning_model][Y_dict['condition'] == condition, gene_index]
    elif prediction_model == 'covariance':
        return Y_dict['Y_confounded'][Y_dict['condition'] == condition, gene_index]    
    
def write_back_data(twosample_object, gene_name, csv_out):
    line = [gene_name.upper(), twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = SP.exp(common)
    individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = SP.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)

def prepare_csv_out(CovFun, out_path, out_file):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    csv_out_file_confounded = open(os.path.join(out_path, out_file), 'wb')
    csv_out_confounded = csv.writer(csv_out_file_confounded)
    header = ["Gene", "Bayes Factor"]
    header.extend(map(lambda x:'Common ' + x, CovFun[0].get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, CovFun[0].get_hyperparameter_names()))
    csv_out_confounded.writerow(header)
    return csv_out_confounded

def plot_and_save_figure(T1, twosample_object, gene_name, savename=None):
    pylab.figure(1)
    pylab.clf()
    plot.plot_results(twosample_object, title='%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()), shift=None, draw_arrows=1)
    pylab.xlim(T1.min(), T1.max())
    if savename is None:
        savename = gene_name
    pylab.savefig("%s" % (savename))
    
def run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, Y0, Y1):
    #create data structure for GPTwwoSample:
    #note; there is no need for the time points to be aligned for all replicates
    #creates score and time local predictions
    twosample_object.set_data_by_xy_data(T1, T2, Y0.reshape(-1, 1), Y1.reshape(-1, 1))
    twosample_object.predict_model_likelihoods()
    twosample_object.predict_mean_variance(Tpredict)

def sample_GP(covariance, covar_hyperparams, X, Nsamples):
    """draw Nsample from a GP with covaraince covaraince, hyperparams and X as inputs"""
    # or draw from a GP:
    sigma = 1e-6
    K = covariance.K(covar_hyperparams, X)
    K += sigma * SP.eye(K.shape[0])
    cholK = cholesky(K)
    #left multiply indepnendet confounding matrix with cholesky:
    Ys = SP.dot(cholK, SP.randn(K.shape[0], Nsamples))
    return Ys
    
def sample_confounders_linear(components, gene_names, n_replicates, gene_length):
    NRT = n_replicates * gene_length 
    X = SP.random.randn(NRT, components)
    W = SP.random.randn(components, len(gene_names)) * 0.5
    Y_conf = SP.dot(X, W)
    return Y_conf.T

def read_data_from_file(cond1_file, cond2_file, fraction=1.0):
    """read raw data and return dict with all data"""
    #1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    # get replicate stuff organized
    gene_names = cond1.keys()
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1 + n_replicates_2
    
    #merge data
    T = SP.tile(T1, n_replicates).reshape(-1, 1)   
    Y1 = SP.array(cond1.values())
    Y2 = SP.array(cond2.values())
    # get Y values for all genes
    Y = SP.concatenate((Y1, Y2), 1).reshape(Y1.shape[0], -1)
    condition = SP.concatenate((0 * SP.ones(n_replicates_1 * T1.shape[0]), 1 * SP.ones(n_replicates_2 * T2.shape[0])))
        
    #create random index and take a subset of these
    Isubset = SP.random.permutation(len(cond1.keys()))
    Isubset = Isubset[0:int(fraction * len(Isubset))]
    #subset Y
    Y = Y[Isubset, :].T
   
    #Y[:,:] = SP.random.randn(Y.shape[0],Y.shape[1])
   
    #gene names on subset:
    gene_names = SP.array(cond1.keys())[Isubset]
    
    #zero mean each gene
    Y -= Y.mean(axis=1)[:, SP.newaxis]
    #set variance to unit variance
    Y /= Y.std()
    
    RV = {'Y': Y, 'gene_names':gene_names, 'T':T, 'condition': condition, 'subset_indices': Isubset}
    return RV

def construct_gp_model(Ydict, model='linear', components=4, explained_variance=1.0):
    #construct covariance model
    if model == 'linear':
        lvm_covariance = LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 1 + components))
        hyperparams = {'covar': SP.log([explained_variance]), 'lik':SP.array([0.1])}
        
    elif model == 'product_linear':
        lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]),
                                  LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 1 + components))), n_dimensions=components + 1)
        hyperparams = {'covar': SP.log([1, 1, explained_variance]), 'lik':SP.array([0.1])}

    X = SP.concatenate((Ydict['T'], SP.random.randn(Ydict['T'].shape[0], components)), axis=1)
   
    #optimization over the latent dimension only (1. Dimension is time)
    hyperparams['x'] = X[:, 1::].copy()
        
    RV = {'covariance':lvm_covariance, 'hyperparams':hyperparams, 'X0':X}
    return RV
    
def add_simulated_confounders(Ydict, gpmodel, components=4, **kw_args):
    """add simulated confounded expression data to dict"""

    #construct X for GP sampling
    # structure consists of [Time, factor1,factor2,..factorN]
    # (time is needed for product_linear covaraince which assume non-perfect independence of measurements
    
    Yconf = sample_GP(gpmodel['covariance'], gpmodel['hyperparams']['covar'], gpmodel['X0'], len(Ydict['gene_names']))

    Ydict['confounder'] = Yconf
    Ydict['Y_confounded'] = Ydict['Y'] + Yconf
    return Ydict
    pass

def learn_confounder_matrix(Ydict, gpmodel, components=4, **kw_args):
    """reconstruct simualted confounding"""

    #1. simple PCA
    # Simulate linear Kernel by PCA estimation:
    Y_confounded = Ydict['Y_confounded']
    X_pca, W_pca = gplvm.PCA(Y_confounded, components)
    #quick checkup: reconstruct Y from Y_confounded using PCA
    Y_reconstruct_PCA = Y_confounded - SP.dot(X_pca, W_pca.T)
    Ydict['X_PCA'] = X_pca.copy()
   
    #2. GPLVM
    likelihood = lik.GaussLikISO()
    
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 1 + components), covar_func=gpmodel['covariance'], likelihood=likelihood, x=gpmodel['X0'], y=Ydict['Y_confounded'])
    bounds = {'lik': SP.array([[-5., 5.]] * Ydict['T'].shape[1])}
    
    # run lvm on data
    print "running standard gplvm"
    hyperparams = gpmodel['hyperparams']
    hyperparams['x'] = X_pca
    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, hyperparams, bounds=bounds, gradcheck=False)
    Ydict['X_GPLVM'] = opt_hyperparams_comm['x'].copy()

    Y_reconstruct_GPLVM = Y_confounded - g.predict(opt_hyperparams_comm, g.x, output=SP.arange(Y_confounded.shape[1]), var=False)

    print "reconstructions using PCA and GPLVM reconstruct:"
    print ((Ydict['Y'] - Y_confounded) ** 2).mean()
    print ((Ydict['Y'] - Y_reconstruct_PCA) ** 2).mean()
    print ((Ydict['Y'] - Y_reconstruct_GPLVM) ** 2).mean()

    Ydict['Y_reconstruct_PCA'] = Y_reconstruct_PCA
    Ydict['Y_reconstruct_GPLVM'] = Y_reconstruct_GPLVM
    pdb.set_trace()
    return Ydict

def get_gptwosample_priors(dim, prediction_model):
    covar_priors_common = []
    covar_priors_individual = [] 
    # SECF amplitude
    covar_priors_common.append([lnpriors.lnGammaExp, [1, .5]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [1, .5]])
    # lengthscale
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [3, 1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [3, 1]])
    # fixedCF amplitude
    if prediction_model=="covariance":
        covar_priors_common.append([lnpriors.lnGauss, [0, 5]])
        covar_priors_individual.append([lnpriors.lnGauss, [0, 5]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .3]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .3]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})
    return priors

def append_to_CovFun(X, CovFun_pca, SECF, noiseCF):
    confounderCF_pca = fixed.FixedCF(SP.dot(X, X.T))
    CovFun_pca.append(SumCF((SumCF((SECF, confounderCF_pca)), noiseCF)))
    return confounderCF_pca

def get_gptwosample_covariance_function(Y_dict, prediction_model, dim):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()

    if prediction_model == 'reconstruct':
        CovFun_gplvm = CovFun_pca = [SumCF((SECF, noiseCF))]*3

    elif prediction_model == 'covariance':
        CovFun_gplvm = []
        CovFun_pca = []
        
        # condition 1:
        append_to_CovFun(Y_dict['X_PCA'][Y_dict['condition'] == 0], CovFun_pca, SECF, noiseCF)
        append_to_CovFun(Y_dict['X_GPLVM'][Y_dict['condition'] == 0], CovFun_gplvm, SECF, noiseCF)

        # condition 2:
        append_to_CovFun(Y_dict['X_PCA'][Y_dict['condition'] == 1], CovFun_pca, SECF, noiseCF)
        append_to_CovFun(Y_dict['X_GPLVM'][Y_dict['condition'] == 1], CovFun_gplvm, SECF, noiseCF)

        # shared:
        append_to_CovFun(Y_dict['X_PCA'], CovFun_pca, SECF, noiseCF)
        append_to_CovFun(Y_dict['X_GPLVM'], CovFun_gplvm, SECF, noiseCF)
    
    return CovFun_pca, CovFun_gplvm

def get_data(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, components):
    dump_file = "%s.pickle" % get_out_path(confounder_model, confounder_learning_model, components)
    if (not os.path.exists(dump_file)) or 'recalc' in sys.argv:
        Y_dict = read_data_from_file(cond1_file, cond2_file, fraction)
        gpmodel = construct_gp_model(Y_dict, model=confounder_model, components=components, explained_variance=2)
        Y_dict = add_simulated_confounders(Y_dict, gpmodel, components=components)
        gpmodel = construct_gp_model(Y_dict, model=confounder_learning_model, components=components, explained_variance=2)
        Y_dict = learn_confounder_matrix(Y_dict, gpmodel, components=components)
        cPickle.dump(Y_dict, open(dump_file, 'wb'), -1)
    else:
        Y_dict = cPickle.load(open(dump_file, 'r'))
        gpmodel = construct_gp_model(Y_dict, model=confounder_model, components=components, explained_variance=2)
    return Y_dict

if __name__ == '__main__':
    cond1_file='./../examples/warwick_control.csv'
    cond2_file='../examples/warwick_treatment.csv'
    confounder_learning_model= 'product_linear'
    confounder_model='product_linear'
    prediction_model='reconstruct'
    run_demo(cond1_file=cond1_file, cond2_file=cond2_file, confounder_model=confounder_model, confounder_learning_model=confounder_learning_model, prediction_model=prediction_model, fraction=1)
#    gptwosample_confounders_standard_prediction.run_demo(cond1_file=cond1_file, cond2_file=cond2_file, confounder_model=confounder_model, confounder_learning_model=confounder_learning_model, prediction_model=prediction_model)
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
