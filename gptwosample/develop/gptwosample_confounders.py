'''
Created on Sep 14, 2011

@author: maxz
'''
import sys
sys.path.append('./../../')

from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
from gptwosample.twosample.twosample_compare import \
    GPTwoSample_individual_covariance, GPTwoSample_share_covariance
from numpy.linalg.linalg import cholesky
from pygp import likelihood as lik, covar
from pygp.covar import se, noise, gradcheck, combinators
from pygp.covar.combinators import ProductCF, ShiftCF, SumCF
from pygp.covar.fixed import FixedCF
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
import scipy as SP
import pylab
from gptwosample import plot

def run_demo(cond1_file, cond2_file,fraction = 0.1,confounder_model='linear',components=4):
    """run demo sscript with condition file on random fractio of all data"""
    
    logging.basicConfig(level=logging.INFO)
    # how many confounders to learn?
    components = 4

    dump_file = 'confounder_run_GP.pickle'
    if (not os.path.exists(dump_file)) or 'recalc' in sys.argv:
        Ydict = read_data_from_file(cond1_file,cond2_file,fraction)
        Ydict = add_simulated_confounders(Ydict,model=confounder_model,components=components)
        Ydict = learn_confounder_matrix(Ydict,model=confounder_model,components=components)


        cPickle.dump([learned_confounders,T1, T2, Tpredict, gene_names, n_replicates, n_replicates_1, n_replicates_2, Y_dict],  open(dump_file, 'wb'), -1)
    else:
        [learned_confounders, T1, T2,
         Tpredict, gene_names,
         n_replicates, n_replicates_1, n_replicates_2,
         Y_dict] = cPickle.load(open(dump_file, 'r'))
    
    #hyperparamters
    dim = 1
    
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    confounderCF = covar.fixed.FixedCF(learned_confounders)
    
    covar_priors_common = []
    covar_priors_individual = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp, [1, .5]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [1, .5]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [3, 1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [3, 1]])
        
    covar_priors_common.append([lnpriors.lnGauss, [0, 5]])
    covar_priors_individual.append([lnpriors.lnGauss, [0, 5]])
 
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .3]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .3]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})

    CovFun = SumCF((ProductCF((SECF, confounderCF)), noiseCF))
    
    out_path = "simulated_confounders_GP_2000_fixed_changed"
    out_file = "%i_confounders.csv" % (components)
    out_file_raw = "%i_confounders_raw.csv" % (components)    
    
    num = 1
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(os.path.join(out_path,"plots")):
        os.mkdir(os.path.join(out_path,"plots"))
    
    csv_out_file_confounded = open(os.path.join(out_path, out_file), 'wb')
    csv_out_confounded = csv.writer(csv_out_file_confounded)

    csv_out_file_raw = open(os.path.join(out_path, out_file_raw), 'wb')
    csv_out_raw = csv.writer(csv_out_file_raw)

    header = ["Gene", "Bayes Factor"]
    
    header.extend(map(lambda x:'Common ' + x, CovFun.get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, CovFun.get_hyperparameter_names()))

    csv_out_confounded.writerow(header)
    csv_out_raw.writerow(header)
    
    twosample_object = GPTwoSample_share_covariance(CovFun, priors=priors)

    T1 = SP.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2, n_replicates_2).reshape(-1, 1)

    # get ground truth genes for comparison:
    gt_names = []
    for [name, val] in csv.reader(open("../examples/ground_truth_random_genes.csv", 'r')):
        gt_names.append(name.upper())
    still_to_go = len(gt_names) - 1
    
    #loop through genes
    for gene_name in gt_names:
        try:
            if gene_name is "input":
                continue
            print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
            run_gptwosample_on_data(twosample_object, Tpredict, T1, T2,
                                    Y_dict[gene_name]['confounded'][:len(T1)],
                                    Y_dict[gene_name]['confounded'][len(T1):])
            write_back_data(twosample_object, gene_name, csv_out_confounded)
            plot_and_save_figure(T1, twosample_object, gene_name, savename=os.path.join(out_path,"plots","%s_conf.png"%gene_name))
            
            run_gptwosample_on_data(twosample_object, Tpredict, T1, T2,
                                    Y_dict[gene_name]['raw'][:len(T1)],
                                    Y_dict[gene_name]['raw'][len(T1):])
            write_back_data(twosample_object, gene_name, csv_out_raw)
            plot_and_save_figure(T1, twosample_object, gene_name, savename=os.path.join(out_path,"plots","%s_raw.png"%gene_name))
    
            still_to_go -= 1
            if still_to_go == len(gt_names) - 1:
                import pdb;pdb.set_trace()
        except:
            still_to_go -= 1

def write_back_data(twosample_object, gene_name, csv_out):
    line = [gene_name.upper(), twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = SP.exp(common)
    individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = SP.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)

def plot_and_save_figure(T1, twosample_object, gene_name, savename = None):
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
    
def sample_confounders_from_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T):
        # or draw from a GP:
    NRT = n_replicates * gene_length 
    X = SP.concatenate((T.copy().T, SP.randn(NRT, components).T)).T
    sigma = 1e-6
    K = lvm_covariance.K(hyperparams['covar'], X) + sigma * SP.eye(NRT)
    cholK = cholesky(K)
    #left multiply indepnendet confounding matrix with cholesky:
    Y_conf = SP.dot(cholK,SP.randn(NRT,len(gene_names)))

    #slow version:
    #Y_conf = SP.array([SP.dot(cholK, SP.randn(NRT, 1)).flatten() for i in range(len(gene_names))])
    return Y_conf


def sample_GP(covariance,covar_hyperparams,X,Nsamples):
    """draw Nsample from a GP with covaraince covaraince, hyperparams and X as inputs"""
    # or draw from a GP:
    sigma = 1e-6
    K = covariance.K(covar_hyperparams, X)
    K += sigma * SP.eye(K.shape[0])
    cholK = cholesky(K)
    #left multiply indepnendet confounding matrix with cholesky:
    Ys = SP.dot(cholK,SP.randn(K.shape[0],Nsamples))
    return Ys


    
def sample_confounders_linear(components, gene_names, n_replicates, gene_length):
    NRT = n_replicates * gene_length 
    X = SP.random.randn(NRT, components)
    W = SP.random.randn(components, len(gene_names)) * 0.5
    Y_conf = SP.dot(X, W)
    return Y_conf.T


def read_data_from_file(cond1_file, cond2_file,fraction=1.0):
    """read raw data and return dict with all data"""
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

    #merge data
    T = SP.tile(T1, n_replicates).reshape(-1, 1)   
    Y1 = SP.array(cond1.values())
    Y2 = SP.array(cond2.values())
    # get Y values for all genes
    Y = SP.concatenate((Y1, Y2), 1).reshape(Y1.shape[0], -1)
    condition = SP.concatenate(( 0*SP.ones(n_replicates_1*T1.shape[0]), 1* SP.ones(n_replicates_2*T2.shape[0])))
        
    #create random index and take a subset of these
    Isubset = SP.random.permutation(len(cond1.keys()))
    Isubset = Isubset[0:int(fraction*len(Isubset))]
    #subset Y
    Y = Y[Isubset,:].T
   
    #gene names on subset:
    gene_names = SP.array(cond1.keys())[Isubset]
    
    #zero mean each gene
    Y-= Y.mean(axis=1)[:,SP.newaxis]
    RV = {'Y': Y,'gene_names':gene_names,'T':T, 'condition': condition, 'subset_indices': Isubset}
    return RV

def construct_gp_model(model='linear',components=4):
    #construct covariance model
    if model=='linear':
        lvm_covariance = LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 1+components))
        hyperparams = {'covar': SP.log([0.5]),'lik':0.1}
        
    elif model=='product_linear':
        lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]),
                                  LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 1+components))), n_dimensions=components + 1)
        hyperparams = {'covar': SP.log([1, 1, .5]),'lik':0.1}

    X = SP.concatenate((Ydict['T'],SP.random.randn(Ydict['T'].shape[0],components)),axis=1)
        
    RV = {'covariance':lvm_covariance,'hyperparams':hyperparams,'X0':X}
    return RV
    
def add_simulated_confounders(Ydict,components=4,**kw_args):
    """add simulated confounded expression data to dict"""

    gpmodel = construct_gp_model(components=4,**kw_args)

    #construct X for GP sampling
    # structure consists of [Time, factor1,factor2,..factorN]
    # (time is needed for product_linear covaraince which assume non-perfect independence of measurements

    Yconf = sample_GP(gpmodel['covariance'],gpmodel['hyperparams']['covar'], gpmodel['X0'], len(Ydict['gene_names']))

    Ydict['confounder'] = Yconf
    Ydict['Y_confounded'] = Yconf + Ydict['Y']
    return Ydict
    pass

def learn_confounder_matrix(Ydict,components=4,**kw_args):
    """reconstruct simualted confounding"""

    #1. simple PCA
    # Simulate linear Kernel by PCA estimation:
    X_pca,W_pca = gplvm.PCA(Ydict['Y_confounded'], components)
    #quick checkup: reconstruct Y from Y_confounded using PCA
    Y_reconstruct_PCA = Y_confounded-SP.dot(X_pca,W_pca.T)

   
    #2. GPLVM
    gpmodel = construct_gp_model(components=components,**kw_args)
    likelihood = lik.GaussLikISO()
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 1+components), covar_func=gpmodel['covariance'], likelihood=likelihood, x=gpmodel['X0'], y=Ydict['Y_confounded'])
    bounds = {}
    bounds['lik'] = SP.array([[-5., 5.]] * Y2.shape[1])
    
    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, gpmodel['hyperparams'], gradcheck=False)
    Y_reconstruct_GPLVM = g.predict(opt_hyperparams_comm,g.x,output=SP.arrange(4),var=False)
    pdb.set_trace()


    print "reconstructions using PCA and GPLVM reconstruct"
    print ((Y-Y_confounded)**2).mean()
    print ((Y-Y_reconstruct_PCA)**2).mean()
    print ((Y-Y_reconstruct_GPLVM)**2).mean()


    pass

def read_and_learn_data_from_files(cond1_file, cond2_file, components=4,fraction=1.0):
    
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
        
    Y1 = SP.array(cond1.values())
    Y2 = SP.array(cond2.values())

    lvm_covariance = ProductCF((SqexpCFARD(n_dimensions=1, dimension_indices=[0]),
                                LinearCFISO(n_dimensions=components, dimension_indices=xrange(1, 5))), n_dimensions=components + 1)
    hyperparams = {'covar': SP.log([1, 1, .5])}

    # get Y values for all genes
    Y = SP.concatenate((Y1, Y2), 1).reshape(Y1.shape[0], -1)

    #subsample random fraction if active:
    #create random index and take a subset of these
    Isubset = SP.random.permutation(len(cond1.keys()))
    Isubset = Isubset[0:int(fraction*len(Isubset))]
    pdb.set_trace()
    #subset Y
    Y = Y[Isubset,:]
    #gene names on subset:
    gene_names = SP.array(cond1.keys())[Isubset]
    
    #zero mean each gene
    Y-= Y.mean(axis=1)[:,SP.newaxis]
    
    non_linear = False
    if non_linear:
        simulated_confounders = sample_confounders_from_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T)
    else:
        simulated_confounders = sample_confounders_linear(components,gene_names,n_replicates,gene_length)

    Y_confounded = Y + simulated_confounders
    
    # from now on Y matrices are transposed:
    Y = Y.T
    Y_confounded = Y_confounded.T

    # Simulate linear Kernel by PCA estimation:
    X_pca,W_pca = gplvm.PCA(Y_confounded, components)

    #quick checkup: reconstruct Y from Y_confounded using PCA
    Y_reconstruct_PCA = Y_confounded-SP.dot(X_pca,W_pca.T)
    
    # Get X right:
    X0 = SP.concatenate((T.copy(), X_pca.copy()), axis=1)
    # optimize X?
    hyperparams['x'] = X_pca.copy()
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.4])
    
    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 5), covar_func=lvm_covariance, likelihood=likelihood, x=X0, y=Y_confounded)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5., 5.]] * Y2.shape[1])
    
    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm, opt_lml2] = opt_hyper(g, hyperparams, gradcheck=False)

    pdb.set_trace()
    Y_reconstruct_GPLVM = g.predict(opt_hyperparams_comm,g.x,output=SP.arrange(4),var=False)
    


    print "reconstructions using PCA and GPLVM reconstruct"
    print ((Y-Y_confounded)**2).mean()
    print ((Y-Y_reconstruct_PCA)**2).mean()
    print ((Y-Y_reconstruct_GPLVM)**2).mean()

    #store things in result dictionary
    Y_dict = dict([[name, {'confounded':Y_confounded[i], 'raw':Y[i]}] for i, name in enumerate(cond1.keys())])



    # Here comes the analysis: >>>>
    # pdb.set_trace()
    # Note: simulated confounders is W*X

    # mean_predicted_confounders = g.predict(opt_hyperparams_comm, g.x) # 
    # Y_wx=SP.tile(mean_predicted_confounders[0].reshape(-1,1),simulated_confounders.shape[0]) # >>> Here might be the mistake <<<
    # Y2 = Y_confounded-Y_wx

    # SP.mean((Y-Y2)**2)
    # SP.mean((Y_confounded-Y)**2)
    # SP.mean((Y_wx - simulated_confounders)**2)
    
    # end of analysis <<<<
    
    # Adjust Confounders for proper kernel usage
    learned_confounders = opt_hyperparams_comm['x'] * SP.exp(opt_hyperparams_comm['covar'][2])
    return learned_confounders, T1, T2, Tpredict, gene_names, n_replicates, n_replicates_1, n_replicates_2, Y_dict

if __name__ == '__main__':
    run_demo(cond1_file='./../examples/warwick_control.csv', cond2_file='../examples/warwick_treatment.csv')
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
