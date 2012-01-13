'''
Created on Sep 14, 2011

@author: maxz
'''
from gptwosample import plot
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
from gptwosample.twosample.twosample_compare import \
    GPTwoSample_individual_covariance
from gptwosample.util.confounder_constants import *
from gptwosample.util.sample_confounders import sample_GP
from pygp import likelihood as lik
from pygp.covar import se, noise, fixed, delta
from pygp.covar.combinators import ProductCF, SumCF
from pygp.covar.linear import LinearCFISO
from pygp.covar.se import SqexpCFARD
from pygp.optimize.optimize_base import opt_hyper
from pygp.priors import lnpriors
from threading import Thread
import cPickle
import csv
import gptwosample.util.confounder_model as conf_module
import gptwosample_confounders_standard_prediction
import logging
import os
import pylab
import scipy as SP
import sys
import threading
import time
sys.path.append('./../../')

def get_path_for_pickle(confounder_model, confounder_learning_model, components):
    return 'sampledfrom-%s_learnedby-%s_conf-%i' % (confounder_model, confounder_learning_model, components)

def find_gene_name_hit(Y_dict, gene_name):
    """Searches for the right index of given gene_name in Y_dict, returns -1 if no gene was hit"""
    gene_name_hit = Y_dict['gene_names'] == gene_name
    if not gene_name_hit.any():
        gene_name = gene_name.upper()
        gene_name_hit = Y_dict['gene_names'] == gene_name
        if not gene_name_hit.any():
            return -1
    gene_index = SP.where(gene_name_hit)[0][0]
    return gene_index

def get_gptwosample_data_for_model(prediction_model, condition, Y_dict, gene_index):
    if prediction_model == reconstruct_model_id:
        return Y_dict['Y_reconstruct'][Y_dict['condition'] == condition, gene_index]
    elif prediction_model == covariance_model_id:
        return Y_dict['Y_confounded'][Y_dict['condition'] == condition, gene_index]    
    
def write_back_data(twosample_object, gene_name, csv_out, csv_out_file):
    line = [gene_name.upper(), twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = SP.exp(common)
    individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = SP.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)
    csv_out_file.flush()

def prepare_csv_out(CovFun, out_path, out_file):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    csv_out_file_confounded = open(os.path.join(out_path, out_file), 'wb')
    csv_out_confounded = csv.writer(csv_out_file_confounded)
    header = ["Gene", "Bayes Factor"]
    header.extend(map(lambda x:'Common ' + x, CovFun[0].get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, CovFun[0].get_hyperparameter_names()))
    csv_out_confounded.writerow(header)
    return csv_out_confounded, csv_out_file_confounded

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

def get_gptwosample_priors(dim, prediction_model):
    covar_priors_common = []
    covar_priors_individual = [] 
    # SECF amplitude
    covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [1, 1]])
    # lengthscale
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [6, 2]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [6, 2]])
    # fixedCF amplitude
    if prediction_model == covariance_model_id:
        covar_priors_common.append([lnpriors.lnGauss, [0, 1]])
        covar_priors_individual.append([lnpriors.lnGauss, [0, 1]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .6]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .6]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})
    return priors

def get_covariance_function(X, SECF, noiseCF):
    confounderCF_pca = fixed.FixedCF(SP.dot(X, X.T))
    return SumCF((SumCF((SECF, confounderCF_pca)), noiseCF))

def get_gptwosample_covariance_function(Y_dict, prediction_model, dim):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()

    if prediction_model == reconstruct_model_id:
        CovFun_gplvm = [SumCF((SECF, noiseCF))] * 3

    elif prediction_model == covariance_model_id:
        CovFun_gplvm = []
        
        # condition 1:
        CovFun_gplvm.append(get_covariance_function(Y_dict['X'][Y_dict['condition'] == 0], SECF, noiseCF))

        # condition 2:
        CovFun_gplvm.append(get_covariance_function(Y_dict['X'][Y_dict['condition'] == 1], SECF, noiseCF))

        # shared:
        CovFun_gplvm.append(get_covariance_function(Y_dict['X'], SECF, noiseCF))
    
    return CovFun_gplvm

def add_simulated_confounders(Ydict, gp_conf_model, components=4, **kw_args):
    """add simulated confounded expression data to dict"""

    #construct X for GP sampling
    # structure consists of [Time, factor1,factor2,..factorN]
    # (time is needed for product_linear covaraince which assume non-perfect independence of measurements
    
    Yconf = sample_GP(gp_conf_model._lvm_covariance, gp_conf_model._lvm_hyperparams['covar'], gp_conf_model._lvm_X, Ydict['Y'].shape[1])

    Ydict['confounder'] = Yconf
    Ydict['Y_confounded'] = Ydict['Y'] + Yconf
    return Ydict
    pass

def get_data(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, components):
    dump_file = "%s.pickle" % get_path_for_pickle(confounder_model, confounder_learning_model, components)
    explained_variance = .5
    if (not os.path.exists(dump_file)) or 'recalc' in sys.argv:
        print "recalculating simulations and reading data"
        Y_dict = read_data_from_file(cond1_file, cond2_file, fraction)
        # simulation model:
        gp_conf_model = conf_module.Confounder_Model(confounder_model, Y_dict['T'], Y_dict['condition'].reshape(-1, 1), components, explained_variance)
        Y_dict = add_simulated_confounders(Y_dict, gp_conf_model, components=components)
        # learning model:
        gp_conf_model = conf_module.Confounder_Model(confounder_learning_model, Y_dict['T'], Y_dict['condition'].reshape(-1, 1), components, explained_variance)
        Y_dict['X'], Y_dict['Y_reconstruct'] = gp_conf_model.learn_confounder_matrix(Y_dict['Y_confounded'])
        # save data:
        cPickle.dump(Y_dict, open(dump_file, 'wb'), -1)
    else:
        print "loading data and simulations from file: %s" % (dump_file)
        # data already exists
        Y_dict = cPickle.load(open(dump_file, 'r'))
        #gpmodel = construct_gp_model(Y_dict, model=confounder_model, components=components, explained_variance=explained_variance)
    return Y_dict

def f(cond1_file, cond2_file, components=4, **kwargs):
    dump_file = "%s.pickle" % get_path_for_pickle(confounder_model, confounder_learning_model, components)
    this_thread = threading.current_thread()
    run_demo(cond1_file, cond2_file, prediction_model=covariance_model_id, **kwargs)
    while(not os.path.exists(dump_file)):
        this_thread.join()
    run_demo(cond1_file, cond2_file, prediction_model=reconstruct_model_id, **kwargs)
    gptwosample_confounders_standard_prediction.run_demo(cond1_file=cond1_file, cond2_file=cond2_file, **kwargs)

def run_gptwosample_and_write_back_threaded(csv_lock, still_to_go, prediction_model, CovFun, priors, Y_dict, csv_out_GPLVM, csv_out_file_GPLVM, T1, T2, Tpredict, gene_name):
    if gene_name == "input":
        return

    gene_index = find_gene_name_hit(Y_dict, gene_name)
    
    if(gene_index == -1):
        print "%s not in random set" % (gene_name)
        still_to_go -= 1
        return
            
    print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
    
    twosample_object_gplvm = GPTwoSample_individual_covariance(CovFun, priors=priors)
    run_gptwosample_on_data(twosample_object_gplvm, Tpredict, T1, T2, 
                            get_gptwosample_data_for_model(prediction_model, 0, Y_dict, gene_index), 
                            get_gptwosample_data_for_model(prediction_model, 1, Y_dict, gene_index))
    csv_lock.acquire()
    write_back_data(twosample_object_gplvm, gene_name, csv_out_GPLVM, csv_out_file_GPLVM)
    csv_lock.release()

def run_demo(cond1_file, cond2_file, fraction=0.1, confounder_model=linear_covariance_model_id, confounder_learning_model=linear_covariance_model_id, prediction_model=reconstruct_model_id, components=4):
    """run demo script with condition file on random fraction of all data"""
    
    print "Sampled from: %s. Learned by: %s. Predicted by: %s" % (confounder_model, confounder_learning_model, prediction_model)
    
    logging.basicConfig(level=logging.INFO)
    # how many confounders to learn?
    components = 4
    
    # get all data needed into Y_dict:
    # This method will construct ans simulate confounded data, if there is not pickle saved.
    Y_dict = get_data(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, components)

    # hyperparamters
    dim = 1
    
    CovFun_gplvm = get_gptwosample_covariance_function(Y_dict, prediction_model, dim)    
    
    # save results into csv:
    out_path = get_path_for_pickle(confounder_model, confounder_learning_model, components)
    out_file_GPLVM = "%s-%i_confounders_GPLVM.csv" % (prediction_model, components)
    csv_out_GPLVM, csv_out_file_GPLVM = prepare_csv_out(CovFun_gplvm, out_path, out_file_GPLVM)
    if not os.path.exists(os.path.join(out_path, "plots")):
        os.mkdir(os.path.join(out_path, "plots"))
      
    # priors to start with:
    priors = get_gptwosample_priors(dim, prediction_model)
    
    twosample_object_gplvm = GPTwoSample_individual_covariance(CovFun_gplvm, priors=priors)

    T1 = Y_dict['T'][Y_dict['condition'] == 0]
    T2 = Y_dict['T'][Y_dict['condition'] == 1]

    Tpredict = get_model_structure(SP.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 96)[:, SP.newaxis], SP.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 2 * 96)[:, SP.newaxis]);

    # get ground truth genes for comparison:
    gt_names = {}
    for [name, val] in csv.reader(open("../examples/ground_truth_random_genes.csv", 'r')):
#    for [name, val] in csv.reader(open("../examples/ground_truth_balanced_set_of_100.csv", 'r')):
        gt_names[name.upper()] = val
    
    still_to_go = int(fraction * (len(gt_names)))
    csv_lock = threading.Lock()
    
    #loop through genes
#    for gene_name in ["CATMA1A24060", "CATMA1A49990"]:
    for gene_name in SP.random.permutation(gt_names.keys())[:still_to_go]:
        try:
            # find the right gene:
#            if gene_name == "input":
#                continue
#
#            gene_index = find_gene_name_hit(Y_dict, gene_name)
#
#            if(gene_index == -1):
#                print "%s not in random set"%(gene_name)
#                still_to_go -= 1
#                continue
#            
#            print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
            
#            run_gptwosample_on_data(twosample_object_pca, Tpredict, T1, T2,
#                                    get_gptwosample_data_for_model(prediction_model, "PCA", 0, Y_dict, gene_index),
#                                    get_gptwosample_data_for_model(prediction_model, "PCA", 1, Y_dict, gene_index))
#            write_back_data(twosample_object_pca, gene_name, csv_out_PCA)
#            plot_and_save_figure(T1, twosample_object_pca, gene_name, savename=os.path.join(out_path, "plots", "%s_%s-PCA.png" % (gene_name, prediction_model)))
            
            while(threading.active_count()>150):
                time.sleep(1)
            
            Thread(group=None, target=run_gptwosample_and_write_back_threaded, name=gene_name, args=(csv_lock, still_to_go, prediction_model, CovFun_gplvm, priors, Y_dict, csv_out_GPLVM, csv_out_file_GPLVM, T1, T2, Tpredict, gene_name)).start()
            
#            run_gptwosample_on_data(twosample_object_gplvm, Tpredict, T1, T2, get_gptwosample_data_for_model(prediction_model, 0, Y_dict, gene_index), get_gptwosample_data_for_model(prediction_model, 1, Y_dict, gene_index))
#            write_back_data(twosample_object_gplvm, gene_name, csv_out_GPLVM, csv_out_file_GPLVM)
            #plot_and_save_figure(T1, twosample_object_gplvm, gene_name, savename=os.path.join(out_path, "plots", "%s_%s.png" % (gene_name, prediction_model)))
    
            still_to_go -= 1
        except Exception as e:
            print e
            still_to_go -= 1

    
if __name__ == '__main__':
    cond1_file = './../examples/warwick_control.csv'
    cond2_file = '../examples/warwick_treatment.csv'
    fraction = 1
    
    for confounder_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
        for confounder_learning_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
#            run_demo(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, reconstruct_model_id, 4)
            Thread(target=f, name="%s>%s" % (confounder_model, confounder_learning_model), args=(cond1_file, cond2_file),
                   kwargs={'confounder_model':confounder_model,
                           'confounder_learning_model':confounder_learning_model,
                           'fraction':fraction
                           }, verbose=False).start()
#            thread.start_new_thread(f, (cond1_file, cond2_file),
#                                    {'confounder_model':confounder_model, 
#                                     'confounder_learning_model':confounder_learning_model, 
#                                     'fraction':fraction
#                                     }
#                                    )
#    gptwosample_confounders_standard_prediction.run_demo(cond1_file=cond1_file, cond2_file=cond2_file, confounder_model=confounder_model, confounder_learning_model=confounder_learning_model, prediction_model=prediction_model)
#            run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
