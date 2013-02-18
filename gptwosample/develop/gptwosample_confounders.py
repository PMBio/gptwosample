'''
Created on Sep 14, 2011

@author: maxz
'''
from gptwosample.data.data_base import get_model_structure
from gptwosample.twosample.twosample_compare import \
    GPTwoSample_individual_covariance
from gptwosample.util.confounder_constants import covariance_model_id,\
    linear_covariance_model_id, reconstruct_model_id,\
    product_linear_covariance_model_id
from pygp.covar import se, noise, fixed
from pygp.covar.combinators import SumCF
from pygp.priors import lnpriors
from threading import Thread
import logging
import os
import scipy
import sys
import threading
from gptwosample.util.warwick_confounder_data_io import get_ground_truth_iterator,\
    get_path_for_pickle, prepare_csv_out, get_data, find_gene_name_hit,\
    write_back_data, read_data_from_file, get_ground_truth_subset_100_iterator
import copy 
sys.path.append('./../../')

def get_gptwosample_data_for_model(prediction_model, condition, Y_dict, gene_index):
    if prediction_model == reconstruct_model_id:
        return Y_dict['Y_reconstruct'][Y_dict['condition'] == condition, gene_index]
    elif prediction_model == covariance_model_id:
        return Y_dict['Y_confounded'][Y_dict['condition'] == condition, gene_index]    
    
def run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, Y0, Y1):
    #create data structure for GPTwwoSample:
    #note; there is no need for the time points to be aligned for all replicates
    #creates score and time local predictions
    twosample_object.set_data_by_xy_data(T1, T2, Y0.reshape(-1, 1), Y1.reshape(-1, 1))
    twosample_object.predict_model_likelihoods()
    twosample_object.predict_mean_variance(Tpredict)

def get_gptwosample_priors(dim, prediction_model):
    covar_priors_common = []
    covar_priors_individual = [] 
    # SECF amplitude
    covar_priors_common.append([lnpriors.lnGammaExp, [3, .8]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [3, .8]])
    # lengthscale
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [6, 2]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [6, 2]])
    # fixedCF amplitude
    if prediction_model == covariance_model_id:
        covar_priors_common.append([lnpriors.lnGammaExp, [3, .8]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [3, .8]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .6]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .6]])
    
    priors = get_model_structure({'covar':scipy.array(covar_priors_individual)}, {'covar':scipy.array(covar_priors_common)})
    return priors

def get_gptwosample_covariance_function(Y_dict, prediction_model, dim):
    def get_covariance_function(X, SECF, noiseCF):
        confounderCF_pca = fixed.FixedCF(scipy.dot(X, X.T))
        return SumCF((SumCF((SECF, confounderCF_pca)), noiseCF))

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

def run_both_prediction_models_and_standard_prediction(Y_dict = {}, **kwargs):
    kwargs.update({"Y_dict":copy.deepcopy(Y_dict),"prediction_model":covariance_model_id})
    Thread(group=None,
           target=run_demo,
           name="run_demo: %s"%(covariance_model_id), 
           args=(), 
           kwargs=kwargs, 
           verbose=False).start()
#    kwargs.update({"Y_dict":copy.deepcopy(Y_dict),"prediction_model":reconstruct_model_id})
#    Thread(group=None,
#           target=run_demo,
#           name="run_demo: %s"%(reconstruct_model_id), 
#           args=(), 
#           kwargs=kwargs, 
#           verbose=False).start()
#    run_demo(Y_dict, prediction_model=reconstruct_model_id, **kwargs)
#    gptwosample_confounders_standard_prediction.run_demo(Y_dict, **kwargs)

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

def run_demo(Y_dict, 
             confounder_model=linear_covariance_model_id, 
             confounder_learning_model=linear_covariance_model_id, 
             prediction_model=reconstruct_model_id, components=4):
    """run demo script with condition file on random fraction of all data"""
    
    print "Sampled from: %s. Learned by: %s. Predicted by: %s" % \
          (confounder_model, confounder_learning_model, prediction_model)
    
    logging.basicConfig(level=logging.INFO)
    # how many confounders to learn?
    components = 4
            
    # get all data needed into Y_dict:
    # This method will construct ans simulate confounded data, if there is not pickle saved.
    Y_dict = get_data(Y_dict, confounder_model, confounder_learning_model, components)

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
    
#    twosample_object_gplvm = GPTwoSample_individual_covariance(CovFun_gplvm, priors=priors)

    T1 = Y_dict['T'][Y_dict['condition'] == 0]
    T2 = Y_dict['T'][Y_dict['condition'] == 1]

    Tpredict = get_model_structure(scipy.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 96)[:, scipy.newaxis], scipy.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 2 * 96)[:, scipy.newaxis]);

    # get ground truth genes for comparison:
    gt_names = {}
    for [name, val] in get_ground_truth_subset_100_iterator():
#    for [name, val] in csv.reader(open("../examples/ground_truth_balanced_set_of_100.csv", 'r')):
        gt_names[name.upper()] = val
    
    still_to_go = len(gt_names)
    csv_lock = threading.Lock()
    
    #loop through genes
#    for gene_name in ["CATMA1A24060", "CATMA1A49990"]:
    for gene_name in gt_names.keys():
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
            
            while(threading.active_count()>16):
                pass
            
            Thread(group=None,
                   target=run_gptwosample_and_write_back_threaded,
                   name=gene_name,
                   args=(csv_lock, still_to_go,
                         prediction_model,
                         CovFun_gplvm, priors,
                         Y_dict, csv_out_GPLVM,
                         csv_out_file_GPLVM,
                         T1, T2,
                         Tpredict,
                         gene_name)).start()
            
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
    fraction = .05
    components = 4
    
    Y_dict = None    
    
    for confounder_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
        for confounder_learning_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
#            run_demo(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, reconstruct_model_id, 4)
            if (Y_dict == None 
                and (not os.path.exists("%s.pickle"%get_path_for_pickle(confounder_model,
                                                                   confounder_learning_model,
                                                                   components)) 
                     or 'recalc' in sys.argv)):
                Y_dict = read_data_from_file(cond1_file, cond2_file, fraction)
            Thread(target=run_both_prediction_models_and_standard_prediction, 
                   name="%s>%s" % (confounder_model, confounder_learning_model), 
                   args=(),
                   kwargs={'confounder_model':confounder_model,
                           'confounder_learning_model':confounder_learning_model,
                           'components':components,
                           'Y_dict':copy.deepcopy(Y_dict)
                           }, verbose=False).start()
#    gptwosample_confounders_standard_prediction.run_demo(cond1_file=cond1_file, cond2_file=cond2_file, confounder_model=confounder_model, confounder_learning_model=confounder_learning_model, prediction_model=prediction_model)
#            run_demo(Y_dict, 
#             confounder_model=confounder_model, 
#             confounder_learning_model=confounder_learning_model, 
#             prediction_model=covariance_model_id, components=4)
