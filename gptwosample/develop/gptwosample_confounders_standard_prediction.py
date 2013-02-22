'''
Created on Sep 14, 2011

@author: maxz
'''
from pygp.covar import se, noise, fixed
from pygp.covar.combinators import SumCF
import cPickle
import gptwosample_confounders as gpc
import logging
import os
from gptwosample.util.warwick_confounder_data_io import write_back_data,\
    get_ground_truth_iterator, prepare_csv_out, get_path_for_pickle,\
    get_ground_truth_subset_100_iterator
import scipy
from threading import Thread
from gptwosample.confounder.confounder_model import reconstruct_model_id,\
    covariance_model_id, linear_covariance_model_id,\
    product_linear_covariance_model_id


def run_demo(Y_dict, confounder_model, confounder_learning_model, fraction=0.1, components=4):
    """run demo sscript with condition file on random fractio of all data"""
    
    logging.basicConfig(level=logging.INFO)
    # how many confounders to learn?
    components = 4

    print "predict ideal and standard learned by: %s predicted by: %s" % (confounder_learning_model, confounder_model)
    
    #hyperparamters
    dim = 1
    
    CovFun_standard = get_gptwosample_covariance_function(Y_dict, reconstruct_model_id, dim)    
    CovFun_ideal = get_gptwosample_covariance_function(Y_dict, covariance_model_id, dim)    
    
    out_path = get_path_for_pickle(confounder_model, confounder_learning_model, components)
    out_file_standard = "%s-%i_confounders_standard.csv" % (confounder_learning_model,components)
    out_file_ideal = "%s-%i_confounders_ideal.csv" % (confounder_learning_model,components)    
#    out_file_confounderless = "%s-%i_confounders_confounderless.csv" % (confounder_learning_model,components)    
    
    print "running standard on \t %s" % (out_file_standard)
    print "running ideal on \t %s" % (out_file_ideal)
    
    csv_out_standard, out_file_standard = prepare_csv_out(CovFun_standard, out_path, out_file_standard)
#    csv_out_confounderless = gpc.prepare_csv_out(CovFun_standard, out_path, out_file_confounderless)
    csv_out_ideal, out_file_ideal = prepare_csv_out(CovFun_ideal, out_path, out_file_ideal)

    if not os.path.exists(os.path.join(out_path, "plots")):
        os.mkdir(os.path.join(out_path, "plots"))
        
    priors_standard = gpc.get_gptwosample_priors(dim, reconstruct_model_id)
    priors_ideal = gpc.get_gptwosample_priors(dim, covariance_model_id)
    
    twosample_object_standard = gpc.GPTwoSample_individual_covariance(CovFun_standard, priors=priors_standard)
    twosample_object_ideal = gpc.GPTwoSample_individual_covariance(CovFun_ideal, priors=priors_ideal)
    
    T1 = Y_dict['T'][Y_dict['condition'] == 0]
    T2 = Y_dict['T'][Y_dict['condition'] == 1]

    Tpredict = scipy.linspace(Y_dict['T'].min(), Y_dict['T'].max(), 96)[:, scipy.newaxis]

    # get ground truth genes for comparison:
    gt_names = []
    for [name, val] in get_ground_truth_subset_100_iterator():
        gt_names.append(name.upper())
#    gt_subset = scipy.random.permutation(gt_names)[:fraction*len(gt_names)]
    gt_subset = gt_names
    still_to_go = len(gt_subset) - 1
    
    #loop through genes
    for gene_name in gt_subset:
        try:
            gene_name_hit = Y_dict['gene_names'] == gene_name
            if gene_name is "input":
                continue
            if not gene_name_hit.any():
                still_to_go -= 1
                continue

            print 'processing %s, genes still to come: %i' % (gene_name, still_to_go)
            gene_index = scipy.where(gene_name_hit)[0][0]
            
            gpc.run_gptwosample_on_data(twosample_object_ideal, Tpredict, T1, T2,
                                    Y_dict['Y_confounded'][Y_dict['condition'] == 0, gene_index],
                                    Y_dict['Y_confounded'][Y_dict['condition'] == 1, gene_index])
            write_back_data(twosample_object_ideal, gene_name, csv_out_ideal, out_file_ideal)
            #gpc.plot_and_save_figure(T1, twosample_object_ideal, gene_name, savename=os.path.join(out_path, "plots", "%s_ideal.png" % gene_name))
    
            gpc.run_gptwosample_on_data(twosample_object_standard, Tpredict, T1, T2,
                                    Y_dict['Y_confounded'][Y_dict['condition'] == 0, gene_index],
                                    Y_dict['Y_confounded'][Y_dict['condition'] == 1, gene_index])
            write_back_data(twosample_object_standard, gene_name, csv_out_standard, out_file_standard)
            #gpc.plot_and_save_figure(T1, twosample_object_standard, gene_name, savename=os.path.join(out_path, "plots", "%s_standard.png" % gene_name))
            
#            gpc.run_gptwosample_on_data(twosample_object_standard, Tpredict, T1, T2,
#                                    Y_dict['Y'][Y_dict['condition'] == 0, gene_index],
#                                    Y_dict['Y'][Y_dict['condition'] == 1, gene_index])
#            gpc.write_back_data(twosample_object_standard, gene_name, csv_out_confounderless)
            #gpc.plot_and_save_figure(T1, twosample_object_standard, gene_name, savename=os.path.join(out_path, "plots", "%s_confounderless.png" % gene_name))
            
        except:
            pass
        finally:
            still_to_go -= 1
            pass
        pass


def get_gptwosample_covariance_function(Y_dict, prediction_model, dim):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()

    if prediction_model == reconstruct_model_id:
        CovFun = [SumCF((SECF, noiseCF))]*3

    elif prediction_model == covariance_model_id:
        CovFun = []
        # condition 1:
        X = Y_dict['confounder'][Y_dict['condition'] == 0]
        confounderCF = fixed.FixedCF(scipy.dot(X,X.T))
        CovFun.append(SumCF((SumCF((SECF, confounderCF)), noiseCF)))
        # condition 2:
        X = Y_dict['confounder'][Y_dict['condition'] == 1]
        confounderCF = fixed.FixedCF(scipy.dot(X,X.T))
        CovFun.append(SumCF((SumCF((SECF, confounderCF)), noiseCF)))
        # shared:
        X = Y_dict['confounder']
        confounderCF = fixed.FixedCF(scipy.dot(X,X.T))
        CovFun.append(SumCF((SumCF((SECF, confounderCF)), noiseCF)))
    
    return CovFun

if __name__ == '__main__':
    cond1_file='./../examples/warwick_control.csv'
    cond2_file='../examples/warwick_treatment.csv'
    components=4

    for confounder_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
        for confounder_learning_model in [product_linear_covariance_model_id, linear_covariance_model_id]:
            dump_file = "%s.pickle"%get_path_for_pickle(confounder_model, confounder_learning_model, components)
            Y_dict = cPickle.load(open(dump_file, 'r'))
            Thread(group=None, 
                   target=run_demo, 
                   name="%s, %s"%(confounder_learning_model, confounder_model), 
                   args=(), 
                   kwargs={"Y_dict":Y_dict, "confounder_model":confounder_model, "confounder_learning_model":confounder_learning_model, "components":components}, 
                   verbose=False).start()

#    confounder_model = linear_covariance_model_id
#    confounder_learning_model = linear_covariance_model_id
#    dump_file = "%s.pickle"%get_path_for_pickle(confounder_model, confounder_learning_model, components)
#    Y_dict = cPickle.load(open(dump_file, 'r'))
#    run_demo(Y_dict, confounder_model=confounder_model, confounder_learning_model=confounder_learning_model, components=components)
