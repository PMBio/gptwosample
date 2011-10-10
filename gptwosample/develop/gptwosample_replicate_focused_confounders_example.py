'''
Small Example application of GPTwoSample with confounder detection
==================================================================

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
'''
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_model_structure, \
    common_id, individual_id
from gptwosample.twosample.twosample_compare import \
    GPTwoSample_individual_covariance, GPTwoSample_share_covariance
from pygp import likelihood as lik
from pygp.covar import linear, se, noise, combinators, gradcheck
from pygp.covar.combinators import ProductCF, SumCF
from pygp.covar.fixed import FixedCF
from pygp.covar.se import SqexpCFARD
from pygp.gp import gplvm
from pygp.optimize.optimize_base import opt_hyper
from pygp.priors import lnpriors
import csv
import logging as LG
import os
import pdb
import scipy
import scipy as SP
import pylab
from gptwosample.plot.plot_basic import plot_results
from pygp.linalg.linalg_matrix import solve_chol
from numpy.linalg.linalg import cholesky


try:
    from gptwosample.data import toy_data_generator
except:
    import sys
    sys.path.append('../../')
    sys.path.append("../../../pygp")
finally:
    from gptwosample.data import toy_data_generator
#import pylab as PL
#from gptwosample.plot.plot_basic import plot_results

def run_demo(cond1_file, cond2_file, components=4, simulate_confounders = False):
    #full debug info:
    LG.basicConfig(level=LG.INFO)

    # Settings:
    timeshift = False
    grad_check = False
    learn_X = True
    print "Number of components: %i"%components
    out_path = 'simulated_learned_confounders_ProductCF_strict_priors'
    out_file = "%i_confounder.csv"%(components)
    print "writing to file: %s/%s"%(out_path,out_file)

    #1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")
    
    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    gene_names = sorted(cond1.keys()) 
    #assert gene_names == sorted(cond2.keys())
    
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1+n_replicates_2
    gene_length = len(T1)    
    
    T = SP.tile(T1,n_replicates).reshape(-1,1)
    
    # init product covariance for right dimensions
    lvm_covariance = ProductCF((SqexpCFARD(dimension_indices=[0]),
                                linear.LinearCFISO(dimension_indices=xrange(1,components+1))),
                               n_dimensions=components+1)
    hyperparams = {'covar': SP.log([1.1,1.2,1])}
    
    # no product for simulation and testing purpose
#    lvm_covariance = linear.LinearCFISO(n_dimensions=components)
#    hyperparams = {'covar': SP.log([.6])}

    
    Y1_conf = SP.array(cond1.values())
    Y2_conf = SP.array(cond2.values())

    Y_comm = SP.concatenate((Y1_conf,Y2_conf),1).reshape(Y1_conf.shape[0],-1)
    
    if simulate_confounders:
        print "simulating confounders"
        if 0:
            Y_comm += get_simulated_confounders_nicolo(T2, gene_names, n_replicates, Y1_conf, Y2_conf)
        else:
            Y_comm += get_simulated_confounders_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T)
            
    Y_dict = dict([[name, Y_comm[i]] for i,name in enumerate(cond1.keys())])
    
    # from now on we need Y_comm transposed:
    Y_comm = Y_comm.T
    
    # Simulate linear Kernel by PCA estimation:        
    print "running pca"
    X_pca = gplvm.PCA(Y_comm, components)[0]
    X_pca += 0.1*SP.random.randn(X_pca.shape[0], X_pca.shape[1])
    #SP.concatenate((X01, X02)).copy()#
        
    # Get X right:
    X0 = SP.concatenate((T.copy(),X_pca.copy()),axis=1)
#    X0 = X_pca.copy()
    if learn_X:
        hyperparams['x'] = X_pca.copy()

    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.2])

    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1,1+components),covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y_comm)
#    g = gplvm.GPLVM(gplvm_dimensions=xrange(components),covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y_comm)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*Y2_conf.shape[1])

    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm,opt_lml2] = opt_hyper(g,hyperparams,gradcheck=grad_check)
    
#     Do gradchecks for covariance function
    if grad_check:
        gradcheck.grad_check_logtheta(lvm_covariance, hyperparams['covar'], X0)
        gradcheck.grad_check_Kx(lvm_covariance, hyperparams['covar'], X0)
    
    if learn_X:
        X_conf_comm = opt_hyperparams_comm['x'] * SP.exp(opt_hyperparams_comm['covar'][2])
#        X_conf_comm = opt_hyperparams_comm['x'] * SP.exp(opt_hyperparams_comm['covar'][0])
    else:
        X_conf_comm = X_pca * SP.exp(opt_hyperparams_comm['covar'][0])
    
    X_len = X_conf_comm.shape[0]
    X_conf_1 = X_conf_comm[:X_len/2]
    X_conf_2 = X_conf_comm[X_len/2:]

    X_conf_1 = SP.dot(X_conf_1, X_conf_1.T)
    X_conf_2 = SP.dot(X_conf_2, X_conf_2.T)
    
    X_conf_comm = SP.dot(X_conf_comm, X_conf_comm.T) 
    
    #hyperparamters
    dim, replicate_indices_1, replicate_indices_2 = calculate_replicate_stuff(timeshift, n_replicates_1, n_replicates_2, gene_length)

    # covariance structure for prediction
    covar, covar_no_conf = get_covariance_functions(timeshift, X_conf_comm, X_conf_1, X_conf_2, n_replicates_1, replicate_indices_1, replicate_indices_2, dim)    
    # priors for prediction
    covar_priors_individual, covar_priors_common, covar_no_conf_priors = get_priors(timeshift, n_replicates_1, n_replicates, dim)
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, 
                                 {'covar':SP.array(covar_priors_common)})
    
    num = 1
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    while os.path.exists(os.path.join(out_path,out_file)):
        out_file = out_file.replace("_%i.csv"%num,".csv")
        num+=1
        out_file = out_file.replace(".csv","_%i.csv"%num)
    csv_out_file = open(os.path.join(out_path, out_file), 'wb')
    csv_out = csv.writer(csv_out_file)
    
    csv_out_file_no_conf = open(os.path.join(out_path, out_file.replace(".csv","_no_conf.csv")), 'wb')
    csv_out_no_conf = csv.writer(csv_out_file_no_conf)
    
    # Prepare header for writing
    header = ["Gene", "Bayes Factor"]    
    header_no_conf = ["Gene", "Bayes Factor"]    
    if timeshift:
        header.extend(map(lambda x:'Common '+x,covar[0].get_hyperparameter_names()))
        header.extend(map(lambda x:'Individual '+x,covar[2].get_hyperparameter_names()))        
        twosample_object = GPTwoSample_individual_covariance(covar,priors=priors)
    else:
        header.extend(map(lambda x:'Common '+x,covar.get_hyperparameter_names()))
        header.extend(map(lambda x:'Individual '+x,covar.get_hyperparameter_names()))
        twosample_object = GPTwoSample_share_covariance(covar,priors=priors)

    twosample_no_conf_object = GPTwoSample_share_covariance(covar_no_conf,
                                                            priors=get_model_structure({'covar':covar_no_conf_priors}))
    header_no_conf.extend(map(lambda x:'Common '+x,covar_no_conf.get_hyperparameter_names()))
    header_no_conf.extend(map(lambda x:'Individual '+x,covar_no_conf.get_hyperparameter_names()))
    csv_out.writerow(header)
    csv_out_no_conf.writerow(header_no_conf)
    
    print 'sorting out genes not in ground truth'
    gt_reader = csv.reader(open('../examples/ground_truth_balanced_set_of_100.csv','r'))
    gt_gene_names = []
    for line in gt_reader:
        gt_gene_names.append(line[0].upper())
    still_to_go = len(gene_names) - 1

    T1 = SP.tile(T1,n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2,n_replicates_2).reshape(-1, 1)
    #loop through genes
    for i,gene_name in enumerate(gt_gene_names):
        try:
            if gene_name is "input":
                continue
            #expression levels: replicates x #time points
            if simulate_confounders:
                Y0 = Y_dict[gene_name][:gene_length*n_replicates_1]
                Y1 = Y_dict[gene_name][gene_length*n_replicates_1:]
            else:
                Y0 = cond1[gene_name]
                Y1 = cond2[gene_name]

            pdb.set_trace()
                
            #create data structure for GPTwwoSample:
            #note; there is no need for the time points to be aligned for all replicates
            #creates score and time local predictions
            twosample_object.set_data_by_xy_data(T1,T2,
                                                 Y0.reshape(-1, 1),
                                                 Y1.reshape(-1, 1))
            print 'processing %s, genes still to come: %i'%(gene_name,still_to_go)
            twosample_object.predict_model_likelihoods()
            twosample_object.predict_mean_variance(Tpredict)
    
            line = [gene_name, twosample_object.bayes_factor()]
            common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
            individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
            if timeshift:
                timeshift_index = scipy.array(scipy.ones_like(common), dtype='bool')
                timeshift_index[dim + 1:dim + 1 + n_replicates_1+n_replicates_2] = 0
                common[timeshift_index] = scipy.exp(common[timeshift_index])
                timeshift_index = scipy.array(scipy.ones_like(individual), dtype='bool')
                timeshift_index[dim + 1:dim + 1 + n_replicates_1] = 0
                individual[timeshift_index] = scipy.exp(individual[timeshift_index])
            else:
                common = scipy.exp(common)
                individual = scipy.exp(individual)
            line.extend(common)
            line.extend(individual)
            csv_out.writerow(line)
            
            ################## plotting >>>>>>>>>>>>>
            pylab.figure(1)
            shift = None
            if timeshift:
                shift=SP.concatenate((replicate_indices_1,replicate_indices_2+n_replicates_1))
            print 'plotting %s'%(gene_name)
            pylab.clf()
            plot_results(twosample_object,
                         title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
                         shift=shift,
                         draw_arrows=1)
            pylab.xlim(T1.min(), T1.max())
            plot_path = os.path.join(out_path,out_file.replace(".csv",""))
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)
            pylab.savefig(os.path.join(plot_path, "%s_%s_confounder.png"%(gene_name,components)),format='png')
            twosample_no_conf_object.set_data_by_xy_data(T1, T2, 
                                                         Y0.reshape(-1,1),
                                                         Y1.reshape(-1,1))
            twosample_no_conf_object.predict_model_likelihoods()
            twosample_no_conf_object.predict_mean_variance(Tpredict)
            
            pylab.clf()
            plot_results(twosample_no_conf_object,
                         title='%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_no_conf_object.bayes_factor()),
                         shift=shift,
                         draw_arrows=1)
            pylab.xlim(T1.min(), T1.max())
            pylab.savefig(os.path.join(plot_path, "%s_%s_standard_prediction.png"%(gene_name,components)),format='png')
            
            line = [gene_name, twosample_no_conf_object.bayes_factor()]
            common = twosample_no_conf_object.get_learned_hyperparameters()[common_id]['covar']
            individual = twosample_no_conf_object.get_learned_hyperparameters()[individual_id]['covar']
            common = scipy.exp(common)
            individual = scipy.exp(individual)        
            line.extend(common)
            line.extend(individual)
            csv_out_no_conf.writerow(line)
            
            twosample_no_conf_object.set_data_by_xy_data(T1, T2, 
                                                         cond1[gene_name].reshape(-1,1),
                                                         cond2[gene_name].reshape(-1,1))
            twosample_no_conf_object.predict_model_likelihoods()
            twosample_no_conf_object.predict_mean_variance(Tpredict)
            
            pylab.clf()
            plot_results(twosample_no_conf_object,
                         title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_no_conf_object.bayes_factor()),
                         shift=shift,
                         draw_arrows=1)
            pylab.xlim(T1.min(), T1.max())
            pylab.savefig(os.path.join(plot_path, "%s_%s_raw.png"%(gene_name,components)),format='png')
            
            line = [gene_name, twosample_no_conf_object.bayes_factor()]
            common = twosample_no_conf_object.get_learned_hyperparameters()[common_id]['covar']
            individual = twosample_no_conf_object.get_learned_hyperparameters()[individual_id]['covar']
            common = scipy.exp(common)
            individual = scipy.exp(individual)        
            line.extend(common)
            line.extend(individual)
            csv_out_no_conf.writerow(line)
            
            ################## plotting <<<<<<<<<<<<<<< 
            
#            yres_1 = g.predict(opt_hyperparams_1,opt_hyperparams_1['x'],var=False,output=components)
#            yres_2 = g.predict(opt_hyperparams_2,opt_hyperparams_2['x'],var=False,output=components)
#            yres_comm = g.predict(opt_hyperparams_comm,opt_hyperparams_comm['x'],var=False,output=components)
#            yres_len = yres_comm.shape[0]
#            yres_1 = yres_comm[:yres_len/2]
#            yres_2 = yres_comm[yres_len/2:]
#            
#            twosample_object.set_data(get_training_data_structure(SP.tile(T1,Y0.shape[0]).reshape(-1, 1),
#                                                                  SP.tile(T2,Y1_conf.shape[0]).reshape(-1, 1),
#                                                                  (Y0.reshape(-1)-yres_1).reshape(-1, 1),
#                                                                  (Y1_conf.reshape(-1)-yres_2).reshape(-1, 1)))
#            twosample_object.predict_model_likelihoods()
#            twosample_object.predict_mean_variance(Tpredict)
#            plot_results(twosample_object,
#                         title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
#                         shift=twosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates],
#                         draw_arrows=1)
#            PL.xlim(T1.min(), T1.max())
#            
#            PL.savefig("out/GPTwoSample_%s_confounder.png"%(gene_name),format='png')
#            ## wait for window close
#            #import pdb;pdb.set_trace()
        except:
            import sys
            print "Caught Failure on gene %s: " % (gene_name),sys.exc_info()
            print "Genes left: %i"%(still_to_go)
        finally:
            still_to_go -= 1

def get_simulated_confounders_nicolo(T2, gene_names, n_replicates, Y1_conf, Y2_conf):
    # Nicolo's way of simulating confounders
    Y2_r = [scipy.random.randn(T2.shape[0]) for i in range(n_replicates)]
    Y2_c = [scipy.random.randn(T2.shape[0] * i) for i in [n_replicates]]
    Y2_all_c = SP.tile(Y2_c, len(gene_names)).reshape(len(gene_names), -1)
    Y2_all_r = SP.tile(Y2_r, len(gene_names)).reshape(Y2_all_c.shape)
    Y2_all = Y2_all_r + Y2_all_c
    Y2_conf = scipy.atleast_2d(Y2_all).T
    Y_comm = SP.concatenate((Y1_conf, Y2_conf)) + Y2_conf
    return Y_comm


def get_simulated_confounders_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T):
    # or draw from a GP:
    NRT = n_replicates * gene_length 
    X = SP.concatenate((T.copy().T,scipy.randn(NRT,components).T)).T
    sigma = 1e-6
    # Y_conf = scipy.array([scipy.dot(cholesky(lvm_covariance.K(hyperparams['covar'], X) + sigma * scipy.eye(NRT)), scipy.randn(NRT, 1)).flatten() for i in range(len(gene_names))])

    X = SP.random.randn(NRT,components)
    W = SP.random.randn(components, len(gene_names))*0.5
    Y_conf = SP.dot(X, W)
    return Y_conf.T


def calculate_replicate_stuff(timeshift, n_replicates_1, n_replicates_2, gene_length):
    dim = 1
    replicate_indices_1 = None
    replicate_indices_2 = None
    if timeshift:
        replicate_indices_1 = []
        for rep in SP.arange(n_replicates_1):
            replicate_indices_1.extend(SP.repeat(rep, gene_length))
        
        replicate_indices_1 = SP.array(replicate_indices_1)
        replicate_indices_2 = []
        for rep in SP.arange(n_replicates_2):
            replicate_indices_2.extend(SP.repeat(rep, gene_length))
        
        replicate_indices_2 = SP.array(replicate_indices_2)
    return dim, replicate_indices_1, replicate_indices_2

def get_covariance_functions(timeshift, X_conf_comm, X_conf_1, X_conf_2, n_replicates_1, replicate_indices_1, replicate_indices_2, dim):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    if timeshift:
        shiftCFInd1 = combinators.ShiftCF(SECF, replicate_indices_1)
        shiftCFInd2 = combinators.ShiftCF(SECF, replicate_indices_2)
        shiftCFCom = combinators.ShiftCF(SECF, 
            SP.concatenate((replicate_indices_1, 
                    replicate_indices_2 + n_replicates_1)))
    if timeshift:
        covar = [combinators.SumCF((combinators.SumCF((shiftCFInd1, 
                            FixedCF(X_conf_1))), 
                    noiseCF)), 
            combinators.SumCF((combinators.SumCF((shiftCFInd2, 
                            FixedCF(X_conf_2))), 
                    noiseCF)), 
            combinators.SumCF((combinators.SumCF((shiftCFCom, 
                            FixedCF(X_conf_comm))), 
                    noiseCF))]
    else:
        covar = combinators.SumCF((combinators.SumCF((SECF, 
                        FixedCF(X_conf_comm))), 
                noiseCF))
    covar_no_conf = combinators.SumCF((SECF, noiseCF))
    return covar, covar_no_conf

def get_priors(timeshift, n_replicates_1, n_replicates, dim):
    covar_priors_common = []
    covar_priors_individual = []
    covar_no_conf_priors = []
    # amplitude
    covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
    covar_no_conf_priors.append([lnpriors.lnGammaExp, [6, .3]])
    # lengthscale
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])
        covar_no_conf_priors.append([lnpriors.lnGammaExp, [30, .1]])
    # timeshift
    if timeshift:
        for i in range(n_replicates):
            covar_priors_common.append([lnpriors.lnGauss, [0, 1]])
        for i in range(n_replicates_1):
            covar_priors_individual.append([lnpriors.lnGauss, [0, 1]])
    # confounders amplitude
    covar_priors_common.append([lnpriors.lnuniformpdf, [1, 1]])
    covar_priors_individual.append([lnpriors.lnuniformpdf, [1, 1]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, 1]])
        covar_no_conf_priors.append([lnpriors.lnGammaExp, [1, 1]])
    
    return covar_priors_individual, covar_priors_common, covar_no_conf_priors            

if __name__ == '__main__':
#    for i in xrange(1,5):
    run_demo(cond1_file = './../examples/warwick_control.csv', cond2_file = '../examples/warwick_treatment.csv',components=4, simulate_confounders=True)
    #run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv', simulate_confounders=True)
