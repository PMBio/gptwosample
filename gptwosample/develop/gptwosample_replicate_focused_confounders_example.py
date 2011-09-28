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
    out_path = 'all_data_no_timeshift_learned_confounders'
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
    
    Y1 = SP.array(cond1.values()).reshape(T1.shape[0]*n_replicates_1,-1)
    Y2 = SP.array(cond2.values()).reshape(T2.shape[0]*n_replicates_2,-1)
#    Y1 = SP.array(cond1.values()).reshape(-1,1)
#    Y2 = SP.array(cond2.values()).reshape(-1,1)

    if simulate_confounders:
#        Y1_r = [scipy.random.randn(T1.shape[0]) for i in range(n_replicates_1)]
#        Y1_c = [scipy.random.randn(T1.shape[0]*i) for i in [n_replicates_1]]
#        
#        Y1_all_r = SP.tile(Y1_r,len(gene_names)).reshape()
#        Y1_all_c = SP.tile(Y1_c,len(gene_names))
#        
#        Y1_all = Y1_all_r + Y1_all_c
#        Y1_all = scipy.atleast_2d(Y1_all).T        
        
        Y2_r = [scipy.random.randn(T2.shape[0]) for i in range(n_replicates_2)]
        Y2_c = [scipy.random.randn(T2.shape[0]*i) for i in [n_replicates_2]]
        
        Y2_all_c = SP.tile(Y2_c,len(gene_names)).reshape(len(gene_names),-1)
        Y2_all_r = SP.tile(Y2_r,len(gene_names)).reshape(Y2_all_c.shape)
        
        Y2_all = Y2_all_r + Y2_all_c
        Y2_all = scipy.atleast_2d(Y2_all).T        
        pass
    
    # Simulate linear Kernel by PCA estimation:
    if simulate_confounders:
        Y_comm = SP.concatenate((Y1+Y2_all,Y2+Y2_all))
        pylab.figure()
        pylab.pcolor(SP.dot(Y2_all,Y2_all.T))
        pylab.colorbar()
        pylab.title(r"confounders")
        import pdb;pdb.set_trace()
    else:
        Y_comm = SP.concatenate((Y1,Y2))
        pylab.figure()
        pylab.pcolor(SP.dot(Y1,Y2.T))
        pylab.colorbar()
        pylab.title(r"Y_1 \cdot Y_2^T")
        import pdb;pdb.set_trace()
    
    X_pca = gplvm.PCA(Y_comm, components)[0]
    #SP.concatenate((X01, X02)).copy()#
    
    # init product covariance for right dimensions
    lvm_covariance = SumCF((SqexpCFARD(n_dimensions=1),
                                linear.LinearCFISO(n_dimensions=components,
                                                   dimension_indices=xrange(1,components+1))),
                               n_dimensions=components+1)
    hyperparams = {'covar': SP.log([1,1,1.2])}
    
    T = SP.tile(T1,n_replicates).reshape(-1,1)
    # Get X right:
    X0 = SP.concatenate((T.copy(),X_pca.copy()),axis=1)
    if learn_X:
        hyperparams['x'] = X_pca.copy()

    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.2])

    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1,components+1),covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y_comm)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*Y2.shape[1])
    
    # Filter for scalar factor
    Ifilter={'covar':SP.array([1,1,1]), 'lik':SP.ones(1), 'x':SP.array([1,1,1,1,1,1])}
    
    # run lvm on data
    print "running standard gplvm"
    [opt_hyperparams_comm,opt_lml2] = opt_hyper(g,hyperparams,gradcheck=grad_check)
    
#     Do gradchecks for covariance function
    if grad_check:
        gradcheck.grad_check_logtheta(lvm_covariance, hyperparams['covar'], X0)
        gradcheck.grad_check_Kx(lvm_covariance, hyperparams['covar'], X0)
    
    if learn_X:
        X_conf_comm = opt_hyperparams_comm['x'] * SP.exp(opt_hyperparams_comm['covar'][2])
    else:
        X_conf_comm = X_pca * SP.exp(opt_hyperparams_comm['covar'][2])
    
    X_len = X_conf_comm.shape[0]
    X_conf_1 = X_conf_comm[:X_len/2]
    X_conf_2 = X_conf_comm[X_len/2:]

    X_conf_1 = SP.dot(X_conf_1, X_conf_1.T)
    X_conf_2 = SP.dot(X_conf_2, X_conf_2.T)
    
    X_conf_comm = SP.dot(X_conf_comm, X_conf_comm.T) 
    
    pylab.figure()
    pylab.pcolor(X_conf_comm)
    pylab.title("learned confounders")
    import pdb;pdb.set_trace()

    #hyperparamters
    dim = 1
    replicate_indices_1 = []
    for rep in SP.arange(n_replicates_1):
        replicate_indices_1.extend(SP.repeat(rep,gene_length))
    replicate_indices_1 = SP.array(replicate_indices_1)
    replicate_indices_2 = []
    for rep in SP.arange(n_replicates_2):
        replicate_indices_2.extend(SP.repeat(rep,gene_length))
    replicate_indices_2 = SP.array(replicate_indices_2)
    #n_replicates = len(SP.unique(replicate_indices))
#    
#    logthetaCOVAR = [1,1]
#    logthetaCOVAR.extend(SP.repeat(SP.exp(1),n_replicates))
#    logthetaCOVAR.extend([sigma1])
#    logthetaCOVAR = SP.log(logthetaCOVAR)#,sigma2])
#    hyperparams = {'covar':logthetaCOVAR}
#    
    SECF = se.SqexpCFARD(dim)
    #noiseCF = noise.NoiseReplicateCF(replicate_indices)
    noiseCF = noise.NoiseCFISO()
    
    if timeshift:
        shiftCFInd1 = combinators.ShiftCF(SECF,replicate_indices_1)
        shiftCFInd2 = combinators.ShiftCF(SECF,replicate_indices_2)
        shiftCFCom = combinators.ShiftCF(SECF,
                                         SP.concatenate((replicate_indices_1,
                                                         replicate_indices_2+n_replicates_1)))
            
    covar_priors_common = []
    covar_priors_individual = []
    covar_priors = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors_individual.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    #shift
    if timeshift:
        for i in range(n_replicates):
            covar_priors_common.append([lnpriors.lnGauss,[0,.5]])
        for i in range(n_replicates_1):
            covar_priors_individual.append([lnpriors.lnGauss,[0,.5]])
        
    covar_priors_common.append([lnpriors.lnuniformpdf,[0,0]])
    covar_priors_individual.append([lnpriors.lnuniformpdf,[0,0]])
    covar_priors.append([lnpriors.lnuniformpdf,[0,0]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, 
                                 {'covar':SP.array(covar_priors_common)})
    #Ifilter = {'covar': SP.ones(n_replicates+3)}
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
    
    num = 1
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    while os.path.exists(out_file):
        out_file = "result_%i.csv"%num
        num+=1
    csv_out_file = open(os.path.join(out_path, out_file), 'wb')
    csv_out = csv.writer(csv_out_file)
    header = ["Gene", "Bayes Factor"]    
    if timeshift:
        header.extend(map(lambda x:'Common '+x,covar[0].get_hyperparameter_names()))
        header.extend(map(lambda x:'Individual '+x,covar[2].get_hyperparameter_names()))
        twosample_object = GPTwoSample_individual_covariance(covar,priors=priors)
    else:
        header.extend(map(lambda x:'Common '+x,covar.get_hyperparameter_names()))
        header.extend(map(lambda x:'Individual '+x,covar.get_hyperparameter_names()))
        twosample_object = GPTwoSample_share_covariance(covar,priors=priors)
    csv_out.writerow(header)

#    print 'sorting out genes not in ground truth'
#    gt_reader = csv.reader(open('../examples/ground_truth_random_genes.csv','r'))
#    gene_names = []
#    for line in gt_reader:
#        gene_names.append(line[0].upper())
    still_to_go = len(gene_names) - 1

    T1 = SP.tile(T1,n_replicates_1).reshape(-1, 1)
    T2 = SP.tile(T2,n_replicates_2).reshape(-1, 1)
    #loop through genes
    for gene_name in gene_names:
        try:
            #PL.close()
            #PL.close()
            if gene_name is "input":
                continue
            #expression levels: replicates x #time points
            Y0 = cond1[gene_name]
            Y1 = cond2[gene_name]
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
            line.extend(common)
            line.extend(individual)
            csv_out.writerow(line)
            
            #print 'plotting %s'%(gene_name)
            #plot_results(twosample_object,
            #             title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
            #             shift=twosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates],
            #             draw_arrows=1)
            #PL.xlim(T1.min(), T1.max())
            #PL.savefig("out/GPTwoSample_%s_raw.png"%(gene_name),format='png')
            #PL.figure()
            
            #yres_1 = g.predict(opt_hyperparams_1,opt_hyperparams_1['x'],var=False,output=components)
            #yres_2 = g.predict(opt_hyperparams_2,opt_hyperparams_2['x'],var=False,output=components)
            #yres_comm = g.predict(opt_hyperparams_comm,opt_hyperparams_comm['x'],var=False,output=components)
            #yres_len = yres_comm.shape[0]
            #yres_1 = yres_comm[:yres_len/2]
            #yres_2 = yres_comm[yres_len/2:]
            
            #twosample_object.set_data(get_training_data_structure(SP.tile(T1,Y0.shape[0]).reshape(-1, 1),
            #                                                      SP.tile(T2,Y1.shape[0]).reshape(-1, 1),
            #                                                      (Y0.reshape(-1)-yres_1).reshape(-1, 1),
            #                                                      (Y1.reshape(-1)-yres_2).reshape(-1, 1)))
            #twosample_object.predict_model_likelihoods()
            #twosample_object.predict_mean_variance(Tpredict)
    #        plot_results(twosample_object,
    #                     title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
    #                     shift=twosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates],
    #                     draw_arrows=1)
    #        PL.xlim(T1.min(), T1.max())
    #        
    #        PL.savefig("out/GPTwoSample_%s_confounder.png"%(gene_name),format='png')
            ## wait for window close
            #import pdb;pdb.set_trace()
        except:
            import sys
            print "Caught Failure on gene %s: " % (gene_name),sys.exc_info()[0]
            print "Genes left: %i"%(still_to_go)
        finally:
            still_to_go -= 1
        pass

if __name__ == '__main__':
    #for i in xrange(1,9):
    #    run_demo(cond1_file = './../examples/warwick_control.csv', cond2_file = '../examples/warwick_treatment.csv',components=i)
    run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv', simulate_confounders=True)
