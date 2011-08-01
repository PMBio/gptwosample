'''
Small Example application of GPTwoSample
========================================

Please run script "generateToyExampleFiles.py" to generate Toy Data.
This Example shows how to apply GPTwoSample to toy data, generated above.

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
'''

from gptwosample.data import toy_data_generator
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_training_data_structure,\
    get_model_structure, common_id
from pygp.covar import linear, se, noise, combinators
from pygp.priors import lnpriors
import logging as LG
import pylab as PL
import scipy as SP
from gptwosample.plot.plot_basic import plot_results
from gptwosample.twosample.twosample_compare import GPTwoSampleMLII, GPTimeShift
from pygp.gp import gplvm
from pygp import likelihood as lik
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar.fixed import FixedCF


def run_demo(cond1_file, cond2_file):
    #full debug info:
    LG.basicConfig(level=LG.INFO)

    #1. read csv file
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

#range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    gene_names = sorted(cond1.keys()) 
    assert gene_names == sorted(cond2.keys())
    
    n_genes = len(gene_names)
    n_replicates = cond1[gene_names[0]].shape[0]
    gene_length = len(T1)
    
    components = 4
    
    Y1 = SP.array(cond1.values()).reshape(T1.shape[0]*n_replicates,-1)
    Y2 = SP.array(cond2.values()).reshape(T2.shape[0]*n_replicates,-1)

    X01 = gplvm.PCA(Y1, components)[0]
    X02 = gplvm.PCA(Y2, components)[0]
    
    lvm_covariance = linear.LinearCFISO(n_dimensions=components)
    
    X0=X01
    hyperparams = {'covar': SP.log([1.2])}
    hyperparams['x'] = X0
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    g = gplvm.GPLVM(covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y1)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*Y1.shape[1])
    hyperparams['x'] = X0
    
    print "running standard gplvm"
    [opt_hyperparams_1,opt_lml2] = opt_hyper(g,hyperparams,gradcheck=False)
    
    X0=X02
    hyperparams = {'covar': SP.log([1.2])}
    hyperparams['x'] = X0
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    g = gplvm.GPLVM(covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y2)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*Y2.shape[1])
    hyperparams['x'] = X0
    
    print "running standard gplvm"
    [opt_hyperparams_2,opt_lml2] = opt_hyper(g,hyperparams,gradcheck=False)
    
    Y_comm = SP.concatenate((Y1,Y2))#.reshape(T1.shape[0]*n_replicates*2,-1)
    X0 = SP.concatenate((X01, X02))#gplvm.PCA(Y_comm, components)[0]
    
    hyperparams = {'covar': SP.log([1.2])}
    hyperparams['x'] = X0
    
    likelihood = lik.GaussLikISO()
    hyperparams['lik'] = SP.log([0.1])
    g = gplvm.GPLVM(covar_func=lvm_covariance,likelihood=likelihood,x=X0,y=Y_comm)
    
    bounds = {}
    bounds['lik'] = SP.array([[-5.,5.]]*Y2.shape[1])
    hyperparams['x'] = X0
    
    print "running standard gplvm"
    [opt_hyperparams_comm,opt_lml2] = opt_hyper(g,hyperparams,gradcheck=False)
    
    X_conf_1 = opt_hyperparams_1['x'] * opt_hyperparams_1['covar']
    X_conf_2 = opt_hyperparams_2['x'] * opt_hyperparams_2['covar']

    X_conf_comm = SP.concatenate((X_conf_1,X_conf_2))#opt_hyperparams_comm['x']

    X_conf_1 = SP.dot(X_conf_1,X_conf_1.T) \
        
    X_conf_2 = SP.dot(X_conf_2,X_conf_2.T) \
    
    X_conf_comm = SP.dot(X_conf_comm, X_conf_comm.T) \
        
        
    
    #hyperparamters
    dim = 1
    replicate_indices = []
    for rep in SP.arange(n_replicates):
        replicate_indices.extend(SP.repeat(rep,gene_length))
    replicate_indices = SP.array(replicate_indices)
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
    
    shiftCFInd1 = combinators.ShiftCF(SECF,replicate_indices)
    shiftCFInd2 = combinators.ShiftCF(SECF,replicate_indices)
    shiftCFCom = combinators.ShiftCF(SECF,SP.concatenate((replicate_indices,replicate_indices+n_replicates)))

    CovFun = combinators.SumCF((SECF,noiseCF))
    
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
    for i in range(2*n_replicates):
        covar_priors_common.append([lnpriors.lnGauss,[0,.5]])
    for i in range(n_replicates):
        covar_priors_individual.append([lnpriors.lnGauss,[0,.5]])
        
    covar_priors_common.append([lnpriors.lnuniformpdf,[0,0]])
    covar_priors_individual.append([lnpriors.lnuniformpdf,[0,0]])
    covar_priors.append([lnpriors.lnuniformpdf,[0,0]])
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})
    #Ifilter = {'covar': SP.ones(n_replicates+3)}
    covar = [combinators.SumCF((combinators.SumCF((shiftCFInd1,FixedCF(X_conf_1))),noiseCF)),
             combinators.SumCF((combinators.SumCF((shiftCFInd2,FixedCF(X_conf_2))),noiseCF)),
             combinators.SumCF((combinators.SumCF((shiftCFCom,
                                                   FixedCF(X_conf_comm))),
                                noiseCF))]
    twosample_object = GPTimeShift(covar,
                                    priors=priors)
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
        twosample_object.set_data(get_training_data_structure(SP.tile(T1,Y0.shape[0]).reshape(-1, 1),
                                                              SP.tile(T2,Y1.shape[0]).reshape(-1, 1),
                                                              Y0.reshape(-1, 1),
                                                              Y1.reshape(-1, 1)))
        twosample_object.predict_model_likelihoods()
        twosample_object.predict_mean_variance(Tpredict)
        plot_results(twosample_object,
                     title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
                     shift=twosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates],
                     draw_arrows=1)
        PL.xlim(T1.min(), T1.max())
        
        PL.savefig("GPTwoSample_%s.png"%(gene_name),format='png')
        import pdb;pdb.set_trace()
        ## wait for window close
        pass

if __name__ == '__main__':
    run_demo(cond1_file = './ToyCondition1.csv', cond2_file = './ToyCondition2.csv')
