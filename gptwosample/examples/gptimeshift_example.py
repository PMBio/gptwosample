"""
Application Example of GPTwosample
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) preprocessed by shiftCF(:py:class`covar.combinators.ShiftCF) and combined with noise
:py:class:`covar.noise.NoiseISOCF` by summing them up
(using :py:class:`covar.combinators.SumCF`).

Created on Apr 28, 2011

@author: maxz
"""

from gptwosample.data.data_base import get_training_data_structure, \
    get_model_structure, common_id
from gptwosample.plot.plot_basic import plot_results
from pygp.covar import se, noise, combinators
from pygp.priors import lnpriors
import logging as LG
import numpy.random as random
import scipy as SP
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.twosample.twosample_base import TwoSampleShare,\
    TwoSampleSeparate

def run_demo(cond1_file, cond2_file):
    LG.basicConfig(level=LG.INFO)
    random.seed(1)

    
    # noise for each replicate
#    sigma1 = 0.15
    
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
    
    #n_genes = len(gene_names)
    n_replicates = cond1[gene_names[0]].shape[0]
    gene_length = len(T1)
    
    #hyperparameters
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
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    
    priors = get_model_structure({'covar':SP.array(covar_priors_individual)}, {'covar':SP.array(covar_priors_common)})
    #Ifilter = {'covar': SP.ones(n_replicates+3)}
    
    for gene_name in gene_names:
        y1 = cond1[gene_name]
        y2 = cond2[gene_name]
        training_data = get_training_data_structure(SP.tile(T1,n_replicates).reshape(-1,1),
                                                    SP.tile(T2,n_replicates).reshape(-1,1),
                                                    y1.reshape(-1,1), y2.reshape(-1,1))
        
        # First, GPTwoSample_individual_covariance:
        gptwosample_object = TwoSampleSeparate(combinators.SumCF((shiftCFInd1,noiseCF)),
                                          combinators.SumCF((shiftCFInd2,noiseCF)),
                                          combinators.SumCF((shiftCFCom,noiseCF)),
                                         priors=priors)
        
        gptwosample_object.predict_model_likelihoods(training_data=training_data)
        gptwosample_object.predict_mean_variance(Tpredict)
        
        #PL.suptitle("Example for GPTwoSample_individual_covariance with simulated data", fontsize=24)
        
        import pylab as PL
        PL.subplot(212)
        plot_results(gptwosample_object, 
                     shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates], 
                     draw_arrows=1,legend=False, plot_old=True,
                     xlabel="Time [h]",ylabel="Expression level",
                     title=r'TimeShift: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gptwosample_object.bayes_factor()))
        ylim = PL.ylim()
        
        # Second, GPTwoSample without timeshift:
        gptwosample_object = TwoSampleShare(CovFun, priors={'covar':SP.array(covar_priors)})
        gptwosample_object.predict_model_likelihoods(training_data=training_data)
        gptwosample_object.predict_mean_variance(Tpredict)
        
        PL.subplot(211)
        plot_results(gptwosample_object,legend=False,
                     xlabel="Time [h]",ylabel="Expression level",
                     title=r'Standard: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gptwosample_object.bayes_factor()))
        
        PL.ylim(ylim)
        xlim = PL.xlim()
        
        PL.subplot(212)
        PL.xlim(xlim)
        
        PL.subplots_adjust(left=.1, bottom=.085, 
        right=.98, top=.92,
        wspace=.4, hspace=.47)
        
        PL.savefig("GPTimeShift_%s.png"%(gene_name),format="png")
        PL.show()
    
if __name__ == "__main__":
    run_demo('./ToyCondition1.csv','./ToyCondition2.csv')
