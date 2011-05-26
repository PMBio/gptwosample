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

from Carbon.QuickTime import pdActionActivateSubPanel
from gptwosample.data.data_base import get_training_data_structure, \
    get_model_structure, individual_id, common_id
from gptwosample.plot.plot_basic import plot_results
from gptwosample.twosample.twosample_compare import GPTwoSampleMLII, GPTimeShift
from pygp.covar import se, noise, combinators
from pygp.gp import GP
from pygp.optimize import opt_hyper
from pygp.plot import gpr_plot
from pygp.priors import lnpriors
import logging as LG
import numpy.random as random
import pylab as PL
import scipy as SP


#import pdb

def get_synthetic_data(n_replicates):
    """
    generate synthetic-replicate-data; just samples from a superposition of a sin + linear trend
    """
    xmin = 1
    xmax = 2.5*SP.pi
    x1 = SP.tile(SP.arange(xmin,xmax,.4), n_replicates)
    x2 = SP.tile(SP.arange(xmin,xmax,.4), n_replicates)
    
    C = 2       #offset
    #b = 0.5
    sigma1 = 0.15
    sigma2 = 0.15
    
    b = 0
    
    y1  = b*x1 + C + 1*SP.sin(x1)
#    dy1 = b   +     1*SP.cos(x1)
    y1 += sigma1*random.randn(y1.shape[0])
    y1-= y1.mean()
    
    y2  = b*x2 + C + 1*SP.sin(x2)
#    dy2 = b   +     1*SP.cos(x2)
    y2 += sigma2*random.randn(y2.shape[0])
    y2-= y2.mean()
    
    x1 = x1[:,SP.newaxis]
    x2 = (x2-2)[:,SP.newaxis]
    
    return x1,x2,y1,y2


def run_demo():
    LG.basicConfig(level=LG.INFO)
    random.seed(1)

    # noise for each replicate
#    sigma1 = 0.15
    # number of replicates
    n_replicates = 4
    
    # get synthetic timeshifted data
    x1,x2,y1,y2 = get_synthetic_data(n_replicates)
    
    #predictions:
    X = SP.linspace(-2,10,100*n_replicates)[:,SP.newaxis]
    
    #hyperparamters
    dim = 1
    replicate_indices = []
    for i,xi in enumerate((x1,x2)):
        for rep in SP.arange(i*n_replicates, (i+1)*n_replicates):
            replicate_indices.extend(SP.repeat(rep,len(SP.unique(xi))))
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
    
    shiftCFInd1 = combinators.ShiftCF(SECF,replicate_indices[:n_replicates*len(SP.unique(x1))])
    shiftCFInd2 = combinators.ShiftCF(SECF,replicate_indices[:n_replicates*len(SP.unique(x1))])
    shiftCFCom = combinators.ShiftCF(SECF,replicate_indices)

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
    
    training_data = get_training_data_structure(x1, x2, y1, y2)
    
    # First, GPTimeShift:
    gptwosample_object = GPTimeShift([combinators.SumCF((shiftCFInd1,noiseCF)),
                                      combinators.SumCF((shiftCFInd2,noiseCF)),
                                      combinators.SumCF((shiftCFCom,noiseCF))],
                                     priors=priors)
    
    gptwosample_object.predict_model_likelihoods(training_data=training_data)
    gptwosample_object.predict_mean_variance(X)
    
    PL.suptitle("Example for GPTimeShift with simulated data", fontsize=28)

    PL.subplot(212)
    plot_results(gptwosample_object, 
                 shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates], 
                 draw_arrows=2,legend=False,
                 xlabel="Time [h]",ylabel="Expression level")
    ylim = PL.ylim()
    
    # Second, GPTwoSample without timeshift:
    gptwosample_object = GPTwoSampleMLII(CovFun, priors={'covar':SP.array(covar_priors)})
    gptwosample_object.predict_model_likelihoods(training_data=training_data)
    gptwosample_object.predict_mean_variance(X)
    
    PL.subplot(211)
    plot_results(gptwosample_object,legend=False,
                 xlabel="Time [h]",ylabel="Expression level")
    
    PL.ylim(ylim)
    
    PL.subplots_adjust(left=.1, bottom=.1, 
    right=.96, top=.8,
    wspace=.4, hspace=.4)
    PL.show()
    
if __name__ == "__main__":
    run_demo()
