# '''
# Created on Feb 16, 2011

# @author: maxz
# '''
import scipy as SP

from pygp.covar import se, combinators, noise, bias
import pygp.priors.lnpriors as lnpriors
from gptwosample.twosample.twosample_base import TwoSampleShare

def get_toy_data(xmin=1, xmax=2.5 * SP.pi, step1=.7, step2=.4,
                 fy1=lambda x, b, C:b * x + C + 1 * SP.sin(x),
                 fy2=lambda x, b, C:(b * x + C + 1 * SP.sin(x)) * b + 1 * SP.cos(x),
                 sigma1=0.1, sigma2=0.1, b=0, C=2):
    """
    Create Toy Data
    """
    x1 = SP.arange(xmin, xmax, step1)
    x2 = SP.arange(xmin, xmax, step2)    

    y1 = fy1(x1, b, C)
    y1 += sigma1 * SP.randn(y1.shape[0])
    
    y2 = fy2(x2, b, C) 
    y2 *= 1 * SP.cos(x2)

    y2 += sigma2 * SP.randn(y2.shape[0])
    
    return x1, x2, y1, y2

def get_twosample(dim=1):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    
    
    CovFun = combinators.SumCF((SECF, noiseCF))
    CovFun = combinators.SumCF((SECF, bias.BiasCF()))

    #covar_priors = []
    #scale
    #covar_priors.append([lnpriors.lnGammaExp, [1, 2]])
    #covar_priors.append([lnpriors.lnGammaExp, [1, 1]])
    #covar_priors.append([lnpriors.lnGammaExp, [1, 1]])

    #priors = {'covar':SP.array(covar_priors)}

    twosample_object = TwoSampleShare(CovFun)

    return twosample_object
    
