'''
Created on Feb 16, 2011

@author: maxz
'''

import scipy as SP

from pygp.covar import se, combinators, noise
import pygp.priors.lnpriors as lnpriors

import gptwosample.twosample.twosample_compare as TS

def get_toy_data(xmin=1, xmax=2.5*SP.pi, step1=.7, step2=.4,
                 fy1 = lambda x,b,C:b*x +C+ 1*SP.sin(x),
                 fy2 = lambda x,b,C:(b*x +C+ 1*SP.sin(x))*b+1*SP.cos(x),
                 sigma1 = 0.1, sigma2=0.1,b=0,C=2):

    x1 = SP.arange(xmin, xmax, step1)
    x2 = SP.arange(xmin, xmax, step2)    

    y1 = fy1(x1,b,C)
    y1 += sigma1 * SP.randn(y1.shape[0])
    y1 -= y1.mean()
    
    y2 =fy2(x2,b,C) 
    y2 *= 1 * SP.cos(x2)

    y2 += sigma2 * SP.randn(y2.shape[0])
    y2 -= y2.mean()
    
    return x1, x2, y1, y2

def get_twosample_objects(dim=1):
    SECF = se.SEARDCF(dim)
    noiseCF = noise.NoiseISOCF()

    CovFun = combinators.SumCF((SECF,noiseCF))

    covar_priors = []
    #scale
    covar_priors.append([lnpriors.lngammapdf,[1,2]])
    covar_priors.append([lnpriors.lngammapdf,[1,1]])
    covar_priors.append([lnpriors.lngammapdf,[1,1]])

    priors = {'covar':SP.array(covar_priors)}

    twosample_object = TS.GPTwoSampleMLII(CovFun, priors = priors)

    return twosample_object

def get_training_data_structure(x1,x2,y1,y2):
    return {'input':{'group_1':x1.reshape(-1,1), 'group_2':x2.reshape(-1,1)},
            'output':{'group_1':y1, 'group_2':y2}}
    