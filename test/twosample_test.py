import sys
#sys.path.append('./../')
sys.path.append("./../../")

#from GPTwoSample.src import GPTwoSample
import GPTwoSample.src.twosample as TS

import unittest
import scipy as SP
import numpy.random as random

import GPTwoSample.pygp.gpr as GPR
from GPTwoSample.pygp.covar import se, noise, combinators
import GPTwoSample.pygp.lnpriors as lnpriors

class TestGPTwoSampleMLII(unittest.TestCase):
    
    def setUp(self):
        self.SECF = se.SEARDCF(1)
        self.covar = combinators.ProductCF((self.SECF,self.SECF))
        self.logtheta = SP.log(SP.array([1,3,3,1]))

        #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
        xmin = 1
        xmax = 2.5*SP.pi
        x1 = SP.arange(xmin,xmax,.7)
        x2 = SP.arange(xmin,xmax,.4)

        C = 2       #offset
        b = 0.5
        sigma1 = 0.1
        sigma2 = 0.1
        n_noises = 1

        b = 0

        y1  = b*x1 + C + 1*SP.sin(x1)
        dy1 = b   +     1*SP.cos(x1)
        y1 += sigma1*random.randn(y1.shape[0])
        y1-= y1.mean()

        y2  = b*x2 + C + 1*SP.sin(x2)
        y2 *= 1*SP.cos(x2)
        dy2 = b   +     1*SP.cos(x2)
        y2 += sigma2*random.randn(y2.shape[0])
        y2-= y2.mean()

        x1 = x1[:,SP.newaxis]
        x2 = x2[:,SP.newaxis]

        #predictions:
        self.X = SP.linspace(0,10,100)[:,SP.newaxis]

        #hyperparamters
        dim = 1
        
        logthetaCOVAR = SP.log([1,1,sigma1])#,sigma2])
        hyperparams = {'covar':logthetaCOVAR}

        SECF = se.SEARDCF(dim)
        #noiseCF = noise.NoiseReplicateCF(replicate_indices)
        noiseCF = noise.NoiseISOCF()

        CovFun = combinators.SumCF((SECF,noiseCF))

        covar_priors = []
        #scale
        covar_priors.append([lnpriors.lngammapdf,[1,2]])
        for i in range(dim):
            covar_priors.append([lnpriors.lngammapdf,[1,1]])
        #noise
        for i in range(n_noises):
            covar_priors.append([lnpriors.lngammapdf,[1,1]])

        priors = {'covar':SP.array(covar_priors)}

        self.twosample_initial_hyperparams = TS.GPTwoSampleMLII(CovFun, initial_hyperparameters=hyperparams)
        self.twosample_initial_priors = TS.GPTwoSampleMLII(CovFun, priors = priors)
        self.twosample = TS.GPTwoSampleMLII(CovFun)

        self.training_data_differential={'input':{'group_1':x1, 'group_2':x2},
                                         'output':{'group_1':y1, 'group_2':y2}}
        self.training_data_same={'input':{'group_1':x1, 'group_2':x1},
                                         'output':{'group_1':y1, 'group_2':y1+.1*random.randn(y1.shape[0])}}
        
    def test_bayes_factor(self):
        self.twosample_initial_hyperparams.predict_model_likelihoods(self.training_data_differential)
        self.twosample.predict_model_likelihoods(self.training_data_differential)

        self.twosample_initial_priors.predict_model_likelihoods(self.training_data_differential)
        bayes_differential = self.twosample_initial_priors.bayes_factor()
        
        self.twosample_initial_priors.predict_model_likelihoods(self.training_data_same)
        bayes_same = self.twosample_initial_priors.bayes_factor()

        self.assertTrue(bayes_differential > 0)
        self.assertTrue(bayes_same < 0)

    def test_plot(self):
        import pylab as PL
        #differential plot
        PL.figure()
        
        self.twosample_initial_priors.predict_model_likelihoods(self.training_data_differential)
        self.twosample_initial_priors.predict_mean_variance(self.X)
        self.twosample_initial_priors.plot_results()

        #non-differential plot
        PL.figure()
        
        self.twosample_initial_priors.predict_model_likelihoods(self.training_data_same)
        self.twosample_initial_priors.predict_mean_variance(self.X)
        self.twosample_initial_priors.plot_results()

        
if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        SECF = se.SEARDCF(1)
        covar = combinators.ProductCF((SECF,SECF))
        logtheta = SP.log(SP.array([1,3,3,1]))

        #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
        xmin = 1
        xmax = 2.5*SP.pi
        x1 = SP.arange(xmin,xmax,.7)
        x2 = SP.arange(xmin,xmax,.4)

        C = 2       #offset
        b = 0.5
        sigma1 = 0.1
        sigma2 = 0.1
        n_noises = 1

        b = 0

        y1  = b*x1 + C + 1*SP.sin(x1)
        dy1 = b   +     1*SP.cos(x1)
        y1 += sigma1*random.randn(y1.shape[0])
        y1-= y1.mean()

        y2  = b*x2 + C + 1*SP.sin(x2)
        dy2 = b   +     1*SP.cos(x2)
        y2 += sigma2*random.randn(y2.shape[0])
        y2-= y2.mean()

        x1 = x1[:,SP.newaxis]
        x2 = -x2[:,SP.newaxis]

        #predictions:
        X = SP.linspace(0,10,100)[:,SP.newaxis]

        #hyperparamters
        dim = 1

        logthetaCOVAR = SP.log([1,1,sigma1])#,sigma2])
        hyperparams = {'covar':logthetaCOVAR}

        SECF = se.SEARDCF(dim)
        noiseCF = noise.NoiseISOCF()

        CovFun = combinators.SumCF((SECF,noiseCF))
        CovFun_same = combinators.SumCF((SECF,noiseCF))

        covar_priors = []
        #scale
        covar_priors.append([lnpriors.lngammapdf,[1,2]])
        for i in range(dim):
            covar_priors.append([lnpriors.lngammapdf,[1,1]])
        #noise
        for i in range(n_noises):
            covar_priors.append([lnpriors.lngammapdf,[1,1]])

        priors = {'covar':SP.array(covar_priors)}

        twosample_initial_hyperparams = TS.GPTwoSampleMLII(CovFun, initial_hyperparameters=hyperparams)
        twosample_initial_priors = TS.GPTwoSampleMLII(CovFun, priors = priors)
        twosample_initial_priors_same = TS.GPTwoSampleMLII(CovFun_same, priors = priors)
        twosample = TS.GPTwoSampleMLII(CovFun)

        training_data_differential={'input':{'group_1':x1, 'group_2':x2},
                                    'output':{'group_1':y1, 'group_2':y2}}
        training_data_same={'input':{'group_1':x1, 'group_2':x1},
                            'output':{'group_1':y1, 'group_2':y1+sigma1*random.randn(y1.shape[0])}}


        model_likelihoods_init_hyper = twosample_initial_hyperparams.predict_model_likelihoods(training_data_differential)
        model_likelihoods_init_priors_differential = twosample_initial_priors.predict_model_likelihoods(training_data_differential)
        model_likelihoods_init_priors_same = twosample_initial_priors.predict_model_likelihoods(training_data_same)
        model_likelihoods = twosample.predict_model_likelihoods(training_data_differential)
