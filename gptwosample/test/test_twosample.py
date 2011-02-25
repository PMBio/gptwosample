import gptwosample.toy_data_generator as toy

import unittest
import scipy as SP
import numpy.random as random

class TestGPTwoSampleMLII(unittest.TestCase):
    
    def setUp(self):
        #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
        x1, x2, y1, y2 = toy.get_toy_data()
        self.X = SP.linspace(-2, 10, 100).reshape(-1,1)

        self.twosample_object = toy.get_twosample_object()

        self.training_data_differential=toy.get_training_data_structure(x1.reshape(-1,1), x2.reshape(-1,1), y1, y2)
        self.training_data_same=toy.get_training_data_structure(x1.reshape(-1,1), x1.reshape(-1,1), y1, y1+.1*random.randn(y1.shape[0]))
        
    def test_bayes_factor(self):
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        bayes_differential = self.twosample_object.bayes_factor()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        bayes_same = self.twosample_object.bayes_factor()

        self.assertTrue(bayes_differential > 0)
        self.assertTrue(bayes_same < 0)

    def test_plot(self):
        import pylab as PL
        #differential plot
        PL.figure()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        self.twosample_object.predict_mean_variance(self.X)
        self.twosample_object.plot_results()

        #non-differential plot
        PL.figure()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        self.twosample_object.predict_mean_variance(self.X)
        self.twosample_object.plot_results()
        
        PL.show()
        
if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        import pylab as PL
        
        twosample_object = toy.get_twosample_object()
        
        sigma1=.1
        x1,x2,y1,y2 = toy.get_toy_data(sigma1=sigma1)
        X = SP.linspace(-2, 10, 100).reshape(-1,1)
        
        training_data_differential=toy.get_training_data_structure(x1, x2, y1, y2)
        training_data_same=toy.get_training_data_structure(x1,x1,y1,y1+SP.randn(y1.shape[0]))

        twosample_object.predict_model_likelihoods(training_data_same)
        twosample_object.predict_mean_variance(X)
        twosample_object.plot_results()
        
        PL.figure()
        
        twosample_object.predict_model_likelihoods(training_data_differential)
        twosample_object.predict_mean_variance(X)
        twosample_object.plot_results()
        
        PL.show()