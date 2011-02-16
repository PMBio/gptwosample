'''
Created on Feb 16, 2011

@author: maxz
'''
from test import toy_data_generator

import unittest
import scipy as SP
import numpy.random as random

import test.toy_data_generator as toy
from gptwosample.twosample import interval_gibbs

class TestGPTwoSampleMLII(unittest.TestCase):
    
    def setUp(self):
        x1, x2, y1, y2 = toy_data_generator.get_toy_data()
        self.X = SP.linspace(-2, 10, 100).reshape(-1,1)

        self.twosample_object = toy_data_generator.get_twosample_objects()

        self.training_data_differential=toy_data_generator.get_training_data_structure(x1, x2, y1, y2)
        self.training_data_same=toy_data_generator.get_training_data_structure(x1, x1, y1, y1+.1*random.randn(y1.shape[0]))

        self.interval_gibbs = interval_gibbs.GPTwoSampleInterval(self.twosample_object)
        
    def test_sampling(self):
        #self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        #self.interval_gibbs.predict_all_x_interval_probabilities();

        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        self.interval_gibbs.predict_all_x_interval_probabilities();
        
    def test_plot(self):
        import pylab as PL
        PL.figure()
        
        #differential plot
        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        self.interval_gibbs.predict_all_x_interval_probabilities();
        self.interval_gibbs.plot_predicted_results()
        
        PL.show()
        
if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        import pylab as PL
        
        twosample_object = toy.get_twosample_objects()
        
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