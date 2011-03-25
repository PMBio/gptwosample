from gptwosample.data import get_training_data_structure
from gptwosample.plot import plot_results
import gptwosample.data.toy_data_generator as toy
import numpy.random as random
import pylab as PL
import scipy as SP
import unittest
from gptwosample.data.data_base import get_model_structure

        

class TestGPTwoSampleMLII(unittest.TestCase):
    
    def setUp(self):
        #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
        x1, x2, y1, y2 = toy.get_toy_data(step2=.7)
        intervals = SP.ones_like(x1)
        intervals = SP.array(intervals, dtype='bool')
        intervals[5:8] = False # for common only
        intervals[0:2] = False # for common only
        
        self.interval_indices = get_model_structure(common=~intervals, individual=intervals)
        
        self.X = SP.linspace(-2, 10, 100).reshape(-1, 1)

        self.twosample_object = toy.get_twosample_object()

        self.training_data_differential = get_training_data_structure(x1.reshape(-1, 1), x2.reshape(-1, 1), y1, y2)
        self.training_data_same = get_training_data_structure(x1.reshape(-1, 1), x1.reshape(-1, 1), y1, y1 + .1 * random.randn(y1.shape[0]))
        
    def test_bayes_factor(self):
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        bayes_differential = self.twosample_object.bayes_factor()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        bayes_same = self.twosample_object.bayes_factor()

        self.assertTrue(bayes_differential > 0)
        self.assertTrue(bayes_same < 0)

    def test_plot(self):
        #differential plot
        PL.figure()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        self.twosample_object.predict_mean_variance(self.X)
        plot_results(self.twosample_object)

        #non-differential plot
        PL.figure()
        
        self.twosample_object.predict_model_likelihoods(self.training_data_same)
        self.twosample_object.predict_mean_variance(self.X)

        plot_results(self.twosample_object)
        
        PL.show()
        
    def test_interval(self):
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        PL.clf()
        self.twosample_object.predict_mean_variance(self.X,
            interval_indices=self.interval_indices)            
        plot_results(self.twosample_object,
            interval_indices=self.interval_indices)
                    
if __name__ == '__main__':
    if(1):
        unittest.main()
    else:
        x1, x2, y1, y2 = toy.get_toy_data(step2=.7)
        X = SP.linspace(-2, 10, 100).reshape(-1, 1)
        twosample_object = toy.get_twosample_object()
        training_data_differential = get_training_data_structure(x1.reshape(-1, 1), x2.reshape(-1, 1), y1, y2)
        
        twosample_object.predict_model_likelihoods(training_data_differential)
        
        intervals = SP.ones_like(x1)
        intervals = SP.array(intervals, dtype='bool')
        intervals[1:3] = False
        intervals[-2:-1] = False
        twosample_object.predict_mean_variance(X.reshape(-1, 1),
               interval_indices=self.interval_indices)            
        plot_results(twosample_object, interval_indices = self.interval_indices)
            
        PL.show()
