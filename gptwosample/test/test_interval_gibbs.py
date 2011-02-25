'''
Created on Feb 16, 2011

@author: maxz
'''
import gptwosample.toy_data_generator as toy_data_generator

import unittest
import scipy as SP
import numpy.random as random

from gptwosample.twosample import interval_gibbs
from pyio import csvex

class TestGPTwoSampleMLII(unittest.TestCase):
    
    def setUp(self):
        data = csvex.readCSV('./demo.csv',',',typeC='str')
        
        #1. header versus data
        col_header = data[:,0]
        #time
        Tc = SP.array(data[0,1::],dtype='float')
        #expression levels
        Yc = SP.array(data[1::,1::],dtype='float')
        #how many unique labels ? 
        gene_names = SP.unique(col_header[1::])
        Ngenes = gene_names.shape[0]
        Nrepl  = Yc.shape[0]/(Ngenes*2)
        
        T  = SP.zeros([Nrepl,Tc.shape[0]])
        T[:,:] = Tc
        
        g = 0
        
        i0 = g*Nrepl*2
        i1 = i0+Nrepl
        #expression levels: replicates x #time points
        Y0 = Yc[i0:i0+Nrepl,:]
        Y1 = Yc[i1:i1+Nrepl,:]
        
#        T, T, Y0, Y1 = toy_data_generator.get_toy_data(step1=.4, step2=.4)
        
        self.X = SP.linspace(T.min(), T.max(), 100).reshape(-1,1)
        self.twosample_object = toy_data_generator.get_twosample_object()

        self.training_data_differential=toy_data_generator.get_training_data_structure(T, T, Y0, Y1)
        self.interval_gibbs = interval_gibbs.GPTwoSampleInterval(self.twosample_object)
        
    def test_sampling(self):
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        self.interval_gibbs.predict_all_x_interval_probabilities();

#        self.twosample_object.predict_model_likelihoods(self.training_data_same)
#        self.interval_gibbs.predict_all_x_interval_probabilities();
        
    def test_plot(self):
        import pylab as PL
        PL.figure()
        
        #differential plot
        self.twosample_object.predict_model_likelihoods(self.training_data_differential)
        self.interval_gibbs.predict_all_x_interval_probabilities();
        self.interval_gibbs.plot_predicted_results()
        
#        PL.figure()
#        
#        #same plot
#        self.twosample_object.predict_model_likelihoods(self.training_data_same)
#        self.interval_gibbs.predict_all_x_interval_probabilities();
#        self.interval_gibbs.plot_predicted_results()
#        
        PL.show()
        
if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        import pylab as PL
        
        twosample_object = toy_data_generator.get_twosample_object()
        
        sigma1=.1
        x1,x2,y1,y2 = toy_data_generator.get_toy_data(sigma1=sigma1)
        X = SP.linspace(-2, 10, 100).reshape(-1,1)
        
        training_data_differential=toy_data_generator.get_training_data_structure(x1, x2, y1, y2)
        training_data_same=toy_data_generator.get_training_data_structure(x1,x1,y1,y1+SP.randn(y1.shape[0]))

        twosample_object.predict_model_likelihoods(training_data_same)
        twosample_object.predict_mean_variance(X)
        twosample_object.plot_results()
        
        PL.figure()
        
        twosample_object.predict_model_likelihoods(training_data_differential)
        twosample_object.predict_mean_variance(X)
        twosample_object.plot_results()
        
        PL.show()