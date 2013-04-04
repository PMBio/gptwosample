'''
Created on Apr 3, 2013

@author: Max
'''
import unittest
import numpy
import pylab
from gptwosample.twosample.twosample import TwoSample
import h5py

class TwoSampleTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with h5py.File("testdata.h5f",'r') as data:
            cls.Y = numpy.array(data['twosample']["Y"])
            cls.T = numpy.array(data['twosample']["T"])
            cls.n,cls.r,cls.t,cls.d = data['twosample'].attrs.values()
        cls.twosample = TwoSample(cls.T, cls.Y)
        
    def setUp(self):
        pass
        

    def testLikelihoods(self):
        self.twosample.predict_likelihoods(self.T, self.Y, "Testing likelihoods...", False)
        pass

    def testPredict(self):
        interpolation_interval = numpy.linspace(self.T.min(), self.T.max(),100)
        indices = numpy.random.randint(self.d, size=(self.d/2))
        self.twosample.predict_means_variances(interpolation_interval, indices, "Testing Predictions")
        pass

    def testBayesFactors(self):
        try:
            self.twosample.bayes_factors()
        except RuntimeError:
            self.twosample.predict_likelihoods(self.T, self.Y)
            self.twosample.bayes_factors()
    
    def testPlot(self):
        interpolation_interval = numpy.linspace(self.T.min(), self.T.max(),100)
        indices = numpy.random.randint(self.d, size=(self.d))
        self.twosample.predict_means_variances(interpolation_interval, indices, "Testing Predictions")
        pylab.ion()
        pylab.figure()
        for p in self.twosample.plot():
            pylab.draw()
            raw_input("enter continue...")

if __name__ == "__main__":
    import sys;sys.argv = ['', 
                           'TwoSampleTest.testLikelihoods',
                           #'TwoSampleTest.testPredict',
                           'TwoSampleTest.testBayesFactors',
                           'TwoSampleTest.testPlot',
                           ]
    unittest.main()