'''
Small Example application of GPTwoSample
========================================

Please run script "generateToyExampleFiles.py" to generate Toy Data.
This Example shows how to apply GPTwoSample to toy data, generated above.

Created on Feb 25, 2011

@author: Max Zwiessele, Oliver Stegle
'''

from gptwosample.data.dataIO import get_data_from_csv
import logging as LG
import scipy as SP
import numpy
from gptwosample.confounder.confounder import TwoSampleConfounder


if __name__ == '__main__':
    cond1_file = './ToyCondition1.csv'; cond2_file = './ToyCondition2.csv'
    
        #full debug info:
    LG.basicConfig(level=LG.INFO)

    #1. read csv file
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    #range where to create time local predictions ? 
    #note: this needs to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    gene_names = cond1.keys() 
    assert gene_names == cond2.keys()
    
    #expression levels: replicates x #time points
    Y0 = numpy.array(cond1.values()).T.swapaxes(0,1)
    Y1 = numpy.array(cond2.values()).T.swapaxes(0,1)
    
    T = numpy.array([numpy.tile(T1[:,None], Y0.shape[0]),
                     numpy.tile(T2[:,None], Y1.shape[0])]).swapaxes(1,2)
    Y = numpy.array([Y0,Y1])
    # Test for nans in data:
    n,r,t,d = Y.shape
    ri = numpy.random.randint    
    for _ in range(4):
        Y[ri(n), ri(r), ri(t), ri(d)] = numpy.nan
    
    confounder_object = TwoSampleConfounder(T, Y, q=2)
    confounder_object.learn_confounder_matrix()
    confounder_object.predict_likelihoods(T,Y)
    
    Tpredict = numpy.linspace(T1.min(), T1.max(), 100)[:,None]
    it = confounder_object.predict_means_variances(Tpredict)
    import pylab
    pylab.ion()
    pylab.figure()
    for _ in confounder_object.plot():
        raw_input("Press Enter to continue...")
    