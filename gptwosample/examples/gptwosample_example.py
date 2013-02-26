'''
Small Example application of GPTwoSample
========================================

Please run script "generateToyExampleFiles.py" to generate Toy Data.
This Example shows how to apply GPTwoSample to toy data, generated above.

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
'''

from gptwosample.data import toy_data_generator
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.data.data_base import get_training_data_structure
import logging as LG
import pylab as PL
import scipy as SP
from gptwosample.plot.plot_basic import plot_results


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
    T1 = cond1["input"]
    T2 = cond2["input"]
    
    gene_names = sorted(cond1.keys()) 
    assert gene_names == sorted(cond2.keys())
    
    twosample_object = toy_data_generator.get_twosample_object()
    #loop through genes
    for gene_name in gene_names:
        if gene_name is "input":
            continue
        #expression levels: replicates x #time points
        Y0 = cond1[gene_name]
        Y1 = cond2[gene_name]
        
        #create data structure for GPTwwoSample:
        #note; there is no need for the time points to be aligned for all replicates
        #creates score and time local predictions
        twosample_object.set_data(get_training_data_structure(SP.tile(T1,Y0.shape[0]).reshape(-1, 1),
                                                              SP.tile(T2,Y1.shape[0]).reshape(-1, 1),
                                                              Y0.reshape(-1, 1),
                                                              Y1.reshape(-1, 1)))
        twosample_object.predict_model_likelihoods()
        twosample_object.predict_mean_variance(Tpredict)
        plot_results(twosample_object,title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()))
        PL.xlim(T1.min(), T1.max())
        
        PL.savefig("GPTwoSample_%s.png"%(gene_name),format='png')
        ## wait for window close
        PL.show()

        pass
