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
from gptwosample.plot.plot_basic import plot_results
import logging as LG
import scipy as SP
import os


if __name__ == '__main__':
    #cond1_file = './ToyCondition1.csv'; cond2_file = './ToyCondition2.csv'
    cond1_file = './gsample1.csv'; cond2_file = './gsample2.csv'
    
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
    gene_names = SP.array(gene_names)
    
    twosample_object = toy_data_generator.get_twosample()
    plots = "plots_{}".format(os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(plots):
        os.makedirs(plots)
    #loop through genes
    ind = SP.where((gene_names=="gene 2") + (gene_names=="gene 14") + (gene_names=="gene 41"))[0]
    
    for gene_name in gene_names[ind]:#SP.random.permutation(gene_names)[:20]:
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
        
        import pylab as PL
        PL.clf()
        plot_results(twosample_object,
                     title=r'%s: $\mathcal B = \ln(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()), xlabel="Time", ylabel="Expression Level")
        PL.xlim(T1.min(), T1.max())

        try:
            PL.tight_layout()
        except:
            pass
        print gene_name
        PL.savefig(os.path.join(plots,"GPTwoSample_%s.pdf"%(gene_name)))
        ## wait for window close
        #PL.show()

        pass
