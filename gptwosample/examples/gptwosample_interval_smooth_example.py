'''
Small Example application of TwoSampleInterval
================================================

Please run script "generateToyExampleFiles.py" to generate Toy Data.
This Example shows how to apply GPTwoSample to toy data, generated above.

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
'''

from gptwosample.data import toy_data_generator
from gptwosample.data.data_base import get_training_data_structure
from gptwosample.plot.interval import plot_results_interval
from gptwosample.data.dataIO import get_data_from_csv
import logging as LG
import scipy as SP, pylab as PL
from gptwosample.twosample.interval_smooth import TwoSampleIntervalSmooth

def run_demo(cond1_file, cond2_file):
    #full debug info:
    LG.basicConfig(level=LG.INFO)
    #if verbose is on the models all create plots visualising the resulting inference resultSP.
#    verbose = True
#    #if we create the figures we can also save them to disk
#    figures_out = './out'
#
#    #0. create gptest object
#    #starting point for parameter optimisation
#    #amplitude variation, length scale, noise level
#    #note: time series are all rescaled from -5 to 5 internally.
#    logtheta0 = SP.log([0.5, 1, 0.4])

    #similarly: hyperparameters of the smooth process for time intervalSP.
    #this GP is not optimised by default and settings should be adapted;noise level is basically 0 (logistic GP).
    logthetaZ = SP.log([.5, 2, .001])
#    #if we now that there is a transtion from non differential to differential genes it is helpful to fix the first few observations as not differntially expressed (initially). This paremter allows to set say the first 2 observations to not differentially expressed
#    fix_Z = SP.array([0, 1])
#    #optimisation of hyperparmeters; should be on
#    opt = True
#    #number of optimisation iterations 
#    maxiter = 20
#    #prior belief in differential expression. it make sense to set that a number < 0.5 if we blief that less than half of the genes differentially expressed.
#    #only the interval test makes use of this setting/prior probability ; the base factor is the raw value without this prior counted in.
#    prior_Z = 0.7
#    #Ngibbs_iteratons: number of gibbs sweeps through all indicatorSP. form 30 on this should be (depending on the data) be converged).
    Ngibbs_iterations = 30

    #1. read csv file
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, SP.newaxis]
    T1 = cond1["input"]
    T2 = cond2["input"]
    
    gene_names = sorted(cond1.keys()) 
    assert gene_names == sorted(cond2.keys())

    lik = SP.log([2, 2])

    twosample_object = toy_data_generator.get_twosample()
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

        gptest = TwoSampleIntervalSmooth(twosample_object, outlier_probability=.1)
        Z = gptest.predict_interval_probabilities(Tpredict, hyperparams={'covar':logthetaZ, 'lik':lik},
                                                  number_of_gibbs_iterations=Ngibbs_iterations)
        plot_results_interval(gptest,title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()))
        PL.xlim(T1.min(), T1.max())
        
        PL.savefig("GPTwoSampleInterval_%s.png"%(gene_name),format='png')
        ## wait for window close
        PL.show()

        pass

if __name__ == '__main__':
    run_demo(cond1_file = './ToyCondition1.csv', cond2_file = './ToyCondition2.csv')