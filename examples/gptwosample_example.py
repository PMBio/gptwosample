'''
Small Example application of GPTwoSample
========================================

Please run script "generateToyExampleFiles.py" to generate Toy Data.
This Example shows how to apply GPTwoSample to toy data, generated above.
'''

from gptwosample.data import toy_data_generator
from gptwosample.data.data_base import get_training_data_structure
from gptwosample.plot.interval import plot_results_interval
from gptwosample.twosample.interval_smooth import GPTwoSampleInterval
import csv
import logging as LG
import scipy as SP, pylab as PL

cond1_file = './ToyCondition1.csv'
cond2_file = './ToyCondition2.csv'

if __name__ == '__main__':
    #full debug info:
    LG.basicConfig(level=LG.INFO)
    #if verbose is on the models all create plots visualising the resulting inference resultSP.
    verbose = True
    #if we create the figures we can also save them to disk
    figures_out = './out'

    #0. create gptest object
    #starting point for parameter optimisation
    #amplitude variation, length scale, noise level
    #note: time series are all rescaled from -5 to 5 internally.
    logtheta0 = SP.log([0.5, 1, 0.4])
    #similarly: hyperparameters of the smooth process for time intervalSP.
    #this GP is not optimised by default and settings should be adapted;noise level is basically 0 (logistic GP).
    logthetaZ = SP.log([.5, 2, .001])
    #if we now that there is a transtion from non differential to differential genes it is helpful to fix the first few observations as not differntially expressed (initially). This paremter allows to set say the first 2 observations to not differentially expressed
    fix_Z = SP.array([0, 1])
    #optimisation of hyperparmeters; should be on
    opt = True
    #number of optimisation iterations 
    maxiter = 20
    #prior belief in differential expression. it make sense to set that a number < 0.5 if we blief that less than half of the genes differentially expressed.
    #only the interval test makes use of this setting/prior probability ; the base factor is the raw value without this prior counted in.
    prior_Z = 0.7
    #Ngibbs_iteratons: number of gibbs sweeps through all indicatorSP. form 30 on this should be (depending on the data) be converged).
    Ngibbs_iterations = 30

    #1. read csv file
    cond1 = csv.reader(open(cond1_file,'r'),delimiter=",")
    cond2 = csv.reader(open(cond2_file,'r'),delimiter=",")

    R1 = []
    for line in cond1:
        R1.append(line)
    R2 = []
    for line in cond2:
        R2.append(line)
    R1 = SP.array(R1)
    R2 = SP.array(R2)
    #time
    Tc1 = SP.array(R1[0, 1::], dtype='float')
    Tc2 = SP.array(R2[0, 1::], dtype='float')
    #expression levels
    Yc1 = SP.array(R1[1::, 1::], dtype='float')
    Yc2 = SP.array(R2[1::, 1::], dtype='float')
    #how many unique labels ? 
    gene_names1 = SP.unique(R1[:,0][1::])
    gene_names2 = SP.unique(R2[:,0][1::])
    Ngenes1 = gene_names1.shape[0]
    Ngenes2 = gene_names1.shape[0]
    Nrepl1 = Yc1.shape[0] / (Ngenes1)
    Nrepl2 = Yc1.shape[0] / (Ngenes2)
    
    assert Ngenes1 == Ngenes2, "For gene to gene comparison same number of genes needed"
    #structure for time
    #replicates x #time points
    T1 = SP.zeros([Nrepl1, Tc1.shape[0]])
    T2 = SP.zeros([Nrepl2, Tc2.shape[0]])
    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(T1.min(), T2.max(), 100)[:, SP.newaxis]

    lik = SP.log([2, 2])
    #loop through genes
    for g in xrange(Ngenes1):
        i1 = g * Nrepl1
        i2 = g * Nrepl2
        #expression levels: replicates x #time points
        Y0 = Yc1[i1:i1 + Nrepl1, :]
        Y1 = Yc2[i2:i2 + Nrepl2, :]
        
        #create data structure for GPTwwoSample:
        #note; there is no need for the time points to be aligned for all replicates
        #creates score and time local predictions
        import pdb;pdb.set_trace()
        twosample_object = toy_data_generator.get_twosample_object()
        twosample_object.set_data(get_training_data_structure(T1.reshape(-1, 1),
                                                              T2.reshape(-1, 1),
                                                              Y0.reshape(-1, 1),
                                                              Y1.reshape(-1, 1)))

        gptest = GPTwoSampleInterval(twosample_object, outlier_probability=.1)
        Z = gptest.predict_interval_probabilities(hyperparams={'covar':logthetaZ, 'lik':lik},
                                                  number_of_gibbs_iterations=Ngibbs_iterations)
        plot_results_interval(gptest)
        ## wait for window close
        PL.show()
        pass
