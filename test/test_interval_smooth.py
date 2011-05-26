"""DEMO script running various versions of the two sample test"""
import sys
sys.path.append('./..')
#append directories to path
#debugger
#import pdb
#IO libraries for reading csv files
import pyio.csvex as pyio, os
#scientific python
import scipy as SP
#pylab - matlab style plotting
import pylab as PL

#log level control
import logging as LG
#two_sample: smooth model implements standard test and time dependent model. 
from gptwosample.twosample.interval_smooth import GPTwoSampleInterval
from gptwosample.data import toy_data_generator
from gptwosample.data.data_base import get_training_data_structure
from gptwosample.plot.plot_basic import plot_results
from gptwosample.plot.interval import plot_results_interval

#expr. levels IN

expr_file = './demo.csv'


if __name__ == '__main__':
    #full debug info:
    LG.basicConfig(level=LG.INFO)

    intervals = True
    #if verbose is on the models all create plots visualising the resulting inference resultSP.
    verbose = True
    #if we create the figures we can also save them to disk
    if verbose:
        save_fig = True
    else:
        save_fig = False

    figures_out = './out'

    #0. create gptest object
    #logtheta0: 
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
    R = pyio.readCSV('./demo.csv', ',', typeC='str')
    #1. header versus data
    col_header = R[:, 0]
    #time
    Tc = SP.array(R[0, 1::], dtype='float')
    #expression levels
    Yc = SP.array(R[1::, 1::], dtype='float')
    #how many unique labels ? 
    gene_names = SP.unique(col_header[1::])
    Ngenes = gene_names.shape[0]
    Nrepl = Yc.shape[0] / (Ngenes * 2)
    #structure for time
    #replicates x #time points
    T = SP.zeros([Nrepl, Tc.shape[0]])
    T[:, :] = Tc
    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    Tpredict = SP.linspace(T.min(), T.max(), 100)[:, SP.newaxis]

    if verbose:
        #if verbose activate interactive plotting
        PL.ion()
        
    lik = SP.log([2, 2])
    #loop through genes
    for g in xrange(Ngenes):
        i0 = g * Nrepl * 2
        i1 = i0 + Nrepl
        #expression levels: replicates x #time points
        Y0 = Yc[i0:i0 + Nrepl, :]
        Y1 = Yc[i1:i1 + Nrepl, :]
        
#        mean = SP.mean((SP.mean(Y0),SP.mean(Y1)))
#        
#        Y0 -= mean
#        Y1 -= mean
        #create data structure for GPTwwoSample:
        #note; there is no need for the time points to be aligned for all replicates
        if intervals:
            #creates score and time local predictions
            twosample_object = toy_data_generator.get_twosample_object()
            twosample_object.set_data(get_training_data_structure(T.reshape(-1, 1),
                                                                  T.reshape(-1, 1),
                                                                  Y0.reshape(-1, 1),
                                                                  Y1.reshape(-1, 1)))

            gptest = GPTwoSampleInterval(twosample_object, outlier_probability=.1)
            
            Z = gptest.predict_interval_probabilities(hyperparams={'covar':logthetaZ, 'lik':lik},
                                                      number_of_gibbs_iterations=Ngibbs_iterations)
            #fix_Z=fix_Z)
#            gptest = GPTwoSampleInterval(twosample_object)
#            
#            Z, interval = gptest.predict_interval_probabilities({'covar':logthetaZ, 'lik':lik}, 30)
#            PL.plot(interval, Z)
#            PL.figure()
#            
#            twosample_object.predict_model_likelihoods()
#            twosample_object.predict_mean_variance(Tpredict)
#            plot_results(twosample_object, 
#                         xlabel="Time/h", ylabel='Expression level', 
#                         title='%s : %f.5'%(gene_names[g],twosample_object.bayes_factor()),
#                         legend=False)
##            
        plot_results_interval(gptest)
        #wait for enter
        PL.show()
        pass
