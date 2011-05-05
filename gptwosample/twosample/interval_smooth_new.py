'''
Created on Apr 21, 2011

@author: maxz
'''
from pygp.covar.combinators import SumCF
from pygp.covar.noise import NoiseCFISO
from pygp.covar.se import SqexpCFARD
from scipy.stats.distributions import norm
import scipy as SP
from pygp.gp.gpcEP import GPCEP
import logging

class GPTwoSampleInterval(object):
    '''
    Sample each x point by gibbs sampling, smoothing indicators by gpcEP with ProbitLikelihood.
    '''
    def __init__(self, twosample_object, indicator_prior = .3, outlier_probability=.1):
        '''
        Predict probability of an input point of twosample_object is more likely
        described by shared or individual model, respectivel.
        
        NOTE: Shapes of inputs of models must agree, otherwise we cannot gibbs sample over them!
    
        **Parameters:**
        
        twosample_object : :py:class:`gptwosample.twosample.GPTwoSample`
            GPTwoSample object to predict for
            
        outlier_probability : probability
            Probability for an input point to be an outlier (for robust
            output_prediction)
        '''
        # save static objects and values:
        self._twosample_object = twosample_object
        self.outlier_probability = outlier_probability
        
        # All x-values, which are given by twosample_object
        self._input_comm = twosample_object.get_data('common')[0]
        self._input_ind1 = twosample_object.get_data('individual',0)[0]
        self._input_ind2 = twosample_object.get_data('individual',1)[0]

        # Unique x-values:
        self._input = SP.unique(self._input_comm).reshape(-1,1)

        # All target-values, which are given by twosample_object
        self._target_comm = twosample_object.get_data('common')[1]
        self._target_ind1 = twosample_object.get_data('individual',0)[1]
        self._target_ind2 = twosample_object.get_data('individual',1)[1]

        # Count replicates:
        self._n_replicates_comm = (self._input_comm == self._input_comm[0]).sum()
        # Individual replicates are common replicates over 2
        #self._n_replicates_ind1 = (self._input_ind1 == self._input_ind1[0]).sum()
        #self._n_replicates_ind2 = (self._input_ind2 == self._input_ind2[0]).sum()

        # set everything up to begin with resampling:
        self.reset()

        secf = SqexpCFARD()
        noise = NoiseCFISO()
        covar = SumCF([secf, noise])
        self._indicator_regressor = GPCEP(covar_func=covar, 
                                          x=self._input.reshape(-1,1), 
                                          y=self._indicators)
        self._indicator_prior = indicator_prior
        
        
    def predict_interval_probabilities(self, hyperparams, number_of_gibbs_iterations=10):
        """
        Predict propability for each input point whether it 
        is more likely described by common, or individual model, respectively.
        
        **Parameters:**
        hyperparams : dict(['covar':covariance hyperparameters, 'lik': robust likelihood hyperparameters], ...)
        
        **Returns:**
        [double] : P(input_values correspond to common model)
        """
        self._twosample_object.predict_model_likelihoods()
        ratio = self._twosample_object.bayes_factor()
        
        probabilities = []
        for iteration in xrange(number_of_gibbs_iterations):
            for interval_index in xrange(self._input.shape[0]):
                self._indicators[interval_index] = self._resample_interval_index(interval_index, hyperparams)
            probabilities.append(self._indicators)
            logging.info("Gibbs Iteration: %i"%(iteration))
            logging.info("Current Indicator: %s"% (self._indicators))
            
        probabilities = SP.array(probabilities)
        logging.info("End: Indicators %s"% (probabilities.mean(0)))
        
        return self._calculate_indicator_mean(probabilities, hyperparams)
        
    def reset(self):
        # indicators for already assigned input values:
        self._indicators = SP.random.rand(self._input.shape[0]) > .5
        
      
    ####### private ######
    
    def _robust_likelihood(self, x, mean, var):
        # non outlier:
        likelihood = (1 - self.outlier_probability) * norm.pdf(x, mean, var)
        # outlier:
        likelihood += (self.outlier_probability) * norm.pdf(x, mean, var)
        return likelihood   
    
    def _resample_interval_index(self, interval_index, hyperparams):
        """
        Resample all input values corresponding to ones in interval_indices, w.r.t. already sampled values.
        
        **Parameters:**
        
        interval_index : int
            Index of input value, which shall be resampled            
        """
        interval_indicator = SP.zeros_like(self._indicators)
        interval_indicator[interval_index] = 1
        interval_indicator = SP.array(interval_indicator,dtype='bool')
        
        # make sure to predict on all data given (including all replicates)
        ind_interval_indicator = self._indicators & ~interval_indicator
        comm_interval_indicator = ~self._indicators & ~interval_indicator

        # predict output at interval_indicator
        target_prediction = self._twosample_object.predict_mean_variance(\
                      self._input[interval_indicator],\
                      interval_indices={'individual': SP.tile(ind_interval_indicator,self._n_replicates_comm/2),\
                                        'common':   SP.tile(comm_interval_indicator, self._n_replicates_comm)})
        
        ind1 = [target_prediction['individual']['mean'][0], target_prediction['individual']['var'][0]]
        ind2 = [target_prediction['individual']['mean'][1], target_prediction['individual']['var'][1]]
        comm = [target_prediction['common']['mean'], target_prediction['common']['var']]
        
        # predict binary indicator
        binary_interval_indicator = SP.ones_like(self._indicators)
        binary_interval_indicator &= interval_indicator
        self._indicator_regressor.setData(self._input[binary_interval_indicator], self._indicators[binary_interval_indicator])
        indicator_prediction = self._indicator_regressor.predict(hyperparams, self._input[interval_indicator])[0]
        
        # Make sure to get all targets of all replicates (Assumption: whole input array is tiled for replicates):
        ind_data_interval_indicator = SP.tile(interval_indicator, self._n_replicates_comm/2)
        comm_data_interval_indicator = SP.tile(interval_indicator, self._n_replicates_comm)
        
        # calculate robust likelihood
        ind1_likelihood = self._robust_likelihood(self._target_ind1[ind_data_interval_indicator], ind1[0], ind1[1]) 
        ind2_likelihood = self._robust_likelihood(self._target_ind2[ind_data_interval_indicator], ind2[0], ind2[1]) 
        comm_likelihood = self._robust_likelihood(self._target_comm[comm_data_interval_indicator], comm[0], comm[1])
        
        ind_likelihood = SP.log(ind1_likelihood).sum(axis=0) + SP.log(ind2_likelihood).sum(axis=0)
        comm_likelihood = SP.log(comm_likelihood).sum(axis=0)
        
        # calculate posterior
        ind_posterior = indicator_prediction * SP.exp(ind_likelihood) * self._indicator_prior
        comm_posterior = (1 - indicator_prediction) * SP.exp(comm_likelihood) * (1 - self._indicator_prior)

#        posterior = SP.zeros([2, 1])
#        posterior[1, :] = indicator_prediction * SP.exp(ind_likelihood) * self._indicator_prior
#        posterior[0, :] = (1 - indicator_prediction) * SP.exp(comm_likelihood) * (1 - self._indicator_prior)
#        posterior /= posterior.sum(axis=0)
#        
        posterior = ind_posterior / (ind_posterior + comm_posterior)

        return SP.rand() <= posterior
        
    def _calculate_indicator_mean(self, probabilities, hyperparams):
        """
        Calculate bernoulli mean for all probabilities given
        """
        min = self._input.min() - 2
        max = self._input.max() + 2
        prediction_interval = SP.linspace(min, max, 100).reshape(-1,1)
        sum_over_means = []
        for interval_indicator in probabilities:
            self._indicator_regressor.setData(self._input, interval_indicator)
            sum_over_means.append(self._indicator_regressor.predict(hyperparams, prediction_interval)[0])
        return SP.mean(sum_over_means, 0), prediction_interval