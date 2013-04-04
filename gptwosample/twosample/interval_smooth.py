'''
TwoSampleInterval
===================

Resamples each input data point and elucidates the properties of both given samples in great detail.
For each input point, a decision for `diferential` or `non-differential` is predicted.

The Results of this class can be plottet by :py:class:`gptwosample.plot.interval`.

Created on Apr 21, 2011

@author: Max Zwiessele, Oliver Stegle
'''
from pygp.covar.combinators import SumCF
from pygp.covar.noise import NoiseCFISO
from pygp.covar.se import SqexpCFARD
import scipy as SP
from pygp.gp.gpcEP import GPCEP
from gptwosample.data.data_base import common_id, individual_id

class TwoSampleIntervalSmooth(object):
    '''
    Sample each x point by gibbs sampling, smoothing indicators by gpcEP with ProbitLikelihood.
    '''
    def __init__(self, twosample_object, indicator_prior = .3, outlier_probability=.1):
        '''
        Predict probability of an input point of twosample_object is more likely
        described by shared or individual model, respectivel. This class
        learns all hyperparameters needed for Gaussian Process Regression.
        
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
        self._input_comm = twosample_object.get_data(common_id)[0]
        self._input_ind2 = twosample_object.get_data(individual_id,0)[0]
        self._input_ind1 = twosample_object.get_data(individual_id,1)[0]

        # Unique x-values:
        self._input = SP.unique(self._input_comm).reshape(-1,1)

        # All target-values, which are given by twosample_object
        self._target_comm = twosample_object.get_data(common_id)[1]
        self._target_ind2 = twosample_object.get_data(individual_id,0)[1]
        self._target_ind1 = twosample_object.get_data(individual_id,1)[1]

        # Count replicates:
        # if there are not replicate indices given:
        self._n_replicates_ind = (self._input_ind1 == self._input_ind1[0]).sum()
        self._n_replicates_comm = self._n_replicates_ind*2
        # Individual replicates are common replicates over 2
        #self._n_replicates_ind1 = (self._input_ind1 == self._input_ind1[0]).sum()
        #self._n_replicates_ind2 = (self._input_ind2 == self._input_ind2[0]).sum()

        # set everything up to begin with resampling:
        self.reset()

        secf = SqexpCFARD(1)
        noise = NoiseCFISO()
        covar = SumCF([secf, noise])
        self._indicator_regressor = GPCEP(covar_func=covar, 
                                          x=self._input.reshape(-1,1), 
                                          y=self._indicators)
        self._indicator_prior = indicator_prior
        
        
    def predict_interval_probabilities(self, prediction_interval, hyperparams, number_of_gibbs_iterations=10, messages=True):
        """
        Predict propability for each input point whether it 
        is more likely described by common, or individual model, respectively.
        
        **Parameters:**
        hyperparams : dict(['covar':covariance hyperparameters for interval regressor, \
                            'lik': robust likelihood hyperparameters], ...)
        
        **Returns:**
        [double] : P(input_values correspond to common model, respectively?),\
        [double] : prediction interval
        """
        self._twosample_object.predict_model_likelihoods(messages=False)
        self._twosample_object.bayes_factor()
        
        probabilities = SP.zeros((number_of_gibbs_iterations, self._input.shape[0]),dtype='bool')
        for iteration in xrange(number_of_gibbs_iterations):
            for interval_index in xrange(self._input.shape[0]):
                self._indicators[interval_index] = self._resample_interval_index(interval_index, hyperparams)
            probabilities[iteration,:] = self._indicators
#            logging.info("Gibbs Iteration: %i"%(iteration))
#            logging.info("Current Indicator: %s"% (self._indicators))
        
        probabilities = SP.array(probabilities, dtype='bool')
        
        #get rid of training runs (first half)
        probabilities = probabilities[SP.ceil(number_of_gibbs_iterations/2):]
#        logging.info("End: Indicators %s"% (probabilities.mean(0)))
        
        self._predicted_model_distribution = self._calculate_indicator_mean(probabilities, hyperparams, prediction_interval)
        self._predicted_indicators = probabilities.mean(0) > .5
        return self._predicted_model_distribution
        
    def reset(self):
        # indicators for already assigned input values:
        self._indicators = SP.random.rand(self._input.shape[0]) > .5
        self._predicted_indicators = None
        
    
    def get_predicted_indicators(self):
        """
        Return the predicted interval_indicators.
        NOTE: If no predction was done, this method will return None!
        """
        return self._predicted_indicators

    def get_predicted_model_distribution(self):
        """
        Return model distribution P(common model explains data best), 
        and prediction interval (on which probabilities was calculated, 
        see :py:class:`pygp.gp.GPEP`).
        NOTE: If no predction was done, this method will return None!
        """
        return self._predicted_model_distribution
    ####### private ######
    
    def _robust_likelihood(self, x, mean, var):
        # non outlier:
        likelihood = (1 - self.outlier_probability) * self._normpdf(x, mean, var)
        # outlier:
        likelihood += (self.outlier_probability) * self._normpdf(x, mean, 1E8)
        return likelihood   
    
    def _normpdf(self, x, mu, v):
        """Normal PDF, x mean mu, variance v"""
        return SP.exp(-0.5 * (x - mu) ** 2 / v) * SP.sqrt(2 * SP.pi / v)

    def _resample_interval_index(self, interval_index, hyperparams):
        """
        Resample all input values corresponding to ones in interval_indices, 
        w.r.t. already sampled values.
        
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
        target_prediction = \
            self._twosample_object.predict_mean_variance(\
                 self._input[interval_indicator],\
                 interval_indices={individual_id: ~SP.tile(ind_interval_indicator,self._n_replicates_ind),\
                                   common_id: ~SP.tile(comm_interval_indicator, self._n_replicates_comm)},
                 )
        
        ind1 = [target_prediction[individual_id]['mean'][0], target_prediction[individual_id]['var'][0]]
        ind2 = [target_prediction[individual_id]['mean'][1], target_prediction[individual_id]['var'][1]]
        comm = [target_prediction[common_id]['mean'], target_prediction[common_id]['var']]
        
        # predict binary indicator
        binary_interval_indicator = SP.ones(self._indicators.shape[0], dtype='bool') & ~interval_indicator
        self._indicator_regressor.setData(self._input[binary_interval_indicator], self._indicators[binary_interval_indicator])
        indicator_prediction = self._indicator_regressor.predict(hyperparams, self._input[interval_indicator])[0]
        
        # Make sure to get all targets of all replicates (Assumption: whole input array is tiled for replicates):
        ind_data_interval_indicator = SP.tile(interval_indicator, self._n_replicates_comm/2)
        comm_data_interval_indicator = SP.tile(interval_indicator, self._n_replicates_comm)
        
        # calculate robust likelihood
        ind1_likelihood = self._robust_likelihood(self._target_ind1[ind_data_interval_indicator].reshape(-1,1),
                                                  ind1[0], ind1[1]) 
        ind2_likelihood = self._robust_likelihood(self._target_ind2[ind_data_interval_indicator].reshape(-1,1),
                                                  ind2[0], ind2[1]) 
        comm_likelihood = self._robust_likelihood(self._target_comm[comm_data_interval_indicator].reshape(-1,1),
                                                  comm[0], comm[1])
        ind_likelihood = SP.log(ind1_likelihood).sum(axis=0) + SP.log(ind2_likelihood).sum(axis=0)
        comm_likelihood = SP.log(comm_likelihood).sum(axis=0)
        
        # calculate posterior
        ind_posterior = indicator_prediction * SP.exp(ind_likelihood) * (1 - self._indicator_prior)
        comm_posterior = (1 - indicator_prediction) * SP.exp(comm_likelihood) * self._indicator_prior

        posterior = ind_posterior / (ind_posterior + comm_posterior)
        
        prediction = SP.rand() <= posterior
        #if ind_interval_indicator.sum() == 1:
        #    prediction = True
        #if comm_interval_indicator.sum() == 1:
        #    prediction = False
        return prediction
        
    def _calculate_indicator_mean(self, probabilities, hyperparams, prediction_interval):
        """
        Calculate bernoulli mean for all probabilities given
        """
        sum_over_means = SP.zeros(prediction_interval.shape[0])
        for interval_indicator in probabilities:
            self._indicator_regressor.setData(self._input, interval_indicator)
            sum_over_means += self._indicator_regressor.predict(hyperparams, prediction_interval)[0]
        return sum_over_means/probabilities.shape[0], prediction_interval
