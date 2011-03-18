'''
Created on Mar 18, 2011

@author: maxz
'''



from pygp.optimize import opt_hyper
import copy
import gptwosample.plot.gptwosample as plot
import scipy as SP
from gptwosample.data.data_base import DataStructureError
    
class GPTwoSample(object):
    """
    Perform GPTwoSample with the given covariance function covar.
    """
    def __init__(self, covar,
                 learn_hyperparameters=True,
                 priors=None,
                 initial_hyperparameters=None, **kwargs):
        """
        Perform GPTwoSample with the given covariance function covar.

        **Parameters**:

            covar : :py:class:`pygp.covar.CovarianceFunction`
                The covariance function this GPTwoSample class works with.

            learn_hyperparameters : bool
                Specifies whether or not to optimize the hyperparameters for the given data

            priors : {'covar': priors for covar, ...}
                Default: None; The prior beliefs you provide for the hyperparamaters of the covariance function.

                
        """

        self._learn_hyperparameters = learn_hyperparameters
        self._covar = covar
        self._priors = priors
        
        self._models = dict()
        self._init_twosample_model(covar)
        
        if priors is not None and initial_hyperparameters is None:
            self._initial_hyperparameters = dict([[name, prior.shape] for name,
                                                  prior in priors.iteritems()])

            if self._initial_hyperparameters.has_key('covar'):
                #logarithmize the right hyperparameters for the covariance
                logtheta = SP.array([p[1][0] * p[1][1] for p in priors['covar']], dtype='float')
                # out since version 1.0.0
                # logtheta[covar.get_Iexp(logtheta)] = SP.log(logtheta[covar.get_Iexp(logtheta)])
                self._initial_hyperparameters['covar'] = logtheta
        elif initial_hyperparameters is not None:
            self._initial_hyperparameters = initial_hyperparameters
        else:
            self._initial_hyperparameters = {'covar':SP.zeros(covar.get_number_of_parameters())}

        self._invalidate_cache()
        
    def set_data(self, training_data):
        """
        Set the data of prediction.
        
        **Parameters:**
        
        training_data : dict traning_data
            The training data to learn from. Input are time-values and
            output are expression-values of e.g. a timeseries.

            Training data training_data has following structure::

                {'input' : {'group 1':[double] ... 'group n':[double]},
                 'output' : {'group 1':[double] ... 'group n':[double]}}

        """
        try:                
            X = training_data['input'].values()
            Y = training_data['output'].values() # set individual model's data
            
            self._models['individual'].setData(X, Y) # set common model's data
            self._models['common'].setData(SP.concatenate(X), SP.concatenate(Y))
        except KeyError:
            print """Please validate training data given. \n
                training_data must have following structure: \n
                {'input' : {'group 1':[double] ... 'group n':[double]},
                'output' : {'group 1':[double] ... 'group n':[double]}}"""
            raise DataStructureError("Training Data must confirm printed structure")

    def predict_model_likelihoods(self, training_data=None, *args, **kwargs):
        """
        Predict the probabilities of the models (individual and common) to describe the data
        It will optimize hyperparameters respectively, if option was chosen at creating this object.
        
        **Parameters**:
        
        training_data : dict traning_data
            The training data to learn from. Input are time-values and
            output are expression-values of e.g. a timeseries. 
            If not given, training data must be given previously by 
            :py:class:`gptwosample.twosample.basic.set_data`.

            Training data training_data has following structure::

                {'input' : {'group 1':[double] ... 'group n':[double]},
                'output' : {'group 1':[double] ... 'group n':[double]}}

        args : [..]
            see :py:class:`pygp.gpr.GP`

        kwargs : {..}
            see :py:class:`pygp.gpr.GP`
        
        """
        if(training_data is not None):
            self.set_data(training_data)

        for name, model in self._models.iteritems():
            if(self._learn_hyperparameters):
                self._learned_hyperparameters[name] = opt_hyper(model,
                                                                self._initial_hyperparameters,
                                                                priors=self._priors, *args, **kwargs)[0]
            self._model_likelihoods[name] = model.LML(self._learned_hyperparameters[name],
                                                      priors=self._priors, *args, **kwargs)

        return self._model_likelihoods

    def predict_mean_variance(self, interpolation_interval,
                              hyperparams=None,
                              interval_indices=None,
                              *args, **kwargs):
        """
        Predicts the mean and variance of both models.
        Returns::

            {'individual':{'mean':[pointwise mean], 'var':[pointwise variance]},
                 'common':{'mean':[pointwise mean], 'var':[pointwise variance]}}

        **Parameters:**

        interpolation_interval : [double]
            The interval of inputs, which shall be predicted

        hyperparams : {'covar':logtheta, ...}
            Default: learned hyperparameters. Hyperparams for the covariance function's prediction.
    
        interval_indices : {'common':[boolean],'individual':[boolean]}
            Indices in which to predict, for each group, respectively.
        """
        if(hyperparams is None):
            hyperparams = self._learned_hyperparameters            
        if(interval_indices is None):
            interval_indices = dict([[name, None] for name in self._models.keys()])
        
        self._predicted_mean_variance = dict([[name, None] for name in self._models.keys()])
        for name, model in self._models.iteritems():
            prediction = model.predict(hyperparams[name], interpolation_interval, var=True, interval_indices=interval_indices[name], *args, **kwargs)
            self._predicted_mean_variance[name] = {'mean':prediction[0], 'var':prediction[1]}

        self._interpolation_interval_cache = interpolation_interval

        return self._predicted_mean_variance

    def bayes_factor(self, model_likelihoods=None):
        """
        Return the Bayes Factor for the given log marginal likelihoods model_likelihoods

        **Parameters:**

        model_likelihoods : {'individual': *the individual likelihoods*, 'common': *the common likelihoods*}
            The likelihoods calculated by
            predict_model_likelihoods(training_data)
            for given training data training_data.
            
        """
        if(model_likelihoods is None):
            model_likelihoods = self._model_likelihoods
        return  model_likelihoods['common'] - model_likelihoods['individual']

    def get_model_likelihoods(self):
        return self._model_likelihood
    def get_learned_hyperparameters(self):
        return self._learned_hyperparameter
    def get_predicted_mean_variance(self):
        """
        Get the predicted mean and variance as::

            {'individual':{'mean':[pointwise mean], 'var':[pointwise variance]},
                 'common':{'mean':[pointwise mean], 'var':[pointwise variance]}}

        If not yet predicted it will return 'individual' and 'common' empty.
        """
        return self._predicted_mean_variance

    def plot_results(self, *args, **kwargs):
        """
        See :py:class:`gptwosample.plot.plot_gptwosample`
        """
        plot.plot_results(self, *args, **kwargs)
        
######### PRIVATE ##############
    def _init_twosample_model(self, covar):
        """
        The initialization of the twosample model with
        the given covariance function covar
        """
        print("please implement twosample model")
        pass

    def _invalidate_cache(self):
        # self._learned_hyperparameters = dict([name,None for name in self._models.keys()])
        self._model_likelihoods = dict([[name, None] for name in self._models.keys()])
        self._learned_hyperparameters = copy.deepcopy(self._model_likelihoods)
        self._interpolation_interval_cache = None
        self._predicted_mean_variance = None
        #self._training_data_cache = {'input': {'group_1':None,'group_2':None},
        #                             'output': {'group_1':None,'group_2':None}}

    def get_data(self, model='common', index=None):
        """
        get inputs of model `model` with index `index`.
        If index is None, the whole model group will be returned.
        """
        if(index is None):
            return self._models[model].getData()
        else:
            return self._models[model].getData()[index]
