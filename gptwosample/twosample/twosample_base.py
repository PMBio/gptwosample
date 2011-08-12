'''
GPTwoSample Base Class
======================

All classes ahndling GPTwoSample tasks should extend this class.

Created on Mar 18, 2011

@author: Max Zwiessele, Oliver Stegle
'''
from gptwosample.data import DataStructureError, get_model_structure
from gptwosample.data.data_base import input_id, output_id, individual_id, \
    common_id, has_model_structure
from pygp.optimize import opt_hyper
import scipy as SP
    
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
        if has_model_structure(priors):
            self._priors = priors
        else:
            self._priors = get_model_structure(priors, priors)
            
        
        self._models = dict()
        self._init_twosample_model(covar, **kwargs)
        
        if initial_hyperparameters is None:
            self._initial_hyperparameters = get_model_structure({}, {});
            for name, prior in self._priors.iteritems():
                if prior.has_key('covar'):
                    #logarithmize the right hyperparameters for the covariance
                    logtheta = SP.array([p[1][0] * p[1][1] for p in prior['covar']], dtype='float')
                    # out since version 1.0.0
                    # logtheta[covar.get_Iexp(logtheta)] = SP.log(logtheta[covar.get_Iexp(logtheta)])
                    self._initial_hyperparameters[name]['covar'] = logtheta
        elif has_model_structure(initial_hyperparameters):
            self._initial_hyperparameters = initial_hyperparameters
        elif initial_hyperparameters is not None:
            self._initial_hyperparameters = get_model_structure(initial_hyperparameters, initial_hyperparameters)
        else:
            self._initial_hyperparameters = get_model_structure({'covar':SP.zeros(covar.get_number_of_parameters())}, {'covar':SP.zeros(covar.get_number_of_parameters())})

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
            #X = training_data['input'].values()#
            X = [training_data[input_id]['group_1'], training_data[input_id]['group_2']]
            #Y = training_data['output'].values()
            Y = [training_data[output_id]['group_1'], training_data[output_id]['group_2']]
            # set individual model's data
            self._models[individual_id].setData(X, Y)
            # set common model's data
            self._models[common_id].setData(SP.concatenate(X), SP.concatenate(Y))
        except KeyError:
            #print """Please validate training data given. \n
            #    training_data must have following structure: \n
            #    {'input' : {'group 1':[double] ... 'group n':[double]},
            #    'output' : {'group 1':[double] ... 'group n':[double]}}"""
            raise DataStructureError("Please use gptwosample.data.data_base.get_training_data_structure for data passing!")

    def predict_model_likelihoods(self, training_data=None, interval_indices=get_model_structure(), *args, **kwargs):
        """
        Predict the probabilities of the models (individual and common) to describe the data.
        It will optimize hyperparameters respectively.
        
        **Parameters**:
        
        training_data : dict traning_data
            The training data to learn from. Input are time-values and
            output are expression-values of e.g. a timeseries. 
            If not given, training data must be given previously by 
            :py:class:`gptwosample.twosample.basic.set_data`.

        interval_indices: :py:class:`gptwosample.data.get_model_structure()`
            interval indices, which assign data to individual or common model,
            respectively.

        args : [..]
            see :py:class:`pygp.gpr.GP`

        kwargs : {..}
            see :py:class:`pygp.gpr.GP`
        
        """
        if(training_data is not None):
            self.set_data(training_data)

        for name, model in self._models.iteritems():
            model.set_active_set_indices(interval_indices[name])
            if(self._learn_hyperparameters):
                self._learned_hyperparameters[name] = opt_hyper(model,
                                                                self._initial_hyperparameters[name],
                                                                priors=self._priors[name],
                                                                *args, **kwargs)[0]
            self._model_likelihoods[name] = model.LML(self._learned_hyperparameters[name],
                                                      priors=self._priors, *args, **kwargs)

        return self._model_likelihoods

    def predict_mean_variance(self, interpolation_interval,
                              hyperparams=None,
                              interval_indices=get_model_structure(),
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
        self._predicted_mean_variance = get_model_structure()
        for name, model in self._models.iteritems():
            model.set_active_set_indices(interval_indices[name])
            prediction = model.predict(hyperparams[name], interpolation_interval, var=True, *args, **kwargs)
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
        return  model_likelihoods[common_id] - model_likelihoods[individual_id]

    def get_model_likelihoods(self):
        return self._model_likelihood
    def get_learned_hyperparameters(self):
        return self._learned_hyperparameters
    def get_predicted_mean_variance(self):
        """
        Get the predicted mean and variance as::

            {'individual':{'mean':[pointwise mean], 'var':[pointwise variance]},
                 'common':{'mean':[pointwise mean], 'var':[pointwise variance]}}

        If not yet predicted it will return 'individual' and 'common' empty.
        """
        return self._predicted_mean_variance
        
    def get_data(self, model=common_id, index=None, interval_indices=get_model_structure()):
        """
        get inputs of model `model` with group index `index`.
        If index is None, the whole model group will be returned.
        """
        if(index is None):
            return self._models[model].getData()[:, interval_indices[model]].squeeze()
        else:
            return self._models[model].getData()[index][:, interval_indices[model]].squeeze()
        
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
        self._model_likelihoods = get_model_structure()
        self._learned_hyperparameters = get_model_structure()
        self._interpolation_interval_cache = None
        self._predicted_mean_variance = None
        #self._training_data_cache = {'input': {'group_1':None,'group_2':None},
        #                             'output': {'group_1':None,'group_2':None}}
