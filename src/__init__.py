"""
Package for using GPTwoSample
=============================

This module allows the user to compare two timelines with respect to diffferential expression.

It compares two timeseries against each other, depicting whether these two
timeseries were more likely drawn from the same function, or from
different ones. This prediction is defined by which covariance function :py:class:`pygp.covar` you use.

"""

import sys
sys.path.append("./../../")
sys.path.append("./../")

from pygp import gpr as GPR, gpr_plot as PLOT

import scipy as SP
import copy

class GPTwoSample(object):
    """
    Perform GPTwoSample with the given covariance function covar.
    """
    def __init__(self, covar,
                 learn_hyperparameters=True,
                 priors = None,
                 initial_hyperparameters = None, **kwargs):
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

        self._init_twosample_model(covar)
        
        if priors is not None and initial_hyperparameters is None:
            self._initial_hyperparameters = dict([[name, prior.shape] for name,
                                                  prior in priors.iteritems()])

            if self._initial_hyperparameters.has_key('covar'):
                #logarithmize the right hyperparameters for the covariance
                logtheta = SP.array([p[1][0]*p[1][1] for p in priors['covar']], dtype='float')
                logtheta[covar.get_Iexp(logtheta)] = SP.log(logtheta[covar.get_Iexp(logtheta)])
                self._initial_hyperparameters['covar'] = logtheta
        elif initial_hyperparameters is not None:
            self._initial_hyperparameters = initial_hyperparameters
        else:
            self._initial_hyperparameters = {'covar':SP.zeros(covar.get_number_of_parameters())}

        self._invalidate_cache()
        

    def predict_model_likelihoods(self, training_data, *args, **kwargs):
        """
        Predict the probabilities of the models (individual and common) to describe the data
        
        **Parameters**:
        
            training_data : dict traning_data
                The training data to learn from. Input are time-values and
                output are expression-values of e.g. a timeseries.

                Training data training_data has following structure::

                    {'input' : {'group 1':[double] ... 'group n':[double]},
                    'output' : {'group 1':[double] ... 'group n':[double]}}

            args : [..]
                see :py:class:`pygp.gpr.GP`

            kwargs : {..}
                see :py:class:`pygp.gpr.GP`
        
        """
        X = training_data['input'].values()
        Y = training_data['output'].values()
        # set individual model's data
        self._models['individual'].setData(X,Y)
        # set common model's data
        self._models['common'].setData(SP.concatenate(X),SP.concatenate(Y))

        for name,model in self._models.iteritems():
            if(self._learn_hyperparameters):
                self._learned_hyperparameters[name] = GPR.optHyper(model,
                                                                   self._initial_hyperparameters,
                                                                   priors=self._priors,*args,**kwargs)[0]
            self._model_likelihoods[name] = model.lMl(self._learned_hyperparameters[name],
                                                       priors=self._priors,*args,**kwargs)

        return self._model_likelihoods

    def predict_mean_variance(self,interpolation_interval,hyperparams=None,*args,**kwargs):
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

        """
        if(hyperparams is None):
            hyperparams = self._learned_hyperparameters
            
        self._predicted_mean_variance = dict([[name,None] for name in self._models.keys()])
        for name,model in self._models.iteritems():
            prediction = model.predict(hyperparams[name], interpolation_interval,var=True,*args,**kwargs)
            self._predicted_mean_variance[name] = {'mean':prediction[0],'var':prediction[1]}

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
        return  model_likelihoods['common']  - model_likelihoods['individual']

    

    def get_model_likelihoods(self):
        return self._model_likelihoods
    
    def get_learned_hyperparameters(self):
        return self._learned_hyperparameters

    def plot_results(self, x_label="input", y_label="ouput", title=None):
        """
        Plot the results given by last prediction.

        Two Instance Plots of comparing two groups to each other:
        
        **Differential Groups:**
        
        .. image:: ../images/plotGPTwoSampleDifferential.png
            :height: 8cm
            
        **Non-Differential Groups:**
        
        .. image:: ../images/plotGPTwoSampleSame.png
            :height: 8cm
        
        """
        if self._predicted_mean_variance is None:
            print "Not yet predicted"
            return
        if title is None:
            title = r"Prediction result: $ \frac{P(individual)} {P(common)} = %.2f $" % (self.bayes_factor())

        import pylab as PL
        plparams = {'axes.labelsize': 20,
            'text.fontsize': 20,
            'legend.fontsize': 18,
            'title.fontsize': 22,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'text.usetex': True}
        PL.rcParams.update(plparams)

        legend_plots = []
        legend_names = []

        for name,value in self._predicted_mean_variance.iteritems():
            mean = value['mean']
            var = SP.sqrt(value['var'])
            if len(mean.shape)>1:
                number_of_groups = mean.shape[0]
                first = True
                for i in range(number_of_groups):
                    col = (.1,.1,.8)#(i/number_of_groups,i/number_of_groups,.8)
                    data = self._models[name].getData()                    
                    PLOT.plot_training_data(
                        data[i][0],data[i][1],
                        format_data={'alpha':.4,
                                     'marker':'.',
                                     'linestyle':'',
                                     'markersize':13,
                                     'color':col})
                    plots = PLOT.plot_sausage(self._interpolation_interval_cache, mean[i], var[i],
                        format_fill={'alpha':0.2,'facecolor':col},
                        format_line={'alpha':1,'color':col}
                                              )[0]
                    if(first):
                        legend_plots.append(plots[0])
                        legend_names.append("%ss" % (name))
                        first=False
            else:
                legend_names.append("%s" % (name))
                col = (.8,.1,.1)
                plots = PLOT.plot_sausage(self._interpolation_interval_cache, mean, var,
                                          format_fill={'alpha':0.2,'facecolor':col},
                                          format_line={'alpha':1,'color':col})[0]
                legend_plots.append(PLOT.CrossRect((0,0),1,1,alpha=.2,fc=col,fill=True))
                
        PL.legend(legend_plots,legend_names,
                  bbox_to_anchor=(0., 0., 1., 0.), loc=3,
                  ncol=2, mode="expand", borderaxespad=0.,
                  fancybox=False, frameon=False)

        PL.xlabel(x_label)
        PL.ylabel(y_label)

#        PL.subplot_ajust(
        
        PL.suptitle(title, fontsize=22)
        
    ####### private #########

    def _init_twosample_model(self, covar):
        """
        The initialization of the twosample model with
        the given covariance function covar
        """
        print("please implement the twosample model")
        pass

    def _invalidate_cache(self):
        # self._learned_hyperparameters = dict([name,None for name in self._models.keys()])
        self._model_likelihoods = dict([[name,None] for name in self._models.keys()])
        self._learned_hyperparameters = copy.deepcopy(self._model_likelihoods)
        self._interpolation_interval_cache = None
        self._predicted_mean_variance = None
        #self._training_data_cache = {'input': {'group_1':None,'group_2':None},
        #                             'output': {'group_1':None,'group_2':None}}

