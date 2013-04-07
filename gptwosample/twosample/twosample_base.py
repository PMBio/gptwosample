'''
Classes to apply GPTwoSample to data
====================================

All classes handling TwoSampleBase tasks should extend this class.

Created on Mar 18, 2011

@author: Max Zwiessele, Oliver Stegle
'''
from gptwosample.data.data_base import input_id, output_id, individual_id, \
    common_id, has_model_structure, get_model_structure, DataStructureError
import scipy
from pygp.gp.gp_base import GP
from pygp.gp.composite import GroupGP
from pygp.optimize.optimize_base import opt_hyper
import numpy
from pygp.plot.gpr_plot import plot_sausage, plot_training_data
from copy import deepcopy
from matplotlib import cm
from pygp.likelihood.likelihood_base import GaussLikISO

class TwoSampleBase(object):
    """
    TwoSampleBase object with the given covariance function covar.
    """
    def __init__(self, learn_hyperparameters=True,
                 priors=None,
                 initial_hyperparameters=None, **kwargs):
        """
        Perform TwoSampleBase with the given covariance function covar.

        **Parameters**:

            covar : :py:class:`pygp.covar.CovarianceFunction`
                The covariance function this TwoSampleBase class works with.

            learn_hyperparameters : bool
                Specifies whether or not to optimize the hyperparameters for the given data

            priors : {'covar': priors for covar, ...}
                Default: None; The prior beliefs you provide for the hyperparamaters of the covariance function.


        """

        self._learn_hyperparameters = learn_hyperparameters
        if has_model_structure(priors):
            self._priors = priors
        else:
            self._priors = get_model_structure(priors, priors)

        self._models = dict()

        if initial_hyperparameters is None and priors is not None:
            self._initial_hyperparameters = get_model_structure({}, {});
            for name, prior in self._priors.iteritems():
                if prior.has_key('covar'):
                    # logarithmize the right hyperparameters for the covariance
                    logtheta = scipy.array([p[1][0] * p[1][1] for p in prior['covar']], dtype='float')
                    # out since version 1.0.0
                    # logtheta[covar.get_Iexp(logtheta)] = SP.log(logtheta[covar.get_Iexp(logtheta)])
                    self._initial_hyperparameters[name]['covar'] = logtheta
        elif has_model_structure(initial_hyperparameters):
            self._initial_hyperparameters = initial_hyperparameters
        elif initial_hyperparameters is not None:
            self._initial_hyperparameters = get_model_structure(initial_hyperparameters, initial_hyperparameters)
        else:
            self._initial_hyperparameters = get_model_structure({})

        for name, hyper in self._initial_hyperparameters.iteritems():
            hyper['lik'] = numpy.log([0.1])
        self._invalidate_cache()

    def set_data_by_xy_data(self, x1, x2, y1, y2):
        #not_missing = (numpy.isfinite(x1) * numpy.isfinite(x2) * numpy.isfinite(y1) * numpy.isfinite(y2)).flatten()
        #x1, x2 = x1[not_missing], x2[not_missing]
        #y1, y2 = y1[not_missing], y2[not_missing]

        X = numpy.array([x1, x2]); Y = numpy.array([y1, y2])
        # set individual model's data
        self._models[individual_id].setData(X, Y)
        # set common model's data
        self._models[common_id].setData(scipy.concatenate(X), scipy.concatenate(Y))

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
            self.set_data_by_xy_data(training_data[input_id]['group_1'],
                                     training_data[input_id]['group_2'],
                                     training_data[output_id]['group_1'],
                                     training_data[output_id]['group_2'])
        except KeyError:
            # print """Please validate training data given. \n
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

        interval_indices: :py:class:`gptwosample.data.data_base.get_model_structure()`
            interval indices, which assign data to individual or common model,
            respectively.

        args : [..]
            see :py:class:`pygp.gpr.gp_base.GP`

        kwargs : {..}
            see :py:class:`pygp.gpr.gp_base.GP`

        """
        if(training_data is not None):
            self.set_data(training_data)

        for name, model in self._models.iteritems():
            model.set_active_set_indices(interval_indices[name])
            try:
                if(self._learn_hyperparameters):
                    opt_hyperparameters = opt_hyper(model,
                                                    self._initial_hyperparameters[name],
                                                    priors=self._priors[name],
                                                    *args, **kwargs)[0]
                    self._learned_hyperparameters[name] = opt_hyperparameters
                else:
                    self._learned_hyperparameters[name] = self._initial_hyperparameters[name]
            except ValueError as r:
                print "caught error:", r.message, "\r",
                self._learned_hyperparameters[name] = self._initial_hyperparameters[name]
            self._model_likelihoods[name] = model.LML(self._learned_hyperparameters[name],
                                                              priors=self._priors)

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
        if interpolation_interval.ndim < 2:
            interpolation_interval = interpolation_interval[:, None]
        if(hyperparams is None):
            hyperparams = self._learned_hyperparameters
        self._predicted_mean_variance = get_model_structure()
        if(not has_model_structure(interpolation_interval)):
            interpolation_interval = get_model_structure(interpolation_interval, interpolation_interval)
        for name, model in self._models.iteritems():
            model.set_active_set_indices(interval_indices[name])
            prediction = model.predict(hyperparams[name], interpolation_interval[name], var=True, *args, **kwargs)
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
        if model_likelihoods is numpy.NaN:
            return numpy.NaN
        if(model_likelihoods is None):
            model_likelihoods = self._model_likelihoods
        return  model_likelihoods[common_id] - model_likelihoods[individual_id]

    def get_covars(self):
        models = self._models
        return {individual_id: models[individual_id].covar, common_id: models[common_id].covar}

    def get_model_likelihoods(self):
        """
        Returns all calculated likelihoods in model structure. If not calculated returns None in model structure.
        """
        return self._model_likelihoods

    def get_learned_hyperparameters(self):
        """
        Returns learned hyperparameters in model structure, if already learned.
        """
        return self._learned_hyperparameters

    def get_predicted_mean_variance(self):
        """
        Get the predicted mean and variance as::

            {'individual':{'mean':[pointwise mean], 'var':[pointwise variance]},
                 'common':{'mean':[pointwise mean], 'var':[pointwise variance]}}

        If not yet predicted it will return 'individual' and 'common' empty.
        """
        return self._predicted_mean_variance

    def get_data(self, model=common_id, index=None):  # , interval_indices=get_model_structure()):
        """
        get inputs of model `model` with group index `index`.
        If index is None, the whole model group will be returned.
        """
        if(index is None):
            return self._models[model].getData()  # [:, interval_indices[model]].squeeze()
        else:
            return self._models[model].getData()[index]  # [:, interval_indices[model]].squeeze()

    def plot(self,
             xlabel="input", ylabel="ouput", title=None,
             interval_indices=None, alpha=None, legend=True,
             replicate_indices=None, shift=None, *args, **kwargs):
        """
        Plot the results given by last prediction.

        Two Instance Plots of comparing two groups to each other:

        **Parameters:**

        twosample_object : :py:class:`gptwosample.twosample`
            GPTwoSample object, on which already 'predict' was called.

        **Differential Groups:**

        .. image:: ../images/plotGPTwoSampleDifferential.pdf
            :height: 8cm

        **Non-Differential Groups:**

        .. image:: ../images/plotGPTwoSampleSame.pdf
            :height: 8cm

        Returns:
            Proper rectangles for use in pylab.legend().
        """
        if self._predicted_mean_variance is None:
            print "Not yet predicted, or not predictable"
            return
        if interval_indices is None:
            interval_indices = get_model_structure(
            common=numpy.array(numpy.zeros_like(self.get_data(common_id)[0]), dtype='bool'),
            individual=numpy.array(numpy.ones_like(self.get_data(individual_id, 0)[0]), dtype='bool'))
        import pylab
        if title is None:
            title = r'Prediction result: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (self.bayes_factor())

    #        plparams = {'axes.labelsize': 20,
    #            'text.fontsize': 20,
    #            'legend.fontsize': 18,
    #            'title.fontsize': 22,
    #            'xtick.labelsize': 20,
    #            'ytick.labelsize': 20,
    #            'usetex': True }

        legend_plots = []
        legend_names = []

        calc_replicate_indices = replicate_indices is None

        alpha_groups = alpha
        if alpha is not None:
            alpha_groups = 1 - alpha

        for name, value in self._predicted_mean_variance.iteritems():
            mean = value['mean']
            var = numpy.sqrt(value['var'])
            if len(mean.shape) > 1:
                number_of_groups = mean.shape[0]
                first = True
                for i in range(number_of_groups):
                    col_num = (i / (2. * number_of_groups))
                    col = cm.jet(col_num)  # (i/number_of_groups,i/number_of_groups,.8) @UndefinedVariable
                    x, y = self.get_data(name, i)
                    x, y = x.squeeze(), y.squeeze()
                    replicate_length = len(numpy.unique(x))
                    number_of_replicates = len(x) / replicate_length
                    if calc_replicate_indices:
                        # Assume replicates are appended one after another
                        replicate_indices = []
                        curr = x[0] - 1
                        rep = 0
                        replicate_length = 0
                        for xi in x:
                            if xi < curr:
                                replicate_indices.append(numpy.repeat(rep, replicate_length))
                                rep += 1
                                replicate_length = 0
                            replicate_length += 1
                            curr = xi
                        replicate_indices.append(numpy.repeat(rep, replicate_length))
                        replicate_indices = numpy.concatenate(replicate_indices)
                    shifti = deepcopy(shift)
                    if shifti is not None:
                        shifti = shift[i * number_of_replicates:(i + 1) * number_of_replicates]
                        # import pdb;pdb.set_trace()
                        plot_sausage(self._interpolation_interval_cache[name] - numpy.mean(shifti), mean[i], var[i], format_fill={'alpha':0.2, 'facecolor':col}, format_line={'alpha':1, 'color':col, 'lw':3, 'ls':'--'}, alpha=alpha_groups)[0]
                    else:
                        plot_sausage(self._interpolation_interval_cache[name],
                                          mean[i], var[i],
                                          format_fill={'alpha':0.2, 'facecolor':col},
                                          format_line={'alpha':1, 'color':col, 'lw':3, 'ls':'--'}, alpha=alpha_groups)[0]
                    plot_training_data(
                            numpy.array(x), numpy.array(y),
                            format_data={'alpha':.8,
                                         'marker':'.',
                                         'linestyle':'--',
                                         'lw':1,
                                         'markersize':6,
                                         'color':col},
                            replicate_indices=replicate_indices,
                            shift=shifti, *args, **kwargs)
                    if(first):
                        legend_plots.append(pylab.Rectangle((0, 0), 1, 1, alpha=.2, fill=True, facecolor=col))
                        legend_names.append("%s %i" % (name, i + 1))
                        # first=False
            else:
                col = cm.jet(1.)  # @UndefinedVariable
                # data = self.get_data(name, interval_indices=interval_indices)
                # PLOT.plot_training_data(
                #        x, y,
                #        format_data={'alpha':.2,
    #                                 'marker':'.',
    #                                 'linestyle':'',
    #                                 'markersize':10,
    #                                 'color':col})
                legend_names.append("%s" % (name))
                plot_sausage(
                    self._interpolation_interval_cache[name], mean, var,
                    format_fill={'alpha':0.2, 'facecolor':col},
                    format_line={'alpha':1, 'color':col, 'lw':3, 'ls':'--'}, alpha=alpha)[0]
                legend_plots.append(pylab.Rectangle((0, 0), 1, 1, alpha=.2, fc=col, fill=True))
        if legend:
            pylab.legend(legend_plots, legend_names,
                      bbox_to_anchor=(0., 0., 1., 0.), loc=3,
                      ncol=2,
                      mode="expand",
                      borderaxespad=0.,
                      fancybox=False, frameon=False)

        pylab.xlabel(xlabel)
        pylab.ylabel(ylabel)

        pylab.subplots_adjust(top=.88)
        pylab.title(title, fontsize=22)

        return legend_plots



######### PRIVATE ##############
#    def _init_twosample_model(self, covar):
#        """
#        The initialization of the twosample model with
#        the given covariance function covar
#        """
#        print("please implement twosample model")
#        pass

    def _invalidate_cache(self):
        # self._learned_hyperparameters = dict([name,None for name in self._models.keys()])
        self._model_likelihoods = get_model_structure()
        self._learned_hyperparameters = get_model_structure()
        self._interpolation_interval_cache = None
        self._predicted_mean_variance = None

class TwoSampleShare(TwoSampleBase):
    """
    This class provides comparison of two Timeline Groups to each other.

    see :py:class:`gptwosample.twosample.twosample_base.TwoSampleBase` for detailed description of provided methods.
    """
    def __init__(self, covar, *args, **kwargs):
        """
        see :py:class:`gptwosample.twosample.twosample_base.TwoSampleBase`
        """
        if not kwargs.has_key('initial_hyperparameters'):
            kwargs['initial_hyperparameters'] = \
                get_model_structure(individual={'covar':numpy.zeros(covar.get_number_of_parameters())},
                                    common={'covar':numpy.zeros(covar.get_number_of_parameters())})
        super(TwoSampleShare, self).__init__(*args, **kwargs)
        gpr1 = GP(deepcopy(covar), likelihood=GaussLikISO())
        gpr2 = GP(deepcopy(covar), likelihood=GaussLikISO())
        # individual = GroupGP([gpr1,gpr2])
        # common = GP(covar)
        # self.covar = covar
        # set models for this TwoSampleBase Test
        self._models = {individual_id:GroupGP([gpr1, gpr2]),
                        common_id:GP(deepcopy(covar), likelihood=GaussLikISO())}

class TwoSampleSeparate(TwoSampleBase):
    """
    This class provides comparison of two Timeline Groups to one another, inlcuding timeshifts in replicates, respectively.

    see :py:class:`gptwosample.twosample.twosample_base.TwoSampleBase` for detailed description of provided methods.

    Note that this model will need one covariance function for each model, respectively!
    """
    def __init__(self, covar_individual_1, covar_individual_2, covar_common, **kwargs):
        """
        see :py:class:`gptwosample.twosample.twosample_base.TwoSampleBase`
        """
        if not kwargs.has_key('initial_hyperparameters'):
            kwargs['initial_hyperparameters'] = \
                get_model_structure(individual={'covar':numpy.zeros(covar_individual_1.get_number_of_parameters())},
                                    common={'covar':numpy.zeros(covar_common.get_number_of_parameters())})
        super(TwoSampleSeparate, self).__init__(**kwargs)
        gpr1 = GP(deepcopy(covar_individual_1), likelihood=GaussLikISO())
        gpr2 = GP(deepcopy(covar_individual_2), likelihood=GaussLikISO())
        # self.covar_individual_1 = covar_individual_1
        # self.covar_individual_2 = covar_individual_2
        # self.covar_common = covar_common
        # set models for this TwoSampleBase Test


        self._models = {individual_id:GroupGP([gpr1, gpr2]), common_id:GP(deepcopy(covar_common), likelihood=GaussLikISO())}
