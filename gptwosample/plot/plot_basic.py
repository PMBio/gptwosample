'''
Plot GPTwoSample predictions
============================

Module for easy plotting of GPTwoSample results.

:py:class:`gptwosample.plot.plot_basic.plot_results` plots
training data, as well as sausage_plots for a GPTwoSample
experiment. You can give interval indices for plotting, if u chose


Created on Feb 10, 2011

@author: maxz
'''

import pygp.plot.gpr_plot as PLOT
import pylab as PL
import scipy as SP
from gptwosample.data.data_base import get_model_structure
from matplotlib import cm
from numpy.ma.core import ceil

def plot_results(twosample_object,
                 ax=None, xlabel="input", ylabel="ouput", title=None,
                 interval_indices=None, alpha=None, legend=True):
    """
    Plot the results given by last prediction.

    Two Instance Plots of comparing two groups to each other:

    **Parameters:**
    
    twosample_object : :py:class:`gptwosample.twosample`
        GPTwoSample object, on which already 'predict' was called.
    
    **Differential Groups:**
    
    .. image:: ../images/plotGPTwoSampleDifferential.png
        :height: 8cm
        
    **Non-Differential Groups:**
    
    .. image:: ../images/plotGPTwoSampleSame.png
        :height: 8cm
    
    """
    if twosample_object._predicted_mean_variance is None:
        print "Not yet predicted"
        return
    if interval_indices is None:
        interval_indices = get_model_structure(
        common=SP.array(SP.zeros_like(twosample_object.get_data('common')[0]), dtype='bool'),
        individual=SP.array(SP.ones_like(twosample_object.get_data('individual', 0)[0]), dtype='bool'))
    
    if title is None:
        title = r'Prediction result: $\log \frac{p(individual)}{p(common)} = %.2f $' % (twosample_object.bayes_factor())

#        plparams = {'axes.labelsize': 20,
#            'text.fontsize': 20,
#            'legend.fontsize': 18,
#            'title.fontsize': 22,
#            'xtick.labelsize': 20,
#            'ytick.labelsize': 20,
#            'usetex': True }

    legend_plots = []
    legend_names = []

    alpha_groups = alpha
    if alpha is not None:
        alpha_groups = 1-alpha

    for name, value in twosample_object._predicted_mean_variance.iteritems():
        mean = value['mean']
        var = SP.sqrt(value['var'])
        if len(mean.shape) > 1:
            number_of_groups = mean.shape[0]
            first = True
            for i in range(number_of_groups):
                col_num=(i / (2.* number_of_groups))
                col = cm.jet(col_num)#(i/number_of_groups,i/number_of_groups,.8)
                data = twosample_object.get_data(name, i)
                PLOT.plot_training_data(
                    data[0], data[1],
                    format_data={'alpha':.5,
                                 'marker':'.',
                                 'linestyle':'',
                                 'markersize':10,
                                 'color':col})
                plots = PLOT.plot_sausage(
                    twosample_object._interpolation_interval_cache,
                    mean[i], var[i],
                    format_fill={'alpha':0.2, 'facecolor':col},
                    format_line={'alpha':1, 'color':col, 'lw':3, 'ls':'--'}, alpha=alpha_groups)[0]
                if(first):
                    legend_plots.append(PL.Rectangle((0, 0), 1, 1, alpha=.2, fill=True, facecolor=col))
                    legend_names.append("%s %i" % (name, i + 1))
                    #first=False
        else:
            col = cm.jet(1.)
            #data = twosample_object.get_data(name, interval_indices=interval_indices)   
            #PLOT.plot_training_data(
            #        data[0], data[1],
            #        format_data={'alpha':.2,
#                                 'marker':'.',
#                                 'linestyle':'',
#                                 'markersize':10,
#                                 'color':col})
            legend_names.append("%s" % (name))
            plots = PLOT.plot_sausage(
                twosample_object._interpolation_interval_cache, mean, var,
                format_fill={'alpha':0.2, 'facecolor':col},
                format_line={'alpha':1, 'color':col, 'lw':3, 'ls':'--'}, alpha=alpha)[0]
            legend_plots.append(PL.Rectangle((0, 0), 1, 1, alpha=.2, fc=col, fill=True))
    if legend:
        PL.legend(legend_plots, legend_names,
                  bbox_to_anchor=(0., 0., 1., 0.), loc=3,
                  ncol=2,
                  mode="expand",
                  borderaxespad=0.,
                  fancybox=False, frameon=True)
    
    PL.xlabel(xlabel)
    PL.ylabel(ylabel)

    PL.subplots_adjust(top=.88)
    PL.suptitle(title, fontsize=22)

    
    
