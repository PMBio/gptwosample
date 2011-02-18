'''
Created on Feb 10, 2011

@author: maxz
'''

import scipy as SP
import pygp.plot.gpr_plot as PLOT

def plot_results(twosample_object, ax=None, x_label="input", y_label="ouput", title=None):
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
        
        if title is None:
            title = r'Prediction result: $\log \frac{p(\text{"individual"})} {p(\text{"common"})} $%.2f $' % (twosample_object.bayes_factor())

        import pylab as PL
#        plparams = {'axes.labelsize': 20,
#            'text.fontsize': 20,
#            'legend.fontsize': 18,
#            'title.fontsize': 22,
#            'xtick.labelsize': 20,
#            'ytick.labelsize': 20}
#        from matplotlib import rc
#        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#        ## for Palatino and other serif fonts use:
#        #rc('font',**{'family':'serif','serif':['Palatino']})
#        rc('text', usetex=True)

        legend_plots = []
        legend_names = []

        for name,value in twosample_object._predicted_mean_variance.iteritems():
            mean = value['mean']
            var = SP.sqrt(value['var'])
            if len(mean.shape)>1:
                number_of_groups = mean.shape[0]
                first = True
                for i in range(number_of_groups):
                    col = (.1,.1,.8)#(i/number_of_groups,i/number_of_groups,.8)
                    data = twosample_object._models[name].getData()                    
                    PLOT.plot_training_data(
                        data[i][0],data[i][1],
                        format_data={'alpha':.4,
                                     'marker':'.',
                                     'linestyle':'',
                                     'markersize':13,
                                     'color':col})
                    plots = PLOT.plot_sausage(twosample_object._interpolation_interval_cache, mean[i], var[i],
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
                plots = PLOT.plot_sausage(twosample_object._interpolation_interval_cache, mean, var,
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
        
        PL.title(title, fontsize=22)