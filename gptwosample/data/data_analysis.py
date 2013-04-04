'''
Created on Sep 16, 2011

@author: maxz
'''
from gptwosample.data.dataIO import get_data_from_csv
import scipy,pylab

def plot_roc_curve(path_to_result,path_to_ground_truth,
                   delimiter_1=',',delimiter_2=',',
                   xlabel="False positive rate",
                   ylabel="True positive rate",
                   upper=False,
                   **kwargs):
    """
    Plot roc curve for given results and ground truth. 
    
    Returns [pylab_plot,area_under_the_ROC]

    **Parameters:**

    path_to_result_file : String
        has to be in training_data_structure (see :py:class:`gptwosample.data.data_base` for details).

    ground_truth : String
        file has to have following structure: 
            first column contains gene_names, 
            second column contains 1 for positive, 0 for negative truth.
            
    delimiter_i : char
        delimiter for result file and ground truth file, respectively.
        
    xlabel, ylabel: string
        The x/ylabel for the plot. See matplotlib for details.
    
    upper : boolean
        True, if all names shall be upper converted. 
    
    kwargs:
        matplotlib kwargs, for adjusting the plot.
    """
    result_data = get_data_from_csv(path_to_result,delimiter=delimiter_1)
    result_data_header = result_data.pop('input')
    ground_truth_data = get_data_from_csv(path_to_ground_truth,delimiter=delimiter_2)
    ground_truth_data_header = ground_truth_data.pop('input')
    
    #### Get scipy arrays for comparison: ####
    result_names = []
    result_results = []
    for gene_name,result in result_data.iteritems():
        if(upper):
            gene_name = gene_name.upper()
        result_names.append(gene_name)
        result_results.append(scipy.ndarray.flatten(result)[0])
    result_names_sorting = scipy.argsort(result_names)
    result_names=scipy.array(result_names)[result_names_sorting]
    result_results=scipy.array(result_results,dtype='float')[result_names_sorting]
    ground_truth_names = []
    ground_truth_results = []
    for gene_name,truth in ground_truth_data.iteritems():
        if(upper):
            gene_name = gene_name.upper()
        ground_truth_names.append(gene_name)
        ground_truth_results.append(scipy.ndarray.flatten(truth)[0])
    ground_truth_sorting = scipy.argsort(ground_truth_names)
    ground_truth_names=scipy.array(ground_truth_names)[ground_truth_sorting]
    ground_truth_results=scipy.array(ground_truth_results,dtype='int')[ground_truth_sorting]
    #### end ####

    # pairwise match names, to get comparison set    
    match_filter = scipy.where(scipy.atleast_2d(result_names)==scipy.atleast_2d(ground_truth_names).T)

    # filter matching pairs for comparison
    labels = ground_truth_results[match_filter[0]]
    predictions = result_results[match_filter[1]]
    
    # get ROC and AUROC 
    try:
        from scikits.learn.metrics.metrics import roc_curve, auc    
        fpr, tpr, _ = roc_curve(labels, predictions)
        auroc_c = auc(fpr, tpr)
    except:
        tpr, fpr = roc(labels, predictions)
        auroc_c = auroc(tp=tpr, fp=fpr)
    
    # plot the curve into existing pylab environment
    if 'label' in kwargs.keys():
        kwargs['label'] = "{0}: AUC={1:.3g}".format(kwargs['label'], auroc_c)
    plot = pylab.plot(fpr,tpr,**kwargs)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    try:
        fig = pylab.gcf()
        fig.tight_layout()
    except:
        pass

    return [plot,auroc_c]
    

def roc(labels, predictions):
    """roc - calculate receiver operator curve
    labels: true labels (>0 : True, else False)
    predictions: the ranking generated from whatever predictor is used"""
    #1. convert to arrays
    labels = scipy.array(labels).reshape([-1])
    predictions = scipy.array(predictions).reshape([-1])

    #threshold
    t = labels>0
    
    #sort predictions in desceninding order
    #get order implied by predictor (descending)
    Ix = scipy.argsort(predictions)[::-1]

    #reorder truth
    t = t[Ix]

    #compute true positiive and false positive rates
    tp = scipy.double(scipy.cumsum(t))/t.sum()
    fp = scipy.double(scipy.cumsum(~t))/(~t).sum()

    #add end points
    tp = scipy.concatenate(([0],tp,[1]))
    fp = scipy.concatenate(([0],fp,[1]))

    return [tp,fp]

def auroc(labels=None,predictions=None,tp=None,fp=None):
    """auroc - calculate area under the curve from a given fp/rp plot"""

    if labels is not None:
        [tp,fp] = roc(labels,predictions)
    n = tp.size
    auc = 0.5*((fp[1:n]-fp[0:n-1]) * (tp[1:n]+tp[0:n-1])).sum()
    return auc
    pass

if __name__ == '__main__':
    [plt,aoc] = plot_roc_curve("../examples/warwick/result.csv", "../examples/ground_truth_random_genes.csv", label="warwick")
    pylab.legend(loc=4)
    print "AUROC: %f" % aoc