'''
Created on Nov 8, 2011

@author: maxz
'''
from gptwosample.data.data_analysis import plot_roc_curve
import os
import pylab
import scipy
import gptwosample_confounders
from gptwosample.util.confounder_constants import *
import sys

def plot_roc_curves_for_sample_model(sampled_from, prediction_model, ground_truth, root_dir='./', upper=False):
    pylab.title("Sampled from %s, %s" % (sampled_from, prediction_model))
    plots = []
    legends = []
    standard_not_done = True
                
    for f in os.listdir(root_dir):
        import pdb; pdb.set_trace()
        if f.startswith('sampledfrom') and os.path.isdir(os.path.join(root_dir,f)):
            [sampledfrom, learnedby, conf] = map(lambda x: x.split('-')[1], f.split("_"))
            if sampled_from == sampledfrom:
                for fi in os.listdir(os.path.join(root_dir, f)):
                    result_file = os.path.join(root_dir, f, fi)
                    if os.path.isfile(result_file) and not fi.startswith("."):
                        [predictionmodel, prop] = fi.split('-')
                        prop = prop.split("_")[-1].split(".")[0]
                        if(prediction_model == predictionmodel):
                            [plot, auc] = plot_roc_curve(result_file, ground_truth, upper=upper)
                            legends.append("%s %s: %.3f" % (learnedby, prop, auc))
                            plots.append(plot)
                        if(prop == 'ideal' and prediction_model == confounder_constants.covariance_model_id):
                            [plot, auc] = plot_roc_curve(result_file, ground_truth, upper=upper)
                            legends.append("%s ideal: %.3f" % (learnedby, auc))
                            plots.append(plot)
                        if(prop == 'standard' and standard_not_done):
                            [plot, auc] = plot_roc_curve(result_file, ground_truth, upper=upper)
                            legends.append("standard: %.3f" % (auc))
                            standard_not_done = False
                            plots.append(plot)
                            
    
    sort = scipy.argsort(legends)
    legends = scipy.array(legends)
    plots = scipy.array(plots)
    pylab.legend(plots[sort], legends[sort], loc=4)


if __name__ == '__main__':
    ground_truth = '../examples/ground_truth_random_genes.csv'
    if len(sys.argv) == 2:
        root_dir=sys.argv[1]
    else:
        root_dir='./'
    for sample_model in [product_linear_covariance_model_id]:#,linear_covariance_model_id]:
        for prediction_model in [covariance_model_id]:#, gptwosample_confounders.reconstruct_model_id]:
            pylab.figure()
            plot_roc_curves_for_sample_model(sample_model,
                                             prediction_model,
                                             ground_truth,
                                             root_dir=root_dir,
                                             upper=True)
            pylab.savefig('%s/%s %s.png' % (root_dir, sample_model, prediction_model))
            pylab.close("all")
