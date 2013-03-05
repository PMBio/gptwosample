'''
Created on Mar 5, 2013

@author: Max
'''
import os
from gptwosample.data.data_analysis import plot_roc_curve
import pylab
from gptwosample.develop.confounder_module_warwick.warwick import finished

root = 'warwick'
s = "plotting roc curves..."
print s
for f in os.listdir(root):
    if os.path.splitext(os.path.basename(f))[0].endswith("bayes"):
        plot_roc_curve(os.path.join(root,f), "../../examples/ground_truth_random_genes.csv", label=os.path.basename(f).split("_")[0])
        
pylab.legend(loc=4)
pylab.savefig("warwick/roc.pdf")
finished(s)