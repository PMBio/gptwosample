'''
Created on Mar 5, 2013

@author: Max
'''
import os
from gptwosample.data.data_analysis import plot_roc_curve


for f in os.listdir('warwick'):
    if f.is_file and f.basename.endswith("bayes"):
        plot_roc_curve(f, "../../examples/ground_truth_random_genes.csv", label=f.basename.split("_")[0])