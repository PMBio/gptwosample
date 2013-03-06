'''
Created on Mar 5, 2013

@author: Max
'''
import os
from gptwosample.data.data_analysis import plot_roc_curve
import pylab
import sys

root = sys.argv[1]
s = "plotting roc curves..."
print s,
for f in os.listdir(root):
    if os.path.splitext(os.path.basename(f))[0].endswith("bayes"):
        plot_roc_curve(os.path.join(root,f), "../../examples/ground_truth_random_genes.csv", label=os.path.basename(f).split("_")[0])
        
pylab.legend(loc=4)
pylab.savefig(os.path.join(root, "roc.pdf"))

try:
    sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '            \n')
except:
    sys.stdout.write(s + " done             \n")
sys.stdout.flush()
