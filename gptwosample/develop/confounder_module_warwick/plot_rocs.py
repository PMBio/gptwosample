'''
Created on Mar 5, 2013

@author: Max
'''
import os
from gptwosample.data.data_analysis import plot_roc_curve
import pylab
import sys
import itertools
import numpy

root = sys.argv[1]
s = "plotting roc curves..."

colors = itertools.cycle(numpy.array([[97,216,76],
[184,91,222],
[199,125,28],
[211,63,92],
[110,122,202],
[191,210,42],
[81,139,44],
[202,83,168]], dtype='float')/255.)

print s,
for parent, folders, files in os.walk(root):
    for f in files:
        if f == "bayes.csv":
            label = parent[len(root):].lstrip("/").replace("/","_")
            try:
                plot_roc_curve(os.path.join(parent,f), "../../examples/ground_truth_random_genes.csv", 
                               label=label, color=colors.next())
            except StopIteration:
                pass
pylab.legend(loc=4)
pylab.xlim(0,.2)
pylab.savefig(os.path.join(root, "roc.pdf"))

try:
    sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '            \n')
except:
    sys.stdout.write(s + " done             \n")
sys.stdout.flush()
