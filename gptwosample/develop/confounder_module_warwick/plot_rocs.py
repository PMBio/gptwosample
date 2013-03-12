'''
Created on Mar 5, 2013

@author: Max
'''
import os
from gptwosample.data.data_analysis import plot_roc_curve
import pylab
import sys
import itertools

root = sys.argv[1]
s = "plotting roc curves..."

colors = itertools.cycle([[1,0.585,1.067],
[144,1.016,1.221],
[224,0.712,1.153],
[80,1.013,1.263],
[276,0.753,1.127],
[99,0.566,1.193],
[187,0.87,1.205],
[53,0.843,1.119],
[109,0.981,1.231],
[154,0.625,1.194]])

print s,
for f in os.listdir(root):
    if os.path.splitext(os.path.basename(f))[0].endswith("bayes"):
        label = os.path.basename(f)[::-1].split("_",1)[1][::-1]
        plot_roc_curve(os.path.join(root,f), "../../examples/ground_truth_random_genes.csv", label=label, color=colors.next())
        
pylab.legend(loc=4)
pylab.xlim(0,.2)
pylab.savefig(os.path.join(root, "roc.pdf"))

try:
    sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '            \n')
except:
    sys.stdout.write(s + " done             \n")
sys.stdout.flush()
