'''
Created on Mar 20, 2013

@author: Max
'''
import sys
import os
import csv
import itertools
import numpy

root = sys.argv[1]
with open(os.path.join(root, 'aurocsall.txt'), 'r') as aurocs:
    out = "\\toprule \n"
    Qs = ["Q"]
    aucs = []
    models = []
    for line in csv.reader(aurocs):
        Qs.append(line[0].split('=')[1])
        line = map(lambda x: x.split('='), line)
        mod = map(lambda x: x[0].replace("_"," "),line[1:])
        if len(mod) > len(models):
            models = mod
        aucs.append(map(lambda x: x[1],line[1:]))
    out += "&".join(Qs)
    out+= '\\\\\n\\midrule\n'
    for model, auc in itertools.izip_longest(models,itertools.izip_longest(*aucs, fillvalue=str(numpy.nan))):
        out += "{}&".format(model)
        out += "&".join(auc)
        out += "\\\\\n"
    out += '\\bottomrule \n'

with open(os.path.join(root, 'aurocs.table'), 'w') as tab:
    tab.write(out)
    