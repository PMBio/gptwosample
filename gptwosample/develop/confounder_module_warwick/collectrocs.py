'''
Created on Mar 20, 2013

@author: Max
'''
import sys
import os
import csv
import itertools
import numpy
from copy import deepcopy

root = sys.argv[1]
with open(os.path.join(root, 'aurocsall.txt'), 'r') as aurocs:
    out = "\\toprule \n"
    Qs = ["Q"]
    Qmax = 0
    aucs = []
    alldict = {}
    allmodelnames = []
    for line in csv.reader(aurocs):
        Q = line[0].split('=')[1]
        if int(Q) > Qmax:
            Qmax = int(Q)
        Qs.append(Q)
        line = map(lambda x: x.split('='), line)
        mod = map(lambda x: x[0].replace("_"," "),line[1:])
        if len(mod) > len(allmodelnames):
            allmodelnames = mod
        aucs = map(lambda x: x[1],line[1:])
        alldicttmp = {}         
        for model, auc in zip(mod, aucs):
            alldicttmp[model] = str(auc)
        alldict[int(Q)] = deepcopy(alldicttmp)
    out += "&".join(Qs)
    out += '\\\\\n\\midrule\n'
    
for mod in allmodelnames:
    aucs = []
    for q in range(Qmax):
        try:
            aucs.append(alldict[q+1][mod])
        except KeyError:
            aucs.append('nan')
    out += "{}&".format(mod.strip(" "))
    out += "&".join(aucs)
    out += "\\\\\n"

out += '\\bottomrule \n'

with open(os.path.join(root, 'aurocs.table'), 'w') as tab:
    tab.write(out)
    