'''
Created on Mar 12, 2013

@author: Max
'''
import os
import sys
import csv
import itertools
from gptwosample.twosample.twosample_base import TwoSampleBase
import pickle

root = sys.argv[1]
if not os.path.exists(root):
    sys.exit(1)

s = "writing back bayes factors... "
twosample = TwoSampleBase()

def write_bayes_factors(writer, gt_names, likelihoods):
    for name, l in itertools.izip(gt_names, likelihoods):
        writer.writerow([name, twosample.bayes_factor(l)])

for parent, folders, files in os.walk(root):
    bayes_file_name = os.path.join(root, 'bayes.csv')
    if "jobs" == os.path.basename(parent):
        with open(os.path.join(os.path.dirname(parent), "bayes.csv"), 'w') as bayesfile:
            writer = csv.writer(bayesfile)
            N = os.path.splitext(files[0])[0].split("_")[-1]
            for Ni in range(N):
                with open(os.path.join(parent,'likelihoods_job_{}_{}.pickle'.format(Ni, N)),'r') as liks, \
                     open(os.path.join(parent,'gt_names_job_{}_{}.pickle'.format(Ni, N)),'r') as gts:
                    write_bayes_factors(writer, pickle.load(gts), pickle.load(liks))
