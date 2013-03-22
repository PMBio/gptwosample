'''
Created on Mar 18, 2013

@author: Max
'''
from gptwosample.data.dataIO import get_data_from_csv
import numpy
import csv

def read_and_handle_gt(cond1_file, cond2_file, gt_file_name, D='all'):
    # 1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")

    # get replicate stuff organized
    gene_names_all = numpy.intersect1d(cond1.keys(), cond2.keys(), True).tolist()
    gene_names = list()

    n_replicates_1 = cond1[gene_names_all[0]].shape[0]
    n_replicates_2 = cond2[gene_names_all[0]].shape[0]

    T = numpy.array([numpy.vstack([T1] * n_replicates_1), numpy.vstack([T2] * n_replicates_2)])

    # data merging and stuff
    gt_names = []
    gt_file = open(gt_file_name, 'r')
    for [name, _] in csv.reader(gt_file):
        gt_names.append(name)
    gt_file.close()

    if D<0 or D=='all':
        D = len(gene_names_all)

    Y1 = numpy.zeros((n_replicates_1, T1.shape[0], D))
    Y2 = numpy.zeros((n_replicates_2, T2.shape[0], D))

    for i, name in enumerate(numpy.random.permutation(gene_names_all)[:D]):
        Y1[:,:,i] = cond1[name.upper()]
        Y2[:,:,i] = cond2[name.upper()]
        gene_names.append(name.upper())
    
    Ygt = numpy.zeros((2, n_replicates_1, T1.shape[0], len(gt_names)))
    
    for i, name in enumerate(gt_names):
        Ygt[0,:,:,i] = cond1[name.upper()]
        Ygt[1,:,:,i] = cond2[name.upper()]

#    for i, name in enumerate(gt_names):
#        gene_names_all.remove(name.upper())
#        gene_names.append(name.upper())
#        Y1[:,:,i] = cond1.pop(name.upper())
#        Y2[:,:,i] = cond2.pop(name.upper())
#
#    # get random entries not from ground truth, to fill until D:
#    gt_len = len(gt_names)
#    for i, name in enumerate(numpy.random.permutation(gene_names_all)[:D - gt_len]):
#        Y1[:,:,i + gt_len] = cond1.pop(name.upper())
#        Y2[:,:,i + gt_len] = cond2.pop(name.upper())
#        gene_names.append(name.upper())

    Y = numpy.vstack((Y1[None], Y2[None]))
    
    return T, Y, gene_names, Ygt, gt_names 


