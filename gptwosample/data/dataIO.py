"""
Data IO tool
============

For convienent usage this module provides IO operations for data

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
"""

import csv, scipy as SP

def get_data_from_csv(path_to_file,delimiter=','):
    '''
    Return data from csv file with delimiter delimiter in form of a dictionary.
    The file format has to fullfill following formation:
    
    ============ =============== ==== ===============
    *arbitrary*  x1              ...  xl
    ============ =============== ==== ===============
    Gene Name 1  y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene Name 1  y1 replicate k1 ...  yl replicate k1

    ...
    
    Gene Name n  y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene Name n  y1 replicate kn ...  yl replicate kn
    ============ =============== ==== ===============

    Returns: {"input":[x1,...,xl], "Gene Name 1":[[y1 replicate 1, ... yl replicate 1], ... ,[y1 replicate k, ..., yl replikate k]]}
    '''
    reader = csv.reader(open(path_to_file,"rb"),delimiter=delimiter)
    d = []
    for line in reader:
        d.append(line)
    data = {"input":SP.array(d[0][1:], dtype="float")}
    d=SP.array(d[1:])
    names = d[:,0]
    d=SP.array(d[:,1:],dtype='float')
    for gene in SP.unique(names):
        data[gene] = d[names==gene,:]
    return data
