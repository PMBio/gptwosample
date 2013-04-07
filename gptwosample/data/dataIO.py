"""
Data IO tool
============

For convienent usage this module provides IO operations for data

Created on Jun 9, 2011

@author: Max Zwiessele, Oliver Stegle
"""

import csv, scipy as SP, sys
import os
import numpy
from re import compile

def get_data_from_csv(path_to_file, delimiter=',', count= -1, verbose=True, message="Reading File", fil=None):
    '''
    Return data from csv file with delimiter delimiter in form of a dictionary.
    Missing Values are all values x which cannot be converted float(x)

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
    def filter_(x):
        try:
            return float(str.strip(x))
        except:
            return numpy.nan
    
    end = float(count_lines(path_to_file))
    
    if fil is not None:
        fil = map(compile,fil)
        filtered = []    
        def matchesin(name, fil):
            for f in fil:
                if f.match(name):
                    return True
            return False
    
    with open(path_to_file, "r") as out_file:
        reader = csv.reader(out_file, delimiter=str(delimiter))
        out = sys.stdout
        current_line = 0
        if verbose:
            message = "{1:s} {0:s}".format(os.path.basename(path_to_file), message)
            mess = lambda x: "{1:s}: {0:.2%}".format(x, message)
            out.write(mess(0) + " \r")
        data = {"input":map(filter_,reader.next()[1:])}
        for line in reader:
            if line:
                gene_name = line[0]
                if fil is not None:
                    if gene_name in filtered:
                        continue
                    if not matchesin(gene_name, fil):
                        filtered.append(gene_name)
                        continue            
                l_filtered = [filter_(x) for x in line[1:]]
                if(data.has_key(gene_name)):
                    data[gene_name].append(l_filtered)
                else:
                    data[gene_name] = [l_filtered]
            current_line += 1
            if verbose:
                out.flush()
                out.write(mess(current_line / end) + "\r")
    #        progress += 1
    #        step_ahead = int((1.*progress/end)*60.)
    #        if(step_ahead > step):
    #            out.write("#"*(step_ahead-step))
    #            step = step_ahead
    out.flush()
    if verbose:
        try:
            out.write(message + " " + '\033[92m' + u"\u2713" + '\033[0m' + '         \r')
        except:
            out.write(message + " done         \r")
    for name, expr in data.iteritems():
        try:
            data[name] = SP.array(expr, dtype='float')
        except:
            if not (name == 'input'):
                print "Caught Failure on dataset with name %s: " % (name)
                print sys.exc_info()[0]
#            else:
#                print "input is header and cannot be converted, this is NO error \r",
    return data

def write_data_to_csv(data, path_to_file, header='GPTwoSample', delimiter=','):
    """
    Write given data in training_data_structure (see :py:class:`gptwosample.data.data_base` for details)
    into file for path_to_file.

    **Parameters:**

    data : dict
        data to write in training_data_structure

    path_to_file : String
        The path to the file to write to

    header : String
        Name of the table

    delimiter : character
        delimiter for the csv file
    """
    data = data.copy()
    if not path_to_file.endswith(".csv"):
        path_to_file.append(".csv")
    out_file = open(path_to_file, "w")
    writer = csv.writer(out_file)
    line = [header]
    line.extend(data.pop("input"))
    writer.writerow(line)
    for name, line in data.iteritems():
        if line.shape[0] > 1:
            l = [[name]] * line.shape[0]
            l = SP.concatenate((l, line), axis=1)
            writer.writerows(l)
        else:
            l = [name]
            l = SP.concatenate((l, line), axis=1)
            writer.writerow(l)
    out_file.flush()

def count_lines(filename):
    f = open(filename)
    lines = 1
    buf_size = 1024 * 1024
    read_f = f.read
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    return lines
