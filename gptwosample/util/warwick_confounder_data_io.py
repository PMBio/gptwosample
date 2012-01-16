'''
Created on Jan 16, 2012

@author: maxz
'''
import scipy
from gptwosample.data.dataIO import get_data_from_csv
import cPickle
import pylab
from gptwosample import plot
import csv
from gptwosample.data.data_base import common_id, individual_id
import os
from gptwosample.util.confounder_model import Confounder_Model
from gptwosample.util.sample_confounders import sample_GP
import sys


def get_path_for_pickle(confounder_model, confounder_learning_model, components):
    return 'sampledfrom-%s_learnedby-%s_conf-%i' % (confounder_model, confounder_learning_model, components)

def find_gene_name_hit(Y_dict, gene_name):
    """Searches for the right index of given gene_name in Y_dict, returns -1 if no gene was hit"""
    gene_name_hit = Y_dict['gene_names'] == gene_name
    if not gene_name_hit.any():
        gene_name = gene_name.upper()
        gene_name_hit = Y_dict['gene_names'] == gene_name
        if not gene_name_hit.any():
            return -1
    gene_index = scipy.where(gene_name_hit)[0][0]
    return gene_index

def write_back_data(twosample_object, gene_name, csv_out, csv_out_file):
    line = [gene_name.upper(), twosample_object.bayes_factor()]
    common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
    common = scipy.exp(common)
    individual =\
        twosample_object.get_learned_hyperparameters()[individual_id]['covar']
    individual = scipy.exp(individual)
    line.extend(common)
    line.extend(individual)
    csv_out.writerow(line)
    csv_out_file.flush()

def prepare_csv_out(CovFun, out_path, out_file):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    csv_out_file_confounded = open(os.path.join(out_path, out_file), 'wb')
    csv_out_confounded = csv.writer(csv_out_file_confounded)
    header = ["Gene", "Bayes Factor"]
    header.extend(map(lambda x:'Common ' + x, CovFun[0].get_hyperparameter_names()))
    header.extend(map(lambda x:'Individual ' + x, CovFun[0].get_hyperparameter_names()))
    csv_out_confounded.writerow(header)
    return csv_out_confounded, csv_out_file_confounded

def get_data(cond1_file, cond2_file, fraction, confounder_model, confounder_learning_model, components):
    dump_file = "%s.pickle" % get_path_for_pickle(confounder_model, confounder_learning_model, components)
    explained_variance = .5
    if (not os.path.exists(dump_file)) or 'recalc' in sys.argv:
        print "recalculating simulations and reading data"
        Y_dict = read_data_from_file(cond1_file, cond2_file, fraction)
        # simulation model:
        gp_conf_model = Confounder_Model(confounder_model, Y_dict['T'], Y_dict['condition'].reshape(-1, 1), components, explained_variance)
        Y_dict = add_simulated_confounders(Y_dict, gp_conf_model, components=components)
        # learning model:
        gp_conf_model = Confounder_Model(confounder_learning_model, Y_dict['T'], Y_dict['condition'].reshape(-1, 1), components, explained_variance)
        Y_dict['X'], Y_dict['Y_reconstruct'] = gp_conf_model.learn_confounder_matrix(Y_dict['Y_confounded'])
        # save data:
        cPickle.dump(Y_dict, open(dump_file, 'wb'), -1)
    else:
        print "loading data and simulations from file: %s" % (dump_file)
        # data already exists
        Y_dict = cPickle.load(open(dump_file, 'r'))
        #gpmodel = construct_gp_model(Y_dict, model=confounder_model, components=components, explained_variance=explained_variance)
    return Y_dict

def read_data_from_file(cond1_file, cond2_file, fraction=1.0):
    """read raw data and return dict with all data"""
    #1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    # get replicate stuff organized
    gene_names = cond1.keys()
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1 + n_replicates_2
    
    #merge data
    T = scipy.tile(T1, n_replicates).reshape(-1, 1)   
    Y1 = scipy.array(cond1.values())
    Y2 = scipy.array(cond2.values())
    # get Y values for all genes
    Y = scipy.concatenate((Y1, Y2), 1).reshape(Y1.shape[0], -1)
    condition = scipy.concatenate((0 * scipy.ones(n_replicates_1 * T1.shape[0]), 1 * scipy.ones(n_replicates_2 * T2.shape[0])))
        
    #create random index and take a subset of these
    Isubset = scipy.random.permutation(len(cond1.keys()))
    Isubset = Isubset[0:int(fraction * len(Isubset))]
    #subset Y
    Y = Y[Isubset, :].T
   
    #Y[:,:] = scipy.random.randn(Y.shape[0],Y.shape[1])
   
    #gene names on subset:
    gene_names = scipy.array(cond1.keys())[Isubset]
    
    #zero mean each gene
    Y -= Y.mean(axis=1)[:, scipy.newaxis]
    #set variance to unit variance
    Y /= Y.std()
    
    RV = {'Y': Y, 'gene_names':gene_names, 'T':T, 'condition': condition, 'subset_indices': Isubset}
    return RV

def plot_and_save_figure(T1, twosample_object, gene_name, savename=None):
    pylab.figure(1)
    pylab.clf()
    plot.plot_results(twosample_object, 
                      title='%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' \
                      % (gene_name, twosample_object.bayes_factor()), shift=None, draw_arrows=1)
    pylab.xlim(T1.min(), T1.max())
    if savename is None:
        savename = gene_name
    pylab.savefig("%s" % (savename))
    
def get_ground_truth_iterator():
    return csv.reader(open("../examples/ground_truth_random_genes.csv", 'r'))

def add_simulated_confounders(Ydict, gp_conf_model, components=4, **kw_args):
    """add simulated confounded expression data to dict"""

    #construct X for GP sampling
    # structure consists of [Time, factor1,factor2,..factorN]
    # (time is needed for product_linear covaraince which assume non-perfect independence of measurements
    
    Yconf = sample_GP(gp_conf_model._lvm_covariance, gp_conf_model._lvm_hyperparams['covar'], gp_conf_model._lvm_X, Ydict['Y'].shape[1])

    Ydict['confounder'] = Yconf
    Ydict['Y_confounded'] = Ydict['Y'] + Yconf
    return Ydict
