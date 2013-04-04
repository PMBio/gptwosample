'''
Created on Jan 6, 2012

@author: maxz
'''
from numpy.linalg.linalg import cholesky
import scipy as SP

def sample_GP(covariance, covar_hyperparams, X, Nsamples):
    """draw Nsample from a GP with covaraince covaraince, hyperparams and X as inputs"""
    # or draw from a GP:
    sigma = 1e-6
    K = covariance.K(covar_hyperparams, X)
    K += sigma * SP.eye(K.shape[0])
    cholK = cholesky(K)
    #left multiply indepnendet confounding matrix with cholesky:
    Ys = SP.dot(SP.randn(Nsamples,K.shape[0]),cholK).T
    return Ys
    
def sample_confounders_linear(components, gene_names, n_replicates, gene_length):
    NRT = n_replicates * gene_length 
    X = SP.random.randn(NRT, components)
    W = SP.random.randn(components, len(gene_names)) * 0.5
    Y_conf = SP.dot(X, W)
    return Y_conf.T
