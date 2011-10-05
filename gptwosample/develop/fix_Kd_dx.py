'''
Created on Sep 29, 2011

@author: maxz
'''
from gptwosample.data.dataIO import get_data_from_csv
from pygp.priors import lnpriors
from gptwosample.data.data_base import get_model_structure
import scipy
from pygp import likelihood
from pygp.covar.combinators import ProductCF
from pygp.covar.fixed import FixedCF
from pygp.covar.linear import LinearCFISO
from pygp.gp import gplvm
from pygp.covar import gradcheck
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar.se import SqexpCFARD
import logging

logging.basicConfig(level=logging.INFO)

def run_demo(cond1_file,cond2_file):
    # settings
    components = 4
    
    # Data Values
    cond1 = get_data_from_csv(cond1_file)
    cond2 = get_data_from_csv(cond2_file)
    # Time Points & replicates
    Tpredict = scipy.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, scipy.newaxis]
    T1 = cond1.pop('input')
    T2 = cond2.pop('input')
    gene_names = sorted(cond1.keys()) 
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    n_replicates = n_replicates_1+n_replicates_2
    gene_length = len(T1)
    T = scipy.tile(T1,n_replicates).reshape(-1,1)
    
    # Get all outputs for traiing
    Y1_conf = scipy.array(cond1.values()).reshape(T1.shape[0]*n_replicates_1,-1)
    Y2_conf = scipy.array(cond2.values()).reshape(T2.shape[0]*n_replicates_2,-1)
    Y_comm = scipy.concatenate((Y1_conf,Y2_conf))

    # starting point for confounder learning 
    X_pca = gplvm.PCA(Y_comm, components)[0]
    X_pca += 0.1*scipy.random.randn(X_pca.shape[0], X_pca.shape[1])
    X0 = scipy.concatenate((T,X_pca.copy()),1)
    
    # LVM covariance
    #lvm_covariance = ProductCF((LinearCFISO(n_dimensions=components),FixedCF(scipy.eye(n_replicates*gene_length))))
    #lvm_covariance = ProductCF((LinearCFISO(n_dimensions=components),LinearCFISO(n_dimensions=components)))
    #lvm_covariance = ProductCF((FixedCF(scipy.eye(n_replicates*gene_length)),FixedCF(scipy.eye(n_replicates*gene_length))))
    lvm_covariance = ProductCF((SqexpCFARD(dimension_indices=[0]),(LinearCFISO(dimension_indices=xrange(1,1+components)))))
    #lvm_covariance = ProductCF((SqexpCFARD(),SqexpCFARD()))
    
    # LVM paramteters
    hyperparams = {'covar': scipy.log([1,1,1])}
    hyperparams['x'] = X_pca.copy()
    # noise likelihood
    lik = likelihood.GaussLikISO()
    hyperparams['lik'] = scipy.log([0.2])
    
    # get GPLVM instance
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1,1+components),covar_func=lvm_covariance,likelihood=lik,x=X0,y=Y_comm)
    
    # run gplvm on data for testing purpose
    gradcheck.grad_check_Kx(lvm_covariance, hyperparams['covar'], X0)
    
    opt_hyperparams = opt_hyper(g,hyperparams,gradcheck=True)[0]
    
    gradcheck.grad_check_Kx(lvm_covariance, opt_hyperparams['covar'], X0)
    
if __name__ == '__main__':
     run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')