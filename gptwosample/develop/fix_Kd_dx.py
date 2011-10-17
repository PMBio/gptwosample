'''
Created on Sep 29, 2011

@author: maxz
'''
from gptwosample.data.dataIO import get_data_from_csv
from pygp.priors import lnpriors
from gptwosample.data.data_base import get_model_structure
import scipy
from pygp import likelihood, covar
from pygp.covar import se,gradcheck
from pygp.gp import gplvm
from pygp.optimize.optimize_base import opt_hyper
import logging
from numpy.linalg.linalg import cholesky
from pygp.covar import dist

logging.basicConfig(level=logging.INFO)

def run_demo(cond1_file,cond2_file):
    # settings
    components = 12
    
#    # Data Values
#    cond1 = get_data_from_csv(cond1_file)
#    cond2 = get_data_from_csv(cond2_file)
#    # Time Points & replicates
#    Tpredict = scipy.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, scipy.newaxis]
#    T1 = cond1.pop('input')
#    T2 = cond2.pop('input')
#    gene_names = sorted(cond1.keys()) 
#    n_replicates_1 = cond1[gene_names[0]].shape[0]
#    n_replicates_2 = cond2[gene_names[0]].shape[0]
#    n_replicates = n_replicates_1+n_replicates_2
#    gene_length = len(T1)
#    T = scipy.tile(T1,n_replicates).reshape(-1,1)

    # LVM covariance
    #lvm_covariance = ProductCF((LinearCFISO(n_dimensions=components),FixedCF(scipy.eye(n_replicates*gene_length))))
    #lvm_covariance = ProductCF((LinearCFISO(n_dimensions=components),LinearCFISO(n_dimensions=components)))
    #lvm_covariance = ProductCF((FixedCF(scipy.eye(n_replicates*gene_length)),FixedCF(scipy.eye(n_replicates*gene_length))))
    #lvm_covariance = ProductCF((SqexpCFARD(dimension_indices=[0]),(LinearCFISO(dimension_indices=xrange(1,1+components)))))
    lvm_covariance = se.SqexpCFARD(n_dimensions=components)
    
    timepoints = 12
    
    X = scipy.tile(scipy.arange(0, timepoints).reshape(-1,1),components)#scipy.randn(timepoints,components)
    sigma = 1e-6
    
    W = sigma*scipy.eye(timepoints)
    Y_comm = scipy.dot(W,X)
    
#    Y_comm = scipy.array([scipy.dot(cholesky(lvm_covariance.K(scipy.log([1 for i in range(components+1)]), X) + sigma 
#                                             * scipy.eye(timepoints)), scipy.eye(timepoints, 1)).flatten() for i in range(timepoints)])
        
    # Get all outputs for training
#    Y1_conf = scipy.array(cond1.values()).reshape(T1.shape[0]*n_replicates_1,-1)
#    Y2_conf = scipy.array(cond2.values()).reshape(T2.shape[0]*n_replicates_2,-1)
#    Y_comm = scipy.concatenate((Y1_conf,Y2_conf))

    # starting point for confounder learning 
    X_pca = gplvm.PCA(Y_comm, components)[0]
    X_pca += 0.1*scipy.random.randn(X_pca.shape[0], X_pca.shape[1])
    X0 = X_pca.copy()#scipy.concatenate((T,X_pca.copy()),1)
    
    # LVM paramteters
    logtheta = scipy.log([3 for i in range(components+1)])
    hyperparams = {'covar': logtheta}
    hyperparams['x'] = X_pca.copy()
    # noise likelihood
    lik = likelihood.GaussLikISO()
    hyperparams['lik'] = scipy.log([0.2])
    
    # get GPLVM instance
    g = gplvm.GPLVM(gplvm_dimensions=xrange(components),covar_func=lvm_covariance,likelihood=lik,x=X0,y=Y_comm)
    
    # run gplvm on data for testing purpose
    gradcheck.grad_check_Kx(lvm_covariance, hyperparams['covar'], X0)
    
    opt_hyperparams = opt_hyper(g,hyperparams,gradcheck=True)[0]
    
    gradcheck.grad_check_Kx(lvm_covariance, opt_hyperparams['covar'], X0)
    
if __name__ == '__main__':
     run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')