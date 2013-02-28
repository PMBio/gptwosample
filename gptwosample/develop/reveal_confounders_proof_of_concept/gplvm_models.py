'''
Created on Feb 21, 2013

@author: Max
'''
from pygp.covar.linear import LinearCF
from pygp.covar.bias import BiasCF
from pygp.covar.combinators import SumCF
import numpy
from pygp.gp import gplvm
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar.fixed import FixedCF
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.covar.se import SqexpCFARD

__doc = """
Y: Expression matrix [NRT x D]
T: TimePoints [N x R x T] -> [Samples x Replicates x Timepoints]
"""

def run_gplvm_with_convariance(Y, T, components, lvm_covariance):
    hyperparams = {'covar':lvm_covariance.get_de_reparametrized_theta(numpy.repeat(.5, lvm_covariance.get_number_of_parameters()))}
    X_pca = gplvm.PCA(Y, components)[0]
    # Get X right:
    X0 = numpy.concatenate((T.copy().reshape(-1,1), X_pca.copy()), axis=1)
    hyperparams['x'] = X_pca.copy()
    lik = GaussLikISO()
    hyperparams['lik'] = numpy.log([0.1])
    # lvm for confounders only:
    g = gplvm.GPLVM(gplvm_dimensions=xrange(1, 1+components), covar_func=lvm_covariance, likelihood=lik, x=X0, y=Y)
    bounds = {}
    bounds['lik'] = numpy.array([[-5., 5.]] * T.shape[2])
    [opt_hyperparams_comm, _] = opt_hyper(g, hyperparams, gradcheck=False, messages=True, maxiter=8000)
    X = opt_hyperparams_comm['x']
    ard = lvm_covariance.get_reparametrized_theta(opt_hyperparams_comm['covar'])
    return numpy.dot(X,(ard*X).T), opt_hyperparams_comm, g
    
def linear_gplvm_confounder(Y, T, components=4):
    __doc__ = __doc
    linear_cf = LinearCF(n_dimensions=components)
    mu_cf = BiasCF()
    lvm_covariance = SumCF((linear_cf,mu_cf))
    return run_gplvm_with_convariance(Y, T, components, lvm_covariance)

def conditional_linear_gplvm_confounder(Y, T, components=4):
    __doc__ = __doc
    linear_cf = LinearCF(n_dimensions=components)
    mu_cf = BiasCF()
    # Get fixed cf encoding twosample structure:
    try:
        ls = T.shape[1] * T.shape[2]  # length of samples
    except IndexError:
        raise IndexError("Error: Shape of Timepoints must be [Samples x Replicates x Timepoints] -> [2, R, T] in twosample cases")        
    condition_indicators = numpy.vstack([numpy.hstack([numpy.ones([ls]*T.shape[0])*(i+1)%2, numpy.ones([ls]*T.shape[0])*i%2]) for i in range(T.shape[0])])
    #lvm_covariance = SumCF((mu_cf, ProductCF([FixedCF(condition_indicators),linear_cf])))
    lvm_covariance = SumCF((linear_cf, mu_cf, FixedCF(condition_indicators)))
    #lvm_covariance = ProductCF([FixedCF(condition_indicators),linear_cf])
    return run_gplvm_with_convariance(Y, T, components, lvm_covariance)

def time_linear_gplvm_confounder(Y, T, components=4):
    __doc__ = __doc
    linear_cf = LinearCF(n_dimensions=components)
    mu_cf = BiasCF()
    se_cf = SqexpCFARD(n_dimensions=1,dimension_indices=numpy.array([0]))
    # Get fixed cf encoding twosample structure:
    lvm_covariance = SumCF((linear_cf, mu_cf, se_cf))
    return run_gplvm_with_convariance(Y, T, components, lvm_covariance)