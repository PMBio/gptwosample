'''
Created on Jan 9, 2012

@author: maxz
'''
import scipy
from pygp.covar.linear import LinearCFISO
from pygp.covar.combinators import SumCF, ProductCF
from pygp.covar import delta
from pygp.covar.se import SqexpCFARD
from gptwosample.util.confounder_constants import linear_covariance_model_id, \
    product_linear_covariance_model_id
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.gp.gplvm import GPLVM
from pygp.optimize.optimize_base import opt_hyper
import threading

class Confounder_Model():
    
    def __init__(self, model, T, conditions, components=4, explained_variance=1.):
        self._components = components
        self._number_of_time_points = T.shape[1]
        self._model = model
        
        if (model == linear_covariance_model_id):
            self.__construct_confounder_model_linear(T, conditions, explained_variance)
        elif (model == product_linear_covariance_model_id):
            self.__construct_confounder_model_product_linear(T, conditions, explained_variance)
    
    def learn_confounder_matrix(self, Y):
        likelihood = GaussLikISO()
        bounds = {'lik': scipy.array([[-5., 5.]] * self._number_of_time_points)}
    
        g = GPLVM(gplvm_dimensions=xrange(1, 1 + self._components),
                  covar_func=self._learning_covariance,
                  likelihood=likelihood,
                  x=self._learning_X,
                  y=Y)
        
        opt_hyperparams, opt_f = opt_hyper(g, self._learning_hyperparams, None, 10000, False, bounds)
        print "%s found optimum of F=%s"%(threading.current_thread().getName(), opt_f)
        
        return self._learning_covariance.K(opt_hyperparams['covar'], g.x), g.predict(opt_hyperparams, g.x, output=scipy.arange(Y.shape[1]), var=False)

    def __construct_confounder_model_product_linear(self, T, conditions, explained_variance=1.0):
        self._lvm_covariance = ProductCF((LinearCFISO(n_dimensions=self._components, dimension_indices=xrange(0, self._components)),
                                    SqexpCFARD(n_dimensions=1, dimension_indices=[self._components])), n_dimensions=self._components + 1)
        self._lvm_hyperparams = {'covar': scipy.log([explained_variance, 1, 1]), 'lik':scipy.array([0.1])}
        self._learning_hyperparams = {'covar': scipy.log([explained_variance, 1, 1, 1]), 'lik':scipy.array([0.1])}        
        
        self._learning_covariance = ProductCF(
                                              (
                                               SumCF(
                                                     (
                                                      LinearCFISO(n_dimensions=self._components, 
                                                                  dimension_indices=xrange(0, self._components))
                                                      ,
                                                      delta.DeltaCFISO(n_dimensions=1, 
                                                                    dimension_indices=[self._components + 1])
                                                      )
                                                     )
                                               ,
                                               SqexpCFARD(n_dimensions=1, 
                                                          dimension_indices=[self._components])
                                               )
                                              )
        self._learning_X = scipy.concatenate((
                               scipy.random.randn(T.shape[0], self._components), # Random initialization for confounders, due to delta included 
                               conditions, # Conditions for seperation of conditions  
                               T), # Time points for lvm time covariance 
                              axis=1)

#        self._learning_hyperparams = {'covar': scipy.log([explained_variance, 1, 1, 1, 1, 1]), 'lik':scipy.array([0.1])}        
#        
#        self._learning_covariance = SumCF(
#                                          (
#                                           ProductCF(
#                                                     (
#                                                      LinearCFISO(n_dimensions=self._components, 
#                                                                  dimension_indices=xrange(0, self._components))
#                                                      ,
#                                                      SqexpCFARD(n_dimensions=1, 
#                                                                 dimension_indices=[self._components + 2])
#                                                      )
#                                                     )
#                                           ,
#                                           ProductCF(
#                                                     (
#                                                      delta.DeltaCFISO(n_dimensions=1,
#                                                                       dimension_indices=[self._components + 1])
#                                                      ,
#                                                      SqexpCFARD(n_dimensions=1, 
#                                                                 dimension_indices=[self._components + 2])
#                                                      )
#                                                     )
#                                           )
#                                          )
#        self._learning_X = scipy.concatenate((
#                               scipy.random.randn(T.shape[0], self._components), # Random initialization for confounders, due to delta included 
#                               T, # Time points for lvm time covariance 
#                               conditions, # Conditions for seperation of conditions  
#                               T), # Time points for lvm time covariance 
#                              axis=1)
        self._lvm_X = scipy.concatenate((self._learning_X[:, :self._components], T), axis=1)
       
        #optimization over the latent dimension only (1. Dimension is time, 2. Dimension is )
        self._learning_hyperparams['x'] = self._learning_X[:, :self._components].copy()
                
    def __construct_confounder_model_linear(self, T, conditions, explained_variance=1.0):
        """
        The gpmodel  
        """
        self._lvm_covariance = LinearCFISO(n_dimensions=self._components, dimension_indices=xrange(0, self._components))
        self._lvm_hyperparams = {'covar': scipy.log([explained_variance]), 'lik':scipy.array([0.1])}
        
        self._learning_hyperparams = {'covar': scipy.log([explained_variance, 1, 1, 1]), 'lik':scipy.array([0.1])} 
        self._learning_covariance = SumCF((
                                 LinearCFISO(n_dimensions=self._components, dimension_indices=xrange(0, self._components))
                                 ,
                                 ProductCF((
                                            delta.DeltaCFISO(n_dimensions=1,
                                                             dimension_indices=[self._components + 1])
                                            ,
                                            SqexpCFARD(n_dimensions=1, dimension_indices=[self._components]))
                                           )
                                 )
                                )
            
        self._learning_X = scipy.concatenate((
                               scipy.random.randn(T.shape[0], self._components), # Random initialization for confounders, due to delta included 
                               T, # Time points for lvm time covariance 
                               conditions, # Conditions for seperation of conditions  
                               T),
                              # Time points for lvm time covariance 
                              axis=1)
         
        self._lvm_X = scipy.concatenate((self._learning_X[:, :self._components], T), axis=1)
       
        #optimization over the latent dimension only (1. Dimension is time, 2. Dimension is )
        self._learning_hyperparams['x'] = self._learning_X[:, :self._components].copy()
    
