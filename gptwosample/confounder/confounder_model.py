'''
Created on Jan 9, 2012

@author: maxz
'''
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.gp.gplvm import GPLVM
from pygp.optimize.optimize_base import opt_hyper
from gptwosample.twosample.twosample_base import GPTwoSample_individual_covariance,\
    AbstractGPTwoSampleBase
from gptwosample.data.data_base import get_model_structure
import numpy
from pygp.covar.linear import LinearCF
from pygp.covar.noise import NoiseCFISO
from pygp.covar.combinators import SumCF
from pygp.covar.fixed import FixedCF
from pygp.covar.se import SqexpCFARD
from pygp.util.pca import PCA
import sys

class Confounder_Model(GPTwoSample_individual_covariance):
    
    def __init__(self, T, Y, components=4, 
                 lvm_covariance=None, initial_hyperparameters=None, **kwargs):
        """
        **Parameters**:
            T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
            Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
            compontents : Number of Confounders to use
            lvm_covariance : optional - set covariance to use in confounder learning
        """
        self.set_data(T, Y)
        
        if lvm_covariance is not None:
            self._lvm_covariance = lvm_covariance
        else:
            sample_structure = numpy.append(numpy.ones(self.r * self.t), numpy.zeros(self.r * self.t))[:,None]
            sample_struct = numpy.dot(sample_structure, sample_structure.T)
            sample_struct += numpy.dot(sample_structure[::-1], sample_structure[::-1].T)
            self._lvm_covariance = SumCF([LinearCF(n_dimensions=components), FixedCF(sample_struct), NoiseCFISO()])
        
        if initial_hyperparameters is None:
            initial_hyperparameters = numpy.zeros(self._lvm_covariance.get_number_of_parameters())

        self._components = components
        self._initialized = False
        super(Confounder_Model, self).__init__(None, None, None, initial_hyperparameters=get_model_structure(), **kwargs)
        
    def learn_confounder_matrix(self):
        self._check_data()
        likelihood = GaussLikISO()
        #bounds = {'lik': scipy.array([[-5., 5.]] * (self.r * self.t))}
        
        Y = self.Y.reshape(numpy.prod(self.n*self.r*self.t), self.Y.shape[3])
        p = PCA(Y)
        self.X = p.project(Y, self._components)
    
        g = GPLVM(gplvm_dimensions=xrange(0, self._components),
                  covar_func=self._lvm_covariance,
                  likelihood=likelihood,
                  x=self.X,
                  y=Y)
        
        hyper = {
                 'lik':numpy.log([.15]),
                 'x':self.X,
                 'covar':numpy.zeros(self._lvm_covariance.get_number_of_parameters())
                 }
        
        lvm_hyperparams, _ = opt_hyper(g, hyper, 
                                       Ifilter=None, maxiter=10000, 
                                       gradcheck=False, bounds=None, 
                                       messages=False)
        
        self._init_conf_matrix(lvm_hyperparams)
        #print "%s found optimum of F=%s" % (threading.current_thread().getName(), opt_f)

    def set_data_by_xy_data(self, x1, x2, y1, y2):
        raise NotImplementedError("Set data by calling set_data(T, Y)")
        return AbstractGPTwoSampleBase.set_data_by_xy_data(self, x1, x2, y1, y2)


    def set_data(self, T, Y):
        """
        Set data by time T and expression matrix Y:
        
        **Parameters**:
            T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
            Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
        """
        self._invalidate_cache()
        self.T = T
        try:
            self.n, self.r, self.t = T.shape
        except ValueError:
            raise ValueError("Timepoints must be given as [n x r x t] matrix!")
        self.Y = Y
        try:
            self.d = Y.shape[3]
        except ValueError:
            raise ValueError("Expression must be given as [n x r x t x d] matrix!")
        assert numpy.prod(T.shape) == numpy.prod(Y.shape[:3]), 'Shape mismatch, must be one nrt timepoints per gene.'

    def predict_model_likelihoods(self, interval_indices=get_model_structure(), message="Predicting Likelihoods:",*args, **kwargs):
        self._check_data()
        assert self._initialized, "confounder matrix not yet learned, try using learn_confounder_matrix() first"
        
        self._likelihoods = list()
        self._hyperparameters = list()
        for i in xrange(self.d):
            sys.stdout.flush()
            sys.stdout.write("{1:s}{0:.2%}           \r".format(float(i)/self.d,message))
            T0, T1, Y0, Y1 = self._get_data_for(i)
            super(Confounder_Model, self).set_data_by_xy_data(T0, T1, Y0, Y1)
            try:
                self._likelihoods.append(super(Confounder_Model, self).predict_model_likelihoods(**kwargs).copy())
            except ValueError:
                self._likelihoods.append(numpy.NaN)
            self._hyperparameters.append(self._learned_hyperparameters.copy())
        sys.stdout.write("{1:s}{0:.2%}        \r".format(float(i)/self.d,message))
        return self._likelihoods
        

    def predict_mean_variance_iter(self, interpolation_interval, 
        hyperparams=None, 
        *args, **kwargs):
        """
        Iterate through all predicted mean variances
        """
        try:
            if self._hyperparameters is None:
                raise ValueError()
        except:
            print "likelihoods not yet predicted, running predict_model_likelihoods..."
            self.predict_model_likelihoods()
        
        for i in xrange(self.d):
            super(Confounder_Model, self).set_data_by_xy_data(*self._get_data_for(i))
            yield AbstractGPTwoSampleBase.predict_mean_variance(self, interpolation_interval, self._hyperparameters[i], *args, **kwargs)

    def bayes_factor_iter(self):
        """
        Iterate through all bayes factors for all predicted genes.
        
        **returns**:
            bayes_factor for each gene in Y
        """
        for likelihood in self._likelihoods:
            yield AbstractGPTwoSampleBase.bayes_factor(self, likelihood)


    def get_model_likelihoods(self):
        return self._likelihoods


    def get_learned_hyperparameters(self):
        return self._hyperparameters
        raise ValueError("Hyperparameters are not saved due to memory issues")
        

    def get_predicted_mean_variance(self):
        return self._mean_variances


    def _invalidate_cache(self):
        self._likelihoods = None
        self._mean_variances = None
        self._hyperparameters = None
        return AbstractGPTwoSampleBase._invalidate_cache(self)


    def _check_data(self):
        try:
            self.T.T
            self.Y.T
        except ValueError:
            raise ValueError("Data has not been set or is None, use set_data(Y,T) to set data")

    def _get_data_for(self, i):
        return self.T[0,:,:].ravel()[:,None], \
            self.T[1,:,:].ravel()[:,None], \
            self.Y[0,:,:,i].ravel()[:,None], \
            self.Y[1,:,:,i].ravel()[:,None]
        
        
    def _init_conf_matrix(self, lvm_hyperparams):
        self._initialized = True
        self.X = lvm_hyperparams['x']
        self._lvm_hyperparams = lvm_hyperparams
        
        K = self._lvm_covariance.K(lvm_hyperparams['covar'], self.X)
        
        covar_common = SumCF([SqexpCFARD(1), FixedCF(K), NoiseCFISO()])
        covar_individual_1 = SumCF([SqexpCFARD(1), FixedCF(K[:self.r*self.t,:self.r*self.t]), NoiseCFISO()])
        covar_individual_2 = SumCF([SqexpCFARD(1), FixedCF(K[self.r*self.t:,self.r*self.t:]), NoiseCFISO()])        
        initial_hyperparameters = \
            get_model_structure(individual={'covar':numpy.zeros(covar_common.get_number_of_parameters())}, 
                                common={'covar':numpy.zeros(covar_individual_1.get_number_of_parameters())})
        super(Confounder_Model, self).__init__(covar_individual_1, covar_individual_2, 
                                               covar_common, 
                                               initial_hyperparameters=initial_hyperparameters)
        self._initialized = True
        self.X = lvm_hyperparams['x']
        self._lvm_hyperparams = lvm_hyperparams
        
