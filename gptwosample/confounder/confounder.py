'''
Confounder Learning and Correction Module
-----------------------------------------

@author: Max Zwiessele
'''
import numpy
from pygp.covar.linear import LinearCFISO
from pygp.covar.combinators import SumCF, ProductCF
from pygp.covar.se import SqexpCFARD
from pygp.gp.gplvm import GPLVM
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar.fixed import FixedCF
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.covar.bias import BiasCF
from pygp.util.pca import PCA
from gptwosample.twosample.twosample import TwoSample

class TwoSampleConfounder(TwoSample):
    """Run GPTwoSample on given Data

    **Parameters**:
        - T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
        - Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
        - q : Number of Confounders to use
        - lvm_covariance : optional - set covariance to use in confounder learning
        - init : [random, pca]


    **Fields**:
        * T: Time Points [n x r x t] [Samples x Replicates x Timepoints]
        * Y: Expression [n x r x t x d] [Samples x Replicates x Timepoints x Genes]
        * X: Confounders [nrt x 1+q] [SamplesReplicatesTimepoints x T+q]
        * lvm_covariance: GPLVM covaraince function used for confounder learning
        * n: Samples
        * r: Replicates
        * t: Timepoints
        * d: Genes
        * q: Confounder Components
    """
    def __init__(self, T, Y, q=4,
                 lvm_covariance=None,
                 init="random",
                 covar_common=None,
                 covar_individual_1=None,
                 covar_individual_2=None):
        """
        **Parameters**:
            T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
            Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
            q : Number of Confounders to use
            lvm_covariance : optional - set covariance to use in confounder learning
            init : [random, pca]
        """
        super(TwoSampleConfounder, self).__init__(T, Y, covar_common=covar_common,
                                                  covar_individual_1=covar_individual_1,
                                                  covar_individual_2=covar_individual_2)
        self.q = q
        self.init = init
        
        self.init_X(Y, init)
        
        if lvm_covariance is not None:
            self._lvm_covariance = lvm_covariance
        else:
            rt = self.r * self.t
            # self.X_r = numpy.zeros((self.n * rt, self.n * self.r))
            # for i in xrange(self.n * self.r):self.X_r[i * self.t:(i + 1) * self.t, i] = 1
            # rep = LinearCFISO(dimension_indices=numpy.arange(1 + q, 1 + q + (self.n * self.r)))
            self.X_s = numpy.zeros((self.n * rt, self.n))
            for i in xrange(self.n):self.X_s[i * rt:(i + 1) * rt, i] = 1
            sam = LinearCFISO(dimension_indices=numpy.arange(1 + q,
                                                              # + (self.n * self.r),
                                                              1 + q + self.n))  # + (self.n * self.r)
            self._lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 2)),
                                          # rep,
                                          sam,
                                          ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                          BiasCF()])

        # if initial_hyperparameters is None:
        #    initial_hyperparameters = numpy.zeros(self._lvm_covariance.get_number_of_parameters()))

        self._initialized = False

    def learn_confounder_matrix(self,
                                ard_indices=None,
                                x=None,
                                messages=True,
                                gradient_tolerance=1E-12,
                                lvm_dimension_indices=None,
                                gradcheck=False,
                                maxiter=10000,
                                ):
        """
        Learn confounder matrix with this model.

        **Parameters**:

            x : array-like
                If you provided an own lvm_covariance you have to specify
                the X to use within GPLVM
    
            lvm_dimension_indices : [int]
                If you specified an own lvm_covariance you have to specify
                the dimension indices for GPLVM
    
            ard_indices : [indices]
                If you provided an own lvm_covariance, give the ard indices of the covariance here,
                to be able to use the correct hyperparameters for calculating the confounder covariance matrix.
        """
        if lvm_dimension_indices is None:
            lvm_dimension_indices = xrange(1, 1 + self.q)
        if ard_indices is None:
            ard_indices = slice(0, self.q)
        self._check_data()
        self.init_X(self.Y, self.init)

        # p = PCA(Y)
        # try:
        #    self.X = p.project(Y, self.q)
        # except IndexError:
        #    raise IndexError("More confounder components then genes (q > d)")

        if x is None:
            x = self._x()
        self._Xlvm = x
        self.gplvm = self._gplvm(lvm_dimension_indices)

        hyper = {
                 'lik':numpy.log([.15]),
                 'x':self.X,
                 'covar':numpy.zeros(self._lvm_covariance.get_number_of_parameters())
                 }

        lvm_hyperparams, _ = opt_hyper(self.gplvm, hyper,
                                       Ifilter=None, maxiter=maxiter,
                                       gradcheck=gradcheck, bounds=None,
                                       messages=messages,
                                       gradient_tolerance=gradient_tolerance)

        self._Xlvm = self.gplvm.getData()[0]
        self._init_conf_matrix(lvm_hyperparams, ard_indices, lvm_dimension_indices)
        self.initialize_twosample_covariance()

    def predict_lvm(self):
        return self.gplvm.predict(self._lvm_hyperparams, self._Xlvm, numpy.arange(self.d))

    def initialize_twosample_covariance(self, covar_common=lambda x: SumCF([SqexpCFARD(1), x, BiasCF()]),
                                            covar_individual_1=lambda x: SumCF([SqexpCFARD(1), x, BiasCF()]),
                                            covar_individual_2=lambda x: SumCF([SqexpCFARD(1), x, BiasCF()]),
                                            ):
        """
        initialize twosample covariance with function covariance(XX), where XX
        is a FixedCF with the learned confounder matrix.

        default is SumCF([SqexpCFARD(1), FixedCF(self.K_conf.copy()), BiasCF()])
        """
        self.covar_comm = covar_common(FixedCF(self.K_conf.copy()))
        self.covar_ind1 = covar_individual_1(FixedCF(self.K_conf.copy()))
        self.covar_ind2 = covar_individual_2(FixedCF(self.K_conf.copy()))

    def init_X(self, Y, init):
        if init == 'pca':
            y = Y.reshape(-1, self.d)
            p = PCA(y)
            self.X = p.project(y, self.q)
            self.X += .1 * numpy.random.randn(*self.X.shape)
        elif init == 'random':
            self.X = numpy.random.randn(numpy.prod(self.n * self.r * self.t), self.q)
        else:
            print "init model {0!s} not known".format(init)

    def _init_conf_matrix(self, lvm_hyperparams, conf_covar_name, lvm_dimension_indices):
        self._initialized = True
        self.X = lvm_hyperparams['x']
        self._lvm_hyperparams = lvm_hyperparams
        # if ard_indices is None:
        #    ard_indices = numpy.arange(1)
        # ard = self._lvm_covariance.get_reparametrized_theta(lvm_hyperparams['covar'])[ard_indices]
        # self.K_conf = numpy.dot(self.X*ard, self.X.T)
        try:
            self.gplvm
        except:
            self.gplvm = self._gplvm(lvm_dimension_indices)
        self.K_conf = self._lvm_covariance.K(self._lvm_hyperparams['covar'], self._Xlvm, self._Xlvm, names=['XX'])

    def _gplvm(self, lvm_dimension_indices):
        self._Xlvm[:, lvm_dimension_indices] = self.X
        Y = self.Y.reshape(numpy.prod(self.n * self.r * self.t), self.Y.shape[3])
        return GPLVM(gplvm_dimensions=lvm_dimension_indices, covar_func=self._lvm_covariance,
            likelihood=GaussLikISO(),
            x=self._Xlvm,
            y=Y)

    def _x(self):
        return numpy.concatenate((self.T.reshape(-1, 1), self.X,  # self.X_r,
                self.X_s),
            axis=1)


if __name__ == '__main__':
    Tt = numpy.arange(0, 16, 2)[:, None]
    Tr = numpy.tile(Tt, 3).T
    Ts = numpy.array([Tr, Tr])

    n, r, t, d = nrtd = Ts.shape + (12,)

    covar = SqexpCFARD(1)
    K = covar.K(covar.get_de_reparametrized_theta([1, 13]), Tt)
    m = numpy.zeros(t)

    try:
        from scikits.learn.mixture import sample_gaussian
    except:
        raise "scikits needed for this example"
        # raise r

    y1 = sample_gaussian(m, K, cvtype='full', n_samples=d)
    y2 = sample_gaussian(m, K, cvtype='full', n_samples=d)

    Y1 = numpy.zeros((t, d + d / 2))
    Y2 = numpy.zeros((t, d + d / 2))

    Y1[:, :d] = y1
    Y2[:, :d] = y2

    sames = numpy.random.randint(0, d, size=d / 2)

    Y1[:, d:] = y2[:, sames]
    Y2[:, d:] = y1[:, sames]

    Y = numpy.zeros((n, r, t, d + d / 2))

    sigma = .5
    Y[0, :, :, :] = Y1 + sigma * numpy.random.randn(r, t, d + d / 2)
    Y[1, :, :, :] = Y2 + sigma * numpy.random.randn(r, t, d + d / 2)

    c = TwoSampleConfounder(Ts, Y)
#    c.__verbose = True

    # lvm_hyperparams_file_name = 'lvm_hyperparams.pickle'

    c.learn_confounder_matrix()
#    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'w')
#    pickle.dump(c._lvm_hyperparams, lvm_hyperparams_file)
#    lvm_hyperparams_file.close()

#    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'r')
#    c._init_conf_matrix(pickle.load(lvm_hyperparams_file), None)
#    lvm_hyperparams_file.close()

    c.predict_likelihoods(Ts, Y)
    c.predict_means_variances(numpy.linspace(0, 24, 100))
    
    import pylab
    pylab.ion()
    pylab.figure()
    for _ in c.plot():
        raw_input("enter to continue")

