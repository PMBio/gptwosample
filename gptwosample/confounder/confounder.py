'''
Confounder Learning and Correction Module
-----------------------------------------

@author: Max Zwiessele
'''
import numpy
from pygp.covar.linear import LinearCFISO, LinearCF
from pygp.covar.combinators import SumCF, ProductCF
from pygp.covar.se import SqexpCFARD
from pygp.gp.gplvm import GPLVM
from pygp.optimize.optimize_base import opt_hyper
from pygp.covar.fixed import FixedCF
from pygp.likelihood.likelihood_base import GaussLikISO
from gptwosample.data.data_base import get_model_structure
from gptwosample.twosample.twosample_base import TwoSampleSeparate, \
    TwoSampleBase
from Queue import Queue
import sys
from threading import Thread, Event
import pickle
import pylab
from pygp.covar.bias import BiasCF
import itertools
from copy import deepcopy

class ConfounderTwoSample():
    """Learn Confounder and run GPTwoSample correcting for confounding variation.

    Fields:
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
    NUM_PROCS = 2 # min(max(1,cpu_count()-2),3)
    SENTINEL = object()
    def __init__(self, T, Y, q=4,
                 lvm_covariance=None,
                 **kwargs):
        """
        **Parameters**:
            T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
            Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
            q : Number of Confounders to use
            lvm_covariance : optional - set covariance to use in confounder learning
        """
        self.set_data(T, Y)
        self.q = q
        self.__verbose = False
        self.__running_event = Event()

        self.X = numpy.random.randn(numpy.prod(self.n * self.r * self.t), self.q)

        if lvm_covariance is not None:
            self._lvm_covariance = lvm_covariance
        else:
            rt = self.r * self.t
            #self.X_r = numpy.zeros((self.n * rt, self.n * self.r))
            #for i in xrange(self.n * self.r):self.X_r[i * self.t:(i + 1) * self.t, i] = 1
            #rep = LinearCFISO(dimension_indices=numpy.arange(1 + q, 1 + q + (self.n * self.r)))
            self.X_s = numpy.zeros((self.n * rt, self.n))
            for i in xrange(self.n):self.X_s[i * rt:(i + 1) * rt, i] = 1
            sam = LinearCFISO(dimension_indices=numpy.arange(1 + q,
                                                              #+ (self.n * self.r),
                                                              1 + q + self.n)) # + (self.n * self.r) 
            self._lvm_covariance = SumCF([LinearCF(dimension_indices=numpy.arange(1, 1 + q)),
                                          #rep,
                                          sam,
                                          ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                          BiasCF()])

        # if initial_hyperparameters is None:
        #    initial_hyperparameters = numpy.zeros(self._lvm_covariance.get_number_of_parameters()))

        self._initialized = False

    def learn_confounder_matrix(self, ard_indices=None, x=None):
        """
        Learn confounder matrix with this model.

        **Parameters**:

        ard_indices : [indices]
            If you provided an own lvm_covariance, give the ard indices of the covariance here,
            to be able to use the correct hyperparameters for calculating the confounder covariance matrix.
        """
        if ard_indices is None:
            ard_indices = slice(0, self.q)
        self._check_data()
        likelihood = GaussLikISO()

        Y = self.Y.reshape(numpy.prod(self.n * self.r * self.t), self.Y.shape[3])
        # p = PCA(Y)
        # try:
        #    self.X = p.project(Y, self.q)
        # except IndexError:
        #    raise IndexError("More confounder components then genes (q > d)")

        if x is None:
            x = numpy.concatenate((self.T.reshape(-1, 1), self.X,
                                   #self.X_r,
                                   self.X_s),
                                  axis=1)

        g = GPLVM(gplvm_dimensions=xrange(1, 1 + self.q),
                  covar_func=self._lvm_covariance,
                  likelihood=likelihood,
                  x=x,
                  y=Y)

        hyper = {
                 'lik':numpy.log([.15]),
                 'x':self.X,
                 'covar':numpy.zeros(self._lvm_covariance.get_number_of_parameters())
                 }

        lvm_hyperparams, _ = opt_hyper(g, hyper,
                                       Ifilter=None, maxiter=10000,
                                       gradcheck=False, bounds=None,
                                       messages=True,
                                       gradient_tolerance=1E-12)

        self._init_conf_matrix(lvm_hyperparams, ard_indices)
        # print "%s found optimum of F=%s" % (threading.current_thread().getName(), opt_f)

    def set_data(self, T, Y):
        """
        Set data by time T and expression matrix Y:

        **Parameters**:

        T : real [n x r x t]
            All Timepoints with shape [Samples x Replicates x Timepoints]

        Y : real [n x r x t x d]
            All expression values given in the form: [Samples x Replicates x Timepoints x Genes]
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
        assert T.shape == Y.shape[:3], 'Shape mismatch, must be n*r*t timepoints per gene.'

    def predict_likelihoods(self,
                            indices=None,
                            message="Predicting Likelihoods: ",
                            messages=False,
                            priors=None,
                            **kwargs):
        """
        Predict all likelihoods for all genes, given in Y

        **parameters**:

        indices : [int]
            list (or array-like) for gene indices to predict, if None all genes will be predicted

        message: str
            printing message

        kwargs: {...}
            kwargs for :py:meth:`gptwosample.twosample.GPTwoSampleBase.predict_model_likelihoods`
        """
        self._check_data()
        assert self._initialized, "confounder matrix not yet learned, use learn_confounder_matrix() first"
        if indices is None:
            indices = xrange(self.d)

        kwargs['messages'] = messages

        self.outq = Queue(3)
        self.inq = Queue(3)

        self._likelihoods = list()
        self._hyperparameters = list()
        processes = list()

#        l = len(indices)
#        count = itertools.count()
#        countlock = Condition()
#        results = list()
#        appendlock = Condition()
#        twosamplelock = Condition()
#
#        def worker(indices_inner, **kwargs):
#            def inner(i, x, **kwargs):
#                with twosamplelock:
#                    twosample = self._TwoSampleObject(priors=priors)
#                    twosample.set_data_by_xy_data(*x)
#                lik = deepcopy(twosample.predict_model_likelihoods(messages=False, **kwargs))
#                hyp = deepcopy(twosample.get_learned_hyperparameters())
#                with countlock:
#                    k = count.next()
#                sys.stdout.flush()
#                sys.stdout.write("{1:s} {2}/{3} {0:.3%}             \r".format(float(k + 1) / l, message, k + 1, l))
#                return i, lik, hyp
#            tmp = [inner(i, self._get_data_for(i)) for i in indices_inner]
#            with appendlock:
#                results.extend(tmp)
#                
#        for indices_inner in numpy.array_split(indices, self.NUM_PROCS):
#            processes.append(Thread(target=worker, args=[indices_inner], kwargs=kwargs))
#        for p in processes:
#            p.start()
#        for p in processes:
#            p.join()
#
#        out_indices, liks, hyps = zip(*results)
#        out_indices = numpy.array(out_indices)
#        sorting = numpy.where(numpy.atleast_2d(out_indices) == numpy.atleast_2d(indices).T)
#
#        self._likelihoods = numpy.array(liks)[sorting[1]]
#        self._hyperparameters = numpy.array(hyps)[sorting[1]]

#        try:
#            sys.stdout.write(message + " " + '\033[92m' + u"\u2713" + '\033[0m' + '                         \n')
#        except:
#            sys.stdout.write(message + " done                      \n")
#
#        assert (out_indices[sorting[1]] == indices).all()

        def distribute(i):
            self.inq.put([i, deepcopy(self._get_data_for(indices[i]))])

        processes.append(Thread(target=self._distributor, args=[distribute, len(indices)], name='distributor', verbose=self.__verbose))

        def collect(d):
            self._likelihoods.append(d[0])
            self._hyperparameters.append(d[1])

        processes.append(Thread(target=self._collector, name='lik collector', args=[collect, message, len(indices)], verbose=self.__verbose))

        for i in xrange(self.NUM_PROCS):
            processes.append(Thread(target=self._lik_worker,
                                    name='lik worker %i' % i,
                                    args=[self._TwoSampleObject(),
                                          priors],
                                    kwargs=kwargs,
                                    verbose=self.__verbose))


        sys.stdout.write(message + '                 \r')
        self._main_loop(processes)

        return self._likelihoods

    def predict_means_variances(self,
                                interpolation_interval,
                                indices=None,
                                message="Predicting means and variances: ",
                                *args, **kwargs):
        """
        Predicts means and variances for all genes given in Y for given interpolation_interval
        """
        if indices is None:
            indices = range(self.d)
        self._mean_variances = list()
        self._interpolation_interval_cache = get_model_structure(interpolation_interval)
        self.inq = Queue(3)
        self.outq = Queue(3)

        try:
            if self._hyperparameters is None:
                raise ValueError()
        except:
            print "likelihoods not yet predicted, running predict_model_likelihoods..."
            self.predict_likelihoods()

        processes = list()

        def distribute(i):
            self.inq.put([i, deepcopy([self._get_data_for(indices[i]),
                              self._hyperparameters[i]])])

        processes.append(Thread(target=self._distributor, args=[distribute, len(indices)], name='distributor', verbose=self.__verbose))

        def collect(d):
            self._mean_variances.append(deepcopy(d))

        processes.append(Thread(target=self._collector, name='ms collector', args=[collect, message, len(indices)], verbose=self.__verbose))

        for i in xrange(self.NUM_PROCS):
            processes.append(Thread(target=self._pred_worker,
                                    name='lik worker %i' % i,
                                    args=[self._TwoSampleObject(),
                                          interpolation_interval],
                                    kwargs=kwargs,
                                    verbose=self.__verbose))

        sys.stdout.write(message + '                 \r')
        self._main_loop(processes)

        return self._mean_variances

    def bayes_factors(self, likelihoods = None):
        """
        get list of bayes_factors for all genes.

        **returns**:
            bayes_factor for each gene in Y
        """
        if likelihoods is None:
            likelihoods = self._likelihoods
        t = TwoSampleBase()
        f = lambda lik:t.bayes_factor(lik)
        return map(f, self._likelihoods)
        

    def plot(self, indices,
             xlabel="input", ylabel="ouput", title=None,
             interval_indices=None, alpha=None, legend=True,
             replicate_indices=None, shift=None, *args, **kwargs):
        """
        iterate through all genes and plot
        """
        try:
            self._interpolation_interval_cache
            if self._interpolation_interval_cache is None:
                raise Exception()
        except:
            print "Not yet predicted, use predict_means_variances before"
            return
        pylab.ion()
        t = self._TwoSampleObject()
        for i in xrange(len(indices)):
            pylab.clf()
            t.set_data_by_xy_data(*self._get_data_for(indices[i]))
            t._predicted_mean_variance = self._mean_variances[i]
            t._interpolation_interval_cache = self._interpolation_interval_cache
            t._learned_hyperparameters = self._hyperparameters[i]
            t._model_likelihoods = self._likelihoods[i]
            yield t.plot(xlabel, ylabel, title, interval_indices,
                   alpha, legend, replicate_indices, shift,
                   *args, **kwargs)

    def get_model_likelihoods(self):
        return self._likelihoods

    def get_learned_hyperparameters(self):
        return self._hyperparameters
        # raise ValueError("Hyperparameters are not saved due to memory issues")

    def get_predicted_means_variances(self):
        return self._mean_variances

    def _invalidate_cache(self):
        self._likelihoods = None
        self._mean_variances = None
        self._hyperparameters = None

    def _check_data(self):
        try:
            self.T.T
            self.Y.T
        except ValueError:
            raise ValueError("Data has not been set or is None, use set_data(Y,T) to set data")

    def _get_data_for(self, i):
        return self.T[0, :, :].ravel()[:, None], \
            self.T[1, :, :].ravel()[:, None], \
            self.Y[0, :, :, i].ravel()[:, None], \
            self.Y[1, :, :, i].ravel()[:, None]

    def _init_conf_matrix(self, lvm_hyperparams, ard_indices):
        self._initialized = True
        self.X = lvm_hyperparams['x']
        self._lvm_hyperparams = lvm_hyperparams
        if ard_indices is None:
            ard_indices = numpy.arange(0, self.q)
        ard = self._lvm_covariance.get_reparametrized_theta(lvm_hyperparams['covar'])[ard_indices]
        self.K_conf = numpy.dot(self.X * ard, self.X.T)

    def _TwoSampleObject(self, priors=None):
        covar_common = SumCF([SqexpCFARD(1), FixedCF(self.K_conf.copy()), BiasCF()])
        covar_individual_1 = SumCF([SqexpCFARD(1), FixedCF(self.K_conf[:self.r * self.t, :self.r * self.t].copy()), BiasCF()])
        covar_individual_2 = SumCF([SqexpCFARD(1), FixedCF(self.K_conf[self.r * self.t:, self.r * self.t:].copy()), BiasCF()])

        return TwoSampleSeparate(covar_individual_1, covar_individual_2,
                                 covar_common, priors=priors)

    def _collector(self, collect, message, l):
        counter = itertools.count()
        try:
            cur = 0
            buff = {}
            # Keep running until we see numprocs SENTINEL messages
            for _ in xrange(self.NUM_PROCS):
                for i, d in iter(self.outq.get, self.SENTINEL):
                    k = counter.next()
                    sys.stdout.flush()
                    sys.stdout.write("{1:s} {2}/{3} {0:.3%}             \r".format(float(k + 1) / l, message, k + 1, l))
                    if not self.__running_event.is_set():
                        continue
                    # verify rows are in order, if not save in buff
                    if i != cur:
                        buff[i] = d
                    else:
                        # if yes are write it out and make sure no waiting rows exist
                        collect(d)
                        # print i, d[0]['common'] - d[0]['individual']
                        cur += 1
                        while cur in buff:
                            collect(buff[cur])
                            # print cur, buff[cur][0]['common'] - buff[cur][0]['individual']
                            del buff[cur]
                            cur += 1
                    self.outq.task_done()

            try:
                sys.stdout.write(message + " " + '\033[92m' + u"\u2713" + '\033[0m' + '                         \n')
            except:
                sys.stdout.write(message + " done                      \n")
        except:
            print "ERROR: Caught Exception in _collector"
            raise

    def _distributor(self, main, l):
        try:
            for i in xrange(l):
                if not self.__running_event.is_set():
                    break
                main(i)
                # time.sleep(min(.3,2./float(NUM_PROCS)))
        except:
            print "ERROR: Caught Exception in _distributor"
            raise
        finally:
            for _ in xrange(self.NUM_PROCS):
                self.inq.put(self.SENTINEL)

    def _lik_worker(self, twosample, priors, **kwargs):
        try:
            for i, da in iter(self.inq.get, self.SENTINEL):
                if self.__running_event.is_set():
                    numpy.random.seed(0)
                    twosample.set_data_by_xy_data(*da)
                    try:
                        lik = deepcopy(twosample.predict_model_likelihoods(**kwargs))
                        hyp = deepcopy(twosample.get_learned_hyperparameters())
                    except ValueError:
                        lik = numpy.nan
                        hyp = None
                    self.outq.put([i, [lik, hyp]])
                else:
                    continue
                self.inq.task_done()
        except:
            print "ERROR: Caught Exception in _lik_worker"
            raise
        finally:
            self.outq.put(self.SENTINEL)

    def _pred_worker(self, twosample, interpolation_interval, **kwargs):
        try:
            for i, [da, hyperparams] in iter(self.inq.get, self.SENTINEL):
                if not self.__running_event.is_set():
                    continue
                numpy.random.seed(0)
                twosample.set_data_by_xy_data(*da)
                try:
                    ms = twosample.predict_mean_variance(interpolation_interval, hyperparams=hyperparams, **kwargs).copy()
                except ValueError:
                    ms = None
                self.outq.put([i, ms])
                self.inq.task_done()
        except:
            print "ERROR: Caught Exception in _pred_worker"
            raise
        finally:
            self.outq.put(self.SENTINEL)

    def _main_loop(self, processes):
        self.__running_event.set()
        for p in processes:
            p.start()
        try:
            while len(processes) > 0:  # Some processes still running
                for p in processes:  # iter through running processes
                    if not p.is_alive():  # Process finished
                        processes.remove(p)  # can be deleted
                    else:
                        p.join(.5)  # Join in process
        except KeyboardInterrupt as r:
            sys.stdout.write("\nstopping threads                                    \r")
            self.__running_event.clear()
            while len(processes) > 0:  # Wait for all processes to terminate
                for p in processes:  # iter through running processes
                    sys.stdout.flush()
                    sys.stdout.write("stopping {} ...                                   \r".format(p.name))
                    if not p.is_alive():  # Process finished
                        processes.remove(p)  # can be deleted
                    else:
                        p.join(.2)  # Join in process
            sys.stdout.write("stopping threads... ")
            try:
                sys.stdout.write(" " + '\033[92m' + u"\u2713" + '\033[0m' + '                         \n')
            except:
                sys.stdout.write(" done                      \n")

            raise r
if __name__ == '__main__':
    Tt = numpy.arange(0, 24, 2)[:, None]
    Tr = numpy.tile(Tt, 4).T
    Ts = numpy.array([Tr, Tr])

    n, r, t, d = nrtd = Ts.shape + (20,)

    covar = SqexpCFARD(1)
    K = covar.K(covar.get_de_reparametrized_theta([1, 13]), Tt)
    m = numpy.zeros(t)

    try:
        from scikits.learn.mixture import sample_gaussian
    except _ as r:
        r.message = "scikits needed for this example"
        raise r
    Y1 = sample_gaussian(m, K, cvtype='full', n_samples=d)
    Y2 = sample_gaussian(m, K, cvtype='full', n_samples=d)

    Y = numpy.zeros(nrtd)

    sigma = .5
    Y[0, :, :, :] = Y1 + sigma * numpy.random.randn(r, t, d)
    Y[1, :, :, :] = Y2 + sigma * numpy.random.randn(r, t, d)

    c = ConfounderTwoSample(Ts, Y)
    c.__verbose = True

    lvm_hyperparams_file_name = 'lvm_hyperparams.pickle'

#    c.learn_confounder_matrix()
#    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'w')
#    pickle.dump(c._lvm_hyperparams, lvm_hyperparams_file)
#    lvm_hyperparams_file.close()

    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'r')
    c._init_conf_matrix(pickle.load(lvm_hyperparams_file), None)
    lvm_hyperparams_file.close()

    c.predict_likelihoods()

    c.predict_means_variances(numpy.linspace(0, 24, 100))

    for _ in c.plot():
        raw_input("enter to continue")

