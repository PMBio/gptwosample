'''
Confounder Learning and Correction Module
-----------------------------------------

@author: Max Zwiessele
'''
import numpy
from pygp.covar.combinators import SumCF
from pygp.covar.se import SqexpCFARD
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
from gptwosample.twosample.twosample_base import TwoSampleSeparate, \
    TwoSampleBase
from Queue import Queue
import sys
from threading import Thread, Event
from pygp.covar.bias import BiasCF
import itertools
from copy import deepcopy

class TwoSample(object):
    """Run GPTwoSample on given data.

    **Parameters**:
        - T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
        - Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
    
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
    NUM_PROCS = 1  # min(max(1,cpu_count()-2),3)
    SENTINEL = object()
    def __init__(self, T, Y,
                 covar_common=None,
                 covar_individual_1=None,
                 covar_individual_2=None):
        """
        **Parameters**:
            T : TimePoints [n x r x t]    [Samples x Replicates x Timepoints]
            Y : ExpressionMatrix [n x r x t x d]      [Samples x Replicates x Timepoints x Genes]
        """
        self.set_data(T, Y)
        self.__verbose = False
        self.__running_event = Event()
        self.covar_comm = covar_common
        self.covar_ind1 = covar_individual_1
        self.covar_ind2 = covar_individual_2

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
                            T,
                            Y,
                            # indices=None,
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

        kwargs['messages'] = messages

        self.outq = Queue(3)
        self.inq = Queue(3)

        self._likelihoods = list()
        self._hyperparameters = list()
        processes = list()

        def distribute(i):
            self.inq.put([i,
                          [T[0, :, :].ravel()[:, None],
                          T[1, :, :].ravel()[:, None],
                          Y[0, :, :, i].ravel()[:, None],
                          Y[1, :, :, i].ravel()[:, None]
                          ]])

        processes.append(Thread(target=self._distributor,
                                args=[distribute, Y.shape[-1]],
                                name='distributor',
                                verbose=self.__verbose))

        def collect(d):
            self._likelihoods.append(d[0])
            self._hyperparameters.append(d[1])

        processes.append(Thread(target=self._collector,
                                name='lik collector',
                                args=[collect, message, Y.shape[-1]],
                                verbose=self.__verbose))

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
        self.indices = indices
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
            self.inq.put([i, deepcopy([self._get_data_for(self.indices[i]),
                              self._hyperparameters[i]])])

        processes.append(Thread(target=self._distributor, args=[distribute, len(self.indices)], name='distributor', verbose=self.__verbose))

        def collect(d):
            self._mean_variances.append(deepcopy(d))

        processes.append(Thread(target=self._collector, name='ms collector', args=[collect, message, len(self.indices)], verbose=self.__verbose))

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

    def bayes_factors(self, likelihoods=None):
        """
        get list of bayes_factors for all genes.

        **returns**:
            bayes_factor for each gene in Y
        """
        if likelihoods is None:
            likelihoods = self._likelihoods
        if likelihoods is None:
            raise RuntimeError("Likelihoods not yet learned, use predict_likelihoods first")
        t = TwoSampleBase()
        f = lambda lik:t.bayes_factor(lik)
        return map(f, self._likelihoods)


    def plot(self,
             xlabel="input", ylabel="ouput", title=None,
             interval_indices=None, alpha=None, legend=True,
             replicate_indices=None, shift=None, timeshift=False, *args, **kwargs):
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
        import pylab
        pylab.ion()
        t = self._TwoSampleObject()
        for i in xrange(len(self.indices)):
            pylab.clf()
            t.set_data_by_xy_data(*self._get_data_for(self.indices[i]))
            t._predicted_mean_variance = self._mean_variances[i]
            t._interpolation_interval_cache = self._interpolation_interval_cache
            t._learned_hyperparameters = self._hyperparameters[i]
            t._model_likelihoods = self._likelihoods[i]
            draw_arrows=None
            if timeshift:
                shift = t.get_covars()[common_id].get_timeshifts(t.get_learned_hyperparameters()[common_id]['covar'])
                draw_arrows=2
            yield t.plot(xlabel, ylabel, title, interval_indices,
                   alpha, legend, replicate_indices, shift,draw_arrows=draw_arrows,
                   *args, **kwargs)

    def get_model_likelihoods(self):
        return self._likelihoods

    def get_learned_hyperparameters(self):
        return self._hyperparameters
        # raise ValueError("Hyperparameters are not saved due to memory issues")

    def get_predicted_means_variances(self):
        return self._mean_variances

    def get_twosample(self, priors=None):
        return self._TwoSampleObject(priors)

    def get_covars(self):
        models = self.get_twosample()._models
        return {individual_id: models[individual_id].covar, common_id: models[common_id].covar}

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

    def _TwoSampleObject(self, priors=None):
        if self.covar_comm is None:
            self.covar_comm = SumCF((SqexpCFARD(1), BiasCF()))
        if self.covar_ind1 is None:
            self.covar_ind1 = SumCF((SqexpCFARD(1), BiasCF()))
        if self.covar_ind2 is None:
            self.covar_ind2 = SumCF((SqexpCFARD(1), BiasCF()))
        covar_common = deepcopy(self.covar_comm)
        covar_individual_1 = deepcopy(self.covar_ind1)
        covar_individual_2 = deepcopy(self.covar_ind2)
        tmp = TwoSampleSeparate(covar_individual_1, covar_individual_2,
                                 covar_common, priors=priors)
        return tmp

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
    Tt = numpy.arange(0, 16, 2)[:, None]
    Tr = numpy.tile(Tt, 3).T
    Ts = numpy.array([Tr, Tr], dtype=float)

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

    n, r, t, d = Y.shape

    for _ in range((n * r * t * d) / 4):
        # Ts[numpy.random.randint(n),numpy.random.randint(r),numpy.random.randint(t)] = numpy.nan
        Y[numpy.random.randint(n), numpy.random.randint(r), numpy.random.randint(t), numpy.random.randint(d)] = numpy.nan



    c = TwoSample(Ts, Y)

    c.predict_likelihoods(Ts, Y)
    c.predict_means_variances(numpy.linspace(Ts.min(), Ts.max(), 100))
    
    import pylab
    pylab.ion()
    pylab.figure()
    for _ in c.plot():
        raw_input("enter to continue")

