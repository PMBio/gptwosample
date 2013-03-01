'''
Created on Sep 14, 2011

@author: maxz
'''
from pygp.covar import se, noise
from pygp.priors import lnpriors
from pygp.covar.fixed import FixedCF
from gptwosample.data.data_base import get_model_structure, common_id, \
    individual_id
import os
import csv
import cPickle as pickle
from gptwosample.plot.plot_basic import plot_results
from numpy.linalg.linalg import cholesky
import logging
import pylab
from gptwosample.data.data_analysis import plot_roc_curve
import sys
import numpy
from pygp.gp import gplvm
from pygp.covar.combinators import SumCF
from gptwosample.data.dataIO import get_data_from_csv
from gptwosample.twosample.twosample_base import GPTwoSample_individual_covariance, \
    GPTwoSample_share_covariance
from gptwosample.develop.reveal_confounders_proof_of_concept.gplvm_models import conditional_linear_gplvm_confounder, \
    linear_gplvm_confounder, time_linear_gplvm_confounder
import itertools
from threading import Thread, Lock, Event
from Queue import Queue
from multiprocessing import cpu_count
import time

# Private variables:
__debug = 1
NUM_CPUS = cpu_count()
STOP = "STOP"

def run_demo(cond1_file, cond2_file, components=4, root='.', data='data'):
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(data):
        os.makedirs(data)
    plots_out_dir = os.path.join(root, "plots")
    if not os.path.exists(plots_out_dir):
        os.mkdir(plots_out_dir)

    D='all'
    for arg in sys.argv:
        if arg.startswith("D="):
            D=int(arg.split('=')[1])            
    
    gt_file_name = "../../examples/ground_truth_random_genes.csv"
    if "GT100" in sys.argv:
        gt_file_name = "../../examples/ground_truth_balanced_set_of_100.csv"
    
    if not os.path.exists(os.path.join(data, "toy_data.pickle")):
        read_files_and_pickle(cond1_file, cond2_file, gt_file_name, data, D=D)
        
    data_file_name = os.path.join(data, "toy_data.pickle")
    (Y, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
     n_replicates, gene_length, T) = pickle.load(open(data_file_name, "r"))
    if 0: import ipdb;ipdb.set_trace()
    print "finished loading data"

    conf_file_name = os.path.join(data, "toy_data_sim_conf.pickle")
    if not os.path.exists(conf_file_name):
        simulated_confounders, X_sim = sample_confounders_linear(components, gene_names, n_replicates, gene_length)
        conf_file = open(conf_file_name, 'w')
        pickle.dump([simulated_confounders, X_sim], conf_file, protocol=0)
    else:
        conf_file = open(conf_file_name, 'r')
        simulated_confounders, X_sim = pickle.load(conf_file)
    conf_file.close()
    K_sim = numpy.dot(X_sim, X_sim.T)
    print "finished simulating"

    # Get variances right:
    Y = Y - Y.mean(1)[:, None]
    Y = Y / Y.std(1)[:, None]
    # simulated_confounders = simulated_confounders-simulated_confounders.mean(1)[:,None]
    # simulated_confounders = simulated_confounders/simulated_confounders.std(1)[:,None]
    Y_confounded = Y + simulated_confounders

    # from now on Y matrices are transposed:
    Y = Y.T
    Y_confounded = Y_confounded.T
    simulated_confounders = simulated_confounders.T

    K_conf_file_name = os.path.join(root, "toy_data_conf_K.pickle")
    if not os.path.exists(K_conf_file_name) or "regplvm" in sys.argv:
        if "condition_model" in sys.argv:
            print "conditional model"
            gplvm_model_function = conditional_linear_gplvm_confounder
        elif "time_model" in sys.argv:
            print "time model"
            gplvm_model_function = time_linear_gplvm_confounder
        else:
            print "linear model"
            gplvm_model_function = linear_gplvm_confounder
        K_learned, hyperparams_gplvm, gplvm_model = gplvm_model_function(Y_confounded, T, components)
        K_conf_file = open(K_conf_file_name, 'w')
        pickle.dump([K_learned, hyperparams_gplvm, gplvm_model], K_conf_file, 2)
    else:
        K_conf_file = open(K_conf_file_name, 'r')
        K_learned, hyperparams_gplvm, gplvm_model = pickle.load(K_conf_file)
    K_conf_file.close()
    print "finished loading GPLVM"

    if __debug and 'ideal' in sys.argv:
        covar_ideal = FixedCF(K_sim)
        X_ideal = numpy.concatenate((T.copy().reshape(-1, 1), X_sim.copy()), axis=1)
        g_ideal = gplvm.GPLVM(gplvm_dimensions=xrange(1, 5), covar_func=covar_ideal, x=X_ideal, y=Y_confounded)
        hyper_ideal = {
                       'covar':numpy.zeros(1),
                       'x':X_sim
                       }

        M_ideal = numpy.zeros_like(Y)
        S_ideal = numpy.zeros_like(Y)

        for i in xrange(Y.shape[1]):
            sys.stdout.flush()
            sys.stdout.write("predicting: {}/{}".format(i, Y.shape[1]))
            mi, si = g_ideal.predict(hyper_ideal, gplvm_model.x, output=i)
            M_ideal[:, i] = mi
            S_ideal[:, i] = si
            sys.stdout.write("\r")
        sys.stdout.flush()

        fig = pylab.figure(figsize=(20, 8))
        fig.suptitle("MSD={0:.5g}".format(((M_ideal - simulated_confounders) ** 2).mean()))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(simulated_confounders)
        ax1.set_title("simulated confounders")

        ax2.imshow(M_ideal)
        ax2.set_title("ideally predicted confounders, meanvar={0:.3g}".format(S_ideal.mean()))
        pylab.draw()

        print "ideal GPLVM loaded and predicted"
        del covar_ideal, X_ideal, g_ideal, hyper_ideal, M_ideal, S_ideal, fig, ax1, ax2


#    Y_subtract_file_name = os.path.join(root,"toy_data_predicted_confounders.pickle")
#    if "repredict" in sys.argv or not os.path.exists(Y_subtract_file_name):
#        # Predict confounder matrix:
#        M = numpy.zeros_like(Y)
#        S = numpy.zeros_like(Y)
#        for i in xrange(Y.shape[1]):
#            sys.stdout.flush()
#            sys.stdout.write("predicting: {}/{}".format(i,Y.shape[1]))
#            mi, si = gplvm_model.predict(hyperparams_gplvm, gplvm_model.x, output=i)
#            M[:,i] = mi
#            S[:,i] = si
#            sys.stdout.write("\r")
#        Y_subtracted = Y-M
#        Y_subtract_file = open(Y_subtract_file_name, 'w')
#        pickle.dump(M, Y_subtract_file)
#    else:
#        Y_subtract_file = open(Y_subtract_file_name, 'r')
#        M = pickle.load(Y_subtract_file)
#        Y_subtracted = Y-M
#    Y_subtract_file.close()
#    print "finished predicting"
#
    # get Y values for all genes
    Y_dict = dict([[name, {
                           # 'subtracted': Y_subtracted[:,i],
                           'confounded':Y_confounded[:, i],
                           'raw':Y[:, i]
                           }] for i, name in enumerate(gene_names)])

    MSD = ((K_sim - K_learned) ** 2).mean()

    if __debug and "plot_confounder" in sys.argv:
        pylab.ion()
        pylab.close('all')
        fig = pylab.figure()
        im = pylab.imshow(K_sim)
        pylab.title("Simulated Covariance Matrix:")
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(pylab.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        pylab.colorbar(im, cax=cax)
        try:
            fig.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(root, "plots", "simulated_confounder_matrix.pdf"))

        fig = pylab.figure()
        im = pylab.imshow(K_learned)
        pylab.title("Predicted Confounder Matrix:")
        divider = make_axes_locatable(pylab.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        pylab.colorbar(im, cax=cax)
        try:
            fig.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(root, "plots", "predicted_confounder_matrix.pdf"))

        fig = pylab.figure()
        im = pylab.imshow(K_sim - K_learned)
        pylab.title("Difference: MSD={0:.4G}".format(MSD))
        divider = make_axes_locatable(pylab.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        pylab.colorbar(im, cax=cax)
        try:
            fig.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(root, "plots", "difference_confounder_matrix.pdf"))

    print "mean squared distance: {0:.3G}".format(MSD)
    print "Y.var()={}, Conf.var()={}".format(Y.var(), simulated_confounders.var())

    out_conf_file_name = os.path.join(root, "conf.csv")
    out_normal_file_name = os.path.join(data, "normal.csv")
    out_raw_file_name = os.path.join(data, "raw.csv")
    out_ideal_file_name = os.path.join(data, "ideal.csv")

    conf = not os.path.exists(out_conf_file_name)
    redata = "redata" in sys.argv
    normal = not os.path.exists(out_normal_file_name) or redata
    raw = not os.path.exists(out_raw_file_name) or redata
    ideal = not os.path.exists(out_ideal_file_name) or redata

    NUM_PROCS = max(1, NUM_CPUS-(conf+normal+raw+ideal))

    dim = 1
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()

    priors_normal = lambda: get_priors(dim, confounders=False)
    priors_conf = lambda: get_priors(dim, confounders=True)

    covar_normal = lambda: SumCF((SECF, noiseCF))
    covar_conf_common = lambda: SumCF((SECF, FixedCF(K_learned), noiseCF))

    r1t = T1.shape[0] * n_replicates_1
    covar_conf_r1 = lambda: SumCF((SECF, FixedCF(K_learned[:r1t, :r1t]), noiseCF))
    covar_conf_r2 = lambda: SumCF((SECF, FixedCF(K_learned[r1t:, r1t:]), noiseCF))

    if ideal:
        covar_ideal_common = lambda: SumCF((SECF, FixedCF(K_sim), noiseCF))
        covar_ideal_r1 = lambda: SumCF((SECF, FixedCF(K_sim[:r1t, :r1t]), noiseCF))
        covar_ideal_r2 = lambda: SumCF((SECF, FixedCF(K_sim[r1t:, r1t:]), noiseCF))

    T1 = numpy.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = numpy.tile(T2, n_replicates_2).reshape(-1, 1)

#    out_predict_file_name = os.path.join(results_out_dir,"predict.csv")

    try:
        run_event = Event()
        run_event.set()
        
        if "retwosample" in sys.argv \
            or conf or raw or normal or ideal:
            # or not os.path.exists(out_predict_file_name):
            # get csv files to write to
    
            if conf:
                out_conf_file = open(out_conf_file_name, 'w')
                write_header(covar_conf_common, covar_conf_r1, out_conf_file)
            if normal:
                out_normal_file = open(out_normal_file_name, 'w')
                write_header(covar_normal, covar_normal, out_normal_file)
            if raw:
                out_raw_file = open(out_raw_file_name, 'w')
                write_header(covar_normal, covar_normal, out_raw_file)
            if ideal:
                out_ideal_file = open(out_ideal_file_name, 'w')
                write_header(covar_conf_common, covar_conf_r1, out_ideal_file)
    
            # get ground truth genes for comparison:
            gt_names = []
            gt_file = open(gt_file_name, 'r')
            for [name, _] in csv.reader(gt_file):
                gt_names.append(name)
            gt_file.close()
            lgt_names = len(gt_names)
    
            # loop through genes asynchronously
            if conf:
                out_conf_queue, out_conf_process = setup_queue(out_conf_file, "conf")
            if normal:
                out_normal_queue, out_normal_process = setup_queue(out_normal_file, "normal")
            if raw:
                out_raw_queue, out_raw_process = setup_queue(out_raw_file, "raw")
            if ideal:
                out_ideal_queue, out_ideal_process = setup_queue(out_ideal_file, "ideal")
    
            # out_normal_process = Process(target=write_csv_file, name="normal_write_out", args=(out_normal_file,out_normal_queue))
            iter_lock = Lock()
            gene_names_iter = iter(gt_names)
            current_iter = itertools.count()
    
            def gptwosample_multiprocess(name):
                while run_event.is_set():
                    try:
                        iter_lock.acquire()
                        gene_name = gene_names_iter.next()
                        current = current_iter.next()
                    except StopIteration:
                        if conf:
                            out_conf_queue.put(STOP)
                        if normal:
                            out_normal_queue.put(STOP)
                        if raw:
                            out_raw_queue.put(STOP)
                        if ideal:
                            out_ideal_queue.put(STOP)
                        return
                    finally:
                        iter_lock.release()
                    if gene_name is "input":
                        continue
                    gene_name = gene_name.upper()
                    if gene_name in Y_dict.keys():
                        sys.stdout.flush()
                        sys.stdout.write('processing {0:s} {1:.3%}                  \r'.format(gene_name, float(current) / lgt_names))
                        inner_processes = []
                        if conf:
                            def gptwosample_fun():
                                twosample_object_conf = GPTwoSample_individual_covariance(covar_conf_r1(), covar_conf_r2(), covar_conf_common(), priors=priors_conf())
                                run_gptwosample_on_data(twosample_object_conf, Tpredict, T1, T2,
                                                        n_replicates_1, n_replicates_2,
                                                        Y_dict[gene_name]['confounded'][:len(T1)],
                                                        Y_dict[gene_name]['confounded'][len(T1):],
                                                        gene_name, os.path.join(plots_out_dir, gene_name + "_conf"))
                                out_conf_queue.put([twosample_object_conf, gene_name])
                            inner_processes.append(Thread(target=gptwosample_fun(), name="inner conf"))
                        if normal:
                            def gptwosample_fun():
                                twosample_object_normal = GPTwoSample_share_covariance(covar_normal(), priors=priors_normal())
                                run_gptwosample_on_data(twosample_object_normal, Tpredict, T1, T2,
                                                        n_replicates_1, n_replicates_2,
                                                        Y_dict[gene_name]['confounded'][:len(T1)],
                                                        Y_dict[gene_name]['confounded'][len(T1):],
                                                        gene_name, os.path.join(plots_out_dir, gene_name + "_normal"))
                                out_normal_queue.put([twosample_object_normal, gene_name])
                            inner_processes.append(Thread(target=gptwosample_fun(), name="inner normal"))
                        if raw:
                            def gptwosample_fun():
                                twosample_object_raw = GPTwoSample_share_covariance(covar_normal(), priors=priors_normal())
                                run_gptwosample_on_data(twosample_object_raw, Tpredict, T1, T2,
                                                        n_replicates_1, n_replicates_2,
                                                        Y_dict[gene_name]['raw'][:len(T1)],
                                                        Y_dict[gene_name]['raw'][len(T1):],
                                                        gene_name, os.path.join(plots_out_dir, gene_name + "_raw"))
                                out_raw_queue.put([twosample_object_raw, gene_name])
                            inner_processes.append(Thread(target=gptwosample_fun(), name="inner raw"))
                        if ideal:
                            def gptwosample_fun():
                                twosample_object_ideal = GPTwoSample_individual_covariance(covar_ideal_r1(), covar_ideal_r2(), covar_ideal_common(), priors=priors_conf())
                                run_gptwosample_on_data(twosample_object_ideal, Tpredict, T1, T2,
                                                        n_replicates_1, n_replicates_2,
                                                        Y_dict[gene_name]['confounded'][:len(T1)],
                                                        Y_dict[gene_name]['confounded'][len(T1):],
                                                        gene_name, os.path.join(plots_out_dir, gene_name + "_ideal"))
                                out_ideal_queue.put([twosample_object_ideal, gene_name])
                            inner_processes.append(Thread(target=gptwosample_fun(), name="inner ideal"))
                        for p in inner_processes:
                            p.start()
                        for p in inner_processes:
                            p.join()
                else:
                    if conf:
                        out_conf_queue.put(STOP)
                    if normal:
                        out_normal_queue.put(STOP)
                    if raw:
                        out_raw_queue.put(STOP)
                    if ideal:
                        out_ideal_queue.put(STOP)
                    return
            # pool = Pool(processes=8)
            # pool.map(gptwosample_multiprocess, gt_names)
    
            processes = []
            if conf:
                processes.append(out_conf_process)
            if normal:
                processes.append(out_normal_process)
            if raw:
                processes.append(out_raw_process)
            if ideal:
                processes.append(out_ideal_process)
    
            for num in range(NUM_PROCS):
                processes.append(Thread(target=gptwosample_multiprocess, args=[num], name=str(num)))
    
            for p in processes:
                p.start()
            for p in processes:
                while p.is_alive():
                    time.sleep(1)
                
    except KeyboardInterrupt as k:
        print "caught keyboard interrupt, stopping threads..."
        run_event.clear()
        for p in processes:
            while p.is_alive():
                p.join()
            print "stopped", p.name
        print "exited without errors"
        raise k
    finally:
        if conf:
            out_conf_file.close()
        if normal:
            out_normal_file.close()
        if raw:
            out_raw_file.close()
        if ideal:
                out_ideal_file.close()

    if "plot_roc" in sys.argv:
        pylab.figure()
        plot_roc_curve(out_conf_file_name, gt_file_name, label="conf")
        plot_roc_curve(out_normal_file_name, gt_file_name, label="normal")
        plot_roc_curve(out_raw_file_name, gt_file_name, label="raw")
        plot_roc_curve(out_ideal_file_name, gt_file_name, label="ideal")
        pylab.legend()
        try:
            pylab.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(plots_out_dir,'roc.pdf'))

def setup_queue(out_file, name):
    out_conf_queue = Queue()
    out_conf_process = Thread(target=write_csv_file_asynchronous, name="{}_write_out".format(name), args=(out_file, out_conf_queue))
    return out_conf_queue, out_conf_process

def write_csv_file_asynchronous(out_file, q):
    for _ in range(NUM_PROCS):
        for twosample_object, gene_name in iter(q.get, "STOP"):
            # print gene_name, twosample_object.bayes_factor()
            write_back_data(twosample_object, gene_name, out_file)
            q.task_done()

def write_header(covar_conf_common, covar_conf_r1, out_conf_file):
    first_line = ["gene name", "bayes factor"]
    first_line.extend(map(lambda x:"Common: " + x, covar_conf_common().get_hyperparameter_names()))
    first_line.extend(map(lambda x:"Individual: " + x, covar_conf_r1().get_hyperparameter_names()))
    out_conf_file.write(",".join(first_line) + os.linesep)

def write_back_data(twosample_object, gene_name, csv_out):
    line = [gene_name, twosample_object.bayes_factor()]
    try:
        twosample_object.covar_common
        common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
        common = twosample_object.covar_common.get_reparametrized_theta(common)
        individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
        individual = twosample_object.covar_individual_1.get_reparametrized_theta(individual)
        line.extend(common)
        line.extend(individual)
    except:
        common = twosample_object.get_learned_hyperparameters()[common_id]['covar']
        common = twosample_object.covar.get_reparametrized_theta(common)
        individual = twosample_object.get_learned_hyperparameters()[individual_id]['covar']
        individual = twosample_object.covar.get_reparametrized_theta(individual)
        line.extend(common)
        line.extend(individual)
    csv_out.write(",".join(map(lambda x:str(x), line)) + os.linesep)

def run_gptwosample_on_data(twosample_object, Tpredict, T1, T2, n_replicates_1, n_replicates_2, Y0, Y1, gene_name, savename=None):
    # create data structure for GPTwwoSample:
    # note; there is no need for the time points to be aligned for all replicates
    # creates score and time local predictions
    twosample_object.set_data_by_xy_data(T1, T2, Y0.reshape(-1, 1), Y1.reshape(-1, 1))
    twosample_object.predict_model_likelihoods(messages=False)
    # twosample_object.predict_mean_variance(Tpredict)

    # pylab.figure(1)
    # pylab.clf()
    # plot_results(twosample_object,
    #     title='%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, twosample_object.bayes_factor()),
    # 	 shift=None,
    # 	 draw_arrows=1)
    # pylab.xlim(T1.min(), T1.max())

    # if savename is None:
    #    savename=gene_name
    # pylab.savefig("%s"%(savename))

def sample_confounders_from_GP(components, gene_names, n_replicates, gene_length, lvm_covariance, hyperparams, T):
        # or draw from a GP:
    NRT = n_replicates * gene_length
    X = numpy.concatenate((T.copy().T, numpy.random.randn(NRT, components).T)).T
    sigma = 1e-6
    Y_conf = numpy.array([numpy.dot(cholesky(lvm_covariance.K(hyperparams['covar'], X) + sigma * numpy.eye(NRT)), numpy.random.randn(NRT, 1)).flatten() for _ in range(len(gene_names))])
    return Y_conf.T

def sample_confounders_linear(components, gene_names, n_replicates, gene_length):
    NRT = n_replicates * gene_length
    X = numpy.random.randn(NRT, components)
    W = numpy.random.randn(components, len(gene_names)) * 0.5
    Y_conf = numpy.dot(X, W)
    return Y_conf.T, X

def get_priors(dim, confounders):
    covar_priors_common = []
    covar_priors_individual = []
    # scale
    covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
    covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
    for _ in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])

    if confounders:
        covar_priors_common.append([lnpriors.lnuniformpdf, [1, 1]])
        covar_priors_individual.append([lnpriors.lnuniformpdf, [1, 1]])
    # noise
    for _ in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp, [1, .5]])
        covar_priors_individual.append([lnpriors.lnGammaExp, [1, .5]])
    return get_model_structure({'covar':numpy.array(covar_priors_individual)})

def read_files_and_pickle(cond1_file, cond2_file, gt_file_name, root, D='all'):
    # 1. read csv file
    print 'reading files'
    cond1 = get_data_from_csv(cond1_file, delimiter=',')
    cond2 = get_data_from_csv(cond2_file, delimiter=",")

    # Time and prediction intervals
    Tpredict = numpy.linspace(cond1["input"].min(), cond1["input"].max(), 100)[:, numpy.newaxis]
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")

    # get replicate stuff organized
    gene_names_all = numpy.intersect1d(cond1.keys(), cond2.keys(), True).tolist()
    gene_names = list()

    n_replicates_1 = cond1[gene_names_all[0]].shape[0]
    n_replicates_2 = cond2[gene_names_all[0]].shape[0]
    n_replicates = n_replicates_1 + n_replicates_2

    gene_length = len(T1)

    T = numpy.array([numpy.vstack([T1] * n_replicates_1), numpy.vstack([T2] * n_replicates_2)])

    # data merging and stuff
    gt_names = []
    gt_file = open(gt_file_name, 'r')
    for [name, _] in csv.reader(gt_file):
        gt_names.append(name)
    gt_file.close()

    if D == 'all':
        D = len(gene_names_all)

    Y1 = numpy.zeros((D, n_replicates_1, T1.shape[0]))
    Y2 = numpy.zeros((D, n_replicates_2, T2.shape[0]))

    for i, name in enumerate(gt_names):
        gene_names_all.remove(name.upper())
        gene_names.append(name.upper())
        Y1[i] = cond1.pop(name.upper())
        Y2[i] = cond2.pop(name.upper())

    # get random entries not from ground truth, to fill until D:
    gt_len = len(gt_names)
    for i, name in enumerate(numpy.random.permutation(gene_names_all)[:D - gt_len]):
        Y1[i + gt_len] = cond1.pop(name.upper())
        Y2[i + gt_len] = cond2.pop(name.upper())
        gene_names.append(name.upper())

    Y = numpy.concatenate((Y1, Y2), 1).reshape(Y1.shape[0], -1)

    data_file_name = os.path.join(root, "toy_data.pickle")
    if not os.path.exists(root):
        os.makedirs(root)
    dump_file = open(data_file_name, "w")

    pickle.dump((Y, Tpredict, T1, T2, gene_names, n_replicates_1, n_replicates_2,
		 n_replicates, gene_length, T), dump_file, -1)
    dump_file.close()


if __name__ == '__main__':
    run_demo(cond1_file='./../../examples/warwick_control.csv', cond2_file='../../examples/warwick_treatment.csv', root=sys.argv[1])
    # run_demo(cond1_file = './../examples/ToyCondition1.csv', cond2_file = './../examples/ToyCondition2.csv')
