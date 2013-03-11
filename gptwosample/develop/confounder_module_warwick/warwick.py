'''
Created on Feb 26, 2013

@author: Max
'''
from gptwosample.data.dataIO import get_data_from_csv
import os
import cPickle
import numpy
import sys
import csv
import itertools
from gptwosample.confounder.confounder import ConfounderTwoSample
import pylab
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pygp.covar.linear import LinearCFISO, LinearCF
from pygp.covar.combinators import SumCF, ProductCF
from pygp.covar.se import SqexpCFARD
from pygp.covar.bias import BiasCF

import logging
logging.basicConfig(level=logging.CRITICAL)
del logging

_usage = """usage: python warwick.py root-dir data_dir out_name
[warwick_control-file warwick_treatment-file]
[redata|regplvm|relikelihood|plot_confounder|gt100]

warwick_control-file and warwick_treatment-file have to be given only in first run - data will be cPickled"""


Q = 4
seed = 0
conf_var = 1

stats_lines = list()

for ar in sys.argv:
    if ar.startswith("cores="):
        num_procs = int(ar.split("=")[1])
        stats_lines.append("Using ", num_procs, "workers")
        print "Using ", num_procs, "workers"
    elif ar.startswith("seed="):
        seed = int(ar.split("=")[1])
        print "seed", seed
    elif ar.startswith("Q="):
        Q = int(ar.split("=")[1])
        print "Q", seed
    elif ar.startswith(("conf_var=")):
        conf_var = int(ar.split("=")[1])
stats_lines.extend(["Q="+Q, "seed="+seed, "conf_var="+conf_var])

#try:
root = sys.argv[1]
if not os.path.exists(root):
    os.mkdir(root)
data = sys.argv[2]
if not os.path.exists(data):
    os.mkdir(data)
outname = sys.argv[3]
#except:
    #print _usage
#    sys.exit(0)

numpy.random.seed(seed)
stats_file_name = os.path.join(root, "stats.txt")
stats_file = open(stats_file_name,'w')
stats_file.writelines(stats_lines)
stats_file.flush()
del stats_lines

def finished(s, process=None):
    if process is not None:
        while p.is_alive():
            p.terminate()
            p.join(1)
    try:
        sys.stdout.write(s + " " + '\033[92m' + u"\u2713" + '\033[0m' + '            \n')
    except:
        sys.stdout.write(s + " done             \n")
    sys.stdout.flush()

def start_mill(s):
    sys.stdout.flush()
    sys.stdout.write("{}".format(s))
    sys.stdout.flush()
    sys.stdout.write("\r")
#    mill_symb = {0:'-',1:'\\',2:"|",3:'/'}
#    def mill():
#        i=-1
#        while True:
#            i = (i+1)%4
#            sys.stdout.flush()
#            sys.stdout.write("{}{}\r".format(s,mill_symb[i]))
#            time.sleep(.3)
#    p = Process(target=mill)
    # p.start()

s = "loading data..."
print s,
data_file_path = os.path.join(data, "./data_seed_" + str(seed) + ".cPickle")
if not os.path.exists(data_file_path) or "redata" in sys.argv:
    sys.stdout.write(os.linesep)
    cond1 = get_data_from_csv(sys.argv[4])  # os.path.join(root,'warwick_control.csv'))
    cond2 = get_data_from_csv(sys.argv[5])  # os.path.join(root,'warwick_treatment.csv'))
    print s + "\r",
    T1 = numpy.array(cond1.pop("input"))[:, None]
    T2 = numpy.array(cond2.pop("input"))[:, None]

    Y1 = numpy.array(cond1.values()).T.swapaxes(0, 1)
    Y2 = numpy.array(cond2.values()).T.swapaxes(0, 1)
    Y = numpy.array([Y1, Y2])

    n, r, t, d = Y.shape

    T1 = numpy.tile(T1, r).T
    T2 = numpy.tile(T2, r).T

    T = numpy.array([T1, T2])

    gene_names = cond1.keys()

    assert T.shape == Y.shape[:3]
    assert gene_names == cond2.keys()

    del T1, T2, Y1, Y2, cond1, cond2

    X_sim = numpy.random.randn(n * r * t, Q)
    # X_sim -= X_sim.mean(0)
    # X_sim /= X_sim.std(0)
    X_sim *= numpy.sqrt(conf_var)
    #si = "standardizing data ..."
    #sys.stdout.write(si + "\r")
    Y -= Y.mean(1).mean(1)[:, None, None, :]
    # Y /= Y.std()
    # Conf_sim -= Conf_sim.reshape(n*r*t,d).mean(0)
    # Conf_sim /= Conf_sim.reshape(n*r*t,d).std(0)
    #finished(si)
    
    K_sim = numpy.dot(X_sim.reshape(n * r * t, Q), X_sim.reshape(n * r * t, Q).T)
    Conf_sim = numpy.dot(X_sim, numpy.random.randn(Q, d))

    data_file = open(data_file_path, 'w')
    cPickle.dump([T, Y, gene_names, K_sim, Conf_sim, X_sim.reshape(n * r * t, Q).T], data_file)
else:
    sys.stdout.write("\r")
    data_file = open(data_file_path, 'r')
    T, Y, gene_names, K_sim, Conf_sim, X_sim = cPickle.load(data_file)
    n, r, t, d = Y.shape
finished(s)
data_file.close()
    
s = "setting up gplvm module..."
print s,
sys.stdout.write("\r")
if not ("raw" in sys.argv):
    Y = Y + Conf_sim.reshape(n, r, t, d)

q = Q
rt = r * t
X_r = numpy.zeros((n * rt, n * r))
for i in xrange(n * r):X_r[i * t:(i + 1) * t, i] = 1
rep = LinearCFISO(dimension_indices=numpy.arange(1 + q, 1 + q + (n * r)))
X_s = numpy.zeros((n * rt, n))
for i in xrange(n):X_s[i * rt:(i + 1) * rt, i] = 1
sam = LinearCFISO(dimension_indices=numpy.arange(1 + q + (n * r), 1 + q + (n * r) + n))

lvm_covariance = None
if "conf" in sys.argv:
    if "rep" in sys.argv:
        lvm_covariance = SumCF([LinearCF(dimension_indices=numpy.arange(1, 1 + q)),
                                  rep,
                                  # sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()])
        learn_name = 'rep'
    elif "sam" in sys.argv:
        lvm_covariance = SumCF([LinearCF(dimension_indices=numpy.arange(1, 1 + q)),
                                  # rep,
                                  sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()])
        learn_name = 'sam'
    else:
        lvm_covariance = SumCF([LinearCF(dimension_indices=numpy.arange(1, 1 + q)),
                                  rep,
                                  sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()])
        learn_name = 'all'
    outname = outname + "_" + learn_name

conf_model = ConfounderTwoSample(T, Y, q=Q, lvm_covariance=lvm_covariance)
conf_model.__verbose = 0
try:
    conf_model.NUM_PROCS = num_procs
except NameError:
    pass
x = numpy.concatenate((T.reshape(-1, 1), conf_model.X, X_r, X_s), axis=1)
finished(s)

lvm_hyperparams_file_name = os.path.join(root, outname + '_lvm_hyperparams.cPickle')
if "ideal" in sys.argv:
    conf_model.X = X_sim
    conf_model.K_conf = K_sim
    conf_model._initialized = True
elif "noconf" in sys.argv or "raw" in sys.argv:
    conf_model.X = numpy.zeros((conf_model.n * conf_model.r * conf_model.t, conf_model.q))
    conf_model.K_conf = numpy.dot(conf_model.X, conf_model.X.T)
    conf_model._initialized = True
elif not os.path.exists(lvm_hyperparams_file_name) or "regplvm" in sys.argv:
    s = 'learning confounder matrix... '
    p = start_mill(s)
    conf_model.learn_confounder_matrix(x=x)
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'w')
    cPickle.dump(conf_model._lvm_hyperparams, lvm_hyperparams_file)
    finished(s, process=p)
else:
    s = "loading confounder matrix..."
    print s,
    sys.stdout.write("\r")
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'r')
    conf_model._init_conf_matrix(cPickle.load(lvm_hyperparams_file), None)
    finished(s)
try:
    lvm_hyperparams_file.close()
except:
    pass

if "plot_confounder" in sys.argv:
    fig = pylab.figure()
    im = pylab.imshow(K_sim)
    pylab.title("Simulated")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname+"simulated.pdf"))

    fig = pylab.figure()
    im = pylab.imshow(conf_model.K_conf)
    pylab.title(r"$\mathbf{XX}$")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname+"XX.pdf"))

    fig = pylab.figure()
    cov = conf_model._lvm_covariance
    K_whole = cov.K(conf_model._lvm_hyperparams['covar'], x)
    im = pylab.imshow(K_whole)
    pylab.title(r"$\mathbf{K}$")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname+"K.pdf"))

if "gt100" in sys.argv:
    gt_file_name = '../../examples/ground_truth_balanced_set_of_100.csv'
else:
    gt_file_name = '../../examples/ground_truth_random_genes.csv'
gt_file = open(gt_file_name, 'r')
gt_read = csv.reader(gt_file)
gt_names = list()
gt_vals = list()
for name, val in gt_read:
    gt_names.append(name.upper())
    gt_vals.append(val)
indices = numpy.where(numpy.array(gt_names)[None, :] == numpy.array(gene_names)[:, None])
gt_names = numpy.array(gt_names)[indices[1]]
gt_vals = numpy.array(gt_vals)[indices[1]]

# priors
#covar_priors_common = []
#covar_priors_individual = []
#covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
#covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
#covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
#covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])
#covar_priors_common.append([lnpriors.lnuniformpdf, [1, 1]])
#covar_priors_individual.append([lnpriors.lnuniformpdf, [1, 1]])
#priors = get_model_structure({'covar':numpy.array(covar_priors_individual)})

likelihoods_file_name = os.path.join(root, outname + '_likelihoods.cPickle')
hyperparams_file_name = os.path.join(root, outname + '_hyperparams.cPickle')

if not os.path.exists(likelihoods_file_name) or "relikelihood" in sys.argv:
    s = "predicting model likelihoods..."
    print s,
    sys.stdout.write("             \r")
    likelihoods = conf_model.predict_likelihoods(messages=False, message=s, indices=indices[0])#, priors=priors)
    hyperparams = conf_model.get_learned_hyperparameters()
    likelihoods_file = open(likelihoods_file_name, 'w')
    hyperparams_file = open(hyperparams_file_name, 'w')
    cPickle.dump(likelihoods, likelihoods_file)
    cPickle.dump(hyperparams, hyperparams_file)
    # finished(s)
else:
    s = "loading model likelihoods... "
    print s,
    sys.stdout.write("\r")
    likelihoods_file = open(likelihoods_file_name, 'r')
    hyperparams_file = open(hyperparams_file_name, 'r')
    conf_model._likelihoods = cPickle.load(likelihoods_file)
    conf_model._hyperparameters = cPickle.load(hyperparams_file)
    finished(s)
likelihoods_file.close()
hyperparams_file.close()

s = "writing back bayes factors..."
bayes_file_name = os.path.join(root, outname + '_bayes.csv')
if not os.path.exists(bayes_file_name) or "rebayes" in sys.argv or "relikelihood" in sys.argv:
    print s,
    sys.stdout.write("\r")
    bayes_file = open(bayes_file_name, 'w')
    writer = csv.writer(bayes_file)
    for row in itertools.izip(gt_names, conf_model.bayes_factors()):
        writer.writerow(row)
    bayes_file.close()
    bayes_file = open(bayes_file_name, 'r')
    bayes_factors = []
    for row in csv.reader(bayes_file):
        bayes_factors.append(row)
    bayes_file.close()
    finished(s)

print ""


# s = "plotting roc curve"
# sys.stdout.write(s + "\r")
# pylab.ion()
# pylab.figure(10)
# plot_roc_curve(bayes_file_name, gt_file_name, label=outname)
# pylab.legend(loc=4)
# finished(s)


