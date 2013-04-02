'''
Created on Feb 26, 2013

@author: Max
'''
# import sys
# sys.path.append('/Users/stegle/research/users/stegle/pygp')
# sys.path.append('./../../../..')
# sys.path.append('./../../..')
import os
import pickle
import numpy
import sys
import pylab
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pygp.covar.linear import LinearCFISO
from pygp.covar.combinators import SumCF, ProductCF
from pygp.covar.se import SqexpCFARD
from pygp.covar.bias import BiasCF

import logging
from gptwosample.confounder import confounder
from numpy.ma.core import ceil
from gptwosample.confounder.data import read_and_handle_gt
from gptwosample.twosample.twosample import TwoSample
logging.basicConfig(level=logging.CRITICAL)
del logging

_usage = """usage: python warwick.py root-dir data_dir out_name
[warwick_control-file warwick_treatment-file]
[redata|regplvm|relikelihood|plot_confounder|gt100]

warwick_control-file and warwick_treatment-file have to be given only in first run - data will be pickled"""


Q = 4
Qsim = 4
seed = 0
conf_var = 2.0
N, Ni = 1, 0
D = 'all'

stats_lines = list()

if 'debug' in sys.argv:
    sys.argv = ['me', 'conf', 'data', 'sam', #'gradcheck', 
                '../../examples/warwick_control.csv', '../../examples/warwick_treatment.csv',
                #'regplvm', 
                'D=2000', 'Q=4', 'gt100',
                'norm_genesamples',
                'debug',
                "plot_confounder", 
                "plot_predict",
                'dolikelihood',
                'unconfounded',
                ]

# parse command line:
for ar in sys.argv:
    if ar.startswith("cores="):
        num_procs = int(ar.split("=")[1])
        stats_lines.append("using {0:g} workers".format(num_procs))
        print "using {0:g} workers".format(num_procs)
    elif ar.startswith("seed="):
        seed = int(ar.split("=")[1])
        print "seed", seed
    elif ar.startswith("Q="):
        Q = int(ar.split("=")[1])
        print "Q", Q
    elif ar.startswith("Qsim="):
        Qsim = int(ar.split("=")[1])
        print "Qsim", Qsim
    elif ar.startswith("conf_var="):
        conf_var = float(ar.split("=")[1])
        print "conf_var", conf_var
    elif ar.startswith("D="):
        D = int(ar.split("=")[1])
        print "D", D
    elif ar.startswith("jobs="):
        N, Ni = map(lambda x: int(x), ar.split("=")[1].split(","))
        Ni -= 1  # to index
        print "Ni/N={1}/{0}".format(N, Ni)

stats_lines.extend(["Q={}\n".format(str(Q)),
                    "Qsim={}\n".format(str(Qsim)),
                    "seed={}\n".format(str(seed)),
                    "conf_var={}\n".format(str(conf_var)),
                    "D={}\n".format(str(D)),
                    ])  # ,
#                    "Ni/N={1}/{0}".format(N,Ni)])

# try:
root = sys.argv[1]
if not os.path.exists(root):
    os.makedirs(root)
data = sys.argv[2]
if not os.path.exists(data):
    os.makedirs(data)
outname = sys.argv[3]
if not os.path.exists(os.path.join(root, outname)):
    os.makedirs(os.path.join(root, outname))
if "gt100" in sys.argv:
    gt_file_name = '../../examples/ground_truth_balanced_set_of_100.csv'
else:
    gt_file_name = '../../examples/ground_truth_random_genes.csv'

# except:
    # print _usage
#    sys.exit(0)

numpy.random.seed(seed)

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
sys.stdout.write(s)
sys.stdout.flush()

if "norm_genesamples" in sys.argv:
    norm = "norm=genesamples"
else:  # norm=genes
    norm = "norm=genes"
stats_lines.append(norm+os.linesep)

data_file_path = os.path.join(data, "./data_seed={0!s}_Qsim={1!s}_D={2!s}_{norm}.pickle".format(seed, Qsim, D, norm=norm))

if not os.path.exists(data_file_path) or "redata" in sys.argv:
    sys.stdout.write(os.linesep)
#    cond1 = get_data_from_csv(sys.argv[4])  # os.path.join(root,'warwick_control.csv'))
#    cond2 = get_data_from_csv(sys.argv[5])  # os.path.join(root,'warwick_treatment.csv'))
#    print s + "\r",
#    T1 = numpy.array(cond1.pop("input"))[:, None]
#    T2 = numpy.array(cond2.pop("input"))[:, None]
#
#    Y1 = numpy.array(cond1.values()).T.swapaxes(0, 1)
#    Y2 = numpy.array(cond2.values()).T.swapaxes(0, 1)
#    Y = numpy.array([Y1, Y2])
    T, Y, gene_names, Ygt, gt_names = read_and_handle_gt(sys.argv[4], sys.argv[5], gt_file_name, D=D)
    n, r, t, d = Y.shape

#    T1 = numpy.tile(T1, r).T
#    T2 = numpy.tile(T2, r).T
#
#    T = numpy.array([T1, T2])

    # gene_names = cond1.keys()

    assert T.shape == Y.shape[:3]
    # assert gene_names == cond2.keys()

    # del T1, T2, Y1, Y2, cond1, cond2

    X_sim = numpy.random.randn(n * r * t, Qsim)
    # X_sim -= X_sim.mean(0)
    # X_sim /= X_sim.std(0)
    X_sim *= numpy.sqrt(1. / float(Qsim))
    # si = "standardizing data ..."
    # sys.stdout.write(si + "\r")
    if norm == "norm=genesamples":
        # ZERO MEAN GENES AND SAMPLES:
        Y -= Y.mean(1).mean(1)[:, None, None, :]
    else:  # "norm=genes"
        # ZERO MEAN GENES:
        Y -= Y.reshape(-1, d).mean(0)[None, None, None, :]

    # Y /= Y.std()
    # Conf_sim -= Conf_sim.reshape(n*r*t,d).mean(0)
    # Conf_sim /= Conf_sim.reshape(n*r*t,d).std(0)
    # finished(si)

    K_sim = numpy.dot(X_sim.reshape(n * r * t, Qsim), X_sim.reshape(n * r * t, Qsim).T)
    Conf_sim = numpy.dot(X_sim, numpy.random.randn(Qsim, d))

    data_file = open(data_file_path, 'w')
    pickle.dump([T, Y, gene_names, 
                 K_sim, Conf_sim, 
                 X_sim.reshape(n * r * t, Qsim),
                 Ygt, gt_names], data_file)
else:
    if "onlydata" in sys.argv:
        sys.exit(0)
    sys.stdout.write("\r")
    data_file = open(data_file_path, 'r')
    T, Y, gene_names, K_sim, Conf_sim, X_sim, Ygt, gt_names = pickle.load(data_file)
    n, r, t, d = Y.shape
finished(s)
data_file.close()

gt_names = numpy.array(gt_names)

if "onlydata" in sys.argv:
    sys.exit(0)

#  gt read moved up to improve exit performance
# gt_file = open(gt_file_name, 'r')
# gt_read = csv.reader(gt_file)
# gt_names = list()
# gt_vals = list()
# for name, val in gt_read:
#    gt_names.append(name.upper())
#    gt_vals.append(val)
# indices = numpy.where(numpy.array(gt_names)[None, :] == numpy.array(gene_names)[:, None])
# gt_names = numpy.array(gt_names)[indices[1]]
# gt_vals = numpy.array(gt_vals)[indices[1]]
#
# # select subset of data to run on:
# joblen = ceil(len(gt_names) / float(N))
# jobslice = slice(Ni * joblen, (Ni + 1) * joblen)
# jobindices = indices[0][jobslice]

joblen = ceil(len(gt_names) / float(N))
jobslice = slice(Ni * joblen, (Ni + 1) * joblen)
jobindices = numpy.arange(len(gt_names))[jobslice]

if not len(jobindices):
    print "no more genes left to run"
    sys.exit(0)

s = "setting up gplvm module..."
print s,
sys.stdout.flush()
sys.stdout.write("\r")
yvar = Y.var()
stats_lines.append("Y.var()={}\n".format(Y.var()))

conf = numpy.sqrt(conf_var * yvar) * Conf_sim.reshape(n, r, t, d) / Conf_sim.std()
stats_lines.append("conf.var()={}\n".format(conf.var()))
stats_lines.append("Y.var()/conf.var()={}\n".format(conf.var() / yvar))
if not ("raw" in sys.argv) and not ("unconfounded" in sys.argv):
    Y = Y + conf

stats_lines.append("Yconf.var()={}\n".format(Y.var()))

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
    if 'prod_sample_sample' in sys.argv:
        lvm_covariance = SumCF([sam, ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()],
                               names=['sam', 'samprod', 'bias'])
        learn_name = 'prod_sample_sample'
    elif "rep" in sys.argv:
        lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 1 + q)),
                                  rep,
                                  # sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()],
                               names=['XX', 'rep', 'samprod', 'bias'])
        learn_name = 'rep'
    elif "sam" in sys.argv:
        lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 1 + q)),
                                  # rep,
                                  sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()],
                               names=['XX', 'sam', 'samprod', 'bias'])
        learn_name = 'sam'
    elif "triv" in sys.argv:
        lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 1 + q)),
                                # rep,
                                #  sam,
                                #  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                BiasCF()],
                               names=['XX', 'bias'])
        learn_name = 'triv'
    elif "sam0" in sys.argv:
        lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 1 + q)),
                                # rep,
                                sam,
                                #  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                BiasCF()],
                               names=['XX', 'sam', 'bias'])
        learn_name = 'sam0'
    else:
        lvm_covariance = SumCF([LinearCFISO(dimension_indices=numpy.arange(1, 1 + q)),
                                  rep,
                                  sam,
                                  ProductCF([sam, SqexpCFARD(dimension_indices=numpy.array([0]))]),
                                  BiasCF()],
                               names=['XX', 'rep', 'sam', 'samprod', 'bias'])
        learn_name = 'all'
    if "unconfounded" in sys.argv:
        learn_name += "_unconf"
    outname = os.path.join(outname, learn_name)
    if not os.path.exists(os.path.join(root, outname)):
        os.mkdir(os.path.join(root, outname))

stats_file_name = os.path.join(root, outname, "stats.txt")
stats_file = open(stats_file_name, 'w')
stats_file.writelines(stats_lines)
stats_file.flush()
del stats_lines

init = 'random'
if "init_pca" in sys.argv:
    init = 'pca'

stats_file.write("init={}\n".format(init))
conf_model = confounder.ConfounderTwoSample(T, Y, q=Q,
                                            lvm_covariance=lvm_covariance,
                                            init=init)
conf_model.__verbose = 0
try:
    conf_model.NUM_PROCS = num_procs
except NameError:
    pass
x = numpy.concatenate((T.reshape(-1, 1), conf_model.X, X_r, X_s), axis=1)
finished(s)

lvm_hyperparams_file_name = os.path.join(root, outname, 'lvm_hyperparams.pickle')

stats_file.write("Conf_model.Y.var()={}\n".format(conf_model.Y.var()))

if "ideal" in sys.argv:
    conf_model.X = X_sim
    conf_model.K_conf = K_sim
    conf_model._initialized = True
elif "noconf" in sys.argv or "raw" in sys.argv:
    conf_model = TwoSample(T, Y)
elif ((not os.path.exists(lvm_hyperparams_file_name)) or
      "regplvm" in sys.argv):
    s = 'learning confounder matrix... '
    p = start_mill(s)
    if 'debug' in sys.argv:   
        pylab.figure(),pylab.imshow(numpy.cov(Y.reshape(-1,d))),pylab.colorbar()
        pylab.figure(),pylab.imshow(numpy.cov(Conf_sim.reshape(-1,d))),pylab.colorbar()
        pylab.ion(), pylab.draw(), pylab.show()
        try:
            import ipdb;ipdb.set_trace()
        except:
            import pdb; pdb.set_trace()
        
    conf_model.learn_confounder_matrix(x=x, 
                                       gradcheck=('gradcheck' in sys.argv), 
                                       maxiter=10000)
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'w')
    pickle.dump(conf_model._lvm_hyperparams, lvm_hyperparams_file)
    finished(s, process=p)
else:
    s = "loading confounder matrix..."
    print s,
    sys.stdout.write("\r")
    lvm_hyperparams_file = open(lvm_hyperparams_file_name, 'r')
    x = numpy.concatenate((T.reshape(-1, 1), conf_model.X, X_r, X_s), axis=1)
    conf_model._Xlvm = x
    conf_model._init_conf_matrix(pickle.load(lvm_hyperparams_file), numpy.arange(1), numpy.arange(1, 1 + Q))
    finished(s)
try:
    lvm_hyperparams_file.close()
except:
    pass

if "plot_predict" in sys.argv and "conf" in sys.argv:
    prediction = conf_model.predict_lvm()
    fig = pylab.figure()
    im = pylab.imshow(prediction[0])
    pylab.title("Prediction, mean var = {0:.3f}".format(prediction[1].mean()))
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "Ypred.pdf"))

    fig = pylab.figure()
    im = pylab.imshow(Y.reshape(-1, d))
    pylab.title("Y")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "Yorig.pdf"))

if "plot_confounder" in sys.argv and "conf" in sys.argv:
    sigma = numpy.exp(2 * conf_model._lvm_hyperparams['lik'][0])
    logtheta = conf_model._lvm_hyperparams['covar']
    theta = conf_model._lvm_covariance.get_reparametrized_theta(logtheta)
    
    print "plotting Xsim"
    fig = pylab.figure()
    im = pylab.imshow(K_sim)
    pylab.title(r"$\mathbf{{XX}}sim$")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "KXXsim.pdf"))

    try:
        print "plotting Conf"
        fig = pylab.figure()
        im = pylab.imshow(conf.reshape(-1,d))
        pylab.title(r"$\mathbf{{C}}sim$")
        divider = make_axes_locatable(pylab.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        pylab.colorbar(im, cax=cax)
        try:
            fig.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(root, outname, "C.pdf"))
    except:
        print "no conf"
        pass
    
    print "plotting cov(Y)"
    fig = pylab.figure()
    im = pylab.imshow(numpy.cov(Y.reshape(-1,d)))
    pylab.title(r"Cov$(\mathbf{{Y}})$")
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "YCov.pdf"))

    print "plotting XX"
    fig = pylab.figure()
    im = pylab.imshow(conf_model.K_conf)
    pylab.title(r"$\mathbf{{XX}}, var={:.3g}, \alpha={:.3g}$".format(numpy.trace(conf_model.K_conf) / conf_model.K_conf.shape[1], theta[0]))
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "KXXconf.pdf"))

    print "plotting K"
    fig = pylab.figure()
    cov = conf_model._lvm_covariance
    K_whole = cov.K(conf_model._lvm_hyperparams['covar'], x)
    im = pylab.imshow(K_whole)
    pylab.title(r"$\mathbf{{K}}, var={:.3g}, \sigma={:.3g}$".format(numpy.trace(K_whole) / K_whole.shape[1], sigma))
    divider = make_axes_locatable(pylab.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    pylab.colorbar(im, cax=cax)
    try:
        fig.tight_layout()
    except:
        pass
    pylab.savefig(os.path.join(root, outname, "K.pdf"))

    for name in cov.names:
        print "plotting K{}".format(name)
        fig = pylab.figure()
        K = cov.K(logtheta, x, x, [name])
        im = pylab.imshow(K)
        pylab.title(r"$\mathbf{{{}}}, var={:.3g}, \alpha={:.3g}$".format(name, numpy.trace(K) / K.shape[1], cov.get_theta_by_names(theta, [name])[0]))
        divider = make_axes_locatable(pylab.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        pylab.colorbar(im, cax=cax)
        try:
            fig.tight_layout()
        except:
            pass
        pylab.savefig(os.path.join(root, outname, "K{}.pdf".format(name)))


#    if "sam" in sys.argv:
#        cov = conf_model._lvm_covariance
#        covarsnparams = [c.get_number_of_parameters() for c in cov.covars]
#        covarsnparams.insert(0, 0)
#        covarslices = [slice(a, b) for a, b in itertools.izip(numpy.cumsum(covarsnparams), numpy.cumsum(covarsnparams)[1:])]
#
#        fig = pylab.figure()
#        K_XXcov = cov.covars[0].K(conf_model._lvm_hyperparams['covar'][covarslices[0]], x)
#        im = pylab.imshow(K_XXcov)
#        import ipdb;ipdb.set_trace()
#        pylab.title(r"$\mathbf{{covXX}}$ var=${:.3f}$, $\alpha={:.3f}$".format(numpy.trace(K_XXcov) / K_XXcov.shape[1], theta[covarslices[0]][0]))
#        divider = make_axes_locatable(pylab.gca())
#        cax = divider.append_axes("right", "5%", pad="3%")
#        pylab.colorbar(im, cax=cax)
#        try:
#            fig.tight_layout()
#        except:
#            pass
#        pylab.savefig(os.path.join(root, outname, "KXXcov.pdf"))
#
#        fig = pylab.figure()
#        K_sam = cov.covars[1].K(conf_model._lvm_hyperparams['covar'][covarslices[1]], x)
#        im = pylab.imshow(K_sam)
#        pylab.title(r"$\mathbf{{sam}}$ var=${:.3f}$, $\alpha={:.3f}$".format(numpy.trace(K_sam) / K_sam.shape[1], theta[covarslices[1]][0]))
#        divider = make_axes_locatable(pylab.gca())
#        cax = divider.append_axes("right", "5%", pad="3%")
#        pylab.colorbar(im, cax=cax)
#        try:
#            fig.tight_layout()
#        except:
#            pass
#        pylab.savefig(os.path.join(root, outname, "K_sam.pdf"))
#
#        fig = pylab.figure()
#        K_sam_prod = cov.covars[2].K(conf_model._lvm_hyperparams['covar'][covarslices[2]], x)
#        im = pylab.imshow(K_sam_prod)
#        pylab.title(r"$\mathbf{{sam prod}}$ var=${:.3f}$, $\alpha={:.3f}$".format(numpy.trace(K_sam_prod) / K_sam_prod.shape[1], theta[covarslices[2]][0]))
#        divider = make_axes_locatable(pylab.gca())
#        cax = divider.append_axes("right", "5%", pad="3%")
#        pylab.colorbar(im, cax=cax)
#        try:
#            fig.tight_layout()
#        except:
#            pass
#        pylab.savefig(os.path.join(root, outname, "K_sam_prod.pdf"))
#
#        fig = pylab.figure()
#        bias = cov.covars[3].K(conf_model._lvm_hyperparams['covar'][covarslices[3]], x)
#        im = pylab.imshow(bias)
#        pylab.title(r"$\mathbf{{Kbias}}$ var=${:.3f}$, $\alpha={:.3f}$".format(numpy.trace(bias) / bias.shape[1], theta[covarslices[3]][0]))
#        divider = make_axes_locatable(pylab.gca())
#        cax = divider.append_axes("right", "5%", pad="3%")
#        pylab.colorbar(im, cax=cax)
#        try:
#            fig.tight_layout()
#        except:
#            pass
#        pylab.savefig(os.path.join(root, outname, "bias.pdf"))

sys.stdout.flush()

# GPTwoSample:

# gt_reading done up to improve exit performance ^
outname = os.path.join(outname, "jobs")
if not os.path.exists(os.path.join(root, outname)):
    os.makedirs(os.path.join(root, outname))

# dataset = h5py.File(os.path.join(root, outname, "gptwosample_job_{}_{}.hdf5".format(Ni,N)), 'w')

# priors
# covar_priors_common = []
# covar_priors_individual = []
# covar_priors_common.append([lnpriors.lnGammaExp, [6, .3]])
# covar_priors_individual.append([lnpriors.lnGammaExp, [6, .3]])
# covar_priors_common.append([lnpriors.lnGammaExp, [30, .1]])
# covar_priors_individual.append([lnpriors.lnGammaExp, [30, .1]])
# covar_priors_common.append([lnpriors.lnuniformpdf, [1, 1]])
# covar_priors_individual.append([lnpriors.lnuniformpdf, [1, 1]])
# priors = get_model_structure({'covar':numpy.array(covar_priors_individual)})

likelihoods_file_name = os.path.join(root, outname, 'likelihoods_job_{}_{}.pickle'.format(Ni, N))
hyperparams_file_name = os.path.join(root, outname, 'hyperparams_job_{}_{}.pickle'.format(Ni, N))
gt_file_name = os.path.join(root, outname, 'gt_names_job_{}_{}.pickle'.format(Ni, N))

if "relikelihood" in sys.argv:
    sys.argv.append("dolikelihood")

if ('dolikelihood' in sys.argv and
    not (os.path.exists(likelihoods_file_name) and
         os.path.exists(hyperparams_file_name) and
         os.path.exists(gt_file_name)) or
    "relikelihood" in sys.argv):
    s = "predicting model likelihoods..."
    print s,
    sys.stdout.flush()
    sys.stdout.write("             \r")
    likelihoods = conf_model.predict_likelihoods(T, Ygt[:, :, :, jobindices],
                                                 messages=False,
                                                 message=s,
                                                 )  # , priors=priors)
    hyperparams = conf_model.get_learned_hyperparameters()
    # dataset.create_dataset(name="L", data=numpy.array(likelihoods), dtype=list, shape=(joblen,))
    # dataset.create_dataset(name="H", data=numpy.array(hyperparams), dtype=list, shape=(joblen,))
    # dataset.create_dataset(name="genes", data=gt_names[jobslice], dtype=list, shape=(joblen,))
    likelihoods_file = open(likelihoods_file_name, 'w')
    hyperparams_file = open(hyperparams_file_name, 'w')
    gt_file = open(gt_file_name, 'w')
    pickle.dump(likelihoods, likelihoods_file)
    pickle.dump(hyperparams, hyperparams_file)
    pickle.dump(gt_names[jobslice], gt_file)
    # finished(s)
    likelihoods_file.close()
    hyperparams_file.close()
    gt_file.close()
# dataset.close()
# else:
#    s = "loading model likelihoods... "
#    print s,
#    sys.stdout.write("\r")
#    #likelihoods_file = open(likelihoods_file_name, 'r')
#    #hyperparams_file = open(hyperparams_file_name, 'r')
#    conf_model._likelihoods = dataset.require_dataset("L")#pickle.load(likelihoods_file)
#    conf_model._hyperparameters = dataset.require_dataset("H")#pickle.load(hyperparams_file)
#    finished(s)



print ""
stats_file.close()

# s = "plotting roc curve"
# sys.stdout.write(s + "\r")
# pylab.ion()
# pylab.figure(10)
# plot_roc_curve(bayes_file_name, gt_file_name, label=outname)
# pylab.legend(loc=4)
# finished(s)


