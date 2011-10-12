'''
Command line tool for usage of GPTwoSample

Created on Jun 15, 2011

@author: Max Zwiessele, Oliver Stegle
'''

__all__ = ['plotting', 'show', 'timeshift', 'interval', 'gibbs_iteratons', 'delim', 'out_dir', 'interpolation']
from gptwosample.data.data_base import get_model_structure, \
    get_training_data_structure, common_id, individual_id
import csv
import getopt
import os
import sys
import scipy

#global plotting
#global show
#global timeshift
#global interval
#global gibbs_iterations
#global delim
#global out_dir

plotting = False
hold = False
timeshift = False
interval = False
gibbs_iterations = 0
verbose = False
interpolation = 100
delim = ','
out_dir = os.path.abspath("./")

def usage():
    usage = []
    usage.append('Usage: gptwosample [options] <file1> <file2>')
    usage.append('')
    usage.append('Perform GPTwoSample on each gene included in both files, file1 and file2. See file format below')
    usage.append('')
    usage.append('Options:')
    usage.append('\t-h|--help\t Show this text')
    usage.append('\t-v|--verbose\t Verbose mode')
    usage.append('\t-p|--plot \t Plot all genes and save to file outdir/{gene_name}.png')
    usage.append('\t-s|--show \t Hold between each gene, showing current results as plot')
    usage.append('\t-o|--out_dir=<out_dir> \t ["./"] Save results to out_dir. See file format below')
    usage.append('\t-t|--timeshift=\t Perform GPTwoSample_individual_covariance on data')
    usage.append('\t-i|--interval=<iterations> \t Perform Interval prediction for iterations gibbs iterations on data')
    usage.append('\t-n|--interpolation=<number> \t [100] Specify Number of time points to interpolate with (smoothness of regression)')
    usage.append('\t-d|--delimiter=<delimiter> \t [","] Specify delimiter for reading and writing')
    usage.append('')
    usage.append("""The file format has to fullfill following formation:
    
    ============ =============== ==== ===============
    *arbitrary*  x1              ...  xl
    ============ =============== ==== ===============
    Gene Name 1  y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene Name 1  y1 replicate k1 ...  yl replicate k1

    ...
    
    Gene Name n  y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene Name n  y1 replicate kn ...  yl replicate kn
    ============ =============== ==== ===============""")

    return '\n'.join(usage)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hvpso:ti:n:d:", ["help", "verbose", "plot", "show", "out_dir=", "timeshift", "interval=", 'interpolation=', "delimiter="])
except getopt.GetoptError:
    print usage()
    sys.exit(2)

for opt, arg in opts:
    if(opt in ('-h', '--help')):
        print usage()
        sys.exit()
    elif opt in ('-o', '--out_dir'):
        try:
            out_dir = os.path.abspath(arg)
        except:
            print "Cannot parse output directory: %s" % (arg)
            sys.exit(2)
        if os.path.isfile(out_dir):
            print "Not a directory: %s" % (arg)
            sys.exit(2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    elif opt in ('-t', '--timeshift'):
        timeshift = True
    elif opt in ('-i', '--interval'):
        interval = True
        try:
            gibbs_iterations = int(arg)
        except:
            print "Need an Integer for option %s, given: %s" % (opt, arg)
            sys.exit(2)
    elif(opt in ('-p', '--plot')):
        plotting = True
    elif(opt in ('-s', '--show')):
        hold = True
        plotting = True
    elif(opt in ('-v', '--verbose')):
        verbose = True
    elif(opt in ('-n', '--interpolation')):
        interpolation = int(arg)
    elif(opt in ('-d', 'delimiter')):
        delim = arg.decode('string-escape')

if len(args) is not 2:
    print usage()
    sys.exit(2)

print "Welcome to GPTwoSample. Loading modules, be patient..."
print "Loading: GPTwoSample Modules..."
print "Loading: scipy..."
print "Loading: csv..."
print "Finished Loading, ready to GPTwoSample!"

def main():
    #print "Modules loaded, starting to GPTwoSample!"
    global plotting
    global hold
    global timeshift
    global interval
    global gibbs_iterations
    global delim
    global out_dir
    global csv_out_file
    
    if verbose: 
        print "GPTwoSample:Chosen Parameters: "
        print "GPTwoSample:-verbose:\tobviously =)"
        print "GPTwoSample:-plotting:\t%s" % (plotting)
        print "GPTwoSample:-hold:\t%s" % (hold)
        print "GPTwoSample:-timeshift:\t%s" % (timeshift)
        print "GPTwoSample:-interval:\t%s, with iterations: %s" % (interval, gibbs_iterations)
        print "GPTwoSample:-interpolation:\t%s" % (interpolation)
        print "GPTwoSample:-delim:\t%s" % (delim.encode('string-escape'))
        print "GPTwoSample:-out_dir:\t%s" % (os.path.relpath(out_dir, "./"))
    
    csv_out_file = open(os.path.join(out_dir, "result"+os.path.splitext(args[0])[1]), 'wb')
    csv_out = csv.writer(csv_out_file, delimiter=delim)
    csv_out_file.flush()
    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    # read data
    from gptwosample.data.dataIO import get_data_from_csv
    try:
        if verbose: print('GPTwoSample:Reading data from file:%s' % (os.path.relpath(args[0])))
        cond1 = get_data_from_csv(args[0], delimiter=delim)
        if verbose: print('GPTwoSample:Reading data from file:%s' % (os.path.relpath(args[1])))
        cond2 = get_data_from_csv(args[1], delimiter=delim)
        # get input from data 
        Tpredict = scipy.linspace(cond1["input"].min(), cond1["input"].max(), interpolation).reshape(-1, 1)
        T1 = cond1.pop("input")
        T2 = cond2.pop("input")
    except:
        print "ERROR:Could not read given files, perhaps wrong delimiter chosen?"
        sys.exit(2)
    del get_data_from_csv
    
    if interval:
        if not(len(T1) == len(T2)):
            print "ERROR:GPTwoSampleInterval:For interval sampling same input dimension is required!"
            print "ERROR:GPTwoSampleInterval:Given: len(%s)=%i and len(%s)=%i" % (os.path.basename(args[0]),len(T1),os.path.basename(args[1]),len(T2))
            sys.exit(2)
    # gene_names for later writing and plotting
    gene_names = sorted(cond1.keys()) 
    if not(gene_names == sorted(cond2.keys())):
        print "ERROR:GPTwoSample:Genes have to be includet in both condition files!"
        sys.exit(2)
    
    if(len(T1)==1 or len(T2)==1) and plotting:
        print "WARNING:Cannot plot with only one time-point in either condition, turning of plotting"
        plotting = False

    # replicate information for shift cf
    n_replicates_1 = cond1[gene_names[0]].shape[0]
    n_replicates_2 = cond2[gene_names[0]].shape[0]
    if timeshift:
        if not(n_replicates_1 == n_replicates_1):
            print "ERROR:GPTwoSample_individual_covariance:For proper timeshift detection same number of replicates is required (try duplicating to get right amount of replicates)!"
            print "ERROR:GPTwoSample_individual_covariance:Given: rep(%s)=%i and rep(%s)=%i" % (os.path.basename(args[0]),n_replicates_1,os.path.basename(args[1]),n_replicates_2)
            sys.exit(2)
        if not(len(T1) == len(T2)):
            print "WARNING:GPTwoSample_individual_covariance:Cannot estimate time shift if time series have different dimensions, turning off timeshift"
            timeshift = False

    
    gene_length = len(T1)
    dim = 1
    # get covariance function right
    CovFun = get_covariance_function(dim, gene_length, n_replicates_1, n_replicates_2)
    # Make shure output file has right header and
    # get gptwosample_object right
    header = ["Gene", "Bayes Factor"]
    if(timeshift):
        header.extend(get_header_for_covar(CovFun[2], CovFun[0]))
        from gptwosample.twosample.twosample_compare import GPTwoSample_individual_covariance
        gptwosample_object = GPTwoSample_individual_covariance(CovFun, priors=get_priors(dim, n_replicates_1, n_replicates_2))
        del GPTwoSample_individual_covariance
    else:
        header.extend(get_header_for_covar(CovFun))
        from gptwosample.twosample.twosample_compare import GPTwoSample_share_covariance
        gptwosample_object = GPTwoSample_share_covariance(CovFun, priors=get_priors(dim, n_replicates_1, n_replicates_2))
        del GPTwoSample_share_covariance
    if(interval):
        header.append("Interval Indicators")
        header.append("Interval Probabilities")
    
    csv_out.writerow(header)
    csv_out_file.flush()
    
    # get format for input:
    T1 = scipy.tile(T1, n_replicates_1).reshape(-1, 1)
    T2 = scipy.tile(T2, n_replicates_1).reshape(-1, 1)
    
    if interval:
        perform_interval(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, csv_out, n_replicates_1, n_replicates_2, dim)
    else:
        # now lets run GPTwoSample on each gene in data:
        perform_gptwosample(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, csv_out, n_replicates_1, n_replicates_2, dim)
    
    print "DONE:Results written to %s" % (os.path.join(out_dir, "result"+os.path.splitext(args[0])[1]))
    if(plotting):
        from pylab import close
        close()
        del close
        print "     Plots written to %s" % (out_dir)


def get_exp_on_timeshift_right(gptwosample_object, n_replicates_1, n_replicates_2, dim):
    global timeshift
    common = gptwosample_object.get_learned_hyperparameters()[common_id]['covar']
    individual = gptwosample_object.get_learned_hyperparameters()[individual_id]['covar']
    if (timeshift):
        timeshift_index = scipy.array(scipy.ones_like(common), dtype='bool')
        timeshift_index[dim + 1:dim + 1 + n_replicates_1+n_replicates_2] = 0
        common[timeshift_index] = scipy.exp(common[timeshift_index])
        timeshift_index = scipy.array(scipy.ones_like(individual), dtype='bool')
        timeshift_index[dim + 1:dim + 1 + n_replicates_1] = 0
        individual[timeshift_index] = scipy.exp(individual[timeshift_index])
    else:
        common = scipy.exp(common)
        individual = scipy.exp(individual)
    return common, individual

def perform_interval(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names,
                     csv_out, n_replicates_1, n_replicates_2, dim):#, gibbs_iterations, plotting, show, timeshift, delim):
    global plotting
    global hold
    global timeshift
    global gibbs_iterations
    global delim
    global out_dir
    global csv_out_file

    logthetaZ = scipy.log([.5, 2, .001])
    lik = scipy.log([2, 2])
    from gptwosample.twosample.interval_smooth import GPTwoSampleInterval
    if plotting:
        from pylab import xlim, show, savefig, clf
        from gptwosample.plot.interval import plot_results_interval
    
    for gene_name in gene_names:
        if verbose: print("GPTwoSampleInterval:Processing %s" % (gene_name))
        Y0 = cond1[gene_name].reshape(-1,1)
        Y1 = cond2[gene_name].reshape(-1,1)
        gptwosample_object.set_data_by_xy_data(T1,T2,
                                               Y0,
                                               Y1)
        gptwosample_object.predict_model_likelihoods()

        line = [gene_name, gptwosample_object.bayes_factor()]
        common, individual = get_exp_on_timeshift_right(gptwosample_object, n_replicates_1, n_replicates_2, dim)
        line.extend(common)
        line.extend(individual)
        gptest = GPTwoSampleInterval(gptwosample_object, outlier_probability=.1)
        Z = gptest.predict_interval_probabilities(Tpredict, hyperparams={'covar':logthetaZ, 'lik':lik},
                                                  number_of_gibbs_iterations=gibbs_iterations)
        if verbose: print("GPTwoSampleInterval:Prediction: %s" % (" ".join(scipy.array(gptest.get_predicted_indicators(), dtype="str"))))
        if(plotting):
            if verbose: print("GPTwoSampleInterval:Plotting")
            if(timeshift):
                plot_results_interval(gptest,
                     shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2 + n_replicates_1+n_replicates_2],
                     draw_arrows=1, legend=False,
                     xlabel="Time [h]", ylabel="Expression level",
                     title=r'TimeShift: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gptwosample_object.bayes_factor()))
            else:
                plot_results_interval(gptest, title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, gptwosample_object.bayes_factor())) 
            xlim(T1.min(), T1.max())
            savefig(os.path.join(out_dir, "%s.png") % (gene_name), format='png')
            if verbose: print("GPTwoSampleInterval:Saving Figure %s" % (os.path.join(out_dir, "%s.png") % (gene_name)))
            if hold:
                print("GPTwoSampleInterval:Hold for Showing (Close figure to continue...")
                show()
            clf()
        if(delim == ","):
            internal_delim = ';'
        else:
            internal_delim = ','
        line.append(internal_delim.join(scipy.array(gptest.get_predicted_indicators(), dtype="str")))
        line.append(internal_delim.join(scipy.array(Z[0], dtype="str")))
        if verbose: print("GPTwoSampleInterval:Writing back..." % (line))
        csv_out.writerow(line)
        csv_out_file.flush()

    del GPTwoSampleInterval
    if plotting:
        del xlim, show, savefig, clf
        del plot_results_interval

def perform_gptwosample(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names,
                        csv_out, n_replicates_1,n_replicates_2, dim):#plotting, show, timeshift):
    global plotting
    global hold
    global timeshift
    global out_dir
    global csv_out_file

    if plotting:
        from pylab import xlim, show, savefig, clf
        from gptwosample.plot.plot_basic import plot_results

    iterations_left = len(gene_names) - 1
    
    for gene_name in gene_names:
        Y0 = cond1[gene_name].reshape(-1,1)
        Y1 = cond2[gene_name].reshape(-1,1)
#        if(plotting):
#            rep0 = get_replicate_indices(Y0)
#            rep1 = get_replicate_indices(Y1)
        print("GPTwoSample:Processing gene %s, genes left: %i" % (gene_name, iterations_left))
        iterations_left -= 1
        
        gptwosample_object.set_data_by_xy_data(T1,T2,
                                               Y0,Y1)
        gptwosample_object.predict_model_likelihoods()
        gptwosample_object.predict_mean_variance(Tpredict)
        if verbose: print("GPTwoSample:Prediction: %s" % (gptwosample_object.bayes_factor()))
        line = [gene_name, gptwosample_object.bayes_factor()]
        common, individual = get_exp_on_timeshift_right(\
            gptwosample_object, n_replicates_1,n_replicates_2, dim)
        line.extend(common)
        line.extend(individual)
        if(plotting):
            if verbose: print("GPTwoSample:Plotting")
            if(timeshift):
                plot_results(gptwosample_object,
                     shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2 + n_replicates_1+n_replicates_2],
                     draw_arrows=2, legend=False,
                     xlabel="Time [h]", ylabel="Expression level",
                     title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f$' % (gene_name, gptwosample_object.bayes_factor()))
            else:
                plot_results(gptwosample_object, title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, gptwosample_object.bayes_factor()))
            xlim(T1.min(), T1.max())
            savefig(os.path.join(out_dir, "%s.png") % (gene_name), format='png')
            if verbose: print("GPTwoSample:Saving Figure %s" % (os.path.join(out_dir, "%s.png") % (gene_name)))
            if hold:
                print("GPTwoSampleInterval:Hold for Showing (Close figure to continue...")
                show()
            clf()
        if verbose: print("GPTwoSample:Writing back" % (line))
        csv_out.writerow(line)
        csv_out_file.flush()

    if plotting:
        del xlim, show, savefig, clf
        del plot_results

def get_covariance_function(dim, gene_length, n_replicates_1, n_replicates_2):
    global timeshift
    if verbose: print "GPTwoSample:Calculating covariance function(s)"
    from pygp.covar import se, noise, combinators
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    if not timeshift:
        CovFun = combinators.SumCF((SECF, noiseCF))
    else:
        replicate_indices_1 = get_replicate_indices(n_replicates_1, gene_length)
        replicate_indices_2 = get_replicate_indices(n_replicates_2, gene_length)
        shiftCFInd1 = combinators.ShiftCF(SECF, replicate_indices_1)
        shiftCFInd2 = combinators.ShiftCF(SECF, replicate_indices_2)
        shiftCFCom = combinators.ShiftCF(SECF, \
                                         scipy.concatenate((replicate_indices_1, \
                                                            replicate_indices_2\
                                                            + n_replicates_1)))
#        importprintb;pdb.set_trace()
        CovFun = [combinators.SumCF((shiftCFInd1, noiseCF)),
                combinators.SumCF((shiftCFInd2, noiseCF)),
                combinators.SumCF((shiftCFCom, noiseCF))]
    del se, noise, combinators
    return CovFun

def get_priors(dim, n_replicates_1, n_replicates_2):
    global timeshift
    if verbose: print "GPTwoSample:Calculating priors"
    from pygp.priors import lnpriors
    if(timeshift):
        covar_priors_common = []
        covar_priors_individual_1 = []
        covar_priors_individual_2 = []
        
        #scale
        covar_priors_common.append([lnpriors.lnGammaExp, [1, 2]])
        covar_priors_individual_1.append([lnpriors.lnGammaExp, [1, 2]])
        covar_priors_individual_2.append([lnpriors.lnGammaExp, [1, 2]])
        #length-scale
        for i in range(dim):
            covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
            covar_priors_individual_1.append([lnpriors.lnGammaExp, [1, 1]])
            covar_priors_individual_2.append([lnpriors.lnGammaExp, [1, 1]])
            
        #shift
        for i in range(n_replicates_1 + n_replicates_2):
            covar_priors_common.append([lnpriors.lnGauss, [0, .5]])    
        for i in range(n_replicates_1):
            covar_priors_individual_1.append([lnpriors.lnGauss, [0, .5]])    
        for i in range(n_replicates_2):
            covar_priors_individual_2.append([lnpriors.lnGauss, [0, .5]])    
        #noise
        for i in range(1):
            covar_priors_common.append([lnpriors.lnGammaExp, [1, 1]])
            covar_priors_individual_1.append([lnpriors.lnGammaExp, [1, 1]])
            covar_priors_individual_2.append([lnpriors.lnGammaExp, [1, 1]])
        priors = get_model_structure({'covar':\
                            scipy.array(covar_priors_individual_1)}, \
                            # scipy.array(covar_priors_individual_2)]}, \
                           {'covar':scipy.array(covar_priors_common)})
    else:
        covar_priors = []
        #scale
        covar_priors.append([lnpriors.lnGammaExp, [1, 2]])
        for i in range(dim):
            covar_priors.append([lnpriors.lnGammaExp, [1, 1]])

        #noise
        for i in range(1):
            covar_priors.append([lnpriors.lnGammaExp, [1, 1]])
        priors = {'covar':scipy.array(covar_priors)}
    del lnpriors
    return priors


def get_replicate_indices(n_replicates, gene_length):
    replicate_indices = []
    for rep in scipy.arange(n_replicates):
        replicate_indices.extend(scipy.repeat(rep, gene_length))
    return scipy.array(replicate_indices)

def get_header_for_covar(CovFun, CovFunInd=None):
    if CovFunInd is None:
        CovFunInd = CovFun
    ret = map(lambda x:"Common " + x, CovFun.get_hyperparameter_names())
    ret.extend(map(lambda x:"Individual " + x, \
                   CovFunInd.get_hyperparameter_names()))
    return ret

if __name__ == '__main__':
    main()

