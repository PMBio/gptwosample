'''
Command line tool for usage of GPTwoSample

Created on Jun 15, 2011

@author: Max Zwiessele, Oliver Stegle
'''
print "Welcome to GPTwoSample. Loading modules, be patient..."
import logging
import getopt
import scipy
import pylab
import sys
import csv
import os
from gptwosample.data.dataIO import get_data_from_csv
from pygp.priors import lnpriors
from pygp.covar import se, noise, combinators
from gptwosample.data.data_base import get_model_structure,\
    get_training_data_structure, common_id, individual_id
from gptwosample.twosample.twosample_compare import GPTimeShift, GPTwoSampleMLII
from gptwosample.plot.plot_basic import plot_results
from gptwosample.plot.interval import plot_results_interval
from gptwosample.twosample.interval_smooth import GPTwoSampleInterval

#global plotting
#global show
#global timeshift
#global interval
#global gibbs_iterations
#global delim
#global out_dir
    
def usage():
    usage = []
    usage.append('Usage: gptwosample [options] <file1> <file2> \n')
    usage.append('\n')
    usage.append('Perform GPTwoSample on each gene included in both files, file1 and file2. See file format below \n')
    usage.append('\n')
    usage.append('Options:\n')
    usage.append('\t-h|--help\t Show this text\n')
    usage.append('\t-v|--verbose\t Verbose mode\n')
    usage.append('\t-p|--plot \t Plot all genes and save to file outdir/{gene_name}.png\n')
    usage.append('\t-s|--show \t Hold between each gene, showing current results as plot \n')
    usage.append('\t-o|--out_dir=<out_dir> \t ["./"] Save results to out_dir. See file format below\n')
    usage.append('\t-t|--timeshift=\t Perform GPTimeShift on data\n')
    usage.append('\t-i|--interval=<iterations> \t Perform Interval prediction for iterations gibbs iterations on data\n')
    usage.append('\t-d|--delimiter=<delimiter> \t [","] Specify delimiter for reading and writing\n')
    usage.append('\n')

    return ''.join(usage)

def main(argv):
    print "Modules loaded, starting to GPTwoSample!"
    plotting = False
    show = False
    timeshift = False
    interval=False
    gibbs_iterations = 0
    delim = ','
    out_dir = os.path.abspath("./")
    
    try:
        opts, args = getopt.getopt(argv, "hvpso:ti:d:", ["help", "verbose", "plot", "show", "out_dir=", "timeshift", "interval=", "delimiter="])         
    except getopt.GetoptError:
        print usage()
        sys.exit(2)
    for opt,arg in opts:
        if(opt in ('-h', '--help')):
            print usage()
            sys.exit()
        elif opt in ('-o', '--out_dir'):
            out_dir = os.path.abspath(arg)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        elif opt in ('-t', '--timeshift'):
            timeshift=True
        elif opt in ('-i', '--interval'):
            interval=True
            gibbs_iterations = int(arg)
        elif(opt in ('-p', '--plot')):
            plotting = True
        elif(opt in ('-s', '--show')):
            show = True
            plotting = True
        elif(opt in ('-v', '--verbose')):
            logging.basicConfig(level=logging.INFO)
        elif(opt in ('-d', 'delimiter')):
            delim = arg
            
    csv_out_file = open(os.path.join(out_dir,"result.csv"),'wb')
    csv_out = csv.writer(csv_out_file,delimiter=delim)
    #range where to create time local predictions ? 
    #note: this need to be [T x 1] dimensional: (newaxis)
    # read data
    logging.info('Reading data from file:\n\t%s'%(os.path.abspath(args[0])))
    cond1 = get_data_from_csv(args[0], delimiter=delim)
    logging.info('Reading data from file:\n\t%s'%(os.path.abspath(args[1])))
    cond2 = get_data_from_csv(args[1], delimiter=delim)
    # get input from data 
    Tpredict = scipy.linspace(cond1["input"].min(), cond1["input"].max(), 100).reshape(-1,1)
    T1 = cond1.pop("input")
    T2 = cond2.pop("input")
    
    # gene_names for later writing and plotting
    gene_names = sorted(cond1.keys()) 
    assert gene_names == sorted(cond2.keys()), "Genes have to be includet in both condition files!"

    # replicate information for shift cf
    n_replicates = cond1[gene_names[0]].shape[0]
    gene_length = len(T1)
    dim = 1
    # get covariance function right
    CovFun = get_covariance_function(dim, gene_length, n_replicates, timeshift)
    # Make shure output file has right header and
    # get gptwosample_object right
    header = ["Gene", "Bayes Factor"]
    if(timeshift):
        header.extend(get_header_for_covar(CovFun[0]))
        gptwosample_object = GPTimeShift(CovFun,priors=get_priors(dim, n_replicates, timeshift))
    else:
        header.extend(get_header_for_covar(CovFun))
        gptwosample_object = GPTwoSampleMLII(CovFun,priors=get_priors(dim, n_replicates, timeshift))
    if(interval):
        header.append("Interval Indicators")
        header.append("Interval Probabilities")
    
    csv_out.writerow(header)
    
    if interval:
        perform_interval(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, csv_out, out_dir, gibbs_iterations, plotting, show, timeshift, n_replicates, delim, dim)
    else:
        # now lets run GPTwoSample on each gene in data:
        perform_gptwosample(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, csv_out, out_dir, plotting, show, timeshift, n_replicates, dim)


def get_exponten_right(gptwosample_object, timeshift, n_replicates, dim):
    common = gptwosample_object.get_learned_hyperparameters()[common_id]['covar']
    individual = gptwosample_object.get_learned_hyperparameters()[individual_id]['covar']
    if (timeshift):
        timeshift_index = scipy.array(scipy.ones_like(common),dtype='bool')
        timeshift_index[dim + 1:dim + 1 + 2 * n_replicates] = 0
        common[timeshift_index] = scipy.exp(common[timeshift_index])
        timeshift_index = scipy.array(scipy.ones_like(individual),dtype='bool')
        timeshift_index[dim + 1:dim + 1 + n_replicates] = 0
        individual[timeshift_index] = scipy.exp(individual[timeshift_index])
    return common, individual

def perform_interval(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, 
                     csv_out, out_dir, gibbs_iterations, plotting, show, timeshift, n_replicates, delim, dim):
    
    logthetaZ = scipy.log([.5, 2, .001])
    lik = scipy.log([2, 2])
    for gene_name in gene_names:
        logging.info("GPTwoSampleInterval:Processing %s"%(gene_name))
        Y0 = cond1[gene_name]
        Y1 = cond2[gene_name]
        gptwosample_object.set_data(get_training_data_structure(scipy.tile(T1,Y0.shape[0]).reshape(-1, 1),
                                                              scipy.tile(T2,Y1.shape[0]).reshape(-1, 1),
                                                              Y0.reshape(-1, 1),
                                                              Y1.reshape(-1, 1)))
        gptwosample_object.predict_model_likelihoods()

        line = [gene_name, gptwosample_object.bayes_factor()]
        common, individual = get_exponten_right(gptwosample_object, timeshift, n_replicates, dim)
        line.extend(common)
        line.extend(individual)
        
        gptest = GPTwoSampleInterval(gptwosample_object, outlier_probability=.1)
        Z = gptest.predict_interval_probabilities(Tpredict, hyperparams={'covar':logthetaZ, 'lik':lik},
                                                  number_of_gibbs_iterations=gibbs_iterations)
        #logging.info("\t\tPrediction %s"%(" ".join(Z)))
        if(plotting):
            logging.info("GPTwoSampleInterval:Plotting")
            if(timeshift):
                plot_results_interval(gptest, 
                     shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates], 
                     draw_arrows=2,legend=False,
                     xlabel="Time [h]",ylabel="Expression level",
                     title=r'TimeShift: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gptwosample_object.bayes_factor()))
            else:
                plot_results_interval(gptest,title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, gptwosample_object.bayes_factor())) 
            pylab.xlim(T1.min(), T1.max())
            pylab.savefig(os.path.join(out_dir,"%s.png")%(gene_name),format='png')
            logging.info("GPTwoSampleInterval:Saving Figure %s"%(os.path.join(out_dir,"%s.png")%(gene_name)))
        if show:
            logging.info("GPTwoSampleInterval:Hold for Showing")
            pylab.show()
        pylab.clf()
        if(delim==","):
            internal_delim=';'
        else:
            internal_delim=','
        line.append(internal_delim.join(scipy.array(gptest.get_predicted_indicators()),dtype="str"))
        line.append(internal_delim.join(scipy.array(Z,dtype="str")))
        logging.info("GPTwoSampleInterval:Writing back %s"%(line))
        csv_out.writerow(line)
    print "GPTwoSample:Written to %s"%(out_dir)

def perform_gptwosample(gptwosample_object, cond1, cond2, Tpredict, T1, T2, gene_names, 
                        csv_out, out_dir, plotting, show, timeshift, n_replicates, dim):
    for gene_name in gene_names:
        Y0 = cond1[gene_name]
        Y1 = cond2[gene_name]
        logging.info("GPTwoSample:Processing %s"%(gene_name))
        gptwosample_object.set_data(get_training_data_structure(scipy.tile(T1,Y0.shape[0]).reshape(-1, 1),
                                                              scipy.tile(T2,Y1.shape[0]).reshape(-1, 1),
                                                              Y0.reshape(-1, 1),
                                                              Y1.reshape(-1, 1)))
        gptwosample_object.predict_model_likelihoods()
        gptwosample_object.predict_mean_variance(Tpredict)
        line = [gene_name, gptwosample_object.bayes_factor()]
        common, individual = get_exponten_right(gptwosample_object, timeshift, n_replicates, dim)
        line.extend(common)
        line.extend(individual)
        if(plotting):
            logging.info("GPTwoSample:Plotting")
            if(timeshift):
                plot_results(gptwosample_object, 
                     shift=gptwosample_object.get_learned_hyperparameters()[common_id]['covar'][2:2+2*n_replicates], 
                     draw_arrows=2,legend=False,
                     xlabel="Time [h]",ylabel="Expression level",
                     title=r'TimeShift: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gptwosample_object.bayes_factor()))
            else:
                plot_results(gptwosample_object,title=r'%s: $\log(p(\mathcal{H}_I)/p(\mathcal{H}_S)) = %.2f $' % (gene_name, gptwosample_object.bayes_factor()))
            pylab.xlim(T1.min(), T1.max())
            pylab.savefig(os.path.join(out_dir,"%s.png")%(gene_name),format='png')
            logging.info("GPTwoSample:Saving Figure %s"%(os.path.join(out_dir,"%s.png")%(gene_name)))
        if show:
            logging.info("GPTwoSample:Hold for Showing")
            pylab.show()
        pylab.clf()
        logging.info("GPTwoSample:Writing back %s"%(line))
        csv_out.writerow(line)
    print "GPTwoSample:Written to %s"%(out_dir)

def get_covariance_function(dim, gene_length, n_replicates, timeshift):
    SECF = se.SqexpCFARD(dim)
    noiseCF = noise.NoiseCFISO()
    if not timeshift:
        return combinators.SumCF((SECF,noiseCF))
    else:
        replicate_indices = get_replicate_indices(n_replicates, gene_length)
        shiftCFInd1 = combinators.ShiftCF(SECF,replicate_indices)
        shiftCFInd2 = combinators.ShiftCF(SECF,replicate_indices)
        shiftCFCom = combinators.ShiftCF(SECF,scipy.concatenate((replicate_indices,replicate_indices+n_replicates)))
        return [combinators.SumCF((shiftCFInd1,noiseCF)),
                combinators.SumCF((shiftCFInd2,noiseCF)),
                combinators.SumCF((shiftCFCom,noiseCF))]
        
def get_priors(dim, n_replicates, timeshift):
    covar_priors_common = []
    covar_priors_individual = []
    covar_priors = []
    #scale
    covar_priors_common.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors_individual.append([lnpriors.lnGammaExp,[1,2]])
    covar_priors.append([lnpriors.lnGammaExp,[1,2]])
    for i in range(dim):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    #shift
    for i in range(2*n_replicates):
        covar_priors_common.append([lnpriors.lnGauss,[0,.5]])    
    for i in range(n_replicates):
        covar_priors_individual.append([lnpriors.lnGauss,[0,.5]])    
    #noise
    for i in range(1):
        covar_priors_common.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors_individual.append([lnpriors.lnGammaExp,[1,1]])
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
    if(timeshift):
        return get_model_structure({'covar':scipy.array(covar_priors_individual)}, {'covar':scipy.array(covar_priors_common)})
    else:
        return {'covar':scipy.array(covar_priors)}
    
def get_replicate_indices(n_replicates, gene_length):
    replicate_indices = []
    for rep in scipy.arange(n_replicates):
        replicate_indices.extend(scipy.repeat(rep,gene_length))
    return scipy.array(replicate_indices)
def get_header_for_covar(CovFun):
    ret = map(lambda x:"Common " + x, CovFun.get_hyperparameter_names())
    ret.extend(map(lambda x:"Individual " + x, CovFun.get_hyperparameter_names()))
    return ret

if __name__ == '__main__':
    main(sys.argv[1:])

