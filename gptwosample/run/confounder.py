'''
Created on Mar 14, 2013

@author: Max
'''
from gptwosample.run import started, finished

def run_confounder_twosample(confoundertwosample):
    s = "learning confounders..."
    started(s)
    restarts = 10
    for r in range(restarts):
        try:
            gtol = 1. / (10 ** (12 - r))
            # sys.stdout.write(os.linesep)
            # sys.stdout.flush()
            confoundertwosample.learn_confounder_matrix(messages=False, gradient_tolerance=gtol)
            break
        except KeyboardInterrupt:
            raise
        except:
            started("{} restart {}".format(s, r + 1))
            pass
    else:
        raise Exception("no confounders found after {} restarts".format(restarts))
    finished(s)
    return confoundertwosample
