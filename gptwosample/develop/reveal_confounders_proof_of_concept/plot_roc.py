'''
Created on Feb 21, 2013

@author: Max
'''
import sys
from gptwosample.develop.reveal_confounders_proof_of_concept import simple_confounders,\
    no_confounders, ideal_confounders
import pylab
import os

if __name__ == '__main__':
    root=sys.argv[1]

    cond1_file='./../../examples/warwick_control.csv'
    cond2_file='../../examples/warwick_treatment.csv'
    

    sys.argv.append('plot_roc')
    
    pylab.ion()

    simple_confounders.run_demo(cond1_file, cond2_file, root=root)
    no_confounders.run_demo(cond1_file, cond2_file, root=root)
    ideal_confounders.run_demo(cond1_file, cond2_file, root=root)
    
    pylab.legend(loc=4)
    pylab.savefig(os.path.join(root,"roc.pdf"))
    