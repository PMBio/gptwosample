'''
Created on Feb 18, 2011

@author: maxz
'''
import pylab as PL
import scipy as SP
import copy as CP

def plot_results_gradient(self,GP,M,X,RX,format_fill,format_line):
        """
        Plot results of resampling of a :py:class:`gptwosample.twosample
        """
        def fill_between(X,Y1,Y2,P=1,**format):
            """fill between Y1 and Y2"""
            _format = CP.copy(format)
            _format['alpha']*=P
            X[0]+=0
            Xp = SP.concatenate((X,X[::-1]))
            Yp = SP.concatenate(((Y1),(Y2)[::-1]))
            PL.fill(Xp,Yp,**_format)
        
        #vectorized version of X for G if needed
        if len(X.shape)<2:
            Xv = X.reshape(X.size,1)
        else:
            Xv = X
        #regression:
        [p_mean,p_std] = self.regress(GP,M,X=Xv)
        
        #plot std errorbars where alpha-value is modulated by RX (a bit of a hack)
        Y1 = p_mean+2*p_std
        Y2 = p_mean-2*p_std
        #set line width to 0 (no boundaries
        format_fill['linewidth']=0
        for i in xrange(X.shape[0]-2):
            fill_between(X[i:i+2],Y1[i:i+2],Y2[i:i+2],P=RX[i],**format_fill)
        #plot contours
        PL.plot(X,Y1,format_fill['facecolor'],linewidth=1,alpha=format_fill['alpha'])
        PL.plot(X,Y2,format_fill['facecolor'],linewidth=1,alpha=format_fill['alpha'])
#        Xp = concatenate((X,X[::-1]))
#        Yp = concatenate(((p_mean+2*p_std),(p_mean-2*p_std)[::-1]))
#        PL.fill(Xp,Yp,**format_fill)
        PL.plot(X,p_mean,**format_line)