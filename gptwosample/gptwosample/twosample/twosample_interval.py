"""Two sample test on intervals or clustered two sample test"""

import scipy as SP
import pylab as PL

from twosample import GPTwoSampleMLII
from plot.hinton import hinton
from pygp.gp.basic_gp import GP
from pygp.gp.composite import GroupGP

class GPTwoSampleInterval(GPTwoSampleMLII):
    __slots__=["prior_Z"]


    def __init__(self,prior_Z=0.5,**kwargin):       
        """__init__(data,prior_Z,**kwargin)
        - data: TSdata object
        - covar: covariance function (instance)
        - logtheta0: start point for optimzation of covariance function
        - maxiter:  iterations for optimization
        - priors:   list with priors for each hyper parameter
        - dimension: data dimension(1)
        - opt: perform optimization?
        - Smean: subtract mean ?
        """
        super(GPTwoSampleInterval, self).__init__(**kwargin)
#        GPTwoSampleMLII.__init__(self,**kwargin)
        self.prior_Z  = SP.array([1-prior_Z,prior_Z])
        

        pass



    def M2GPxy(self,M):
        """M2GPxy(M)
        - conversion function returning x/y in arrays suitable for the gpr module
        - the standard version of this helper function does not really do much, but see the cluster variant to find out what this is for"""
        x1 = M[0]
        y1 = M[1]
        x = SP.zeros([0,2])
        y = SP.zeros([0])
        #1. append cluster index and add to input data
        for i in range(len(x1)):
            _y = y1[i]
            _x = x1[i]
            _x = _x.reshape([_x.shape[0],1])
            _x = SP.concatenate((_x,SP.ones_like(_x)),axis=1)
            #index of replicate
            _x[:,1] = i
            x  = SP.concatenate((x,_x),axis=0)
            y  = SP.concatenate((y,_y),axis=0)
        return [x,y]


    def test_interval(self,M0,M1,verbose=True,opt=None,Ngibbs_iterations=None):
        """test for differential expression with clustering model
        - returns a data structure which reflects the time of sepataoin (posterior over Z)
        """

        def updateGP():
            """update the GP datasets and re-evaluate the Ep approximate likelihood"""
            #0. update the noise level in accordance with the responsibilities
            for t in range(T):
                XS[:,t:R*T:T,-1] = 1/(Z[1,t]+1E-6)
                XJ[:,t:R*T:T,-1] = 1/(Z[0,t]+1E-6)

            GPS.setData(XS,Y)
            #here we joint the two conditions
            GPJ.setData(SP.concatenate(XJ,axis=0),SP.concatenate(Y,axis=0))
            #1. set the data to both processes

        MJ = [SP.concatenate((M0[0],M1[0]),axis=0),SP.concatenate((M0[1],M1[1]),axis=0)]


        C  = 2              #conditions
        R  = M0[0].shape[0] #repl.
        T  = M0[0].shape[1] #time
        D  = 2              #dim.


        #Responsibilities: components(2) x time                        
        Z  = 0.5*SP.ones((2,T))

        #Data(X/Y): conditions x replicates x time x 2D
        X  = SP.zeros((C,R*T,D))
        Y  = SP.zeros((C,R*T))
        #unique times
        XT = SP.ones((T,2))
        XT[:,0] = M0[0][0,:]

        [x0,y0] = self.M2GPxy(M0)
        [x1,y1] = self.M2GPxy(M1)
        
        X[0,:,0:2] = x0
        X[1,:,0:2] = x1
        Y[0,:]     = y0
        Y[1,:]     = y1
        #create indicator vector to identify unique time points

        
        #create one copy of the input per process as this is used for input dependen noise
        XS       = X.copy()
        XJ       = X.copy()

        #get hyperparameters form standard test:
        ratio = self.test(M0,M1,verbose=False,opt=opt)
        logtheta_s = self.gpr_0.logtheta
        logtheta_j = self.gpr_join.logtheta
        #initialize the two GPs
        if self.logtheta0 is None:
            logtheta = self.covar.getDefaultParams()

        #the two indv. GPs
        GP0 = GP(self.covar,Smean=self.Smean,logtheta=logtheta_s)
        GP1 = GP(self.covar,Smean=self.Smean,logtheta=logtheta_s)
        #the group GP summarising the two indiv. processes
        GPS = GroupGP([GP0,GP1])
        #the joint process
        GPJ = GP(self.covar,Smean=self.Smean,logtheta=logtheta_j)
        #update the GP
        updateGP()


        debug_plot = True

#        for i in range(1):
        ###iterations###
        #1. get predictive distribution for both GPs
        ##debug
        #introduce the additional dimension to accom. the per obs. noise model
        #get prediction for all time points
        Yp0 = GP0.predict(GP0.logtheta,XT)
        Yp1 = GP1.predict(GP1.logtheta,XT)
        Ypj = GPJ.predict(GPJ.logtheta,XT)
        #considere residuals
        D0  = ((M0[1]-Yp0[0])**2 * (1/Yp0[1])).sum(axis=0)
        D1  = ((M1[1]-Yp1[0])**2 * (1/Yp1[1])).sum(axis=0)
        DJ  = ((MJ[1]-Ypj[0])**2 * (1/Ypj[1])).sum(axis=0)
        #the indiv. GP is the sum
        DS  = D0+D1
        #now use this to restimate Q(Z)
        ES  = SP.exp(-DS)
        EJ  = SP.exp(-DJ)
        #

        Z[0,:] =self.prior_Z[0]*EJ
        Z[1,:] =self.prior_Z[1]*ES
        Z     /=Z.sum(axis=0)
#             pydb.set_trace()
        updateGP()


        if verbose:
            PL.clf()
            labelSize = 15
            tickSize  = 12
            
            #1. plot the gp predictions
            ax1=PL.axes([0.1,0.1,0.8,0.7])
            Xt_ = SP.linspace(0,XT[:,0].max()+2,100)
            Xt  = SP.ones((Xt_.shape[0],2))
            Xt[:,0] = Xt_

            self.plotGPpredict(GP0,M0,Xt,{'alpha':0.1,'facecolor':'r'},{'linewidth':2,'color':'r'})
            self.plotGPpredict(GP1,M0,Xt,{'alpha':0.1,'facecolor':'g'},{'linewidth':2,'color':'g'})
            self.plotGPpredict(GPJ,M0,Xt,{'alpha':0.1,'facecolor':'b'},{'linewidth':2,'color':'b'})
            PL.plot(M0[0].T,M0[1].T,'r.--')
            PL.plot(M1[0].T,M1[1].T,'g.--')
            
            PL.xlim([Xt.min(),Xt.max()])
            #remove last ytick to avoid overlap
            yticks = ax1.get_yticks()[0:-2]
            ax1.set_yticks(yticks)
            PL.xlabel('Time/h',size=labelSize)
            PL.ylabel('Log expression level',size=labelSize)

            #now plot hinton diagram with responsibilities on top
            ax2=PL.axes([0.1,0.715,0.8,0.2],sharex=ax1)
#            ax2=PL.axes([0.1,0.7,0.8,0.2])
            #PL.plot(XT[:,0],Z[1,:])
            #swap the order of Z for optical purposes
            Z_= SP.ones_like(Z)
            Z_[1,:] = Z[0,:]
            Z_[0,:] = Z[1,:]
            hinton(Z_,X=M0[0][0])
            PL.ylabel('diff.')
            
            #hide axis labels
            PL.setp( ax2.get_xticklabels(), visible=False)
            #font size
            PL.setp( ax1.get_xticklabels(), fontsize=tickSize)
            PL.setp( ax1.get_yticklabels(), fontsize=tickSize)
            PL.setp( ax2.get_xticklabels(), fontsize=tickSize)
            #axes label
        return Z
        pass


    def plotGPpredict(self,GP,M,X,format_fill,format_line):
        """plotGPpredict(GP,X,format_fill,format_line)
        - GP regression and plotting of std as +/2 2std and mean prediction"""
        #vectorized version of X for G if needed
        if len(X.shape)<2:
            Xv = X.reshape(X.size,1)
        else:
            Xv = X
        #regression:
        [p_mean,p_std] = self.regress(GP,M,X=Xv)
        #plot std
        X = X[:,0:1]
        
        Xp = SP.concatenate((X,X[::-1]))
        Yp = SP.concatenate(((p_mean+2*p_std),(p_mean-2*p_std)[::-1]))
        PL.fill(Xp,Yp,**format_fill)
        PL.plot(X,p_mean,**format_line)



