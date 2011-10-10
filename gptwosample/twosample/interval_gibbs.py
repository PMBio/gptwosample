"""
Two sample test on intervals
============================

This version uses proper gibbs resampling of the indicator variables.

"""

#path for pygp, stats
from gptwosample.plot import plot_results
from gptwosample.plot.hinton import hinton

import scipy as SP
import pylab as PL

#import pydb
#import copy

import logging

class GPTwoSampleInterval(object):
    __slots__=["prior_Z", "gptwosample_object", "PZ", "_unique_x_intervals"]

    def __init__(self, gptwosample_object, prior_Z=0.5,**kwargin):
        """
        Create instance of GPTwoSample for gibbs resampling all x values of 
        prediction. This class predicts on any instance of
        :py:class:`gptwosample.twosample.basic` object.
        
        **Parameters:**
        
        gptwosample_object : :py:class:`gptwosample.twosample.basic`
            The twosample object to sample from.
        
        
        __init__(data,covar=None,logtheta0=None,maxiter=20,,priors=None,dimension=1)
        - data: TSdata object
        - covar: covariance function (instance)
        - logtheta0: start point for optimzation of covariance function
        - maxiter:  iterations for optimization
        - priors:   list with priors for each hyper parameter
        - dimension: data dimension(1)
        - opt: perform optimization?
        - Smean: subtract mean ?
        - robust: use robust likelihood (False)
        """
        self.gptwosample_object = gptwosample_object
        self._unique_x_intervals = None
        #prior for missing proportions
        self.prior_Z  = SP.array([1-prior_Z,prior_Z])
        self.PZ = None
        pass

    def predict_all_x_interval_probabilities(self,Ngibbs_iterations=10,*args,**kwargs):
        """
        Predict for each x interval, given by training_data of gptwosample_object,
        which model is more likely: 'common', or 'individual', 
        see :py:class:`gptwosample.twosample.basic`
        
        **Parameters:**
        
        Ngibbs_iterations : int
            Number of iterations to sample and optimize
        """
        self.gptwosample_object.predict_model_likelihoods(*args,**kwargs)
        bayes_factor = self.gptwosample_object.bayes_factor()
        
        #store original data
        original_group_0 = self.gptwosample_object._get_data('individual',0)
        original_group_1 = self.gptwosample_object._get_data('individual',1)
        original_common  = self.gptwosample_object._get_data('common')

        

        self._unique_x_intervals = SP.unique(original_common[0]).reshape(-1,1)
        number_of_x_intervals = self._unique_x_intervals.shape[0]
        
        interval_expert_groups = SP.ones([number_of_x_intervals],dtype='bool')
        interval_expert_common = SP.ones([number_of_x_intervals],dtype='bool')
        
        # Start with random interval_experts
        Z_ = SP.random.rand(interval_expert_groups.shape[0])>0.5
        self.PZ= SP.zeros([2,interval_expert_groups.shape[0]])
        Z = SP.zeros([Ngibbs_iterations,interval_expert_groups.shape[0]],dtype='bool')
        
        interval_expert_groups[~Z_] = False
        interval_expert_common[Z_]  = False
        
        # Now sample 
        for n in range(Ngibbs_iterations):
#            for i in random.permutation(ISP.shape[0]):
            for i in SP.arange(interval_expert_groups.shape[0]):
                #resample a single indicator
                interval_indices = SP.zeros(number_of_x_intervals, dtype='bool')
                interval_indices[i] = True
                [Z_,PZ_]  = self._sample_x_interval_probability(interval_indices,
                                                                interval_expert_groups,
                                                                interval_expert_common,
                                                                original_group_0,
                                                                original_group_1,
                                                                original_common)
                #save prob. and indicator value
                self.PZ[:,i] = SP.squeeze(PZ_)
                Z[n,i]  = SP.squeeze(Z_)
                #update indicators
                interval_expert_groups[i] = Z_
                interval_expert_common[i] = ~Z_
            logging.debug("Gibbs iteration:%d" % (n))
            logging.debug(self.PZ)
            logging.debug(interval_expert_groups)
            
        return [bayes_factor, self.PZ]
        
    def plot_predicted_results(self, PZ=None, title="", *args, **kwargs):
        if PZ is None:
            PZ=self.PZ
        self.gptwosample_object.predict_mean_variance(SP.linspace(self._unique_x_intervals.min(), self._unique_x_intervals.max(), 100).reshape(-1,1),*args,**kwargs)
        
        labelSize = 15
        tickSize  = 12
        
        #1. plot the gp predictions
        ax1 = PL.axes([0.1,0.1,0.8,0.7])
        ax1.xlim([self._unique_x_intervals.min(),self._unique_x_intervals.max()])
        
        plot_results(self.gptwosample_object, title=title, *args, **kwargs)
        
        #remove last ytick to avoid overlap
        #yticks = ax1.get_yticks()[0:-2]
        #ax1.set_yticks(yticks)
#        Ymax = MJ[1].max()
#        Ymin = MJ[1].min()
#        DY   = Ymax-Ymin
#        PL.ylim([Ymin-0.1*DY,Ymax+0.1*DY])

        #now plot hinton diagram with responsibilities on top
        ax2=PL.axes([0.1,0.83,0.8,0.1], sharex=ax1)
        
        PL.axes(ax2)
#        Z_= SP.ones_like(Z)
#        Z_[1,:] = Z[0,:]
#        Z_[0,:] = Z[1,:]
        PZ_= SP.ones_like(PZ)
        PZ_[1,:] = PZ[0,:]
        PZ_[0,:] = PZ[1,:]
        hinton(PZ_,X=self._unique_x_intervals)
        PL.ylabel('diff.')
        #hide axis labels
        PL.setp( ax2.get_xticklabels(), visible=False)
        #PL.setp( ax1.get_xticklabels(), fontsize=tickSize)
        #PL.setp( ax1.get_yticklabels(), fontsize=tickSize)
        #PL.setp( ax2.get_xticklabels(), fontsize=tickSize)
############## PRIVATE ############


    def _sample_x_interval_probability(self, interval_indices, 
                                      interval_expert_groups, 
                                      interval_expert_common, 
                                      original_group_0,
                                      original_group_1,
                                      original_common,
                                      *args, **kwargs):
        """
        Predict which model is more likely for given interval_indices.
        
        **Parameters:**
        
        interval_indices : [ bool]
            Scipy array-like of boolean indicators denoting 
            which x indices to predict from data. 
            
        prediction_x_indices : [int]
            scipy array-like, which holds all x indices from which to sample from.
        """
        number_of_xs = interval_indices.sum()

        interval_expert_groups &= ~interval_indices
        interval_expert_common &= ~interval_indices 
        
        #interval_expert_group_0 = interval_expert_groups[original_common[0] == original_group_0[0]] 
        #interval_expert_group_1 = interval_expert_groups[original_common[0] == original_group_1[0]] 
        
        PZ = SP.zeros([2,number_of_xs])

        # predict in interval given
        prediction = self.gptwosample_object.predict_mean_variance(self._unique_x_intervals[interval_indices],*args,**kwargs)
        
        mean_group_0 = prediction['individual']['mean'][0]#self.gpr_0.predict(self.gpr_0.logtheta,XT[I],mean=False)
        mean_group_1 = prediction['individual']['mean'][1]
        mean_common  = prediction['common']['mean']

        var_group_0 = prediction['individual']['var'][0]#self.gpr_0.predict(self.gpr_0.logtheta,XT[I],mean=False)
        var_group_1 = prediction['individual']['var'][1]
        var_common  = prediction['common']['var']

        #compare to hypothesis
        #calculate log likelihoods for every time step under different models

        #D0  = -0.5*(M0R[1][:,I]-Yp0[0])**2/Yp0[1] -0.5*SP.log(Yp0[1])
        #D1  = -0.5*(M1R[1][:,I]-Yp1[0])**2/Yp1[1] -0.5*SP.log(Yp1[1])
        #DJ  = -0.5*(MJR[1][:,I]-Ypj[0])**2/Ypj[1] -0.5*SP.log(Ypj[1])
        
        #robust likelihood
        c =0.9
        
        #import pdb;pdb.set_trace()
        import pdb;pdb.set_trace()
        
        D0  = c*SP.exp(-0.5*(original_group_0[1].transpose()[interval_expert_groups].transpose()-mean_group_0)**2/var_group_0)        
        D0 *= 1/SP.sqrt(2*SP.pi*var_group_0)
        D0 += (1-c)* SP.exp(-0.5*(original_group_0[1].transpose()[interval_expert_groups].transpose()-mean_group_0)**2/1E8)
        D0 /= SP.sqrt(2*SP.pi*SP.sqrt(1E8))
        D0 = SP.log(D0)
        
        D1  = c*SP.exp(-0.5*(original_group_1[1].transpose()[interval_expert_groups].transpose()-mean_group_1)**2/var_group_1)
        D1 *= 1/SP.sqrt(2*SP.pi*var_group_1)
        D1 += (1-c)* SP.exp(-0.5*(original_group_1[1].transpose()[interval_expert_groups].transpose()-mean_group_1)**2/1E8)
        D1 /= SP.sqrt(2*SP.pi*SP.sqrt(1E8))
        D1 = SP.log(D1)
        
        DJ  = c*SP.exp(-0.5*(original_common[1].transpose()[interval_expert_common].transpose()-mean_common)**2/var_common)
        DJ *= 1/SP.sqrt(2*SP.pi*var_common)
        DJ += (1-c)* SP.exp(-0.5*(original_common[1].transpose()[interval_expert_common].transpose()-mean_common)**2/1E8)
        DJ /= SP.sqrt(2*SP.pi*SP.sqrt(1E8))
        DJ = SP.log(DJ)
        
        DS  = D0.sum(axis=0) + D1.sum(axis=0)
        DJ  = DJ.sum(axis=0)
        ES  = SP.exp(DS)
        EJ  = SP.exp(DJ)
        
        PZ[0,:] = self.prior_Z[0]*EJ
        PZ[1,:] = self.prior_Z[1]*ES
        
        PZ      /= PZ.sum(axis=0)
        #sample indicators
        Z       = SP.rand(number_of_xs)<=PZ[1,:]
        if(interval_expert_groups.sum()==1):
            Z = True
        if(interval_expert_common.sum()==1):
            Z = False
        return [Z,PZ]


#    def test_interval(self,M0,M1,verbose=False,opt=None,Ngibbs_iterations=10):
#        """test with interval sampling test
#        M0: first expresison condition
#        M1: second condition
#        verbose: produce plots (False)
#        opt: optimise hyperparameters(True)
#        Ngibbs_iterations (10)
#        """
#
#        def rescaleInputs(X,gpr):
#            """copy implementation of input rescaling
#            - this version rescales in a dum way assuming everything is float
#            """
#            X_ = (X-gpr.minX)*gpr.scaleX - 5
#            return X_
#        
#
#        def sampleIndicator(I,take_out=True):
#            """sample all indicators that are true in the index vector I
#            take_out: take indices out of the dataset first (yes)
#            """
#            PZ = SP.zeros([2,I.sum()])
#            SP.
#            if take_out:
#                IS_ = IS & (~I)
#                IJ_ = IJ & (~I)
#            else:
#                IS_ = IS
#                IJ_ = IJ
#            #set datasets
#            self.gpr_0.setData(SP.concatenate((M0R[0][:,IS_]),axis=0).reshape([-1,1]),
#                               SP.concatenate((M0R[1][:,IS_]),axis=1),process=False)
#            self.gpr_1.setData(SP.concatenate((M1R[0][:,IS_]),axis=0).reshape([-1,1]),
#                               SP.concatenate((M1R[1][:,IS_]),axis=1),process=False)
#            self.gpr_join.setData(SP.concatenate((MJR[0][:,IJ_]),axis=0).reshape([-1,1]),
#                                  SP.concatenate((MJR[1][:,IJ_]),axis=1),process=False)
#
#            
#            Yp0 = self.gpr_0.predict(self.gpr_0.logtheta,XT[I],mean=False)
#            Yp1 = self.gpr_1.predict(self.gpr_0.logtheta,XT[I],mean=False)
#            Ypj = self.gpr_join.predict(self.gpr_0.logtheta,XT[I],mean=False)
#            #compare to hypothesis
#            #calculate log likelihoods for every time step under different models
#
#            #D0  = -0.5*(M0R[1][:,I]-Yp0[0])**2/Yp0[1] -0.5*SP.log(Yp0[1])
#            #D1  = -0.5*(M1R[1][:,I]-Yp1[0])**2/Yp1[1] -0.5*SP.log(Yp1[1])
#            #DJ  = -0.5*(MJR[1][:,I]-Ypj[0])**2/Ypj[1] -0.5*SP.log(Ypj[1])
#            
#            #robust likelihood
#            c =0.9
#            D0  = c    * SP.exp(-0.5*(M0R[1][:,I]-Yp0[0])**2/Yp0[1])*1/SP.sqrt(2*SP.pi*Yp0[1])
#            D0 += (1-c)* SP.exp(-0.5*(M0R[1][:,I]-Yp0[0])**2/1E8)*1/SP.sqrt(2*SP.pi*SP.sqrt(1E8))
#            D0 = SP.log(D0)
#            D1  = c    * SP.exp(-0.5*(M1R[1][:,I]-Yp1[0])**2/Yp1[1])*1/SP.sqrt(2*SP.pi*Yp1[1])
#            D1 += (1-c)* SP.exp(-0.5*(M1R[1][:,I]-Yp1[0])**2/1E8)*1/SP.sqrt(2*SP.pi*SP.sqrt(1E8))
#            D1 = SP.log(D1)
#            DJ  = c    * SP.exp(-0.5*(MJR[1][:,I]-Ypj[0])**2/Ypj[1])*1/SP.sqrt(2*SP.pi*Ypj[1])
#            DJ += (1-c)* SP.exp(-0.5*(MJR[1][:,I]-Ypj[0])**2/1E8)*1/SP.sqrt(2*SP.pi*SP.sqrt(1E8))
#            DJ = SP.log(DJ)
#            
#            DS  = D0.sum(axis=0) + D1.sum(axis=0)
#            DJ  = DJ.sum(axis=0)
#            ES  = SP.exp(DS)
#            EJ  = SP.exp(DJ)
#            PZ[0,:] = self.prior_Z[0]*EJ
#            PZ[1,:] = self.prior_Z[1]*ES
#            
#            PZ      /= PZ.sum(axis=0)
#            #sample indicators
#            Z       = SP.rand(I.sum())<=PZ[1,:]
#            if(IS_.sum()==1):
#                Z = True
#            if(IJ_.sum()==1):
#                Z = False
#            return [Z,PZ]
#            pass
#        pass
#
#        #1. use the standard method to initialise the GP objects
#        ratio = self.test(M0,M1,verbose=verbose,opt=opt)
##        GP0  = CP.deepcopy(self.gpr_0)
##        GP1  = CP.deepcopy(self.gpr_1)
##        GPJ  = CP.deepcopy(self.gpr_join)
#        
#        #2. initialise gibbs samplign for time-local approximation
#        #get set of unique time coordinates
#        XT = M0[0][0].reshape([-1,1])
#        #rescale all datasets inputs
#        MJ = [SP.concatenate((M0[0],M1[0]),axis=0),SP.concatenate((M0[1],M1[1]),axis=0)]
#        MJR = CP.deepcopy(MJ)
#        M0R = CP.deepcopy(M0)
#        M1R = CP.deepcopy(M1)
#        #rescale and 0 mean
#        MJR[0] = rescaleInputs(MJ[0],self.gpr_join)
#        M0R[0] = rescaleInputs(M0[0],self.gpr_0)
#        M1R[0] = rescaleInputs(M1[0],self.gpr_1)
#        MJR[1]-= self.gpr_join.mean
#        M0R[1]-= self.gpr_0.mean
#        M1R[1]-= self.gpr_1.mean
#        
#        #index vector assigning each time point to either expert
#        IS = SP.ones([XT.shape[0]],dtype='bool')
#        IJ = SP.ones([XT.shape[0]],dtype='bool')
#        
#        #1. sample all indicators conditioned on current GP approximation
##        Z_ = sampleIndicator(SP.ones(XT.shape[0],dtype='bool'),take_out=False)
#        Z_ = SP.random.rand(IS.shape[0])>0.5
#        PZ= SP.zeros([2,IS.shape[0]])
#        Z = SP.zeros([Ngibbs_iterations,IS.shape[0]],dtype='bool')
#        #update the datasets in the GP, attention: rescaling might cause trouble here..
#        IS[~Z_] = False
#        IJ[Z_]  = False
#        #sample indicators one by one
#        for n in range(Ngibbs_iterations):
##            for i in random.permutation(ISP.shape[0]):
#            for i in SP.arange(IS.shape[0]):
#                #resample a single indicator
#                I = SP.zeros([IS.shape[0]],dtype='bool')
#                I[i] = True
#                [Z_,PZ_]  = sampleIndicator(I)
#                #save prob. and indicator value
#                PZ[:,i] = SP.squeeze(PZ_)
#                Z[n,i]  = SP.squeeze(Z_)
#                #update indicators
#                IS[i] = Z_
#                IJ[i] = ~Z_
#            LG.debug("Gibbs iteration:%d" % (n))
#            LG.debug(PZ)
#            LG.debug(IS)
#        n_ = round(Ngibbs_iterations)/2
#        Z  = Z[n_::]
#        PZ[1,:] = Z.mean(axis=0)
#        PZ[0,:] = 1-PZ[1,:]
#        GP0 = self.gpr_0
#        GP1 = self.gpr_1
#        GPJ = self.gpr_join
#        if verbose:
#            PL.clf()
#            #1. plot the gp predictions
#            ax1=PL.axes([0.1,0.1,0.8,0.7])
#            Xt_ = SP.linspace(0,XT[:,0].max()+2,100)
#            Xt  = Xt_.reshape([-1,1])
#            self.plotGPpredict(GP0,M0,Xt,{'alpha':0.1,'facecolor':'r'},{'linewidth':2,'color':'r'})
#            self.plotGPpredict(GP1,M0,Xt,{'alpha':0.1,'facecolor':'g'},{'linewidth':2,'color':'g'})
#            self.plotGPpredict(GPJ,M0,Xt,{'alpha':0.1,'facecolor':'k'},{'linewidth':2,'color':'k'})
#            PL.plot(M0[0].T,M0[1].T,'r.--')
#            PL.plot(M1[0].T,M1[1].T,'g.--')
#
#            
#            PL.xlim([Xt.min(),Xt.max()])
#            #remove last ytick to avoid overlap
#            yticks = ax1.get_yticks()[0:-2]
#            ax1.set_yticks(yticks)
#            PL.xlabel('Time/h')
#            PL.ylabel('Log expression level')
#            Ymax = MJ[1].max()
#            Ymin = MJ[1].min()
#            DY   = Ymax-Ymin
#            PL.ylim([Ymin-0.1*DY,Ymax+0.1*DY])
#
#            #now plot hinton diagram with responsibilities on top
#            ax2=PL.axes([0.1,0.715,0.8,0.2],sharex=ax1)
##           ax2=PL.axes([0.1,0.7,0.8,0.2])
#            #PL.plot(XT[:,0],Z[1,:])
#            #swap the order of Z for optical purposes
##            Z_= SP.ones_like(PZ)
##            Z_[1,:] = PZ[0,:]
##            Z_[0,:] = PZ[1,:]
#            hinton((PZ[::-1]),X=M0[0][0])
#            PL.ylabel('diff.')
#            #hide axis labels
#            PL.setp( ax2.get_xticklabels(), visible=False)
#            #font size
#            #setp( ax1.get_xticklabels(), fontsize=tickSize)
#            #setp( ax1.get_yticklabels(), fontsize=tickSize)
#            #setp( ax2.get_xticklabels(), fontsize=tickSize)
#            #PL.ylim(Ymin-0.1*DY,Ymax+0.1*DY)
#        return [ratio,PZ]
#        pass
#
#    
#        
#
#
#    def test_interval_old(self,gene_name,verbose=True,opt=None):
#        """test for differential expression with clustering model
#        - returns a data structure which reflects the time of sepataoin (posterior over Z)
#        """
#
#        def updateGP():
#            """update the GP datasets and re-evaluate the Ep approximate likelihood"""
#            #0. update the noise level in accordance with the responsibilities
#            for t in range(T):
#                XS[:,t:R*T:T,-1] = 1/(Z[1,t]+1E-6)
#                XJ[:,t:R*T:T,-1] = 1/(Z[0,t]+1E-6)
#
#            GPS.setData(XS,Y)
#            #here we joint the two conditions
#            GPJ.setData(SP.concatenate(XJ,axis=0),SP.concatenate(Y,axis=0))
#            #1. set the data to both processes
#
#        M0 = self.data.getExpr(gene_name,0)
#        M1 = self.data.getExpr(gene_name,1)
#        MJ = [SP.concatenate((M0[0],M1[0]),axis=0),SP.concatenate((M0[1],M1[1]),axis=0)]
#
#
#        C  = 2              #conditions
#        R  = M0[0].shape[0] #repl.
#        T  = M0[0].shape[1] #time
#        D  = 2              #dim.
#
#
#        #Responsibilities: components(2) x time                        
#        Z  = 0.5*SP.ones((2,T))
#
#        #Data(X/Y): conditions x replicates x time x 2D
#        X  = SP.zeros((C,R*T,D))
#        Y  = SP.zeros((C,R*T))
#        #unique times
#        XT = SP.ones((T,2))
#        XT[:,0] = M0[0][0,:]
#
#        [x0,y0] = self.M2GPxy(M0)
#        [x1,y1] = self.M2GPxy(M1)
#        
#        X[0,:,0:2] = x0
#        X[1,:,0:2] = x1
#        Y[0,:]     = y0
#        Y[1,:]     = y1
#        #create indicator vector to identify unique time points
#
#        
#        #create one copy of the input per process as this is used for input dependen noise
#        XS       = X.copy()
#        XJ       = X.copy()
#
#
#        #initialize the two GPs
#        if self.logtheta0 is None:
#            logtheta = self.covar.getDefaultParams()
#
#        #the two indv. GPs
#        GP0 = GPR.GP(self.covar,Smean=self.Smean,logtheta=self.logtheta0)
#        GP1 = GPR.GP(self.covar,Smean=self.Smean,logtheta=self.logtheta0)
#        #the group GP summarising the two indiv. processes
#        GPS = GroupGP([GP0,GP1])
#        #the joint process
#        GPJ = GPR.GP(self.covar,Smean=self.Smean,logtheta=self.logtheta0)
#        #update the GP
#        updateGP()
#
#
#        debug_plot = True
#
#        for i in range(1):
#            ###iterations###
#            #1. get predictive distribution for both GPs
#            ##debug
#            #introduce the additional dimension to accom. the per obs. noise model
#            #get prediction for all time points
#            Yp0 = GP0.predict(GP0.logtheta,XT)
#            Yp1 = GP1.predict(GP1.logtheta,XT)
#            Ypj = GPJ.predict(GPJ.logtheta,XT)
#            #considere residuals
#            D0  = ((M0[1]-Yp0[0])**2 * (1/Yp0[1])).sum(axis=0)
#            D1  = ((M1[1]-Yp1[0])**2 * (1/Yp1[1])).sum(axis=0)
#            DJ  = ((MJ[1]-Ypj[0])**2 * (1/Ypj[1])).sum(axis=0)
#            #the indiv. GP is the sum
#            DS  = D0+D1
#            #now use this to restimate Q(Z)
#            ES  = SP.exp(-DS)
#            EJ  = SP.exp(-DJ)
#            #
#
#            Z[0,:] =self.prior_Z[0]*EJ
#            Z[1,:] =self.prior_Z[1]*ES
#            Z     /=Z.sum(axis=0)
##             pydb.set_trace()
#            updateGP()
#
#
#        if verbose:
#            PL.clf()
#            labelSize = 15
#            tickSize  = 12
#            
#            #1. plot the gp predictions
#            ax1=PL.axes([0.1,0.1,0.8,0.7])
#            Xt_ = SP.linspace(0,XT[:,0].max()+2,100)
#            Xt  = SP.ones((Xt_.shape[0],2))
#            Xt[:,0] = Xt_
#
#            self.plotGPpredict(GP0,M0,Xt,{'alpha':0.1,'facecolor':'r'},{'linewidth':2,'color':'r'})
#            self.plotGPpredict(GP1,M0,Xt,{'alpha':0.1,'facecolor':'g'},{'linewidth':2,'color':'g'})
#            self.plotGPpredict(GPJ,M0,Xt,{'alpha':0.1,'facecolor':'b'},{'linewidth':2,'color':'b'})
#            PL.plot(M0[0].T,M0[1].T,'r.--')
#            PL.plot(M1[0].T,M1[1].T,'g.--')
#            
#            PL.xlim([Xt.min(),Xt.max()])
#            #remove last ytick to avoid overlap
#            yticks = ax1.get_yticks()[0:-2]
#            ax1.set_yticks(yticks)
#            PL.xlabel('Time/h',size=labelSize)
#            PL.ylabel('Expression level',size=labelSize)
#
#            #now plot hinton diagram with responsibilities on top
#            ax2=PL.axes([0.1,0.715,0.8,0.2],sharex=ax1)
##            ax2=PL.axes([0.1,0.7,0.8,0.2])
#            #PL.plot(XT[:,0],Z[1,:])
#            #swap the order of Z for optical purposes
#            Z_= SP.ones_like(Z)
#            Z_[1,:] = Z[0,:]
#            Z_[0,:] = Z[1,:]
#            hinton(Z_,X=M0[0][0])
#            PL.ylabel('diff.')
#            
#            #hide axis labels
#            PL.setp( ax2.get_xticklabels(), visible=False)
#            #font size
#            PL.setp( ax1.get_xticklabels(), fontsize=tickSize)
#            PL.setp( ax1.get_yticklabels(), fontsize=tickSize)
#            PL.setp( ax2.get_xticklabels(), fontsize=tickSize)
#            #axes label
#        return Z
#        pass
#
#
#
