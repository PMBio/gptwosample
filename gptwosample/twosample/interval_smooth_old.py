"""Two sample test on intervals
- This verison predicts smooth transitiosn using another GP on the indicators
"""

#from pygp.gpcEP import *
from gptwosample.data.data_base import get_training_data_structure
from pygp.covar.combinators import SumCF
from pygp.covar.noise import NoiseCFISO
from pygp.covar.se import SqexpCFARD
import copy as CP
import logging
import pygp.gp.gpcEP as gpcEP
import pylab as PL
import scipy as SP
from gptwosample.plot.plot_basic import plot_results
from pygp.optimize.optimize_base import opt_hyper

#import pygp.gp.basic_gp as GPR

#import pydb
#log gamma priors for GP hyperparametersf



#import pdb




class GPTwoSampleInterval(object):
    __slots__ = ["prior_Z", "gpZ", "_twosample_object"]


    def __init__(self, twosample_object, prior_Z={'covar':0.5}, **kwargin):
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
        #twosample.GPTwoSampleMLII.__init__(self,**kwargin)
        #prior for missing proportions
        prior_Z = prior_Z['covar']
        self.prior_Z = {'covar':SP.array([1 - prior_Z, prior_Z])}
        self._twosample_object = twosample_object
        pass
    

    def test_interval(self, M0, M1, verbose=False, opt=None, Ngibbs_iterations=10, XPz=None, logtrafo=False, rescale=False, logthetaZ=SP.log([2, 2, 1E-5]), fix_Z=SP.array([0])):
        """test with interval sampling test
        M0: first expresison condition
        M1: second condition
        verbose: produce plots (False)
        opt: optimise hyperparameters(True)
        Ngibbs_iterations (10)
        PZ: discretization for predictions of indicators (None)
        """

        def normpdf(x, mu, v):
            """Normal PDF, x mean mu, variance v"""
            return SP.exp(-0.5 * (x - mu) ** 2 / v) * SP.sqrt(2 * SP.pi / v)
        
        def rescaleInputs(X, minX, scaleX):
            """copy implementation of input rescaling
            - this version rescales in a dum way assuming everything is float
            """
            return X
            X_ = (X - minX) * scaleX - 5
            return X_

        def debug_plot():
            PL.ion()
            PL.clf()
            X_ = SP.linspace(XT.min(), XT.max(), 100).reshape([-1, 1])
            ZP_ = self.gpZ.predict(self.gpZ.logtheta, X_)[0]
            self.plotGPpredict(self.gpr_0, M0, X_, {'alpha':0.4, 'facecolor':'r'}, {'linewidth':2, 'color':'r'})
            self.plotGPpredict(self.gpr_1, M0, X_, {'alpha':0.4, 'facecolor':'g'}, {'linewidth':2, 'color':'g'})
            self.plotGPpredict(self.gpr_join, M0, X_, {'alpha':0.4, 'facecolor':'k'}, {'linewidth':2, 'color':'k'})
            PL.plot(M0[0].T, M0[1].T, 'r.--')
            PL.plot(M1[0].T, M1[1].T, 'g.--')
            PL.plot(X_, ZP_, 'r-')
            PL.plot(XT, 1 * Z, 'k+')
            PL.plot(XT[I], 1 * Z[I], 'r+')
            print XT[I]
            print Z_
            print Z
            PL.draw()
            fdsd = raw_input()

        def predictIndicator(Z, Xp):
            """create predictions from the indicator variables, averaging over the samples"""
            #at the moment we only average the posterior probabilities over the indicators; the rest is just "cosmetics".
            B = SP.zeros([Z.shape[0], Xp.shape[0]])
            for i in range(Z.shape[0]):
                #set data to GPC
                self.gpZ.setData(XTR, Z[i])
                #predict
                B[i] = self.gpZ.predict(self.gpZ.logtheta, Xp)[0]
            return B
            

        def verbose_plot(Z, Xp):
            """create a nice verbose plot based on the a set of indicator samples Z
            Z:  n x XR : array of indicator vectors of mixture responsibilities
            Xp: X-coordinates for evaluation of predictive distributions
            """
            #get predictions from the indicators:
            B = predictIndicator(Z, Xp)          
            #mean bernoulli
            Bm = B.mean(axis=0)

            #mean of indicators
            Zm = Z.mean(axis=0) > 0.5
            IS = Zm
            IJ = ~Zm
            
            #updata datasets
            #self.gpr_0.set_data(SP.concatenate((M0R[0][:,IS]),axis=0).reshape([-1,1]),SP.concatenate((M0R[1][:,IS]),axis=1),process=False)           
            #self.gpr_1.set_data(SP.concatenate((M1R[0][:,IS]),axis=0).reshape([-1,1]),SP.concatenate((M1R[1][:,IS]),axis=1),process=False)
            #self.gpr_join.set_data(SP.concatenate((MJR[0][:,IJ]),axis=0).reshape([-1,1]),SP.concatenate((MJR[1][:,IJ]),axis=1),process=False)
            
            
            IS=SP.tile(~IS,4)
            IJ=SP.tile(~IJ,8)
            #now plot stuff
            PL.clf()
            ax1 = PL.axes([0.15, 0.1, 0.8, 0.7])
            #plot marginal GP predictions
            alpha = 0.18
            #self.plotGPpredict_gradient(self.gpr_0,M0,Xp,Bm,{'alpha':alpha,'facecolor':'r'},{'linewidth':2,'color':'r'})
            #self.plotGPpredict_gradient(self.gpr_1,M0,Xp,Bm,{'alpha':alpha,'facecolor':'g'},{'linewidth':2,'color':'g'})
            #self.plotGPpredict_gradient(self.gpr_join,M0,Xp,(1-Bm),{'alpha':alpha,'facecolor':'b'},{'linewidth':2,'color':'b'})             
            likelihoods = self._twosample_object.predict_model_likelihoods(interval_indices={'individual':IS, 'common':IJ})
            mean_var = self._twosample_object.predict_mean_variance(Xp,interval_indices={'individual':IS, 'common':IJ})
            plots = plot_results(self._twosample_object, 
                                 alpha=Bm, 
                                 legend=False,interval_indices={'individual':IS, 'common':IJ})
            
            #import pdb;pdb.set_trace()
            
            #PL.plot(M0[0].T,M0[1].T,'b.--')
            #PL.plot(M1[0].T,M1[1].T,'b.--')
            #set xlim
            PL.xlim([Xp.min(), Xp.max()])
            yticks = ax1.get_yticks()[0:-2]
            ax1.set_yticks(yticks)
            PL.xlabel('Time/hr')
            PL.ylabel('Log expression level')
            Ymax = MJ[1].max()
            Ymin = MJ[1].min()
            DY = Ymax - Ymin
            PL.ylim([Ymin - 0.1 * DY, Ymax + 0.1 * DY])
            #2nd. plot prob. of diff
            ax2 = PL.axes([0.15, 0.8, 0.8, 0.10], sharex=ax1)
            PL.plot(Xp, Bm, 'k-', linewidth=2)
            PL.ylabel('$P(z(t)=1)$')
#            PL.yticks([0.0,0.5,1.0])
            PL.yticks([0.5])           
            #horizontal bar
            PL.axhline(linewidth=0.5, color='#aaaaaa', y=0.5)
            PL.ylim([0, 1])
            PL.setp(ax2.get_xticklabels(), visible=False)
            pass

        def sampleIndicator(Z, I, take_out=True):
            """sample all indicators that are true in the index vector I
            take_out: take indices out of the dataset first (yes)
            """

            #create indicator vectors for joint & single GP as well as classifier
            
            PZ = SP.zeros([2, I.sum()])
            IS = Z
            IJ = ~Z
            IZ = SP.ones(Z.shape[0], dtype='bool')
            if take_out:
                #take out the Ith observation from each GP
                IS = IS & (~I)
                IJ = IJ & (~I)
                IZ = IZ & (~I)

            #updata datasets
#            self.gpr_0.set_data(SP.concatenate((M0R[0][:,IS]),axis=0).reshape([-1,1]),SP.concatenate((M0R[1][:,IS]),axis=1),process=False)           
#            self.gpr_1.set_data(SP.concatenate((M1R[0][:,IS]),axis=0).reshape([-1,1]),SP.concatenate((M1R[1][:,IS]),axis=1),process=False)
#            self.gpr_join.set_data(SP.concatenate((MJR[0][:,IJ]),axis=0).reshape([-1,1]),SP.concatenate((MJR[1][:,IJ]),axis=1),process=False)
            self.gpZ.setData(XTR[IZ], Z[IZ])
            
            IS=SP.tile(~IS,4)
            IJ=SP.tile(~IJ,8)

            #GP predictions
#            Yp0 = self.gpr_0.predict(self.gpr_0.logtheta,XT[I],mean=False)
#            Yp1 = self.gpr_1.predict(self.gpr_1.logtheta,XT[I],mean=False)
#            Ypj = self.gpr_join.predict(self.gpr_0.logtheta,XT[I],mean=False)

            prediction = self._twosample_object.predict_mean_variance(XT[I],
                                                                      interval_indices={'individual':IS, 'common':IJ})
            Yp0 = [prediction['individual']['mean'][0], prediction['individual']['var'][0]]
            Yp1 = [prediction['individual']['mean'][1], prediction['individual']['var'][1]]
            Ypj = [prediction['common']['mean'], prediction['common']['var']]
            
            #prdict binary variable
            Zp = self.gpZ.predict(self.gpZ.logtheta, XT[I])[0]
                      
            #robust likelihood
            c = 0.9
            D0 = c * normpdf(M0R[1][:, I], Yp0[0], Yp0[1])
            D0 += (1 - c) * normpdf(M0R[1][:, I], Yp0[0], 1E8)
            D0 = SP.log(D0)
            
            D1 = c * normpdf(M1R[1][:, I], Yp1[0], Yp1[1])
            D1 += (1 - c) * normpdf(M1R[1][:, I], Yp1[0], 1E8)
            D1 = SP.log(D1)
            
            DJ = c * normpdf(MJR[1][:, I], Ypj[0], Ypj[1])
            DJ += (1 - c) * normpdf(MJR[1][:, I], Ypj[0], 1E8)
            DJ = SP.log(DJ)
            #sum over logs
            DS = D0.sum(axis=0) + D1.sum(axis=0)
            DJ = DJ.sum(axis=0)
            #calc posterior 
            PZ[0, :] = (1 - Zp) * SP.exp(DJ) * self.prior_Z['covar'][0]
            PZ[1, :] = Zp * SP.exp(DS) * self.prior_Z['covar'][1]
            PZ /= PZ.sum(axis=0)
            Z_ = SP.rand(I.sum()) <= PZ[1, :]
            #sample indicators
            if(IS.sum() == 1):
                Z_ = True
            if(IJ.sum() == 1):
                Z_ = False
            return [Z_, PZ]
            pass
        pass

        #0. apply preprocessing etc.
        #[M0,M1] = self.preprocess(M0,M1,logtrafo=logtrafo,rescale=rescale)
        M0[0] = SP.array(M0[0])
        M0[1] = SP.array(M0[1])
        M1[0] = SP.array(M1[0])
        M1[1] = SP.array(M1[1])

        #1. use the standard method to initialise the GP objects
        self._twosample_object.predict_model_likelihoods()
        ratio = self._twosample_object.bayes_factor()
        PL.close()
        
        #2. initialise gibbs samplign for time-local approximation
        #get set of unique time coordinates
        XT = M0[0][0].reshape([-1, 1])
        #rescale all datasets inputs
        MJ = [SP.concatenate((M0[0], M1[0]), axis=0), SP.concatenate((M0[1], M1[1]), axis=0)]
        MJR = CP.deepcopy(MJ)
        M0R = CP.deepcopy(M0)
        M1R = CP.deepcopy(M1)

        #rescale and 0 mean
        data_join = self._twosample_object.get_data(model='common')
        data_0 = self._twosample_object.get_data(model='individual', index=0)
        data_1 = self._twosample_object.get_data(model='individual', index=1)
        
        data_join = [x.reshape(-1) for x in data_join]
        data_0 = [x.reshape(-1) for x in data_0]
        data_1 = [x.reshape(-1) for x in data_1]
                
        MJR[0] = rescaleInputs(MJ[0],
                               min(data_join[0]),
                               (max(data_join[0]) - min(data_join[0])) / len(data_join[0]))
        M0R[0] = rescaleInputs(M0[0],
                               min(data_0[0]),
                               (max(data_0[0]) - min(data_0[0])) / len(data_0[0]))
        M1R[0] = rescaleInputs(M1[0],
                               min(data_1[0]),
                               (max(data_1[0]) - min(data_1[0])) / len(data_1[0]))
        #MJR[1] -= data_join[1].mean()
        #M0R[1] -= data_0[1].mean()
        #M1R[1] -= data_1[1].mean()

        #TODO !!!!
        #self._twosample_object.set_data(get_training_data_structure(M0R[0],
        #                                               M1R[0],
        #                                               M0R[1],
        #                                               M1R[1]))

        #3. sample all indicators conditioned on current GP approximation
        Z = SP.random.rand(XT.shape[0]) > 0.5
        if fix_Z is not None:
            Z[fix_Z] = False
        
        #4. initialise the GP for the indicators
        secf = SqexpCFARD()
        noise = NoiseCFISO()
        covar = SumCF([secf, noise])
        #logtheta = self.gpr_0.logtheta.copy()
        if logthetaZ is not None:
            logtheta = logthetaZ

#        logtheta[0] = SP.log(2)
#        logtheta[1] = SP.log(1)
#        logtheta[-1]= 1E-5
        self.gpZ = gpcEP.GPCEP(covar_func=covar)
        self.gpZ.logtheta = logtheta
        self.gpZ.setData(XT, Z)
        
        #self.gpZ.logtheta = opt_hyper(self.gpZ, logthetaZ)
        
        logtheta0 = SP.log([0.5, 1, 0.4])
        if 0:
            #HACK

            self.gpr_join.logtheta = logtheta0
            self.gpr_0.logtheta = logtheta0
            self.gpr_1.logtheta = logtheta0

            #set noise lvel of separate one to joint
            self.gpr_join.logtheta[-1] = self.gpr_0.logtheta[-1]
            self.gpr_join.logtheta[-2] = self.gpr_0.logtheta[-2]
        

        if 0:
            X_ = SP.linspace(XT.min(), XT.max(), 100).reshape([-1, 1])
            XP_ = self.gpZ.predict(self.gpZ.logtheta, X_)

        XTR = rescaleInputs(XT, min(XT), max(XT))
        
        #posterior for Gibbs iterations
        #TODO: add other distributions as we see fit
        Q = {'Z': SP.zeros([Ngibbs_iterations, XT.shape[0]], dtype='bool')}

        #sample indicators one by one
        for n in range(Ngibbs_iterations):
            #perm = SP.random.permutation(XT.shape[0])
            perm = SP.arange(0, XT.shape[0])
            for i in perm:
                #resample a single indicator
                I = SP.zeros([XT.shape[0]], dtype='bool')
                I[i] = True
                Z_ = sampleIndicator(Z, I)
                #save prob. and indicator value
                Z[i] = SP.squeeze(Z_[0])
                if 0 and i % 20 == 0:
                    debug_plot()
            #update indicators
            #save in Q
            Q['Z'][n, :] = Z
            logging.debug("Gibbs iteration:%d" % (n))
            logging.debug("Interval_smooth: current indicator: %s" % (Q['Z'].mean(0)))
            #import pdb;pdb.set_trace()
        n_ = round(Ngibbs_iterations) / 2
        Zp = SP.zeros([2, XT.shape[0]])
        Zp[1, :] = Q['Z'].mean(axis=0)
        Zp[0, :] = 1 - Zp[1, :]
        if 1:
            Xp = SP.linspace(0, XT[:, 0].max() + 2, 100).reshape([-1, 1])
            verbose_plot(Q['Z'][n_::], Xp)

        #create the return structure for the indicators:
        if XPz is None:
            #if no discretization gibven, use data resolution:
            XPz = XT
        #obtain predictions
        
        Zpr = predictIndicator(Q['Z'][n_::], XPz).mean(axis=0)
        return [ratio, Zpr]
        pass

    
        


