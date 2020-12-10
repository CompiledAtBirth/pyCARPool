#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:31:45 2020

@author: chartier
"""
import numpy as np
import math
from scipy import stats
from scipy import signal
import scikits.bootstrap as boot
from tqdm import tqdm

#%% DEFINITION OF THE CARPool class and the CARPoolTest inner class

class CARPool:
    
    def __init__(self, description, simData, surrData, simTruth = None):
        
        self.description = description
        
        self.simData = simData
        self.P = simData.shape[0]
        
        self.surrData = surrData
        self.Q = simData.shape[0]
        
        self.simTruth = simTruth
        
        # Append instances of the CARPoolTest class here
        self.testList = []
        
    @property
    def muSurr(self):
        return self._muSurr
        
    @muSurr.setter
    def muSurr(self, muC):
    
        if muC.shape[0] == self.surrData.shape[0]:
            print("Setting the surrogate mean")
            self._muSurr = muC
        else:
            errMessage="The surrogate mean must have the same vector size as the surrogate samples"
            raise ValueError(errMessage)
            
    def createTest(self, name, N4Est, Nmax, p, q, Incremental = True):
        
        Ntot = max([self.simData.shape[1], self.surrData.shape[1]])
        Nmax = Ntot if Nmax > Ntot else Nmax
        
        errMessage = " Number of samples asked for is too large: Nmax = %i"%Nmax
        if N4Est > Nmax:
            raise ValueError(errMessage)
        
        if  Incremental == True:
            nTests = math.floor(Nmax/N4Est) # math.floor returns int contrarily to np.floor
            Nsamples = np.arange(start = N4Est, stop = (nTests + 1) * N4Est, step = N4Est, dtype = np.int)
        else:
            Nsamples = N4Est
            nTests = math.floor(Nmax/N4Est)
    
        return self.CARPoolTest(name, p, q, self.P, self.Q, Nsamples, nTests, Incremental)
    
    def appendTest(self, test):
        self.testList.append(test)
     
    
    class CARPoolTest:
        
        def __init__(self, testName, p, q, P, Q, Nsamples, nTests, Incremental):
            
            self.testName = testName
            self.P = P
            self.Q = Q
            self.p = 1 if p > P else p
            self.q = 1 if q > Q else q
            self.Nsamples = Nsamples
            self.nTests = nTests
            self.Incremental = Incremental #to comment for explanations
            
            self.smDict = {"smBool": False,"wtype":"flat", "wlen": 5, "indSm":None}
                    
        def set_Univariate(self):
            self.p = 1
            self.q = 1
            
        def set_Multivariate(self):
            self.p = self.P
            self.q = self.Q
         
        # The default value of "cauldron" input is the identity function   
        identity  = lambda X: X
        
        # MAIN METHOD: the reason for the class to exist
        def computeTest(self, simData, surrData, muSurr, methodCI = "None", alpha = 5, B = 1000, cauldron = identity):
            
            strStart = "INCREMENTAL" if self.Incremental == True else "FIXED"
            print("STARTING CARPool %s TEST"%strStart)
            
            # Initialize the attributes hosting results
            self.muCARPool = np.zeros((self.P, self.nTests), dtype = np.float)
            self.betaList = []
            
            if methodCI != "None":
                self.lowCARPoolCI = np.zeros((self.P, self.nTests), dtype = np.float)
                self.upCARPoolCI = np.zeros((self.P, self.nTests), dtype = np.float)
                
            # Proceed to estimation ; framework depends on the integers p and q
            
            # Univariate framework
            if self.p == 1 and self.q == 1:
                
                print("UNIVARIATE ESTIMATION")
                
                for k in range(self.nTests):
                    
                    if self.Incremental == True:
                        indStart = 0
                        indEnd = self.Nsamples[k] # Nsmaples is an array of integers
                    else:
                        indStart = k * self.Nsamples # Nsamples is a single integer
                        indEnd = (k + 1) * self.Nsamples
                    
                    # np.ndarray.copy() produces a deep copy (necessary if cauldron != identity)
                    simSamples = cauldron(simData[:,indStart:indEnd].copy())
                    surrSamples = cauldron(surrData[:,indStart:indEnd].copy())
                    
                    # Here beta is a 1D array of floats
                    beta = uvCARPool_Beta(simSamples, surrSamples, self.smDict)
                    self.betaList.append(beta)
                    
                    empSim = np.mean(simSamples, axis = 1)
                    empSurr = np.mean(surrSamples, axis = 1)
                    
                    muX = uvCARPool_Mu(empSim, empSurr, muSurr, beta)
                    self.muCARPool[:,k] = muX
                    
                    # Estimate confidence intervals of estimated means if required
                    if methodCI != "None":
                        
                        lowCI, upCI = uvCARPool_CI(simSamples, surrSamples, 
                                                            muSurr, beta, methodCI, alpha, B)
                        
                        print("CIs for test %i over %i finished"%(k+1, self.nTests))
                        self.lowCARPoolCI[:,k] = lowCI
                        self.upCARPoolCI[:,k] = upCI
                            
                    else:
                        pass
             
            # Hybrid framework
            elif self.p == 1 and self.q > 1:
                
                print("HYBRID ESTIMATION")
                
                for k in range(self.nTests):
                    
                    if self.Incremental == True:
                        indStart = 0
                        indEnd = self.Nsamples[k] # Nsmaples is an array of integers
                    else:
                        indStart = k * self.Nsamples # Nsamples is a single integer
                        indEnd = (k + 1) * self.Nsamples
                    
                    # np.ndarray.copy() produces a deep copy (necessary if cauldron != identity)
                    simSamples = cauldron(simData[:,indStart:indEnd].copy())
                    surrSamples = cauldron(surrData[:,indStart:indEnd].copy())
                        
                    # Here beta is a 1D array of floats
                    muX, beta = hbCARPool_Est(simSamples, surrSamples, muSurr, self.q)
                    
                    self.betaList.append(beta)
                    self.muCARPool[:,k] = muX
                    
                    if methodCI != "None":
                        
                        lowCI, upCI = hbCARPool_CI(simSamples, surrSamples,
                        muSurr, self.q, beta, methodCI, alpha, B)
                        
                        print("CIs for test %i over %i finished"%(k+1, self.nTests))
                        self.lowCARPoolCI[:,k] = lowCI
                        self.upCARPoolCI[:,k] = upCI
                        
                    else:
                        pass
                    
            # Multivariate framework        
            elif  self.p == self.P and self.q == self.Q:
                
                print("MULTIVARIATE ESTIMATION")
                
                for k in range(self.nTests):
                    
                    if self.Incremental == True:
                        indStart = 0
                        indEnd = self.Nsamples[k] # Nsmaples is an array of integers
                    else:
                        indStart = k * self.Nsamples # Nsamples is a single integer
                        indEnd = (k + 1) * self.Nsamples
                        
                    # np.ndarray.copy() produces a deep copy (necessary if cauldron != identity)
                    simSamples = cauldron(simData[:,indStart:indEnd].copy())
                    surrSamples = cauldron(surrData[:,indStart:indEnd].copy())
                    
                    beta = mvCARPool_Beta(simSamples, surrSamples)
                    self.betaList.append(beta)
                    
                    empSim = np.mean(simSamples, axis = 1)
                    empSurr = np.mean(surrSamples, axis = 1)
                    
                    muX = mvCARPool_Mu(empSim, empSurr, muSurr, beta)
                    self.muCARPool[:,k] = muX
                    
                    if methodCI != "None":
                        
                        lowCI, upCI = mvCARPool_CI(simSamples, surrSamples,
                        muSurr, beta, methodCI, alpha, B)
                        
                        print("CIs for test %i over %i finished"%(k+1, self.nTests))
                        self.lowCARPoolCI[:,k] = lowCI
                        self.upCARPoolCI[:,k] = upCI
                        
                    else:
                        pass
                    
            print("TEST FINISHED")
        
        # Compute the variance of CARPool samples by generating them with a list if control matrices
        def varianceAnalysis(self, simSamples, surrSamples, muSurr, safeguard = 1.0):
            
            if self.nTests != len(self.betaList):
                print("The number of beta matrixes is not the same as nTests. Check consistency")                
            
            logdetXX = np.zeros((self.nTests,), dtype = np.float)
            signXX = np.zeros((self.nTests,), dtype = np.float)
            sigma2XX = np.zeros((simSamples.shape[0], self.nTests), dtype = np.float)
            
            for k in range(self.nTests):
                
                xColl = CARPoolSamples(simSamples, surrSamples, muSurr, self.p, self.q, self.betaList[k])
                sigmaXX = np.cov(xColl, rowvar = True, bias = False)
                (sign, logdet) = np.linalg.slogdet(sigmaXX)
                signXX[k] = sign
                logdetXX[k] = logdet
                sigma2XX[:,k] = np.diag(sigmaXX)
                
            return sigma2XX, logdetXX, signXX            
                
#%% ESTIMATION TOOLS

########################
#FUNCTIONS FOR p = q = 1
#########################

def crossCovUni(Y, C):
    '''
    Function for cross covariance, p = q = 1
    Parameters
    ----------
    Y : Numpy array of shape (P,N)
    C : Numpy array of shape (P,N) (P=Q)
    
    Returns the (P, 1) array of cross-covariances between each element of Y and C
    '''
    covYC = 0.0
    muY = np.mean(Y, axis = 1)
    muC = np.mean(C, axis = 1)
    
    assert Y.shape[1] == C.shape[1], "y and c should have the same number of samples"
    N = Y.shape[1]
    
    for k in range(N):
        covYC += (Y[:,k] - muY) * (C[:,k] - muC)
        
    return 1.0/(N - 1.0) * covYC


def uvCARPool_Beta(simSamples, surrSamples, smDict):
    
    # divisor of the empirical sum is N - ddof
    sigmaC2 = np.var(surrSamples, axis = 1, ddof = 1.0)
    
    # Cross-correlation coefficeints
    covYC = crossCovUni(simSamples, surrSamples)
    
    beta = covYC/sigmaC2 # element-wise numpy division here (univariate setting)
    
    if smDict["smBool"]:
        sig = beta[smDict["indSmooth"]].copy()
        y = smooth1D(sig, smDict["wlen"], smDict["wname"])
        beta[smDict["indSmooth"]] = y
    
    return beta

def uvCARPool_Mu(empSim, empSurr, muSurr,beta):
    
    betaMat = np.diag(beta) # because we are in the p = q = 1 framework
    
    # It's easier to treat beta as a diagonal matrix than to do a loop
    muCARP = empSim - np.matmul(betaMat, empSurr - muSurr)
    
    return muCARP
    
def uvCARPool_CI(simSamples, surrSamples, muSurr, beta, method, alpha, B = 1000):
    
    betaMat = np.diag(beta)
    
    # Apply the given beta to all samples to create a "collection" of estimates, each using a fixed beta
    collCARP = simSamples - np.matmul(betaMat, surrSamples - muSurr[:,np.newaxis])
    
    lowCI, upCI = confidenceInt(collCARP, method, alpha, B)
    
    return lowCI, upCI
    
#################################
#FUNCTIONS FOR p = 1, 1 < q <= Q 
#################################
def crossCovHyb(y, C):
    '''
    Function for cross covariance, p = q = 1
    Parameters
    ----------
    N is the number of sample
    Y : Numpy array of shape (1,N)
    C : Numpy array of shape (Q,N)
    correction : Divisor of the estimator is N - correction

    Returns the (Q, 1) array of cross-correlation coefficients between y and each variable in C
    '''
    muy = np.mean(y)
    muC = np.mean(C, axis = 1)
    
    assert len(y) == C.shape[1], "y and c should have the same number of samples"
    N = len(y)
    q = C.shape[0]
    
    covyC = np.zeros((q,), dtype = np.float)
    for k in range(N):
        covyC += (y[k] - muy) * (C[:,k] - muC)
    
    return 1.0/(N - 1.0) * covyC


def hbCARPool_beta(simVar, surrSubset):
    '''
    Parameters
    ----------
    simSamples : Array of shape (1,N) with N samples of the scalar y
    surrSamples : Array of shape (q,N)
    correction : optional
        DESCRIPTION. Divisor of the estimator is N - correction. The default is 1.0.


    Returns
    -------
    beta array of size  q
    '''
    
    SigmaCC = np.cov(surrSubset, rowvar = True, bias = False) # (q,q)
    covYC = crossCovHyb(simVar, surrSubset) # (q,1)
    
    preCC = np.linalg.pinv(SigmaCC, hermitian = True)
    
    beta = np.matmul(preCC, covYC) # shape (q,1)
    
    return beta

# For simplicity, this will also be a function direclty available to users for single computations
def hbCARPool_Est(simSamples, surrSamples, muSurr, q):
    
    shift = math.floor(q/2) # math.floor gives an int
    nBins = simSamples.shape[0]
    
    betaLength = nBins * q - 2 * shift
    
    betaAgg = np.zeros((betaLength,), dtype = np.float)
    xMu = np.zeros((nBins,), dtype = np.float)
    
    for n in range(nBins):
        
        a = n - shift
        b = n + shift
        cStart = a if a >= 0 else 0
        cEnd = b + 1 if b + 1 < nBins else nBins
        
        yVar = simSamples[n,:]
        cSub = surrSamples[cStart:cEnd,:]
        
        beta = hbCARPool_beta(yVar, cSub)
        
        bStart = n * q - 1 if n > 0 else 0
        bEnd = (n + 1) * q - 1 if n < nBins - 1 else betaLength
        betaAgg[bStart:bEnd] = beta
        
        yMu = np.mean(yVar)
        cMu = np.mean(cSub, axis = 1)
        
        xMu[n] = yMu - np.matmul(beta, cMu - muSurr[cStart:cEnd])
        
    return xMu, betaAgg

def hbCARPool_CI(simSamples, surrSamples, muSurr, q, beta, method, alpha, B):
    
    shift = math.floor(q/2)
    nBins = simSamples.shape[0]
    
    betaLength = nBins * q - 2 * shift
    
    assert betaLength == beta.shape[0], "debug hybrid beta computation"
    
    xColl = np.zeros((nBins, simSamples.shape[1]), dtype = np.float)
    
    for n in range(nBins):
        
        a = n - shift
        b = n + shift
        cStart = a if a >= 0 else 0
        cEnd = b + 1 if b + 1 < nBins else nBins
        
        bStart = n * q - 1 if n > 0 else 0
        bEnd = (n + 1) * q - 1 if n < nBins - 1 else betaLength
        
        betaMat = beta[bStart:bEnd]
        
        xColl[n,:] = simSamples[n,:] - np.matmul(betaMat, surrSamples[cStart:cEnd,:] - muSurr[cStart:cEnd, np.newaxis])
        
    lowCI, upCI = confidenceInt(xColl, method, alpha, B)
    
    
    return lowCI, upCI
    
    
#####################################
#FUNCTIONS FOR p = P >1 and q = Q > 1
#####################################

def mvCARPool_Beta(simSamples, surrSamples):
    
    P = simSamples.shape[0]
    Q = surrSamples.shape[0]
    
    assert simSamples.shape[1] == surrSamples.shape[1], "Y and C should have the same number of samples"
    N = simSamples.shape[1]
    
    SigmaCC = np.cov(surrSamples, rowvar = True, bias = False)
    preCC = np.linalg.pinv(SigmaCC, hermitian = True)
    
    empSim = np.mean(simSamples, axis = 1)
    empSurr = np.mean(surrSamples, axis = 1)
    covYC = np.zeros((P, Q), dtype = np.float)
    
    for k in np.arange(0,N):
        covYC += np.outer(simSamples[:,k] - empSim, surrSamples[:,k] - empSurr)
    covYC = covYC/(N - 1.0)
    
    beta = np.matmul(covYC, preCC)
    
    return beta
    
def mvCARPool_Mu(empSim, empSurr, muSurr, betaMat):
    
    muCARP = empSim - np.matmul(betaMat, empSurr - muSurr)
    
    return muCARP
    

def mvCARPool_CI(simSamples, surrSamples, muSurr, beta, method, alpha, B):
    
    # Apply the given beta to all samples to create a "collection" of estimates, each using a fixed beta
    collCARP = simSamples - np.matmul(beta, surrSamples - muSurr[:,np.newaxis])
    
    lowCI, upCI = confidenceInt(collCARP, method, alpha, B)
    
    return lowCI, upCI
    
#%%
#####################################
#CONFIDENCE INTERVALS & CIE
#####################################

def zscoreCI(scalarRV, alpha):
    '''
    Parameters
    ----------
    scalarRV : scalarRV is a (1,N) array of N scalar samples
    alpha : float between 0.0 and 100.0.

    Returns
    -------
    The lower and upper confidence bounds around the sample mean
    '''
    N = scalarRV.shape[0]
    
    lowZ, upZ = stats.norm.interval(1.0 - alpha/100.0, loc = np.mean(scalarRV), 
                                    scale = np.std(scalarRV, ddof=1)/np.sqrt(N))
    
    return lowZ, upZ

def tscoreCI(scalarRV, alpha):
    '''
    Parameters
    ----------
    scalarRV : scalarRV is a (1,N) array of N scalar samples
    alpha : float between 0.0 and 100.0.

    Returns
    -------
    The lower and upper confidence bounds around the sample mean
    '''
    N = scalarRV.shape[0]
    df = N - 1 # degrees of freedom
    
    lowT, upT = stats.t.interval(1.0 - alpha/100.0, df, loc = np.mean(scalarRV), 
                                   scale = stats.sem(scalarRV, ddof = 1))
    
    # Equivalent code
    # t = stats.t.ppf(1.0 - 0.5*alpha/100, df)   # t-critical value for 95% CI
    # s = np.std(scalarRV, ddof=1)   # empirical standard deviation, with ddof = 1 for the unbiased estimator
    # lowT = np.mean(scalarRV) - (t * s / np.sqrt(N))
    # upT = np.mean(scalarRV) + (t * s / np.sqrt(N))
    
    return lowT, upT

# ESTIMATE CONFIDENCE INTERVALS VIA BOOTSTRAP
def bootstrapCI_piv(scalarRV, B, alpha):
    '''
    Parameters
    ----------
    scalarRV : scalarRV is a (1,N) array of N scalar samples
    B : int, number of samples with replacement
    alpha : float between 0.0 and 100.0.

    Returns
    -------
    The lower and upper confidence bounds around the sample mean
    '''
    
    N = scalarRV.shape[0]
    muEmp = np.mean(scalarRV)
    
    reSample = np.random.choice(scalarRV, (N, B), replace = True)
    meanB = np.mean(reSample, axis = 0) # accros axis 0 here !
    
    lower, upper = np.percentile(meanB, [alpha/2.0, 100.0 - alpha/2.0])
    lowPivCI = 2.0 * muEmp - upper
    upPivCI = 2.0 * muEmp - lower
    
    return lowPivCI, upPivCI

def bootstrapCI_BCA(scalarRV, B, alpha):
    '''
    Parameters
    ----------
    scalarRV : scalarRV is a (1,N) array of N scalar samples
    B : int, number of samples with replacement
    alpha : float between 0.0 and 100.0.

    Returns
    -------
    The lower and upper confidence bounds around the sample mean
    '''
    
    (lowBCACI, upBCACI) = boot.ci(scalarRV, statfunction = np.mean, 
                            alpha = alpha/100.0, n_samples = B, method = "bca")
    
    return lowBCACI, upBCACI
    

# This one will be accessible to users    # METHOD CI optionnal
def confidenceInt(dataSamples, method, alpha, B = 1000, progress = False):
    
    nBins = dataSamples.shape[0]
    upCI = np.zeros((nBins,), dtype = np.float)
    lowCI = np.zeros((nBins,), dtype = np.float)
    
    if method == "zscore":
        for n in range(nBins):
            lowCI[n], upCI[n] = zscoreCI(dataSamples[n,:], alpha)
    elif method == "tscore":
         for n in range(nBins):
             lowCI[n], upCI[n]= tscoreCI(dataSamples[n,:], alpha)
    elif method == "bootstrapPiv":
         if progress:
             pbar = tqdm(total = nBins)
         for n in range(nBins):
            lowCI[n], upCI[n] = bootstrapCI_piv(dataSamples[n,:],B, alpha)
            if progress:
                pbar.update(n = 1)
            if progress and n == nBins - 1:
                pbar.close()
    elif method == "bootstrapBCA":
         if progress:
             pbar = tqdm(total = nBins)
         for n in range(nBins):
            lowCI[n], upCI[n] = bootstrapCI_BCA(dataSamples[n,:], B, alpha)
            if progress:
                pbar.update(n = 1)
            if progress and n == nBins - 1:
                pbar.close()
    else:
        print("method string invalid, using z-score CI by default")
        for n in range(nBins):
            lowCI[n], upCI[n] = zscoreCI(dataSamples[n,:], alpha)
            
    return lowCI, upCI

#%% Other handy functions, accessible outside of the class in the package

# The default value of "cauldron" input is the identity function   
def identity(X):
    return X

# Compute a single CARPool estimate with samples provided as inputs
def CARPoolEstimator(simData, surrData, muSurr, p, q, smDict = None, cauldron = identity):
    
    simSamples = cauldron(simData)
    surrSamples = cauldron(surrData)
    
    P = simSamples.shape[0]
    Q = surrSamples.shape[0]
    
    empSim = np.mean(simSamples, axis = 1)
    empSurr = np.mean(surrSamples, axis = 1)
    
    print("CARPool estimate with p = %i and q = %i"%(p, q))
    
    if p == 1 and q == 1:
        
        assert P == Q, "P and Q must be the same for this framework"
        # Same smDict by default as the CARPoolTest class
        if smDict == None:
             smDict = {"smBool": False,"wtype":"flat", "wlen": 5, "indSm":None}
        beta = uvCARPool_Beta(simSamples, surrSamples, smDict)       
        muX = uvCARPool_Mu(empSim, empSurr, muSurr, beta)
        
    elif p == 1 and q > 1:
        
        assert P == Q, "P and Q must be the same for this framework"
        muX, beta = hbCARPool_Est(simSamples, surrSamples, muSurr, q)
        
    elif p == P and q == Q:
        
        beta = mvCARPool_Beta(simSamples, surrSamples)
        muX = mvCARPool_Mu(empSim,empSurr, muSurr, beta)
        
    else:   
        raise ValueError("Case not handled for chosen p and q")
        
    return muX

# Generate a collection of x_n(beta) samples, with beta a control matrix to be provided as input
def CARPoolSamples(simSamples, surrSamples, muSurr, p, q, beta):
    
    P = simSamples.shape[0]
    Q = surrSamples.shape[0]
    
    if p == 1 and q == 1:
        
        if P != Q:
            raise ValueError("P and Q must be the same for this framework")
            
        betaMat = np.diag(beta)
        
        xSamples = simSamples - np.matmul(betaMat, surrSamples - muSurr[:,np.newaxis])
        
    elif p == 1 and q > 1:
        
        if P != Q:
            raise ValueError("P and Q must be the same for this framework")
        
        shift = math.floor(q/2)
        nBins = simSamples.shape[0]
        
        betaLength = nBins * q - 2 * shift
        
        if betaLength != beta.shape[0]:
            raise ValueError("beta has not the right length")
        
        xSamples = np.zeros((nBins, simSamples.shape[1]), dtype = np.float)
        
        for n in range(nBins):
            
            a = n - shift
            b = n + shift
            cStart = a if a >= 0 else 0
            cEnd = b + 1 if b + 1 < nBins else nBins
            
            bStart = n * q - 1 if n > 0 else 0
            bEnd = (n + 1) * q - 1 if n < nBins - 1 else betaLength
            
            betaMat = beta[bStart:bEnd]
            
            xSamples[n,:] = simSamples[n,:] - np.matmul(betaMat, surrSamples[cStart:cEnd,:] - muSurr[cStart:cEnd, np.newaxis])
        
    elif p == P and q == Q:
        
        xSamples = simSamples - np.matmul(beta, surrSamples - muSurr[:,np.newaxis])
        
    else:   
        raise ValueError("Case not handled for chosen p and q")
    
    return xSamples

#%% ADDITIONNAL TRICKS & TOOLS (TO BE UPDATED)

# Additional tools for p = q = 1
# Smooth 1D numpy array
def smooth1D(sigArr, window_len, window):
    
    if sigArr.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if sigArr.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return sigArr
    
    # if not window in ['flat', 'hann']:
    #     raise ValueError("Window is one of 'flat', 'hann")
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('scipy.signal.windows.'+window+'(window_len)')
    
    padd = int(np.floor(window_len/2))
    n = sigArr.shape[0]
    s = np.zeros((int(2*padd + n)),dtype = np.float)
    s[0:padd] = sigArr[0]
    s[padd:n+padd] = sigArr
    s[n+padd:] = sigArr[-1]

    y = signal.convolve(s, w, mode='same')/np.sum(w)
    out = y[padd:padd+n]
    return out