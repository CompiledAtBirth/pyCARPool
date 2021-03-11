#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:56:32 2021

@author: chartier
"""

import os
# dataDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/data/z0p5/"
workDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/examples/matterCorr/"
dataDir = "/home/chartier/Documents/VarianceSlayer/Data/Concatenated/matterCorr/"
gadgetCSV="quijoteGadget_Corr_5200.csv"
colaCSV= "lPicola_Corr_5200.csv"
os.chdir(workDir)

import numpy as np
import pandas as pd
import warnings

# Custom functions
import tools_exampleCorr as toolsCorr
from pyCARPool import CARPool, confidenceInt

#%% ANALYSIS SETUP: run only to create new .csv data from .txt simulation outputs

# Simulation
z = 0.5 # redshift for analysis ;  must correspond to the format for Gadget Pk files
scope = [5.0, 200.0] # minimum and maximum r values for the analysis

# Gadget Data
gadgetDir = "/home/chartier/Documents/VarianceSlayer/Data/Data_matter_Quijote/matterCorr"
gadgetFile = "CF_m_1024_z=%s.txt"%z 

# L-Picola data
zDict = {0:"0p000", 0.5:"0p500", 1:"1p000"}

#r, Nmodes, xi0, xi2, xi4 = surrogate columns
colaDir = "/home/chartier/Documents/VarianceSlayer/Data/Data_matter_picola/Collection_2"
colaFile = "Corr_matter_z" + zDict[z] + "_grid1024.txt"

gadgetData, nBins_Y, nSeeds_Y = toolsCorr.assembleCorr_data(gadgetDir, gadgetFile, np.arange(0,15000), scope, "GADGET")
gadgetData.to_csv(dataDir + gadgetCSV , sep = ",", index = False)

#%%
colaData, nBins_C, nSeeds_C = toolsCorr.assembleCorr_data(colaDir, colaFile, np.arange(0,3100), scope, "LPICOLA")
colaData.to_csv(dataDir + colaCSV , sep = ",", index = False)

#%% I) GADGET N-body raw data

rootRes = "/home/chartier/Documents/VarianceSlayer/pyCARPool/examples/matterCorr/Figures/"
if not(os.path.exists(rootRes)):
    os.mkdir(rootRes)

rlim = 160.0

gadgetData = pd.read_csv(dataDir+gadgetCSV, sep = ",", header = 0, dtype = {"Seed" : np.int, "r" : np.float, "CF" : np.float})
gadgetData = gadgetData[gadgetData.r<=rlim]

seedArr_Y = np.sort(gadgetData.Seed.unique())
nSeeds_Y = len(seedArr_Y)
nBins_Y = len(gadgetData[gadgetData.Seed == seedArr_Y[0]].index)

simMat_Tot = toolsCorr.data2Mat(gadgetData, "CF", seedArr_Y, nBins_Y)

# True mean and covariance matrix of 15000 samples
truth = np.mean(simMat_Tot, axis = 1)
sigmaYY = np.cov(simMat_Tot, rowvar = True, bias = False)

del simMat_Tot

#%% II) COLA raw data

colaData = pd.read_csv(dataDir+colaCSV, sep = ",", header = 0, dtype = {"Seed" : np.int,"r" : np.float, "Nmodes" : np.int,
"CF" : np.float, "qCF" : np.float, "hCF" : np.float})

colaData = colaData[colaData.r <= rlim]

seedArr_C = np.sort(colaData.Seed.unique())
nSeeds_C = len(seedArr_C)
nBins_C = len(colaData[colaData.Seed == seedArr_C[0]].index)

Nmodes3D = colaData[colaData.Seed == seedArr_C[0]].Nmodes.to_numpy()
rmod = colaData[colaData.Seed == seedArr_C[0]].r.to_numpy()

if nBins_C == nBins_Y:
    print("Same number of bins")
    nBins = nBins_Y
    del nBins_Y, nBins_C
else:
    warnings.warn("The simulation and the surrogate have not the same number of bins. Is it intentional?")

#%% III) Data setup

# Seeds for the Control Variates algorithm
seeds_Est = np.arange(0, 500, dtype = np.int)
gadgetCorr = toolsCorr.data2Mat(gadgetData, "CF", seeds_Est, nBins)
colaCorr = toolsCorr.data2Mat(colaData, "CF", seeds_Est, nBins)

# Estimate muC for Pk (1500 L-PICOLA sampless)
startMuC = 500; endMuC = 2000
seeds_muC = np.arange(startMuC, endMuC, dtype = np.int)
del startMuC, endMuC

colaMat_muC = toolsCorr.data2Mat(colaData,"CF", seeds_muC, nBins)
corrMuC = np.mean(colaMat_muC, axis = 1)
SigmaMuC = np.cov(colaMat_muC, rowvar = True, bias = False)
del colaMat_muC

#%% IV) Initialize CARPool instance
cvStep = 5
p = 1
q = 1
N = seeds_Est.shape[0]

# Confidence Intervals
methodCIFix = "None"
alpha = 5

# In case of smoothing
wname = "flat"
wlen = 5
rl = 5 ; indSmoothing = np.where(rmod >= rl)[0]

# Create CARPool framework instance
myCARPool = CARPool("matterCorr : Quijote GADGET + L-PICOLA 20 steps", gadgetCorr, colaCorr, truth)
myCARPool.muSurr = corrMuC

#%% V) "Opponent" estimator : sample mean of nSampSim simulation realizations
nSampSim = 500
ciY = "bootstrapBCA"
bootCard = 5000
muY = np.mean(gadgetCorr, axis = 1)
lowY, upY = confidenceInt(gadgetCorr, ciY, alpha, bootCard, progress = True)


#%% VI) Performance with NFix samples
Nfix = 5
testFix = myCARPool.createTest("sets of %i"%Nfix, Nfix, N, p, q, Incremental = False)
testFix.smDict = {"smBool": False,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testFix.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, 'tscore', alpha)

nCV = 20
indTest = int(nCV/cvStep) - 1
betaMat = np.diag(testFix.betaList[indTest])

# Additional (co)variance brought by the estimation of muC (equation 12 in 2009.08970)
SigmaBetaAdd = np.linalg.multi_dot([betaMat, SigmaMuC, betaMat.T]) * 1.0/len(seeds_muC)
stdBetaAdd = np.sqrt(np.diag(SigmaBetaAdd, k = 0))

barBool = True
factErrFix = 1.0
percBool = True
trueBool = True
epsilonBool = True
boolStd = True
xlog = False
titleBool = False

toolsCorr.plot_cvRatiosBlues(muY, nSampSim, lowY, upY, barBool, testFix.muCARPool,
                 stdBetaAdd,boolStd, Nfix, rmod, factErrFix, percBool, trueBool,
                 xlog, titleBool,truth, saveBool = True, saveFold = rootRes, saveStr = "corrDisp%i"%Nfix)

#%% VIII) Performance for an increasing number of samples
methodCIInc = "tscore"
NInc = 20

testInc = myCARPool.createTest("Every 5 additional samples", cvStep, NInc, p, q, Incremental = True)
testInc.smDict = {"smBool": True,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIInc, alpha, bootCard)

# Reproduce figure 2 of 2009.08970 (if methodCIInc = bootstrapBCA ; bootstrap results may slightly differ)
# For figure B2, use methodCIInc = tscore
nSampCARP = 5
indCV = int(nSampCARP/cvStep) - 1
factErrInc = 20
zoomBool = False
zoomFact = 3.50
TupleLim = (rmod[45], rmod[-1]) # hard coded xlim for inserted zoom plot
xlog = False ; ylog = False
tupleCor1 = (1,4)
reduced = True

toolsCorr.Comparison_errBars(muY, testInc.muCARPool[:,indCV] , rmod, nSampSim, nSampCARP, 
                           lowY, upY, testInc.lowMeanCI[:,indCV], testInc.upMeanCI[:,indCV], reduced,
                       factErrInc,zoomBool, TupleLim,zoomFact,tupleCor1, xlog, ylog, True,rootRes, saveStr = "corr%iCVSm"%nSampCARP)

#%% Reproduce Figure 5 of 2009.08970
seeds_varTest = np.arange(500, 2300)
gad_varTest = toolsCorr.data2Mat(gadgetData, "CF", seeds_varTest, nBins)
cola_varTest = toolsCorr.data2Mat(colaData, "CF", seeds_varTest, nBins)

safeguard = 0.00001 # for overflowing determinant computation

# For the simulations only
(signYY, logdetYY) = np.linalg.slogdet(sigmaYY)
detYY = signYY * np.exp(logdetYY/safeguard)
sigYY = np.sqrt(np.diag(sigmaYY))

# CARPool tests for variance reduction study
NInc2 = 500
meth = "None"
testInc_varU = myCARPool.createTest("Every 5 additional samples", cvStep, NInc2, p, q, Incremental = True)
testInc_varU.smDict = {"smBool": True,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_varU.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth)

# Apply beta estimated with 5, 10, 15... samples to generate 3000 CARPool samples and get a rather good
# estimate of the covariate of x(beta).
sigma2XXU, logdetXXU, signXXU = testInc_varU.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXU = signXXU * np.exp(logdetXXU/safeguard)
reducVarU = np.power(detXXU/detYY, safeguard)

# Same for the multivariate (M) case : p = q = nBins
testInc_varM = myCARPool.createTest("Every 5 additional samples", cvStep, NInc2, nBins, nBins, Incremental = True)
testInc_varM.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth)

sigma2XXM, logdetXXM, signXXM = testInc_varM.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXM = signXXM * np.exp(logdetXXM/safeguard)
reducVarM = np.power(detXXM/detYY, safeguard)

# We take the best estimate of beta we have --> variance reduction to expect at best
sigmaReducM = np.sqrt(sigma2XXM[:,-1])/sigYY
sigmaReducU =  np.sqrt(sigma2XXU[:,-1])/sigYY # actual variance reduction we tend to

# "App" <--> approximate ; we take an "improper" beta to compare with the variance reduction of a "good" beta
nApp = 10 # figure 10
indApp  = int(nApp/cvStep) - 1
sigmaReducUApp = np.sqrt(sigma2XXU[:,indApp])/sigYY # impact of an "improper" 

toolsCorr.plotVarReduc_Cov(reducVarM, reducVarU, sigmaReducM, sigmaReducU, sigmaReducUApp ,testInc_varM.Nsamples, rmod)

#%% VI) New test with q = 3
q2 = 3
testInc_q2 = myCARPool.createTest("Every 5 additional samples, q = 3", cvStep, NInc2, p, q2, Incremental = True)
testInc_q2.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q2.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, "None", alpha)


#%% VII) New test with q = 5

q3 = 5
testInc_q3 = myCARPool.createTest("Every 5 additional samples, q = 5", cvStep, NInc2, p, q3, Incremental = True)
testInc_q3.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q3.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, "None", alpha)


#%% VIII) Extended variance analysis

sigma2XXq2, logdetXXq2, signXXq2 = testInc_q2.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXq2 = signXXq2 * np.exp(logdetXXq2)
reducVarq2 = detXXq2/detYY

sigma2XXq3, logdetXXq3, signXXq3 = testInc_q3.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXq3 = signXXq3 * np.exp(logdetXXq3)
reducVarq3 = detXXq3/detYY

sigmaReducq2 = np.sqrt(sigma2XXq2[:,-1])/sigYY
sigmaReducq3 =  np.sqrt(sigma2XXq3[:,-1])/sigYY

toolsCorr.plotVarReduc_new(reducVarM, reducVarU, reducVarq2,reducVarq3, sigmaReducM,
                         sigmaReducU,sigmaReducq2, sigmaReducq3, testInc_q2.Nsamples, rmod, q2, q3)