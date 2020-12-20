#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:25:20 2020
@author: chartier
"""
import os
dataDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/data/z0p5/"
workDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/examples/matterPk/"
os.chdir(workDir)

import numpy as np
import pandas as pd
import warnings

# Custom functions
import tools_examplePk as toolsPk
from pyCARPool import CARPool, confidenceInt

#%% I) GADGET N-body raw data

gadgetFile="quijoteGadget_Pk95bins_kmax1p20.csv"
gadgetData = pd.read_csv(dataDir+gadgetFile, sep = ",", header = 0, dtype = {"Seed" : np.int, "k3D" : np.float, "Pk3D" : np.float})

seedArr_Y = np.sort(gadgetData.Seed.unique())
nSeeds_Y = len(seedArr_Y)
nBins_Y = len(gadgetData[gadgetData.Seed == seedArr_Y[0]].index)

simMat_Tot = toolsPk.data2Mat(gadgetData, "Pk3D", seedArr_Y, nBins_Y)

# True mean and covariance matrix of 15000 samples
truth = np.mean(simMat_Tot, axis = 1)
sigmaYY = np.cov(simMat_Tot, rowvar = True, bias = False)

del simMat_Tot

#%% II) COLA raw data

colaFile="lPicola_Pk95bins_kmax1p20.csv"
colaData = pd.read_csv(dataDir+colaFile, sep = ",", header = 0, dtype = {"Seed" : np.int,"k3D" : np.float, "Nmodes" : np.float,
"Pk3D" : np.float, "quadPk" : np.float, "hexaPk" : np.float})

seedArr_C = np.sort(colaData.Seed.unique())
nSeeds_C = len(seedArr_C)
nBins_C = len(colaData[colaData.Seed == seedArr_C[0]].index)

Nmodes3D = colaData[colaData.Seed == seedArr_C[0]].Nmodes.to_numpy()
k3D = colaData[colaData.Seed == seedArr_C[0]].k3D.to_numpy()

if nBins_C == nBins_Y:
    print("Same number of bins")
    nBins = nBins_Y
    del nBins_Y, nBins_C
else:
    warnings.warn("The simulation and the surrogate have not the same number of bins. Is it intentional?")

#%% III) Data setup

# Seeds for the Control Variates algorithm
seeds_Est = np.arange(0, 500, dtype = np.int)
gadgetPk = toolsPk.data2Mat(gadgetData, "Pk3D", seeds_Est, nBins)
colaPk = toolsPk.data2Mat(colaData, "Pk3D", seeds_Est, nBins)

# Estimate muC for Pk (1500 L-PICOLA sampless)
startMuC = 1000; endMuC = 2500
seeds_muC = np.arange(startMuC, endMuC, dtype = np.int)
del startMuC, endMuC

colaMat_muC = toolsPk.data2Mat(colaData,"Pk3D", seeds_muC, nBins)
PkMuC = np.mean(colaMat_muC, axis = 1)
SigmaMuC = np.cov(colaMat_muC, rowvar = True, bias = False)
del colaMat_muC

#%% IV) Initialize CARPool instance
pkStep = 5
p = 1
q = 1
N = seeds_Est.shape[0]

# Confidence Intervals
methodCIFix = "None"
alpha = 5

# In case of smoothing
wname = "flat"
wlen = 5
kl = 0.3 ; indSmoothing = np.where(k3D >= kl)[0]

# Create CARPool framework instance
myCARPool = CARPool("matterPk : Quijote GADGET + L-PICOLA 20 steps", gadgetPk, colaPk, truth)
myCARPool.muSurr = PkMuC

#%% V) "Opponent" estimator : sample mean of nSampSim simulation realizations
nSampSim = 500
ciY = "bootstrapBCA"
bootCard = 5000
muY = np.mean(gadgetPk, axis = 1)
lowY, upY = confidenceInt(gadgetPk, ciY, alpha, bootCard, progress = True)

#%% VI) Performance with 5 samples

testFix = myCARPool.createTest("Every 5 additional samples", pkStep, N, p, q, Incremental = False)
testFix.smDict = {"smBool": True ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testFix.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIFix, alpha)

# Reproduce Figure 3 (bottom panel) of 2009.08970
nCV = 5
indTest = int(nCV/pkStep) - 1
betaMat = np.diag(testFix.betaList[indTest])

# Additional (co)variance brought by the estimation of muC (equation 12 in 2009.08970)
SigmaBetaAdd = np.linalg.multi_dot([betaMat, SigmaMuC, betaMat.T]) * 1.0/len(seeds_muC)
stdBetaAdd = np.sqrt(np.diag(SigmaBetaAdd, k = 0))

rootRes = "/home/chartier/Documents/VarianceSlayer/pyCARPool/examples/matterPk/Figures/"
if not(os.path.exists(rootRes)):
    os.mkdir(rootRes)

barBool = True
factErrFix = 1.0
percBool = True
trueBool = True
epsilonBool = True
boolStd = True
xlog = True
titleBool = False

toolsPk.plot_cvRatiosBlues(muY, nSampSim, lowY, upY, barBool, testFix.muCARPool,
                 stdBetaAdd,boolStd, pkStep, k3D, factErrFix, percBool, trueBool,
                 xlog, titleBool,truth, saveBool = False, saveFold = rootRes, saveStr = "Figure3")

#%% VIII) Performance for an increasing number of samples
methodCIInc = "tscore"
NInc = 20

testInc = myCARPool.createTest("Every 5 additional samples", pkStep, NInc, p, q, Incremental = True)
testInc.smDict = {"smBool": False,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIInc, alpha, bootCard)

# Reproduce figure 2 of 2009.08970 (if methodCIInc = bootstrapBCA ; bootstrap results may slightly differ)
# For figure B2, use methodCIInc = tscore
nSampCARP = 5
indCV = int(nSampCARP/pkStep) - 1
factErrInc = 20
zoomBool = True
zoomFact = 3.50
kTupleLim = (k3D[45], k3D[-1]) # hard coded xlim for inserted zoom plot
xlog = True ; ylog = True 
reducedPk = True ; powerK = 1
tupleCor1 = (1,4)

toolsPk.Comparison_errBars(muY, testInc.muCARPool[:,indCV] , k3D, nSampSim, nSampCARP, 
                           lowY, upY, testInc.lowMeanCI[:,indCV], testInc.upMeanCI[:,indCV],
                       factErrInc,zoomBool, kTupleLim,zoomFact,tupleCor1, xlog, ylog, reducedPk, powerK)

#%% Reproduce Figure 5 of 2009.08970
seeds_varTest = np.arange(3000, 6000)
gadPk_varTest = toolsPk.data2Mat(gadgetData, "Pk3D", seeds_varTest, nBins)
colaPk_varTest = toolsPk.data2Mat(colaData, "Pk3D", seeds_varTest, nBins)

# For the simulations only
(signYY, logdetYY) = np.linalg.slogdet(sigmaYY)
detYY = signYY * np.exp(logdetYY)
sigYY = np.sqrt(np.diag(sigmaYY))

# CARPool tests for variance reduction study
NInc2 = 500
meth = "None"
testInc_varU = myCARPool.createTest("Every 5 additional samples", pkStep, NInc2, p, q, Incremental = True)
testInc_varU.smDict = {"smBool": True,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_varU.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth)

# Apply beta estimated with 5, 10, 15... samples to generate 3000 CARPool samples and get a rather good
# estimate of the covariate of x(beta).
sigma2XXU, logdetXXU, signXXU = testInc_varU.varianceAnalysis(gadPk_varTest, colaPk_varTest, myCARPool.muSurr)
detXXU = signXXU * np.exp(logdetXXU)
reducVarU = detXXU/detYY

# Same for the multivariate (M) case : p = q = nBins
testInc_varM = myCARPool.createTest("Every 5 additional samples", pkStep, NInc2, nBins, nBins, Incremental = True)
testInc_varM.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth)

sigma2XXM, logdetXXM, signXXM = testInc_varM.varianceAnalysis(gadPk_varTest, colaPk_varTest, myCARPool.muSurr)
detXXM = signXXM * np.exp(logdetXXM)
reducVarM = detXXM/detYY

sigmaReducM = np.sqrt(sigma2XXM[:,-1])/sigYY
sigmaReducU =  np.sqrt(sigma2XXU[:,-1])/sigYY # actual variance reduction we tend to
sigmaReducUApp = np.sqrt(sigma2XXU[:,0])/sigYY # impact of an "improper" beta diagonal

toolsPk.plotVarReduc_Cov(reducVarM, reducVarU, sigmaReducM, sigmaReducU, sigmaReducUApp ,testInc_varM.Nsamples, k3D)

#%% VI) New test with q = 3
q2 = 3
testInc_q2 = myCARPool.createTest("Every 5 additional samples, q = 3", pkStep, NInc2, p, q2, Incremental = True)
testInc_q2.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q2.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIInc, alpha)

# Reproduce Figure 3 (bottom panel) of 2009.08970
nCV2 = 10
indTest2 = int(nCV2/pkStep) - 1

toolsPk.Comparison_errBars(muY, testInc_q2.muCARPool[:,indTest2] , k3D, nSampSim, nCV2, 
                           lowY, upY, testInc_q2.lowMeanCI[:,indTest2], testInc_q2.upMeanCI[:,indTest2],
                       factErrInc,zoomBool, kTupleLim,zoomFact,tupleCor1, xlog, ylog, reducedPk, powerK)

#%% VII) New test with q = 5

q3 = 5
testInc_q3 = myCARPool.createTest("Every 5 additional samples, q = 5", pkStep, NInc2, p, q3, Incremental = True)
testInc_q3.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q3.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIInc, alpha)

# Reproduce Figure 3 (bottom panel) of 2009.08970
nCV3 = 10
indTest3 = int(nCV3/pkStep) - 1

toolsPk.Comparison_errBars(muY, testInc_q3.muCARPool[:,indTest3] , k3D, nSampSim, nCV3, 
                           lowY, upY, testInc_q3.lowMeanCI[:,indTest3], testInc_q3.upMeanCI[:,indTest3],
                       factErrInc,zoomBool, kTupleLim,zoomFact,tupleCor1, xlog, ylog, reducedPk, powerK)

#%% Extended variance analysis

sigma2XXq2, logdetXXq2, signXXq2 = testInc_q2.varianceAnalysis(gadPk_varTest, colaPk_varTest, myCARPool.muSurr)
detXXq2 = signXXq2 * np.exp(logdetXXq2)
reducVarq2 = detXXq2/detYY

sigma2XXq3, logdetXXq3, signXXq3 = testInc_q3.varianceAnalysis(gadPk_varTest, colaPk_varTest, myCARPool.muSurr)
detXXq3 = signXXq3 * np.exp(logdetXXq3)
reducVarq3 = detXXq3/detYY

sigmaReducq2 = np.sqrt(sigma2XXq2[:,-1])/sigYY
sigmaReducq3 =  np.sqrt(sigma2XXq3[:,-1])/sigYY

toolsPk.plotVarReduc_new(reducVarM, reducVarU, reducVarq2,reducVarq3, sigmaReducM,
                         sigmaReducU,sigmaReducq2, sigmaReducq3, testInc_q2.Nsamples, k3D, q2, q3)