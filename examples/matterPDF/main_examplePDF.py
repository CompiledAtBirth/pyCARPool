#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:44:31 2021

@author: chartier
"""
import os
dataDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/data/z0p5/"
workDir = "/home/chartier/Documents/VarianceSlayer/pyCARPool/examples/matterPDF/"
os.chdir(workDir)

import numpy as np
import pandas as pd
import warnings

# Custom functions
import tools_examplePDF as toolsPDF
from pyCARPool import CARPool, confidenceInt

#%% I) GADGET N-body raw data

gadgetFile="quijoteGadget_PDF_R5.csv"
gadgetData = pd.read_csv(dataDir+gadgetFile, sep = ",", header = 0, dtype = {"Seed" : np.int,"OD" : np.float,  "pdfVal" : np.float})

seedArr_Y = np.sort(gadgetData.Seed.unique())
nSeeds_Y = len(seedArr_Y)
nBins_Y = len(gadgetData[gadgetData.Seed == seedArr_Y[0]].index)

simMat_Tot = toolsPDF.data2Mat(gadgetData, "pdfVal", seedArr_Y, nBins_Y)

# True mean and covariance matrix of 15000 samples
truth = np.mean(simMat_Tot, axis = 1)
sigmaYY = np.cov(simMat_Tot, rowvar = True, bias = False)

del simMat_Tot

#%% II) COLA raw data

colaFile="lPicola_PDF_R5.csv"
colaData = pd.read_csv(dataDir+colaFile, sep = ",", header = 0, dtype = {"Seed" : np.int,"OD" : np.float,  "pdfVal" : np.float})

seedArr_C = np.sort(colaData.Seed.unique())
nSeeds_C = len(seedArr_C)
nBins_C = len(colaData[colaData.Seed == seedArr_C[0]].index)

overdensity = colaData[colaData.Seed == seedArr_C[0]].OD.to_numpy()

if nBins_C == nBins_Y:
    print("Same number of bins")
    nBins = nBins_Y
    del nBins_Y, nBins_C
else:
    warnings.warn("The simulation and the surrogate have not the same number of bins. Is it intentional?")

#%% III) Data setup

# Seeds for the Control Variates algorithm
seeds_Est = np.arange(0, 500, dtype = np.int)
gadgetPDF = toolsPDF.data2Mat(gadgetData, "pdfVal", seeds_Est, nBins)
colaPDF = toolsPDF.data2Mat(colaData, "pdfVal", seeds_Est, nBins)

# Estimate muC for Pk (1500 L-PICOLA sampless)
startMuC = 500; endMuC = 2000
seeds_muC = np.arange(startMuC, endMuC, dtype = np.int)
del startMuC, endMuC

colaMat_muC = toolsPDF.data2Mat(colaData,"pdfVal", seeds_muC, nBins)
PDFMuC = np.mean(colaMat_muC, axis = 1)
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
odl = 0.3
indSmoothing = np.where(overdensity >= odl)[0]

# Create CARPool framework instance
myCARPool = CARPool("matterPk : Quijote GADGET + L-PICOLA 20 steps", gadgetPDF, colaPDF, truth)
myCARPool.muSurr = PDFMuC

#%% V) "Opponent" estimator : sample mean of nSampSim simulation realizations
nSampSim = 500
ciY = "bootstrapBCA"
bootCard = 5000
muY = np.mean(gadgetPDF, axis = 1)
lowY, upY = confidenceInt(gadgetPDF, ciY, alpha, bootCard, progress = True)

#%% VI) Performance with 5 samples
NFix = 50
testFix = myCARPool.createTest("Every 5 additional samples", NFix, N, p, q, Incremental = False)
testFix.smDict = {"smBool": True ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testFix.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIFix, alpha)

# Reproduce Figure 3 (bottom panel) of 2009.08970
nCV = 50
indTest = int(nCV/cvStep) - 1
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

toolsPDF.plot_cvRatiosBlues(muY, nSampSim, lowY, upY, barBool, truth, testFix.muCARPool, stdBetaAdd,
                 boolStd, NFix, overdensity, factErrFix, percBool, trueBool,
                 xlog, titleBool)

#%% VIII) Performance for an increasing number of samples
methodCIInc = "bootstrapPiv"

testInc = myCARPool.createTest("Every 5 additional samples", cvStep, N, p, q, Incremental = True)
testInc.smDict = {"smBool": True,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, methodCIInc, alpha, bootCard)

# Reproduce figure 2 of 2009.08970 (if methodCIInc = bootstrapBCA ; bootstrap results may slightly differ)
# For figure B2, use methodCIInc = tscore
nCARP = 50
indCV = int(nCARP/cvStep) - 1
factErrInc = 40
zoomBool = False
zoomFact = 3.50
odTupleLim = (overdensity[42], overdensity[60]) # hard coded xlim for inserted zoom plot
xlog = True ; ylog = True 
tupleCor1 = (1,4)

toolsPDF.Comparison_errBars(muY, testInc.muCARPool[:,indCV], truth, overdensity, nSampSim, 
                        nCARP, lowY, upY, testInc.lowMeanCI[:,indCV], testInc.upMeanCI[:,indCV],
                       factErrInc,zoomBool, odTupleLim,zoomFact, xlog, ylog)

#%% Reproduce Figure 5 of 2009.08970
seeds_varTest = np.arange(500, 2300)
gad_varTest = toolsPDF.data2Mat(gadgetData, "pdfVal", seeds_varTest, nBins)
cola_varTest = toolsPDF.data2Mat(colaData, "pdfVal", seeds_varTest, nBins)

safeguard = 1000.0 # for overflowing determinant computation

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
nApp = 30 # figure 10
indApp  = int(nApp/cvStep) - 1
sigmaReducUApp = np.sqrt(sigma2XXU[:,indApp])/sigYY # impact of an "improper" 

toolsPDF.plotVarReduc_Cov(reducVarM, reducVarU, sigmaReducM, sigmaReducU, sigmaReducUApp ,testInc_varM.Nsamples, overdensity, nApp)

#%% VI) New test with q = 3
q2 = 3
meth_q2 = "bootstrapPiv"

testInc_q2 = myCARPool.createTest("Every 5 additional samples, q = 3", cvStep, NInc2, p, q2, Incremental = True)
testInc_q2.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q2.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth_q2, alpha)

nCARP_q2 = 40
indCV_q2 = int(nCARP_q2/cvStep) - 1

toolsPDF.Comparison_errBars(muY, testInc_q2.muCARPool[:,indCV_q2], truth, overdensity, nSampSim, 
                        nCARP_q2, lowY, upY, testInc.lowMeanCI[:,indCV_q2], testInc.upMeanCI[:,indCV_q2],
                       factErrInc,zoomBool, odTupleLim,zoomFact, xlog, ylog)

#%% VII) New test with q = 5
q3 = 5
meth_q3 = "bootstrapPiv"

testInc_q3 = myCARPool.createTest("Every 5 additional samples, q = 5", cvStep, NInc2, p, q3, Incremental = True)
testInc_q3.smDict = {"smBool": False ,"wname":wname, "wlen": wlen, "indSmooth":indSmoothing}
testInc_q3.computeTest(myCARPool.simData, myCARPool.surrData, myCARPool.muSurr, meth_q3, alpha)

nCARP_q3 = 40
indCV_q3 = int(nCARP_q3/cvStep) - 1

toolsPDF.Comparison_errBars(muY, testInc_q2.muCARPool[:,indCV_q3], truth, overdensity, nSampSim, 
                        nCARP_q3, lowY, upY, testInc.lowMeanCI[:,indCV_q3], testInc.upMeanCI[:,indCV_q3],
                       factErrInc,zoomBool, odTupleLim,zoomFact, xlog, ylog)

#%% VIII) Extended variance analysis

sigma2XXq2, logdetXXq2, signXXq2 = testInc_q2.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXq2 = signXXq2 * np.exp(logdetXXq2/safeguard)
reducVarq2 = np.power(detXXq2/detYY, safeguard)

sigma2XXq3, logdetXXq3, signXXq3 = testInc_q3.varianceAnalysis(gad_varTest, cola_varTest, myCARPool.muSurr)
detXXq3 = signXXq3 * np.exp(logdetXXq3/safeguard)
reducVarq3 = np.power(detXXq3/detYY, safeguard)

# We take the best estimate of beta we have --> variance reduction to expect at best
sigmaReducq2 = np.sqrt(sigma2XXq2[:,-1])/sigYY
sigmaReducq3 =  np.sqrt(sigma2XXq3[:,-1])/sigYY

nApp_q = 30 # beta diagonal with 10 samples, as in 2009.08970 figure 10
indApp_q  = int(nApp_q/cvStep) - 1
sigmaReducApp_q = np.sqrt(sigma2XXq2[:,indApp_q])/sigYY # impact of an "improper"
qApp = q2

toolsPDF.plotVarReduc_new(reducVarM, reducVarU, reducVarq2,reducVarq3, sigmaReducM,
                         sigmaReducU,sigmaReducq2, sigmaReducq3, sigmaReducApp_q, testInc_q2.Nsamples, overdensity, q2, q3, qApp, nApp_q)