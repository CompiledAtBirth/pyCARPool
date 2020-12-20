#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:25:45 2020

@author: chartier
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%%

# FUNC : COMPRESS PANDAS DATAFRAMES TO LARGER BINS
def compressDATA_Pk(dataBase, seedList, target_nbins, Nmodes3D):
    
    # REMARK :Here we drop any addiional entry in COLA data (choice of convenience, for now)
    newData = pd.DataFrame(columns = ["Seed", "k3D", "Nmodes", "Pk3D"])
    
    # Precess for each seed separately
    for seed in seedList:
        
        dataS = dataBase[dataBase.Seed == seed]
        k3D = dataS["k3D"].to_numpy()
        Pk3D = dataS["Pk3D"].to_numpy()
        
        # Process the data and append to main dataset
        k3D_new, Pk3D_new, Nmodes_new = regroup_PkBins(target_nbins, k3D, Pk3D, Nmodes3D)
        
        dataTemp = pd.DataFrame(data={"k3D" : k3D_new, "Nmodes" : Nmodes_new, "Pk3D" : Pk3D_new})
        dataTemp.insert(0, "Seed", seed)
        
        newData = newData.append(dataTemp, ignore_index = True)
        
    return newData

# FUNC : Reduce the Pk resolution by using larger bin intervals
def regroup_PkBins(target_nbins, k3D, Pk3D, Nmodes3D):
    """
    Pk3D and k3D, in this version, are numpy arrays of only one seed extracted from a pandas dataframe
    with the to_numpy() method
    """
    N = k3D.shape[0]
    n = int(N/target_nbins)
    
    k3D_new = np.zeros((target_nbins,), dtype = float)
    Pk3D_new = np.zeros((target_nbins,), dtype = float)
    Nmodes_new = np.zeros((target_nbins,), dtype = float)
    
    for p in np.arange(0, target_nbins):
        
        # Central kbin value
        k3D_new[p] = 0.5*(k3D[p * n] + k3D[(p + 1) * n - 1])
        
        # Could be more elegant, but does the job
        weightedPk = Nmodes3D[p * n : (p + 1) * n] * Pk3D[p * n : (p + 1) * n ]
        sumModes = np.sum(Nmodes3D[p * n : (p + 1) * n])
        Pk3D_new[p] = 1/(sumModes)*np.sum(weightedPk)
        Nmodes_new[p] = sumModes
        
#    # Sanity check
#    if np.sum(Pk3D_new) != np.sum(Pk3D):
#        print("Warning : The compressed vector has not the same total power as the original")
#        diff = np.sum(Pk3D_new) - np.sum(Pk3D)
#        print("Difference is %1.4f" %diff)


# Transform datasets into array, while respecting the ordering of seeds
def data2Mat(pandaData, colName, seeds, n_bins):
        
    dataMat = np.empty((n_bins,len(seeds)))
    n = 0
    
    for seed in seeds:
        #dataMat[:, n] = pandaData[pandaData.Seed == seeds[n]].Pk3D.to_numpy()
        dataMat[:, n] = pandaData[pandaData.Seed == seeds[n]][colName].to_numpy()
        n+=1
        
    return dataMat

# FUNC : Estimate the mean of the "cheap" estimator COLA on a separate set of seeds than for control variates

def muCPk(colaData, seedList, n_bins):
    
    colaMat = np.empty((n_bins, len(seedList)))
    n = 0
    
    for seed in seedList:
        colaMat[:, n] = colaData[colaData.Seed == seedList[n]].Pk3D.to_numpy()
        n += 1
    
    # Deviation from the estimated muC    
    muFid = np.mean(colaMat, axis = 1)
    #diffC = colaMat - np.reshape(muFid,(n_bins,1))
    
    SigmaCC= np.cov(colaMat, rowvar = True, bias = False)
    
    #return muFid, sigmaDiff, sigmaAlt
    return muFid, SigmaCC

def list2array(mylist):
    p = len(mylist)
    N = mylist[0].shape[0]
    arr = np.zeros((p,N), dtype  = np.float32)
    
    for k in np.arange(0,p):
        arr[k,:] = mylist[k]
    
    return arr

# Color ratio for cv sets plot
def colorRatioTable(cvTable, divider):
    
    nBins = cvTable.shape[0]
    nTests = cvTable.shape[1]
    
    percArr = np.zeros((nBins, nTests), dtype = np.float)
    arrOnes = np.ones((nBins,), dtype = np.float)
    
    for n in np.arange(0, nTests):
        percArr[:,n] = (cvTable[:,n]/divider - arrOnes) * 100.0
        
    outer16 = np.percentile(percArr, 16.0, axis = 1) # 1D arrays of length the number of bins
    outer84 = np.percentile(percArr, 84.0, axis = 1)  # idem
    
    colorsList = []
    for k in np.arange(0,nTests):
        blueList = []
        for n in np.arange(0,nBins):
            if percArr[n,k] <= outer16[n] or percArr[n,k] >= outer84[n]:
                blueList.append("deepskyblue")
            else:
                blueList.append("blue")
        colorsList.append(blueList)
    
    return colorsList

# Gadget with error bars, means of few samples for control variates
def plot_cvRatiosBlues(gadPk, nSamplesGad, gadLow, gadUp, barBool, cvTable, stdMuC,boolStd,testSize, 
                       k3D, factErr, percBool, trueBool, xlog, titleBool, truePk,
                       saveBool = False, saveFold = None, saveStr = None):
        
    nTests = cvTable.shape[1]
    nBins = cvTable.shape[0]
    plt.figure(figsize = (15,9.0))
    
    ylab = r'$P(k)$ error  $[\%]$'
    
    arrOnes = np.ones((nBins,), dtype = np.float)
    
    if trueBool:
        divider = truePk
    else:
        divider = gadPk
    
    blues = colorRatioTable(cvTable, divider)
    
    upErrG = factErr*(gadUp/divider - gadPk/divider) * 100.0
    lowErrG = factErr*(gadPk/divider - gadLow/divider) * 100.0
    
    lab1 = ["%i"%nSamplesGad,"%i"%testSize]
    if testSize >=10:
       lab1[1] = '{:>4}'.format(lab1[1])
    else:
       lab1[1] = '{:>5}'.format(lab1[1])
    lab1[0] = '{:>3}'.format(lab1[0])
    lab1.append(r'$95\%$')
    lab2 = ["GADGET sims","GADGET w/ CARPool ({} sets)".format(nTests)]
    lab2.append(r' error band due to $\mathbf{\overline{\mu}_c}$')
    
    if barBool == False:
        if percBool:
            l1 = plt.plot(k3D, gadPk/divider - arrOnes, color = "red", linestyle ="dashed", linewidth=1.8, zorder = 1)
        else:
            l1 = plt.plot(k3D, gadPk/divider, color = "red", linestyle ="dashed", linewidth=1.8, zorder = 1)
    else:
        if percBool:
            l1 = plt.errorbar(k3D, gadPk/divider - arrOnes, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3.8, zorder = 1,linewidth = 1.8, elinewidth = 1.5, capsize=8)
        else:
            l1 = plt.errorbar(k3D, gadPk/divider, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3.8, zorder = 1,linewidth = 1.8,  elinewidth = 1.5, capsize=8)  
    
    for n in np.arange(0,nTests):
        
        if n < nTests -1:
            if percBool:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider - arrOnes) * 100.0, c = blues[n], linestyle ="None",  zorder = -1, 
                     marker = "_", s = 50, linewidth = 2.2)
            else:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider), color = blues[n], linestyle ="None",  zorder = -1, 
                     marker = "_", s =50, linewidth = 2.2)    
        else:
            if percBool:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider - arrOnes) * 100.0, c = blues[n], linestyle ="None",marker = "_", s = 50,
                     zorder = -1, linewidth = 2.2)
            else:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider), color = blues[n], linestyle ="None",marker = "_", s = 50,
                     zorder = -1, linewidth = 2.2)
    
    if boolStd:
        plt.plot(k3D,1.96*stdMuC/divider*100.0, color = "grey", linestyle = "dashed")
        l3, = plt.plot(k3D,-1.96*stdMuC/divider*100.0, color = "grey", linestyle = "dashed")
            
    plt.xlabel(r'$k$ $[h{\rm Mpc^{-1}}]$', fontsize = 36)
    plt.ylabel(ylab, fontsize = 48)
    
    if xlog:
        plt.xscale("log")
        
    plt.gca().text(0.03, 0.8,"%i"%factErr + r'$\times95\%$ confidence intervals', 
            va='bottom', transform=plt.gca().transData,fontsize=28)
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize =25, which = "minor")
    # Hack to align legend without using monospace (which looks ugly)
    l4, = plt.plot(np.nan, np.nan, '.', ms=0)
    l5, = plt.plot(np.nan, np.nan, '.', ms=0)
    l6, = plt.plot(np.nan, np.nan, '.', ms=0)
    
    plt.legend((l1,l2,l3, l4, l5, l6),[lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2]],fontsize = 28,
               loc = "lower right", handletextpad=0.2, markerscale = 2.2, ncol = 2, columnspacing = -1.9)
    leg = plt.gca().get_legend()
    leg.legendHandles[1].set_color('blue')
    plt.tight_layout()
    plt.grid()
    
    plt.show()
    
    if saveBool:
        plt.savefig(saveFold + saveStr + ".png")
    
# Plot two different estimation results and their estimated 95% Confidence intervals
def Comparison_errBars(gadPk, cvPk , k3D, nSamplesGad, nSamplesCV, lowBarGad, upBarGad, lowBarCV, upBarCV,
                       factErr,zoomBool, kTupleZoom,zoomF,tupleCorners, xlog, ylog, reducedPk, powK):
    
    if reducedPk:
        factor = np.power(k3D,powK) * 1/(2*np.power(np.pi,2))
        gadPk = factor * gadPk
        cvPk = factor * cvPk
        lowBarGad = factor * lowBarGad
        upBarGad = factor * upBarGad
        lowBarCV = factor * lowBarCV
        upBarCV = factor * upBarCV
        #ylab = r"$\frac{kP(k)}{2\pi^2}$ $[{\rm Mpc^{2}}]$" # cannot find a way to incorporate %i and %powK in the string
        ylab = r"$\frac{kP(k)}{2\pi^2}$ $[(h^{-1}{\rm Mpc})^2]$"
    else:
        ylab = r'$P(k)$ $[{\rm Mpc^{3}}]$'
        
    upErrG = factErr*(upBarGad - gadPk)
    lowErrG = factErr*(gadPk - lowBarGad)
    upErrCV = factErr*(upBarCV - cvPk)
    lowErrCV = factErr *(cvPk - lowBarCV)
    
    lab1 = ["%i"%nSamplesGad,"%i"%nSamplesCV]
    if nSamplesCV >=10:
       lab1[1] = '{:>4}'.format(lab1[1])
    else:
       lab1[1] = '{:>5}'.format(lab1[1])
    if nSamplesGad>=10:
        lab1[0] = '{:>3}'.format(lab1[0])
    else:
        lab1[0] = '{:>4}'.format(lab1[0])
    lab2 = ["GADGET","GADGET w/ CARPool"]
    #labTests = ['{:<3} GADGET{:<15}'.format(lab1[idx], lab2[idx]) for idx in range(len(lab1))] %sol1
    
    lgdSize = 27.0
    
    plt.figure(figsize = (15,9.0))
    l1 = plt.errorbar(k3D, gadPk, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3, zorder = 1,linewidth = 1.8, elinewidth = 1.2, capsize=7)
    l2 = plt.errorbar(k3D, cvPk, xerr = None, yerr = np.stack([lowErrCV, upErrCV], axis = 0), color = "blue",
                 marker = "x", markersize = 3, zorder = 2, linewidth = 1.4, linestyle = "dashed" , 
                 elinewidth = 1.2, capsize=4.0)
    plt.xlabel(r'$k$ $[h{\rm Mpc^{-1}}]$', fontsize = 36)
    plt.ylabel(ylab, fontsize = 48)
    if xlog:
        plt.xscale("log")
    if ylog:  
        plt.yscale("log")
#    plt.xticks(fontsize = 24)
#    plt.yticks(fontsize = 24)
#    plt.gca().annotate("%i"%factErr + r'$\times95\%$ confidence intervals',
#            xy=(k3D[1], cvPk[1] - lowErrCV[1]), xycoords='data',
#            xytext=(0.03, 12.), textcoords='data',
#            arrowprops=dict(arrowstyle="->",
#                            connectionstyle="arc3"), fontsize=24)
    plt.gca().text(  # position text relative to data
    0.02, 12., "%i"%factErr +r'$\times$$95\%$ confidence intervals',  # x, y, text,
    va='bottom',   # text alignment,
    transform=plt.gca().transData, fontsize = 28)  # coordinate system transformation
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize = 25, which = "minor")
#    plt.legend(labTests, loc = "upper right", handletextpad = 0.3,
#               prop = {'family':'monospace', 'size':lgdSize}) #if prop is used, fontsize is ignored;monospace is ugly
    
    # Hack to align legend without using monospace
    l3, = plt.plot(np.nan, np.nan, '.', ms=0)
    l4, = plt.plot(np.nan, np.nan, '.', ms=0)
    plt.legend([l1,l2,l3,l4],[lab1[0],lab1[1],lab2[0],lab2[1]], ncol = 2,columnspacing = -1.9, handletextpad = 0.2,
               fontsize = lgdSize, loc = "upper right")

    plt.grid()
    
    #Zoom plot on smaller scales
    if zoomBool:
        ax = plt.gca()
        axins = zoomed_inset_axes(ax, zoomF, loc="lower center")
        axins.errorbar(k3D, gadPk, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 2, zorder = 1,linewidth = 1.8, elinewidth = 1.1, capsize=7)
        axins.errorbar(k3D, cvPk, xerr = None, yerr = np.stack([lowErrCV, upErrCV], axis = 0), color = "blue",
                 marker = "x", markersize = 2, zorder = 2, linewidth = 1.2, linestyle = "dashed", 
                 elinewidth = 1.1, capsize=4.0)
        axins.set_xlim(kTupleZoom[0], kTupleZoom[1]) # apply the x-limits
        indLim = [(k3D >= kTupleZoom[0]) & (k3D <= kTupleZoom[1])]
        indLim = tuple(indLim) # optional, but the following "filtering" will be deprecated apparently as of 3.7
        if ylog:
            axins.set_yscale("log")
        zyMin = min(np.min(gadPk[indLim] - lowErrG[indLim]), np.min(cvPk[indLim] - lowErrCV[indLim]))
        zyMax = max(np.max(gadPk[indLim] + upErrG[indLim]), np.max(cvPk[indLim] + upErrCV[indLim]))
        axins.set_ylim(zyMin, zyMax)
        plt.setp(axins.get_xticklabels(), visible=False)
        plt.setp(axins.get_yticklabels(), visible=True)
        axins.tick_params(axis = "y", which = "minor",labelsize = 16)
        #axins.xaxis.set_visible('False') ALSO WORKS
        #axins.yaxis.set_visible('True')
        mark_inset(ax, axins, loc1=tupleCorners[0], loc2=tupleCorners[1], fc="none", ec="0.5")
    plt.tight_layout()
    plt.show()
    

def plotVarReduc_Cov(reducVar, reducVarDiag, sigmaReduc, sigmaReducDiag, sigmaReducApp ,nSamplesList ,k3D_binned):
    
    A = plt.figure(figsize=(12,7.5))
    plt.semilogy(nSamplesList, reducVar, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.semilogy(nSamplesList, reducVarDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel("Number of simulation pairs", fontsize = 27, style = "italic", y = -0.1)
    plt.ylabel(r'$\frac{det \left(  \mathbf{\Sigma_{xx}}(\mathbf{\hat{\beta}})  \right)}{det \left(  \mathbf{\Sigma_{yy}} \right)}$', 
                        fontsize = 40, rotation = 90)
    plt.legend(fontsize = 27, loc = "best")
    plt.tight_layout()
    plt.grid()
    plt.show()

    
    #%%Plot variance ratio on diagonal only
    
    B = plt.figure(figsize=(12,7.5))
    plt.loglog(k3D_binned, sigmaReduc, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.loglog(k3D_binned, sigmaReducApp, linestyle = "dashed" , linewidth = 1.1, color = "grey", marker = "x", markersize = 4)
    plt.loglog(k3D_binned, sigmaReducDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$k$ $[h{\rm Mpc^{-1}}]$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 27, loc = "lower right")
    plt.tight_layout()
    plt.grid()
    plt.show()
    
def plotVarReduc_new(reducVar, reducVarDiag, reducVarq, reducVarq2, sigmaReduc,
                     sigmaReducDiag, sigmaReducq, sigmaReducq2 ,nSamplesList,k3D_binned, q, q2):
    
    A = plt.figure(figsize=(12,7.5))
    plt.semilogy(nSamplesList, reducVar, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.semilogy(nSamplesList, reducVarDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.semilogy(nSamplesList, reducVarq, label = "CARPool q=%i"%q, linewidth = 1.1, color = "orangered", marker = "x", markersize  = 5)
    plt.semilogy(nSamplesList, reducVarq2, label = "CARPool q=%i"%q2, linewidth = 1.1, color = "indigo", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel("Number of simulation pairs", fontsize = 27, style = "italic", y = -0.1)
    plt.ylabel(r'$\frac{det \left(  \mathbf{\Sigma_{xx}}(\mathbf{\hat{\beta}})  \right)}{det \left(  \mathbf{\Sigma_{yy}} \right)}$', 
                        fontsize = 40, rotation = 90)
    plt.legend(fontsize = 27, loc = "best")
    plt.tight_layout()
    plt.grid()
    plt.show()

    
    #%%Plot variance ratio on diagonal only
    
    B = plt.figure(figsize=(12,7.5))
    plt.loglog(k3D_binned, sigmaReduc, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.loglog(k3D_binned, sigmaReducDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.loglog(k3D_binned, sigmaReducq, label = "CARPool q=%i"%q, linewidth = 1.1, color = "orangered", marker = "x", markersize  = 5)
    plt.loglog(k3D_binned, sigmaReducq2, label = "CARPool q=%i"%q2, linewidth = 1.1, color = "indigo", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$k$ $[h{\rm Mpc^{-1}}]$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 27, loc = "lower right")
    plt.tight_layout()
    plt.grid()
    plt.show()