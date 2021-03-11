#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:44:32 2021

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

# Gadget with error bars, means of few samples for control variates
def plot_cvRatiosBlues(gadPDF, nSamplesGad, gadLow, gadUp, barBool, truePDF, cvTable, stdMuC,boolStd,testSize, k3D, factErr, percBool, trueBool,
                 xlog, titleBool):
        
    nTests = cvTable.shape[1]
    nBins = cvTable.shape[0]
    plt.figure(figsize = (15,9.0))
    
    ylab = r'PDF error  $[\%]$'
    
    arrOnes = np.ones((nBins,), dtype = np.float)
    
    if trueBool:
        divider = truePDF
    else:
        divider = gadPDF
    
    upErrG = factErr*(gadUp/divider - gadPDF/divider) * 100.0
    lowErrG = factErr*(gadPDF/divider - gadLow/divider) * 100.0
    
    lab1 = ["%i"%nSamplesGad,"%i"%testSize]
    if testSize >=10 and testSize < 100:
       lab1[1] = '{:>4}'.format(lab1[1])
    elif testSize >= 100 :
        lab1[1] = '{:>3}'.format(lab1[1])
    else:     
       lab1[1] = '{:>5}'.format(lab1[1])
    lab1[0] = '{:>3}'.format(lab1[0])
    lab1[0] = '{:>3}'.format(lab1[0])
    lab1.append(r'$95\%$')
    lab2 = ["GADGET","GADGET w/ CARPool ({} sets)".format(nTests)]
    lab2.append(r' error band due to $\mathbf{\overline{\mu}_c}$')
    
    if barBool == False:
        if percBool:
            l1 = plt.plot(k3D, gadPDF/divider - arrOnes, color = "red", linestyle ="dashed", linewidth=1.8, zorder = 1)
        else:
            l1 = plt.plot(k3D, gadPDF/divider, color = "red", linestyle ="dashed", linewidth=1.8, zorder = 1)
    else:
        if percBool:
            l1 = plt.errorbar(k3D, gadPDF/divider - arrOnes, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3.8, zorder = 1,linewidth = 1.8, elinewidth = 1.5, capsize=8)
        else:
            l1 = plt.errorbar(k3D, gadPDF/divider, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3.8, zorder = 1,linewidth = 1.8,  elinewidth = 1.5, capsize=8)  
    
    for n in np.arange(0,nTests):
        
        if n < nTests -1:
            if percBool:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider - arrOnes) * 100.0, c = "blue", linestyle ="None",  zorder = -1, 
                     marker = "_", s = 50, linewidth = 2.2)
            else:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider), color = "blue", linestyle ="None",  zorder = -1, 
                     marker = "_", s =50, linewidth = 2.2)    
        else:
            if percBool:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider - arrOnes) * 100.0, c = "blue", linestyle ="None",marker = "_", s = 50,
                     zorder = -1, linewidth = 2.2)
            else:
                l2 = plt.scatter(k3D, (cvTable[:,n]/divider), color = "blue", linestyle ="None",marker = "_", s = 50,
                     zorder = -1, linewidth = 2.2)
    
    if boolStd:
        plt.plot(k3D,1.96*stdMuC/divider*100.0, color = "black", linestyle = "dashed")
        l3, = plt.plot(k3D,-1.96*stdMuC/divider*100.0, color = "black", linestyle = "dashed")
            
    plt.xlabel(r'$\rho/\bar{\rho}$', fontsize = 44)
    plt.ylabel(ylab, fontsize = 48)
    if xlog:
        plt.xscale("log")
    #print(factErr*(gadPDF[1]/divider[1] - gadLow[1]/divider[1]) * 100.0)
#    plt.gca().annotate("%i"%factErr + r'$\times95%$ confidence intervals',
#            xy=(k3D[1], -factErr*(gadPDF[1]/divider[1] - gadLow[1]/divider[1]) * 100.0), xycoords='data',
#            xytext=(0.03, -0.7), textcoords='data',
#            arrowprops=dict(arrowstyle="->",
#                            connectionstyle="arc3"), fontsize=23)
    plt.gca().text(0.1, -10,"%i"%factErr + r'$\times95\%$ confidence intervals', 
            va='bottom', transform=plt.gca().transData,fontsize=28)
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize =25, which = "minor")
    # Hack to align legend without using monospace
    l4, = plt.plot(np.nan, np.nan, '.', ms=0)
    l5, = plt.plot(np.nan, np.nan, '.', ms=0)
    l6, = plt.plot(np.nan, np.nan, '.', ms=0)
    
    plt.legend((l1,l2,l3, l4, l5, l6),[lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2]],fontsize = 28,
               loc = "upper center", handletextpad=0.2, markerscale = 2.2, ncol = 2, columnspacing = -1.9)
    leg = plt.gca().get_legend()
    leg.legendHandles[1].set_color('blue')
    plt.tight_layout()
    plt.grid()
    
    plt.show()
    
# Plot two estimate with error bars           
def Comparison_errBars(gadPDF, cvPDF, truth, od, nSamplesGad, nSamplesCV, lowBarGad, upBarGad, lowBarCV, upBarCV,
                       factErr,zoomBool, kTupleZoom,zoomF, xlog, ylog):
    
    upErrG = factErr*(upBarGad - gadPDF)
    lowErrG = factErr*(gadPDF - lowBarGad)
    upErrCV = factErr*(upBarCV - cvPDF)
    lowErrCV = factErr *(cvPDF - lowBarCV)
    
    lab1 = ["%i"%nSamplesGad,"%i"%nSamplesCV]
    if nSamplesCV >=10 and nSamplesCV < 100:
       lab1[1] = '{:>4}'.format(lab1[1])
    elif nSamplesCV >= 100 :
        lab1[1] = '{:>3}'.format(lab1[1])
    else:     
       lab1[1] = '{:>5}'.format(lab1[1])
    lab1[0] = '{:>3}'.format(lab1[0])
    lab2 = ["GADGET","GADGET w/ CARPool"]
    
    ylab = r"Matter PDF"
    
    plt.figure(figsize = (15,9.0))
    l1 = plt.errorbar(od, gadPDF, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3, zorder = 1,linewidth = 1.8, elinewidth = 1.2, capsize=7)
    l2 = plt.errorbar(od, cvPDF, xerr = None, yerr = np.stack([lowErrCV, upErrCV], axis = 0), color = "blue",
                 marker = "x", markersize = 3, zorder = 2, linewidth = 1.4, linestyle = "dashed", 
                  elinewidth = 1.2, capsize=4.0)
    plt.xlabel(r'$\rho/\bar{\rho}$', fontsize = 44)
    plt.ylabel(ylab, fontsize = 38)
    if xlog:
        plt.xscale("log")
    if ylog:  
        plt.yscale("log")
    else:
        plt.ylim(-2e-3, np.max(gadPDF) + 1e-2)
    #plt.xticks(fontsize = 24)
    #plt.yticks(fontsize = 24)
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize = 25, which = "minor")
    plt.grid()
    # Hack to align legend without using monospace
    l3, = plt.plot(np.nan, np.nan, '.', ms=0)
    l4, = plt.plot(np.nan, np.nan, '.', ms=0)
    plt.legend([l1,l2,l3,l4],[lab1[0],lab1[1],lab2[0],lab2[1]], ncol = 2,columnspacing = -1.9, handletextpad = 0.2,
               fontsize = 28, loc = "best")
#    if ylog:
#        plt.legend(labTests,fontsize = 27, loc = "best", handletextpad = 0.2)
#    else:
#        plt.legend(fontsize = 27, loc = "upper left", handletextpad = 0.2)
    
    #Zoom plot on smaller scales
    if zoomBool:
        ax = plt.gca()
        axins = zoomed_inset_axes(ax, zoomF, loc="upper right")
        axins.errorbar(od, gadPDF, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 2, zorder = 1,linewidth = 1.8,  elinewidth = 1.1, capsize=7)
        axins.errorbar(od, cvPDF, xerr = None, yerr = np.stack([lowErrCV, upErrCV], axis = 0), color = "blue",
                 marker = "x", markersize = 2, zorder = 2, linewidth = 1.2, linestyle = "dashed",  
                 elinewidth = 1.1, capsize=4.0)
        axins.set_xlim(kTupleZoom[0], kTupleZoom[1]) # apply the x-limits
        indLim = [(od >= kTupleZoom[0]) & (od <= kTupleZoom[1])]
        indLim = tuple(indLim) # optional, but the following "filtering" will be deprecated apparently as of 3.7
        if ylog:
            axins.set_yscale("log")
        zyMin = min(np.min(gadPDF[indLim] - lowErrG[indLim]) , np.min(cvPDF[indLim] - lowErrCV[indLim]))
        zyMax = max(np.max(gadPDF[indLim] + upErrG[indLim]), np.max(cvPDF[indLim] + upErrCV[indLim]))
        axins.set_ylim(zyMin, zyMax)
        plt.setp(axins.get_xticklabels(), visible=False)
        plt.setp(axins.get_yticklabels(), visible=True)
        axins.tick_params(axis = "y", which = "minor",labelsize = 12)
        #axins.xaxis.set_visible('False') ALSO WORKS
        #axins.yaxis.set_visible('True')
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.gca().text(  # position text relative to data
    0.6, 1e-5, "%i"%factErr + r'$\times$$95\%$ confidence intervals',  # x, y, text,
    va='bottom',   # text alignment,
    transform=plt.gca().transData, fontsize = 28)      # coordinate system transformation
    plt.tight_layout()
    plt.show()

def plotVarReduc_Cov(reducVar, reducVarDiag, sigmaReduc, sigmaReducDiag, sigmaReducApp ,nSamplesList ,overdensity, nApp):
    
    plt.figure(figsize=(12,7.5))
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
    
    plt.figure(figsize=(12,7.5))
    plt.loglog(overdensity, sigmaReduc, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.loglog(overdensity, sigmaReducApp, label= "Univariate CARPool, %i samples"%nApp, linestyle = "dashed" , linewidth = 1.1, color = "grey", marker = "x", markersize = 4)
    plt.loglog(overdensity, sigmaReducDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$\rho/\bar{\rho}$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 27, loc = "upper right")
    plt.tight_layout()
    plt.grid()
    plt.show()
    
def plotVarReduc_new(reducVar, reducVarDiag, reducVarq, reducVarq2, sigmaReduc,
                     sigmaReducDiag, sigmaReducq, sigmaReducq2, sigmaReducApp, nSamplesList,overdensity, q, q2, qApp, nApp):
    
    plt.figure(figsize=(12,7.5))
    plt.semilogy(nSamplesList, reducVar, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.semilogy(nSamplesList, reducVarDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.semilogy(nSamplesList, reducVarq, label = "CARPool q=%i"%q, linewidth = 1.1, color = "red", marker = "x", markersize  = 5)
    plt.semilogy(nSamplesList, reducVarq2, label = "CARPool q=%i"%q2, linewidth = 1.1, color = "orange", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel("Number of simulation pairs", fontsize = 27, style = "italic", y = -0.1)
    plt.ylabel(r'$\frac{det \left(  \mathbf{\Sigma_{xx}}(\mathbf{\hat{\beta}})  \right)}{det \left(  \mathbf{\Sigma_{yy}} \right)}$', 
                        fontsize = 40, rotation = 90)
    plt.legend(fontsize = 24, loc = "best")
    plt.tight_layout()
    plt.grid()
    plt.show()

    
    #%%Plot variance ratio on diagonal only
    
    plt.figure(figsize=(12,7.5))
    plt.loglog(overdensity, sigmaReduc, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.loglog(overdensity, sigmaReducDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.loglog(overdensity, sigmaReducq, label = "CARPool q=%i"%q, linewidth = 1.1, color = "red", marker = "x", markersize  = 5)
    plt.loglog(overdensity, sigmaReducq2, label = "CARPool q=%i"%q2, linewidth = 1.1, color = "orange", marker = "x", markersize  = 5)
    plt.loglog(overdensity, sigmaReducApp, label = "CARPool q=%i, %i samples"%(qApp, nApp), linestyle = "dashed" , linewidth = 1.1, color = "grey", marker = "x", markersize = 4)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$\rho/\bar{\rho}$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 24, loc = "upper right")
    plt.tight_layout()
    plt.grid()
    plt.show()