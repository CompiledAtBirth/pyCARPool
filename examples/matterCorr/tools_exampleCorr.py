#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:56:45 2021

@author: chartier
"""

#%%
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%%
# FUNC : Retrieve Pk data and returns a matrix of which each column is a realization
def assembleCorr_data(dirName, fileName, seedList, scope, simulation):
    '''
    Recover simulation matter Corr data and concatenate in a numpy array
    '''
    n_bins = 0
    n_files = 0 # initialize number of actual data files found and in the required seed list
    
    dirTree = os.walk(dirName, topdown = True) #dirTree is a generator object
    # list(dirTree) would give a list of 3-tuples (rootdir of files, subdirs, files) = (str,list,list) for each file
    
    # Extract seed number and determine find delimiter
    if simulation == "GADGET":
        # fullName should be "rootlist/1485/Pk_matter_z=0.5.txt", for instance
        findSeed = lambda pwd, file : pwd.split("/" + file)[0].split("/")[-1]
        fileDelimiter = "\t"
        colLabels = ["Seed","r", "CF"]
        typeList = [np.int, np.float, np.float]
    elif simulation == "LPICOLA":
        # fullName should be "rootlist/Collection_0/subset_3/fid_389/matterPk/Pk_matter_z0p500_grid1024.txt", for instance
        findSeed = lambda pwd, file : pwd.split("/matterCorr/" + file)[0].split("fid_")[-1]
        fileDelimiter = ","
        colLabels = ["Seed","r","Nmodes", "CF", "qCF", "hCF"]
        typeList = [np.int, np.float, np.int, np.float, np.float]
    else:
        sys.exit("Simulation type unkown")
        
    # Initalize empty dataframe with labels
    dataColl = pd.DataFrame(columns = colLabels)
    sanitySeed  = []   
    # EXPLORE THE FILE HIERARCHY AND EXTRACT DESIRE SEEDS
    for rootlist, sublist, filelist in dirTree:        
        
        # Check the required data file exist
        if fileName in filelist:
            
            # Retrieve seed number in data
            fullName = rootlist + "/" + fileName
            seed = findSeed(fullName, fileName)
            seed = int(seed)
            
            # Check if the seed is in the required ones
            if seed in seedList:
                sanitySeed.append(seed)
                dataFile = pd.read_csv(fullName, sep = fileDelimiter, header = None, names = colLabels[1:])
                dataFile.insert(0, "Seed", seed)
                
                dataColl = dataColl.append(dataFile, ignore_index = True)
                
                if n_files == 0: #first file loaded : initialize the dataframe
                    n_bins = len(dataColl.index) #n_bins is fixed for the first file found
                
                n_files += 1
            
                if n_files % 50==0:
                    print("n_files = %d"%n_files)
                    #print("seed = %i"%seed)
     
    if len(dataColl.index) % n_bins !=0:
        print("WARNING : Row number is not a multiple of n_bins : not all the power spectra have the same bins")         
    
    # Sanity data formating    
    for k in np.arange(0, len(typeList)):
        dataColl.iloc[:,k] = dataColl.iloc[:,k].astype(typeList[k])

    # Take only data until tailMax
    dataColl = dataColl[(dataColl.r.to_numpy() <= scope[1]) & (dataColl.r.to_numpy() >= scope[0])]
    
    # Number of bins until kmax for analysis (risky since we assume every seed has the same bins, which should be the case)
    seedCheck = dataColl["Seed"].unique()[0]
    n_bins_R = len(dataColl[dataColl.Seed == seedCheck].index)
    
    return dataColl, n_bins_R, n_files


# Transform datasets into array, while respecting the ordering of seeds
def data2Mat(pandaData, colName, seeds, n_bins):
        
    dataMat = np.empty((n_bins,len(seeds)))
    n = 0
    
    for seed in seeds:
        #dataMat[:, n] = pandaData[pandaData.Seed == seeds[n]].Pk3D.to_numpy()
        dataMat[:, n] = pandaData[pandaData.Seed == seeds[n]][colName].to_numpy()
        n+=1
        
    return dataMat

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
    
    im = plt.imread(get_sample_data('/home/chartier/Documents/VarianceSlayer/pyCARPool/data/z0p5/../' + "68Fig.png"))
    
    nTests = cvTable.shape[1]
    nBins = cvTable.shape[0]
    plt.figure(figsize = (15,9.2))
    
    ylab = r'$\xi(r)$ error  $[\%]$'
    
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
        plt.plot(k3D,1.96*stdMuC/divider*100.0, color = "black", linestyle = "dashed")
        l3, = plt.plot(k3D,-1.96*stdMuC/divider*100.0, color = "black", linestyle = "dashed")
            
    plt.xlabel(r'$r$ $[h^{-1}{\rm Mpc}]$', fontsize = 34)
    plt.ylabel(ylab, fontsize = 48)
    plt.ylim(-20.0,20.0)
    
    if xlog:
        plt.xscale("log")
        
    plt.gca().text(30, 10,"%i"%factErr + r'$\times95\%$ confidence intervals', 
            va='bottom', transform=plt.gca().transData,fontsize=28)
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize =25, which = "minor")
    # Hack to align legend without using monospace (which looks ugly)
    l4, = plt.plot(np.nan, np.nan, '.', ms=0)
    l5, = plt.plot(np.nan, np.nan, '.', ms=0)
    l6, = plt.plot(np.nan, np.nan, '.', ms=0)
    
    plt.legend((l1,l2,l3, l4, l5, l6),[lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2]],fontsize = 27,
               loc = "lower left", handletextpad=0.2, markerscale = 2.2, ncol = 2, columnspacing = -1.9)
    leg = plt.gca().get_legend()
    leg.legendHandles[1].set_color('blue')
    plt.tight_layout()
    plt.grid(zorder=2)
    
    newax = plt.gcf().add_axes([0.40, 0.78, 0.24, 0.24], zorder=1)
    newax.imshow(im)
    newax.axis('off')
    
    plt.show()
    
    if saveBool:
        plt.savefig(saveFold + saveStr + ".png")
    
# Plot two different estimation results and their estimated 95% Confidence intervals
def Comparison_errBars(gadPk, cvPk , k3D, nSamplesGad, nSamplesCV, lowBarGad, upBarGad, lowBarCV, upBarCV, reducedPk,
                       factErr,zoomBool, kTupleZoom,zoomF,tupleCorners, xlog, ylog, saveBool, saveFold, saveStr):
    
    
    if reducedPk:
        factor = np.power(k3D,2.0)
        gadPk = factor * gadPk
        cvPk = factor * cvPk
        lowBarGad = factor * lowBarGad
        upBarGad = factor * upBarGad
        lowBarCV = factor * lowBarCV
        upBarCV = factor * upBarCV
        ylab = r'$r^2 \xi (r)$'
    else:
        ylab = r'$\xi (r)$'
        
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
    
    plt.figure(figsize = (15.2,9.4))
    l1 = plt.errorbar(k3D, gadPk, xerr = None, yerr = np.stack([lowErrG, upErrG], axis = 0), color = "red",
                 marker = "x", markersize = 3, zorder = 1,linewidth = 1.8, elinewidth = 1.2, capsize=7)
    l2 = plt.errorbar(k3D, cvPk, xerr = None, yerr = np.stack([lowErrCV, upErrCV], axis = 0), color = "blue",
                 marker = "x", markersize = 3, zorder = 2, linewidth = 1.4, linestyle = "dashed" , 
                 elinewidth = 1.2, capsize=4.0)
    plt.xlabel(r'$r$ $[h^{-1}{\rm Mpc}]$', fontsize = 34)
    plt.ylabel(ylab, fontsize = 38)
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
    20.0, -3.0, "%i"%factErr +r'$\times$$95\%$ confidence intervals',  # x, y, text,
    va='bottom',   # text alignment,
    transform=plt.gca().transData, fontsize = 28)  # coordinate system transformation
    plt.tick_params(axis="both", labelsize = 31, which = "major")
    plt.tick_params(axis="both", labelsize = 25, which = "minor")
    
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
    #plt.xlim(40.0, 220.0)
    #plt.ylim(top=np.max(cvPk[k3D>40.0]), bottom = -0.001)
    plt.tight_layout()
    plt.show()
    if saveBool:
        plt.savefig(saveFold + saveStr + ".png")
    

def plotVarReduc_Cov(reducVar, reducVarDiag, sigmaReduc, sigmaReducDiag, sigmaReducApp ,nSamplesList ,k3D_binned):
    
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
    plt.loglog(k3D_binned, sigmaReduc, label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.loglog(k3D_binned, sigmaReducApp, linestyle = "dashed" , linewidth = 1.1, color = "grey", marker = "x", markersize = 4)
    plt.loglog(k3D_binned, sigmaReducDiag, label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$r$ $[h^{-1}{\rm Mpc}]$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 27, loc = "upper right")
    plt.tight_layout()
    plt.grid()
    plt.show()
    
def plotVarReduc_new(reducVar, reducVarDiag, reducVarq, reducVarq2, sigmaReduc,
                     sigmaReducDiag, sigmaReducq, sigmaReducq2 ,nSamplesList,k3D_binned, q, q2):
    
    plt.figure(figsize=(12,7.5))
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
    
    plt.figure(figsize=(12,7.5))
    #plt.semilogx(k3D_binned, np.power(sigmaReduc,2), label = "Multivariate CARPool", linewidth = 1.1, marker = "x", markersize = 5, color = "blue")
    plt.semilogx(k3D_binned, np.power(sigmaReducDiag,2), label = "Univariate CARPool", linewidth = 1.1, color = "black", marker = "x", markersize  = 5)
    plt.semilogx(k3D_binned, np.power(sigmaReducq,2), label = "CARPool q=%i"%q, linewidth = 1.1, color = "orangered", marker = "x", markersize  = 5)
    plt.semilogx(k3D_binned, np.power(sigmaReducq2,2), label = "CARPool q=%i"%q2, linewidth = 1.1, color = "indigo", marker = "x", markersize  = 5)
    plt.xticks(fontsize=27, rotation = 0)
    plt.yticks(fontsize=27, rotation = 45)
    plt.xlabel(r'$r$ $[h^{-1}{\rm Mpc}]$', fontsize = 32, style = "italic", y = -0.1)
    plt.ylabel(r"$\frac{\sigma_x}{\sigma_y}$", fontsize = 48, rotation = 0.0, labelpad = 27)
    plt.legend(fontsize = 25, loc = "upper right")
    plt.tight_layout()
    plt.grid()
    plt.show()