import pandas as pd
import numpy as np
import scipy.stats as st

def myknn_reg(XTr, yTr, XTe, k):

    ## Reshape 1D array to 2D
    if len(XTe.shape)==1:
        XTe = np.reshape(XTe, (1, XTe.shape[0]))
        
    numTe = XTe.shape[0]
    
    ## Create output vector
    yTe = np.zeros(numTe)
    
    ## Find nearest neighbour for each testing sample
    for i in np.arange(0, XTe.shape[0]):
        print('Test sample no: ' + str(i))
        vTe = XTe[i,:]
        distAll = np.linalg.norm(XTr - vTe, axis=1)
        indNN = distAll.argsort()[:k]
        yTe[i] = yTr[indNN].mean()

    ## Return predictions
    return yTe

def myknn_class(XTr, yTr, XTe, k):

    ## Reshape 1D array to 2D
    if len(XTe.shape)==1:
        XTe = np.reshape(XTe, (1, XTe.shape[0]))
        
    numTe = XTe.shape[0]
    
    ## Create output vector
    yTe = np.zeros(numTe)
    
    ## Find nearest neighbour for each testing sample
    for i in np.arange(0, XTe.shape[0]):
        print('Test sample no: ' + str(i))
        vTe = XTe[i,:]
        distAll = np.linalg.norm(XTr - vTe, axis=1)
        indNN = distAll.argsort()[:k]
        yTe[i] = st.mode(yTr[indNN]).mode[0]

    ## Return predictions
    return yTe
