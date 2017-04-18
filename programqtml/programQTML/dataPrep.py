# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:45:55 2016

@author: Sornpon Wichaidit

"""
import statsmodels.formula.api as sm
import numpy as np
import talib as tl
from sklearn.preprocessing import StandardScaler

def degree():
    degree = 3              # degree of polynomail regression predictor
    return degree
    
#Create Feature
def createFeature(df):
    df['HL_PCT'] = (df['High'] - df['Low'])/df['Close']*100.0
    df['PCT_change'] = (df['Close'] - df['Close'].shift(1))/df['Close'].shift(1)*100.0
    df['ATR'] = tl.ATR(np.asarray(df['High']),np.asarray(df['Low']),np.asarray(df['Close']),14)
    df['ADX'] = tl.ADX(np.asarray(df['High']), np.asarray(df['Low']), np.asarray(df['Close']), timeperiod=14)
    df['+DI'] = tl.PLUS_DI(np.asarray(df['High']), np.asarray(df['Low']), np.asarray(df['Close']), timeperiod=14)
    df['-DI'] = tl.MINUS_DI(np.asarray(df['High']), np.asarray(df['Low']), np.asarray(df['Close']), timeperiod=14)
    df['Label'] = df['PCT_change'].apply(labelUD) 

    return df
    
       
def lagCreate(df,d):    
    df['PCT_change-1'] = df['PCT_change'].shift(d)
    df['Open-1'] = df['Open'].shift(d)
    df['High-1'] = df['High'].shift(d)
    df['Low-1'] = df['Low'].shift(d)
    df['Close-1'] = df['Close'].shift(d)
    #df['Volume-1'] = df['Volume'].shift(d)
    df['HL_PCT-1'] = df['HL_PCT'].shift(d)
    df['PCT_change-1'] = df['PCT_change'].shift(d)
    df['ATR-1'] = df['ATR'].shift(d)
    df['ADX-1'] = df['ADX'].shift(d)
    df['+DI-1'] = df['+DI'].shift(d)
    df['-DI-1'] = df['-DI'].shift(d)
    
    return df




def backwardElimination(x, SL):
    
    numVars = len(x[0])
    temp = np.zeros((891,10)).astype(int)
    for i in range(0, numVars):
        # STEP 2 of Backward Elimination
        # Fit the model with predictors
        regressor_OLS = sm.OLS(y, x).fit()
 
        # STEP 3 of Backward Elimination
        # Find the predictor with the highest P-Value
        # if P > SL, go to STEP 4...else FIN
        maxVar = max(regressor_OLS.pvalues).astype(float)
 
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
                
        if maxVar > SL:
            for j in range(0,numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    # store the column being removed in a temp matrix
                    temp[:,j] = x[:, j]
 
                    # STEP 4 of Backward Elimination
                    # Remove the highest predictor found in STEP 3
                    x = np.delete(x, j, 1)
 
                    # check to see if the adjusted r-squared value was larger
                    # before or after the deletion
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
 
                    # if the model has gotten worse, reinsert the previously
                    # removed variable and return that version of the matrix
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
 
                    
    # Model Complete
    print(regressor_OLS.summary())
    return x
    
    
#cal UP or DOWN
def labelUD(x):
    if x > 0:
        return 1
    elif x < 0 :
        return -1
    else:   
        return 0

def feaScaling(x):
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x
    
    
def mergeDF(dfIndex, df):

    merged = df.join(dfIndex,rsuffix='_y')
    merged.drop(merged.columns[[0, 1, 2, 3, 4]], axis=1) 
    #X = merged[['Open', 'High', 'Low', 'Close', 'Volume']]
    X = merged[['Open', 'High', 'Low', 'Close']]
    
    return X
    

def dataClean(df):    
    dflen = len(df)
    missAmount = dflen*0.98
    op, hi, lo, cl = 0, 0, 0, 0
    bound = 0.3
    for loop in range(0,dflen):
    
        if loop > 0 : 
          
           if ((df.Open[loop]-df.Open[loop-1])/df.Open[loop-1]) > bound:
               #df.Open[loop] = (df.Open[loop-1]+df.Open[loop+1])/2
               df.Open[loop] = df.Open[loop-1]
               op = op + 1
            
           if ((df.High[loop]-df.High[loop-1])/df.High[loop-1]) > bound:
               #df.High[loop] = (df.High[loop-1]+df.High[loop+1])/2
               df.High[loop] = df.High[loop-1]
               hi = hi+1
               
           if ((df.Low[loop]-df.Low[loop-1])/df.Low[loop-1]) > bound:
               #df.Low[loop] = (df.Low[loop-1]+df.Low[loop+1])/2
               df.Low[loop] = df.Low[loop-1]
               lo = lo+1
               
           if ((df.Close[loop]-df.Close[loop-1])/df.Close[loop-1]) > bound:
               #df.Close[loop] = (df.Close[loop-1]+df.Close[loop+1])/2
               df.Close[loop] = df.Close[loop-1]
               cl = cl+1
           
        if (op > missAmount or hi > missAmount or lo > missAmount or cl > missAmount):
             df = missAmount
           
    return df
    
def createResult(X, start_test):
    
    result = np.zeros((len(X) - start_test, 8)) 
    
    return result

def createYpred(X, start_test):
    
    result = np.zeros((len(X) - start_test,1)) 
    
    return result

    
    