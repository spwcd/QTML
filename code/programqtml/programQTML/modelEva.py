import numpy as np
from sklearn import metrics
import pandas as pd
import bt
'''
def mEvaluate(actResult,testResult,returns,periods=252): 
    #MAE = metrics.mean_absolute_error(actResult,testResult)
    #MSE = metrics.mean_squared_error(actResult,testResult)
    RMSE = round(np.sqrt(metrics.mean_absolute_error(actResult,testResult)),5)
    #varScore = metrics.explained_variance_score(actResult,testResult)      
    #print (MAE, MSE, RMSE, varScore)
    
    Shp = round(np.sqrt(periods) * (np.mean(returns)) / np.std(returns),5)
    mstd = round(np.std(returns),5)
    profit = round(np.sum(returns),5)
    
    return RMSE, Shp, mstd, profit   
'''

def PWLcal(predictor,y_test):

    ylen = len(y_test)
    result = np.zeros((len(y_test),1))
    result2 = np.zeros((len(y_test),1))
    sumWin = 0
    
    
    for i in range(0,len(y_test)):
    
        if(predictor[i] > 0) :
            result[i] =  1
        if(predictor[i] < 0) :
            result[i] =  -1
        if(predictor[i] == 0)  :
            result[i] =  0
     
        if(y_test[i] > 0) :
            result2[i] =  1
        if(y_test[i] < 0) :
            result2[i] =  -1
        if(y_test[i] == 0) :
            result2[i] =  0
          
        if result[i] == result2[i]:
            sumWin = sumWin +1
            
    pWin =  (sumWin/ylen)
 
    return pWin*100
    
# flgType was created in order to separate the pwin calculation of other predictors and the model    
# flgType = 0 --> it is a predictor
# flgType = 1 --> it is a model    
def mEvaluate(actResult,testResult,returns,flgType,periods=252):
    window =  len(returns)
    #MAE = metrics.mean_absolute_error(actResult,testResult)
    #MSE = metrics.mean_squared_error(actResult,testResult)
    RMSE = round(np.sqrt(metrics.mean_absolute_error(actResult,testResult)),5)
    #varScore = metrics.explained_variance_score(actResult,testResult)      
    #print (MAE, MSE, RMSE, varScore)
    
    Shp = round(np.sqrt(periods) * (np.mean(returns)) / np.std(returns),5)
    mstd = round(np.std(returns),5)
    profit = round(np.sum(returns),5)
    temp = np.cumsum(returns)+100    
    
    Roll_Max = pd.rolling_max(temp, window, min_periods=1)
    Daily_Drawdown = temp/Roll_Max - 1.0

    Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)
    
    numWin = 0
    perWin = 0;
    
    if flgType == 0:
        Pwin = PWLcal(testResult, actResult)
    elif flgType == 1:
        for i in range(0,len(returns)):
            if returns[i] > 0 :
                numWin = numWin+1
            
        perWin = numWin/len(returns)*100
        Pwin = perWin; 
    
    return RMSE, Shp, mstd, profit, Pwin, Daily_Drawdown, Max_Daily_Drawdown  
       

    
'''def signalCal(x):
    
    signal = np.arange(len(x))
    for i in range(len(x)):
        if x[i] > 0 :
           signal[i] = 1
        elif x[i] < 0 :
            signal[i] = -1
        else :
            signal[i] = signal[i-1]
    
    return signal'''
    
    
    
 # Create signal 
def signalCal1(x):
    if x > 0 :
        signal = 1
    elif x < 0 :
        signal = -1
    else :
        signal = 0
    
    return signal
 

# Create signal 
def signalCal2(x, y):
    if x > 0 :
        signal = 1
    elif x < 0 :
        signal = -1
    else :
        signal = y
    
    return signal
 
# Calculate weight
def weightCal(w):
    if w > 0:
        score = 1
    elif w < 0:
        score = -1
    else :
        score = 0
        
    return score
  
# Calculate ensemble signal
#def ensembleDir(enDir):

def ensembleDir(enDir, pre_pos): 
    print(enDir)

    cntLong = 0
    cntShort = 0
    
    # put all values to an array for looping
    direction = np.array(enDir)
    n = len(enDir)
    
    for i in range(0,n):
        if direction[i] == -1:
            cntShort = cntShort +1
        elif direction[i] == 1:
            cntLong = cntLong +1
        else :
            cntShort = cntShort
            cntLong = cntLong       
        
    if cntShort > cntLong:
        return -1
    elif cntLong > cntShort:
        return 1
    else:
        return pre_pos
     
    
def ensembleRet(enRet, direction):
    
    # create an arry
    position = np.array(enRet)
    n = len(enRet)
    
    longPos = 0
    cntLong = 0
    
    shortPos = 0
    cntShort = 0
    
    allPos = 0
    
    if direction == 1:
        for i in range(0,n):
            if position[i] > 0:
                longPos = longPos + position[i]
                cntLong = cntLong +1
        if cntLong>0:
            return longPos/cntLong
        else :
            return 0
        
    elif direction == -1:
        for i in range(0,n):
            if position[i] < 0:
                shortPos = shortPos + position[i]
                cntShort = cntShort +1
        if cntShort >0:
            return shortPos/cntShort
        else :
            return 0 
    elif direction == 0:
        for i in range(0,n):
            allPos = allPos + position[i] 
        
        return allPos/n
             
    else:
        return 0
    
from sklearn.metrics import confusion_matrix
              
def confustionM(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm
    
    
def modelEvalute(y_test,P_y_pred,P_result,flgType):
    eva_n = 7
    Peva = np.zeros((1, eva_n))  
    Peva = mEvaluate(y_test, P_y_pred,P_result,flgType)
    
    return Peva
    
