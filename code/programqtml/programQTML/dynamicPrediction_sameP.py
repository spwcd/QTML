# ***** start testing at the same day to evaluate result for our research ***** #
#import sys
#sys.path.append('/Users/junenyjune/anaconda/lib/python3.5/site-packages')

from P1_multiRegr import *
from P2_polyRegr import *
from P3_SVRegr import *
from P4_logisClass import *
from P5_SVMClass import *
from P6_randomForestRegr import *
from P7_SVRegrLin import *
from P8_SVRegrPoly import *
from P9_KnnClass import *
from P10_naiveBaysClass import *
from scoreSet import bordaScore
from dataPrep import *
from modelEva import *
import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
import talib as ta
from matplotlib import style
import time
import os
#import talib as tl
style.use('ggplot')
'''
poon = '/home/poon/Downloads/Project/'
path = poon
'''

dum = '/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/ProgramQTML/programQTML/'
path = dum


'''
dum = '/home/sw991/Research/'
path = dum
'''

### ----- Parameters setting ----- ###

d = 1;                       # lagged day
#initial_train = 0.2        # how big the initial training set  
numTrain = 20;               #trainingSize in Days
increaseSize = 10;            #how much you want to increase trainingSize per round
maxtrain  = 500;              #How many round in main Loop

numPredictors = 10;      
selectPredictor = 0.1;       #in case want to select predictors by number
thPredictor = 1;             # threshold to select predictors that have scores more than x.x percent of maximum score
scoreChoice = 1;             # 1:easyScore, 2:bordaEMA, 3:marjority voting
numMainPred = int(numPredictors * selectPredictor);     # Number of selected predictors

### ----- Download Data and name the Columns ----- ###
#df = quandl.get('GOOG/FRA_DBK', start_date="2016-06-03 00:00:00", end_date="2016-11-13 00:00:00")
df = pd.read_csv("Data/Artificial_data/updown.csv", index_col=0, parse_dates=True)
#df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df[['Open', 'High', 'Low', 'Close']]
df = df.dropna()
#df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]


### ----- Maching Data by Index ----- ###
#df = mergeDF(dfIndex, df)
#df = dataClean(df)


### ----- create feature and data cleaning ----- ###
df = createFeature(df)

df = lagCreate(df,d)
df_dropNA = df.dropna()
# flgStart = df_dropNA.index[0]  --> if want to know the table drop until what day

### ----- create train-test and call the predictor ----- ###
X = np.array(df_dropNA.drop(['Open', 'High', 'Low', "Close", "HL_PCT", "PCT_change", "ATR", "ADX", "+DI", "-DI", "Label"],1))
#X = np.array(df_dropNA.drop(['Open', 'High', 'Low', "Close", "Volume", "HL_PCT", "PCT_change", "ATR", "ADX", "+DI", "-DI", "Label"],1))
    
#scaling feature
X = feaScaling(X)
y = np.array(df_dropNA['PCT_change'])
    
#Label data for classification
y_label = np.array(df_dropNA['Label'])
start_time = time.time()
#for numTPR in range(numTrain, maxtrain+increaseSize, increaseSize):
 
start_test = maxtrain +1;
sizeOfX = len(X)        
testSize = len(X) - start_test
y_test = y[start_test:] 
    
for numTPR in range(numTrain, maxtrain+increaseSize, increaseSize):
    
    filename = str(path) +"Result/detailResult"+ str(numTPR) +"/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    count = 0
    border = 10
    
    # create repository for storing final signal and postion from the model
    model_result = np.zeros((len(X) - start_test, 3))      
    
    # create repository for storing y predictors and results for all predictors
    for x in range(1, numPredictors+1):
        globals()['P%s_y_pred' % x] =  createYpred(X, start_test) 
        globals()['P%s_result' % x] =  createResult(X, start_test) 
        
    #import time
    #start_time = time.time()
    
    for LoopPred in range(start_test,sizeOfX):
    
        X_train = X[LoopPred - numTPR:LoopPred,:]
        y_train = y[LoopPred - numTPR:LoopPred]

        y_train_label = y_label[LoopPred - numTPR:LoopPred]
        X_test = X[LoopPred,:]
    
        # ******** Calling the predictor ************* #
        # Px_result(actual y_test, predicted signal, col0*col1, easySocre, accEasyscore )
    
        for pNum in range(1, numPredictors+1):
            
            if pNum == 4 or pNum == 5 or pNum == 9 or pNum == 10:
                globals()['P%s_y_pred' % pNum][count] = globals()['P%s_modelfiting' % pNum](X_train, y_train_label, X_test)      # predicted y 
            else :
                globals()['P%s_y_pred' % pNum][count] = globals()['P%s_modelfiting' % pNum](X_train, y_train, X_test)            # predicted y 
                
            globals()['P%s_result' % pNum][count,0] = y_test[count]                                                              # the actual y  
            
            if count == 0:
                   globals()['P%s_result' % pNum][count,1] = signalCal1(globals() ['P%s_y_pred' % pNum][count])                  # Signal
            else :     
                   globals()['P%s_result' % pNum][count,1] = signalCal2(globals() ['P%s_y_pred' % pNum][count], globals() ['P%s_result' % pNum][count-1,1])                      # signal from predicted y
            
            globals()['P%s_result' % pNum][count,2] = globals() ['P%s_result' % pNum][count,0] * globals() ['P%s_result' % pNum][count,1] # daily profit
               
        
        # ************** select predictors (maybe main and support) to sent to ensemble *********** #
        listEnDir = []
        listEnRet = []
        listScore = []
        listBorda = []
        
        if scoreChoice == 1:
            # 1) calculate weights
            for acPro in range(1,numPredictors+1):
                globals()['P%s_result' % acPro][count,3] = weightCal(globals() ['P%s_result' % acPro][count,2])                      # wieght from that day
                
                # update accummulated score
                if count > 1:
                    globals()['P%s_result' % acPro][count,4] = globals()['P%s_result' % acPro][count-1,4] + globals()['P%s_result' % acPro][count,3]
                else:
                    globals()['P%s_result' % acPro][count,4] = 0
                
                listScore.append(globals()['P%s_result' % acPro][count,4])
               
            # 2) select only predictors with enough score    
            meanScore = max(listScore) * thPredictor
    
            for thScore in range(1,numPredictors+1):
                if globals()['P%s_result' %thScore][count,4] >= meanScore:
                    listEnDir.append(globals()['P%s_result' % thScore][count,1])
                    listEnRet.append(globals()['P%s_y_pred' % thScore][count])
                 
            if count ==0:                               # set the previous postion to send to ensembleDir
                pre_pos = 0
            else:
                pre_pos = model_result[count-1,0]
            
            # 3) produce results
            model_result[count,0] = ensembleDir(listEnDir,pre_pos) 
            model_result[count,1] = ensembleRet(listEnRet, model_result[count,0])
            del pre_pos
                
        elif scoreChoice == 2 :
            
            if count >= border :
                
                # 1) calculate weights
                for polyScore in range(1,numPredictors+1):
                    globals()['P%s_result' % polyScore][count,5] = ta.EMA(np.asarray(globals()['P%s_result' %polyScore][count-border:count,2]),border)[border-1]
                    listBorda.append(globals()['P%s_result' % polyScore][count,5]) 
                
                
                # calculate borda score
                P1_result[count,6],P2_result[count,6],P3_result[count,6],P4_result[count,6],P5_result[count,6],P6_result[count,6],P7_result[count,6],P8_result[count,6],P9_result[count,6],P10_result[count,6] = bordaScore(listBorda)
                
                # update accummulated score
                for updScore in range(1,numPredictors+1):
                    globals()['P%s_result' % updScore][count,7] = globals()['P%s_result' % updScore][count-1,7] + globals()['P%s_result' % updScore][count,6]
                    listScore.append(globals()['P%s_result' % updScore][count,7]) 
                
                # ************** select main predictors ************* #
                # case 1 -> seclect predictors which have score more than the average
                
                # 2) select only predictors with enough score 
                meanScore = max(listScore) * thPredictor
        
                for thScore in range(1,numPredictors+1):
                    if globals()['P%s_result' %thScore][count,7] >= meanScore:
                        listEnDir.append(globals()['P%s_result' % thScore][count,1])
                        listEnRet.append(globals()['P%s_y_pred' % thScore][count])            
        
                # case 2 -> seclect predictors by using the defined number 
                # ************** Ensemble prediction **************** #
                # 1) for direction
                # 2) for actual return
                
                if count ==0:                               # set the previous postion to send to ensembleDir
                    pre_pos = 0
                else:
                    pre_pos = model_result[count-1,0]
                 # 3) produce results
                model_result[count,0] = ensembleDir(listEnDir,pre_pos) 
                model_result[count,1] = ensembleRet(listEnRet, model_result[count,0])
                del pre_pos
                
        elif scoreChoice == 3:
            
            for thScore in range(1,numPredictors+1):
                listEnDir.append(globals()['P%s_result' % thScore][count,1])
                listEnRet.append(globals()['P%s_y_pred' % thScore][count])
                 
            if count ==0:                               # set the previous postion to send to ensembleDir
                pre_pos = 0
            else:
                pre_pos = model_result[count-1,0]
            
            # 3) produce results
            model_result[count,0] = ensembleDir(listEnDir,pre_pos) 
            #model_result[count,1] = ensembleRet(listEnRet, model_result[count,0])
            model_result[count,1] = ensembleRet(listEnRet, 0)
                
            del pre_pos
        
        count = count + 1
        
        #timer = (time.time() - start_time)
        del X_train 
        del y_train
        del y_train_label
        del X_test 
        
    #print(timer)
    
    # ********* calculate profit of the model ********** #    
    y_test = y_test.reshape(testSize,1)
    model_result = np.concatenate((y_test, model_result), axis =1)
    model_result[:,3] = model_result[:,0] * model_result[:,1]
    
    for x in range(1, numPredictors+1):
        globals()['P%s_eva' % x] =  modelEvalute(y_test[border:],  globals()['P%s_y_pred' % x][border:], globals()['P%s_result' % x][border:,2],0) 
        globals()['P%s_eva_df' % x]  = pd.DataFrame(list(globals()['P%s_eva' % x]))
        print(globals()['P%s_eva' % x])
        
    
    
    model_evluation = modelEvalute(y_test[border:],model_result[border:,0],model_result[border:,3],1)
    
    reRMSE = (P1_eva[0]+P2_eva[0]+P3_eva[0]+P6_eva[0]+P7_eva[0]+P8_eva[0])/6
  
    mydata = [{'model' : model_evluation[0]}, {'model' : model_evluation[1]}, {'model' : model_evluation[2]}, {'model' : model_evluation[3]}, {'model' : model_evluation[4]}, {'model' : model_evluation[5]}, {'model' : model_evluation[6]} ]
    results = pd.concat([P1_eva_df, P2_eva_df, P3_eva_df, P4_eva_df, P5_eva_df, P6_eva_df, P7_eva_df, P8_eva_df, P9_eva_df, P10_eva_df, pd.DataFrame(mydata)], axis=1)
    results.to_csv(str(path) +'Result/detailResult'+ str(numTPR) +'/d'+str(numTPR)  +'.csv')
    
    # save detail of all results
    for saveR in range(1, numPredictors+1):
        pd.DataFrame(globals()['P%s_result' % saveR]).to_csv(str(path) +'Result/detailResult'+ str(numTPR)+'/P' + str(saveR) +'.csv')
     
    pd.DataFrame(model_result).to_csv(str(path) +'Result/detailResult'+ str(numTPR) + '/model' +str(numTPR)+ '.csv')

    '''
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(P1_result[border:,2]), color= 'black', label='P1')
    ax.plot(np.cumsum(P2_result[border:,2]), color = 'red', label='P2')
    ax.plot(np.cumsum(P3_result[border:,2]), color = 'blue', label='P3')
    ax.plot(np.cumsum(P4_result[border:,2]), color = 'green',label='P4')
    ax.plot(np.cumsum(P5_result[border:,2]), color = 'purple',label='P5')
    ax.plot(np.cumsum(P6_result[border:,2]), color = 'pink',label='P6')
    ax.plot(np.cumsum(P7_result[border:,2]), color = 'darkgray',label='P7')
    ax.plot(np.cumsum(P8_result[border:,2]), color = 'gold',label='P8')
    ax.plot(np.cumsum(P9_result[border:,2]), color = 'blueviolet',label='P9')
    ax.plot(np.cumsum(P10_result[border:,2]), color = 'lightskyblue',label='P10')
    ax.plot(np.cumsum(model_result[border:,3]), 'k--', color = 'yellow', label='Our model', linewidth = 3.0)
    plt.xlabel('Trading day')
    plt.ylabel('Profit')
    fig.savefig(str(path) +"Result/detailResult" + str(numTPR) + '/profit' +str(numTPR)+ ".png", dpi=fig.dpi)    
'''    
timer = time.time() - start_time  
print("time usage " +str(timer))    



