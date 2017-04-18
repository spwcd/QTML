# read csv file of all results then analyse and plot graphs

import sys
sys.path.append('/Users/junenyjune/anaconda/lib/python3.5/site-packages')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modelEva import *
import csv


resultPath = '/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/TMB_10_500_inc10_sameP/'

analysisPath = '/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/Analysis/TMB_10_500_inc10_sameP/'

dataPath = '/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/ProgramQTML/programQTML/Data/'

datafile = 'TMB'

numTrain = 20;                   #trainingSize in Days
increaseSize = 10;               #how much you want to increase trainingSize per round
maxtrain  = 500;
numPredictors = 10; 
casePlot = 1;                    # 1: accumulated profit 2: AccPlot 3: plot the same day
  
# read original data to plot closing price

df = pd.read_csv(str(dataPath) + str(datafile) + ".csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots()
plt.grid(True)
plt.plot(df['Close'], color = 'blue', label='Close')
plt.xlabel('Trading day')
plt.ylabel('price')

fig.savefig(str(analysisPath) + str(datafile) + ".png", dpi=fig.dpi, format='png', bbox_inches='tight')    

# ***** Plot accumulated return ***** #    
# read all data in dataframe and collect mean profit of all predictors

# **** load model result ***** #
# avg, max, std of everpredictor model

avgProfit = [];    
maxProfit = [];
stdProfit = []

# avg, max and std of model

m_avgProfit = [];
m_maxProfit = [];
m_stdProfit = [];

for numP in range(1, 11):         # run through each predictor
    
    fig, ax = plt.subplots()
    plt.grid(True)
    
    # plot performace of each predictor through every window sizes
    pProfit = [];
    mProfit = [];
    
    pStdProfit = [];
    mStdProfit = [];
    
    for numTPR in range(numTrain, maxtrain+increaseSize, increaseSize):
        print("predictor" + str(numP) + ": window" + str(numTPR))
        
        globals()['R%s' %numTPR] = pd.read_csv(str(resultPath) + "detailResult" + str(numTPR) + "/P" + str(numP) + ".csv", index_col=0, parse_dates=True)
        globals()['M%s' %numTPR] = pd.read_csv(str(resultPath) + "detailResult" + str(numTPR) + "/model" + str(numTPR) + ".csv", index_col=0, parse_dates=True)

        ax.plot(np.cumsum(globals()['R%s' %numTPR]['2']) )
        ax.plot(np.cumsum(globals()['M%s' %numTPR]['3']),'k--', color = 'yellow', label='Our model', linewidth = 3.0)
    
        pProfit.append(sum(globals()['R%s' %numTPR]['2']));
        mProfit.append(sum(globals()['M%s' %numTPR]['3']));
                           
        pStdProfit.append(np.std(globals()['R%s' %numTPR]['2']))
        mStdProfit.append(np.std((globals()['M%s' %numTPR]['3'])))
        
        
    maxProfit.append(max(pProfit));
    avgProfit.append(np.mean(pProfit));
    stdProfit.append(np.mean(pStdProfit));
    
    m_maxProfit.append(max(mProfit));
    m_avgProfit.append(np.mean(mProfit));
    m_stdProfit.append(np.mean(mStdProfit));
    
    plt.ylabel('Profit')
    plt.xlabel('Trading day')
    plt.title('All plots of predictor' + str(numP) + '- of predictor' + str(numP) + '-data-' + str(datafile))
    fig.savefig(str(analysisPath)+ "P" + str(numP) + "_allProfit.png", dpi=fig.dpi)    
    
    #set size for creating groups of windows
    # case train 20 - 500
    
    startWin = [20, 90, 160, 230, 300, 370, 440, 510];
        
    # Plot performance of each predictor for small groups of windows
    globals()['profit_p%s' % numP] =  [];
    
    for i in range(0,7):         # number of groups
        globals()['profit_range%s' % i] =  0;       # for finding average profit
        cnt = 0
        
        fig, ax = plt.subplots()
        plt.grid(True)
        
        for numTPR in range(startWin[i], startWin[i+1], increaseSize):
            cnt = cnt +1;
            print("P" + str(numP) + "-win" + str(numTPR))
            
            ax.plot(np.cumsum(globals()['R%s' %numTPR]['2']), label="P" + str(numP) + "-" + str(numTPR))
            ax.plot(np.cumsum(globals()['M%s' %numTPR]['3']),'k--', color = 'yellow', label='model_w' + str(numTPR) , linewidth = 3.0)
        
            globals()['profit_range%s' % i] = globals()['profit_range%s' % i] + np.sum(globals()['R%s' %numTPR]['2']);
                
        globals()['profit_range%s' % i] = globals()['profit_range%s' % i]/cnt;
        globals()['profit_p%s' % numP].append(globals()['profit_range%s' % i]);
        
        plt.ylabel('Profit')
        plt.xlabel('Trading day')
        plt.title('Plots win' + str(startWin[i]) + 'to' + str(startWin[i+1]-increaseSize) + '- of predictor' + str(numP) + 'data-' + str(datafile))
            
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
        
        fig.savefig(str(analysisPath) + "P" + str(numP) + "win" + str(startWin[i]) + 'to' + str(startWin[i+1]-increaseSize) + ".png", dpi=fig.dpi, format='png', bbox_inches='tight')    



