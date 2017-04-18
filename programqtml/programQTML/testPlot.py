import sys
sys.path.append('/Users/junenyjune/anaconda/lib/python3.5/site-packages')

import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
import talib as ta
from matplotlib import style


# *** plot learning score for every predictor *** #
model = pd.read_csv("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/Different_window/Window_best_for_Alldata/CPF/model.csv", index_col=0, parse_dates=True)
d = pd.read_csv("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/Different_window/Window_best_for_Alldata/CPF/d.csv", index_col=0, parse_dates=True)
pStd = 0;
for numP in range(1, 11):  
    globals()['P%s_result' %numP] = pd.read_csv("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/Different_window/Window_best_for_Alldata/CPF/P" + str(numP) + ".csv", index_col=0, parse_dates=True)
    
    
fig, ax = plt.subplots()                
ax.plot(np.cumsum(P1_result['3']), color= 'black', label='P1')
ax.plot(np.cumsum(P2_result['3']), color = 'red', label='P2')
ax.plot(np.cumsum(P3_result['3']), color = 'blue', label='P3')
ax.plot(np.cumsum(P4_result['3']), color = 'green',label='P4')
ax.plot(np.cumsum(P5_result['3']), color = 'purple',label='P5')
ax.plot(np.cumsum(P6_result['3']), color = 'pink',label='P6')
ax.plot(np.cumsum(P7_result['3']), color = 'darkgray',label='P7')
ax.plot(np.cumsum(P8_result['3']), color = 'gold',label='P8')
ax.plot(np.cumsum(P9_result['3']), color = 'blueviolet',label='P9')
ax.plot(np.cumsum(P10_result['3']), color = 'lightskyblue',label='P10') 


plt.xlabel('Trading day')
plt.ylabel('Cumsum Score')
fig.savefig("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/Different_window/Window_best_for_Alldata/CPF/CPFLScore.png", dpi=fig.dpi)    


  
# **** plot data ****** #
df = pd.read_csv("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/ProgramQTML/programQTML/Data/CPF.csv", index_col=0, parse_dates=True)
#df['Close'] = df['Adj Close']
df = df[508:]  #only for CPN because it was partitioned
#dfDown = pd.read_csv("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Data/Artificial_data/down.csv", index_col=0, parse_dates=True)



fig, ax = plt.subplots()
#ax.plot(df['Open'], color= 'black', label='P1')
#ax.plot(df['High'], color = 'red', label='P2')
#ax.plot(df['Low'], color = 'green', label='P3')
ax.plot(df['Close'], color = 'blue',label='P4')
   
plt.ylabel('Profit')

fig.savefig("/Users/junenyjune/Documents/1_Research/2_QTML/0_QTML_Progs/Result/EasyScore/Different_window/Window_best_for_Alldata/CPF/dataPredict.png", dpi=fig.dpi)    
