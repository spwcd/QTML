#import sys
#sys.path.append("/home/poon/anaconda3/pkgs/quandl-3.0.1-py35_0/lib/python3.5/site-packages")
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns
#import quandl

data.DataReader('DEXJPUS', 'fred')
#quandl.get('GOOG/BKK_STA')
def macrossover(stock, shortMA, longMA, start_date, end_date):
    #collect data from yahoo finance
    stock = data.DataReader(ticker, 'yahoo', start=start_date, end=end_date)
    stock['shortMA'] = np.round(stock['Close'].rolling(window=shortMA).mean(), 4)
    stock['longMA'] = np.round(stock['Close'].rolling(window=longMA).mean(), 4)
    stock.dropna(inplace=True)
    stock['position'] = np.where(stock['shortMA'] > stock['longMA'], 1, 0)
    stock['Rets'] = np.log(stock['Close'] / stock['Close'].shift(1))
    stock['strategy'] = stock['position'].shift(1) * stock['Rets']
    sharpe = np.sqrt(252) * (stock['strategy'].mean() / stock['strategy'].std())
    rets = stock['strategy'].cumsum()[-1]

    return rets, sharpe


short = np.linspace(10,30,2,dtype=int)
long = np.linspace(40,60,2,dtype=int)
profit = np.zeros((len(short),len(long)))
sharperatio = np.zeros((len(short),len(long)))

ticker = 'CPALL.BK'
# define start and end date
start_date = '2013/01/23'
end_date = '2017/01/23'

for i, s in enumerate(short):
    for j, l in enumerate(long):
        rets, sharpe = macrossover(ticker, s, l,start_date ,end_date)
        profit[i,j] = rets
        sharperatio[i,j] = sharpe

plt.pcolor(short, long, sharperatio)
plt.colorbar()
plt.show()

plt.pcolor(short,long,profit)
plt.colorbar()
plt.show()


f = open('/home/poon/Downloads/AKB48-Team-8-3rd-Anniversary-Koen-Rundown-MCs-170406.txt',"r",encoding='utf-8', errors='ignore') #opens file with name of "test.txt"
x = f.readlines()
len(x)

import pandas as pd
import numpy as np
y = pd.Series(len(x))
yy=np.arange(len(x))

for i in range(742):
    y[i] = x[741-i]

y.to_csv('/home/poon/Downloads/s.csv')

np.savetxt(r'/home/poon/Downloads\np.txt', y.values)

yy = y.to_string()








