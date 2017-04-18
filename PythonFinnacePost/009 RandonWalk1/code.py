
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from random import randrange
from array import array
import talib
def getDF():
    path =r'/home/poon/PycharmProjects/Quant/Data/set-archive_EOD_UPDATE' # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return frame

def getStock(name, frame):
    stock = frame.loc[frame['Symbol'] == name]
    stock = stock.sort_values(by='Date')
    stock['Date'] = pd.to_datetime(stock['Date'], format='%Y%m%d')
    stock = stock.set_index(['Date'])
    del stock['Symbol']
    return stock

nba = pd.read_csv('nba.csv')
height, weight= nba.values[:,1], nba.values[:,3]
weight = weight/2.20462
height = height*2.54

plt.plot(weight, height, 'o')
plt.title('Height vs Weight of NBA Player 2014')
plt.xlabel('Weight kg')
plt.ylabel('Height cm')
plt.ylim(160,230)

frame = getDF()
df = getStock('SET', frame)

df['Close']['2013':'2017'].plot()
df['Close']['2013':'2017'].pct_change().plot()[1:]


def signalCal1(x):
    if x > 0:
        signal = 1
    elif x < 0:
        signal = -1
    else:
        signal = 0

    return signal
dfroc = df['Close'].pct_change()
for i in range(len(df['Close'])):
    print(signalCal1(dfroc[i]))

plt.hist(df['Close']['2013':'2017'].pct_change()[1:])
ADX = talib.ADX(high=df['High'].__array__(), low=df['Low'].__array__(), close=df['Close'].__array__())
ATR = talib.ATR(high=df['High'].__array__(), low=df['Low'].__array__(), close=df['Close'].__array__())



stocklen = len(df['Close']['2006':'2017'])
plt.plot(ATR[-stocklen-1:-1]/ATR[-stocklen-1]-1, df['Close']['2006':'2017'].pct_change(),'o')
plt.title('SET Price Analysis')
plt.xlabel('Some indicator %Change')
plt.ylabel('SET %Change')

stocklen = len(df['Close']['2013':'2017'])
plt.plot(ATR[-stocklen-1:-1]/ATR[-stocklen-1]-1, df['Close']['2013':'2017'].pct_change(), 'o')
plt.title('SET Price Analysis')
plt.xlabel('Some indicator %Change')
plt.ylabel('SET %Change')

fit = np.polyfit(ATR[-stocklen:]/ATR[-stocklen]-1, df['Close']['2006':'2017'].pct_change(), deg=1)

ranstock = pd.Series(1000)
for i in range(1000):
    ranstock[i] = randint(-300, 300)

ranstock = pd.Series(100000)
for i in range(100000):
    ranstock[i] = randint(-1, 1)

ranstock = ranstock/100
ranstock.cumsum().plot()

ranstock.plot()
df['Close']['2013':'2017'].pct_change()[1:].plot()

rans = pd.Series(100)
for i in range(100):
    rans[i] = randint(1,10)

ranl = pd.Series(100000)
for i in range(100000):
    ranl[i] = randint(1,10)