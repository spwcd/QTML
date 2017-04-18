import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings; warnings.simplefilter('ignore')
from datetime import datetime

end = datetime.now()
#end = datetime(end.year,end.month,end.day)
start = datetime(end.year - 15,end.month,end.day)

symbols = ['BANPU.BK', 'SCCC.BK', 'TOP.BK', 'AKR.BK', 'PTT.BK']
stocks = pd.DataFrame()

for x in symbols:
    stocks[x] = web.DataReader(x, data_source='yahoo', start=start, end=end)['Adj Close']

stocks = stocks.dropna()

stocks.head()


for x in symbols:
    stocks[x + 'SMA20'] = stocks[x].rolling(window=20).mean()
    stocks[x + 'SMA40'] = stocks[x].rolling(window=40).mean()
    stocks[x + 'Rets'] = np.log(stocks[x] / stocks[x].shift(1))
    stocks[x + 'Position'] = np.where(stocks[x + 'SMA20'] >  stocks[x + 'SMA40'], 1, -1)
    stocks[x + 'Strategy'] = stocks[x + 'Position'].shift(1) * stocks[x + 'Rets']

stocks[['BANPU.BKRets', 'SCCC.BKRets', 'TOP.BKRets', 'AKR.BKRets', 'PTT.BKRets']].cumsum().apply(np.exp).plot(figsize=(15, 10))
stocks[['BANPU.BKStrategy', 'SCCC.BKStrategy', 'TOP.BKStrategy', 'AKR.BKStrategy', 'PTT.BKStrategy']].cumsum().apply(np.exp).plot(figsize=(15, 10))
stocks[['BANPU.BK', 'SCCC.BK', 'TOP.BK', 'AKR.BK', 'PTT.BK']].plot(figsize=(15, 10))

normStocks = pd.DataFrame()
log_return = pd.DataFrame()
for x in symbols:
    normStocks[x] = (stocks[x] / stocks[x].ix[0] * 100)
    log_return[x] = np.log(normStocks[x] / normStocks[x].shift(1))

normStocks[['BANPU.BK', 'SCCC.BK', 'TOP.BK', 'AKR.BK', 'PTT.BK']].plot(figsize=(15, 10))
log_return[['BANPU.BK', 'SCCC.BK', 'TOP.BK', 'AKR.BK', 'PTT.BK']].cumsum().apply(np.exp).plot(figsize=(15, 10))


rets = log_return.mean() * 252

strategyRets = stocks[['BANPU.BKStrategy', 'SCCC.BKStrategy', 'TOP.BKStrategy', 'AKR.BKStrategy', 'PTT.BKStrategy']].mean()*252

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

np.dot(weights, rets)

#log_return = stocks[['BANPU.BKStrategy', 'SCCC.BKStrategy', 'TOP.BKStrategy', 'AKR.BKStrategy', 'PTT.BKStrategy']]
log_return.cov()
log_return.corr()


pvar = np.dot(weights.T, np.dot(log_return.cov() * 252, weights))
pvar

#np.std(log_return)
#len(log_return)

pvol = pvar ** 0.5
pvol

weights = np.random.random(5)  # random numbers
weights /= np.sum(weights)  # normalization to 1
weights
np.dot(weights, rets)

np.dot(weights.T, np.dot(log_return.cov() * 252, weights))

prets = []
pvols = []
for p in range(10000):
    weights = np.random.random(5)
    weights /= np.sum(weights)
    prets.append(np.sum(log_return.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                        np.dot(log_return.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)
portfolio = pd.DataFrame({'return': prets, 'volatility': pvols})

portfolio.plot(x='volatility', y='return', kind='scatter', fontsize=16, figsize=(15, 10))
plt.xlabel('volatility', fontsize=20)
plt.ylabel('return', fontsize=20)
plt.title("Portfolio ",fontsize=25)




