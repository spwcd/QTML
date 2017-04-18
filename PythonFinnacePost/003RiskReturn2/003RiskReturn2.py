import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

symbols = ['BANPU.BK', 'SCCC.BK', 'TOP.BK', 'AKR.BK', 'PTT.BK']

stocks = pd.DataFrame()
for x in symbols:
    stocks[x] = web.DataReader(x, data_source='yahoo')['Adj Close']

normStocks = (stocks / stocks.ix[0] * 100)
log_return = np.log(normStocks / normStocks.shift(1))
rets = log_return.dropna()

rets.std()
rets.mean()

#print(rets.mean(),rets.std())

area = np.pi*100

plt.scatter(rets.std(), rets.mean(),alpha = 0.5,s =area)
plt.ylabel('Expected returns')
plt.xlabel('Risk')

for label, x, y in zip(rets.columns, rets.std(), rets.mean()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
    plt.xlim(0.010, 0.035)
    plt.ylim(0.0002,0.0008)

yearly_rest = rets.mean()*252

plt.scatter(rets.std(), yearly_rest, alpha=0.5, s=area)
plt.xlabel('Risk')
plt.ylabel('Expected returns')

for label, x, y in zip(rets.columns, rets.std(), yearly_rest):
    plt.annotate(
        label,
        xy=(x, y), xytext=(50, 50),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))

    plt.xlim(0.012, 0.033)
    plt.ylim(0.05,0.20)



plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)
plt.xlabel('Risk')
plt.ylabel('Expected returns')


for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
    plt.xlim(0.0002,0.0007)
    plt.ylim(0.010, 0.035)

yearly_rest = rets.mean()*252

plt.scatter(yearly_rest, rets.std(),alpha = 0.5,s =area)

plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, yearly_rest, rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
