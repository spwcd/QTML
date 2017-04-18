
# coding: utf-8

# In[1]:

import numpy as np  # array operations
import pandas as pd  # time series management
import pandas.io.data as web
import matplotlib.pyplot as plt  # standard plotting library

# put all plots in the notebook itself
get_ipython().magic(u'matplotlib inline')


# In[2]:

SPX = web.DataReader('^GSPC', data_source='yahoo', start='2005-1-1')


# In[5]:

SPX.tail(3)  # the final five rows


# # Generating Trading Signals
# We want to implement a trading strategy based on simple moving averages (SMA). 
# We work with two SMAs:
# short-term SMA over 42 days (SMA42)
# long-term SMA over 252 days (SMA252)
# We distinguish two constellations:
# SMA42 > SMA252: buy signal, being long the market
# SMA42 < SMA252: sell signal, being short the market
# We calculate the two SMAs as follows.

# In[6]:

SPX['SMA42'] = pd.rolling_mean(SPX['Adj Close'], window=42)
SPX['SMA252'] = pd.rolling_mean(SPX['Adj Close'], window=252)
SPX.dropna(inplace=True)  # drop rows with NaN values


# In[10]:

SPX[['Adj Close', 'SMA42', 'SMA252']].plot(figsize=(15, 12));


# This need to be formalized for the calculations to come. We represent "being long the market" by 1 and "being short the market" by -1.

# In[11]:

# vectorized evaluation of the trading condition/signal generation
SPX['position'] = np.where(SPX['SMA42'] > SPX['SMA252'], 1, -1)


# In[12]:

SPX[['Adj Close', 'position']].plot(subplots=True, figsize=(10, 6))
plt.ylim(-1.1, 1.1)  # adjust y-axis limits


# # Backtesting = Judging Performance¶
# Let us calculate the log returns as in the first module. These are needed to judge the performance, i.e. to backtest, our SMA-based trading strategy. We call the column market since these are the market returns.
# 

# In[13]:

# vectorized calculation of log returns
SPX['market'] = np.log(SPX['Adj Close'] / SPX['Adj Close'].shift(1))


# # Next, we can use the market returns to derive the strategy returns in vectorized fashion. Note the shift of the position column by one day, i.e. we have entered/maintained a position yesterday and and today's returns. It now becomes clear, why 1 represents a long position and -1 a short position: we get the market return when we are long and -1 times the market return when we are short. All this makes obviously a number of simplifying assumptions (e.g. no transaction costs).

# In[14]:

# vectorized calculation of strategy returns
SPX['strategy'] = SPX['position'].shift(1) * SPX['market']


# In[15]:

SPX[['market', 'strategy']].cumsum().apply(np.exp).tail()


# In[16]:

SPX[['market', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# # Risk and Return
# Final consideration: what about the relation between risk & return? Let us quickly do the math. The annualized return of the strategy is obviously higher that from the market ...

# In[17]:

arets = SPX[['market', 'strategy']].mean() * 252  # annualized returns
arets


# ... while the annualized volatility is more or less the same. The higher returns do not lead to higher risk in this case.

# In[18]:

astds = SPX[['market', 'strategy']].std() * 252 ** 0.5  # annualized volatility
astds


# In[ ]:



