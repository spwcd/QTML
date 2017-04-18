
# coding: utf-8

# In[3]:

import numpy as np  # array operations
import pandas as pd  # time series management


# In[4]:

import pandas.io.data as web


# In[5]:

goog = web.DataReader('GOOG', data_source='yahoo',start='3/14/2009', end='4/14/2014')


# In[6]:

goog.head()


# In[7]:

goog.info() 


# In[8]:

goog[['Open', 'Close']].tail()


# In[10]:

goog.iloc[:2] 


# In[11]:

get_ipython().magic(u'matplotlib inline')


# In[14]:

goog[['Adj Close']].plot(figsize=(15, 10))


# In[16]:

rets = np.log(goog['Adj Close'] / goog['Adj Close'].shift(1))


# In[17]:

rets.hist(figsize=(10, 6), bins=35);


# In[19]:

goog['MA50'] = pd.rolling_mean(goog['Adj Close'], window=50)


# In[20]:

goog[['Adj Close', 'MA50']].plot(figsize=(10, 6));

