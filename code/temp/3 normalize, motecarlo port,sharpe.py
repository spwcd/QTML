
# coding: utf-8

# In[1]:

import numpy as np  # array operations
import pandas as pd  # time series management
import pandas.io.data as web
import matplotlib.pyplot as plt  # standard plotting library

# put all plots in the notebook itself
get_ipython().magic(u'matplotlib inline')
import warnings; warnings.simplefilter('ignore')


# # Retrieving Stock Price Data
# This module is about the Markowitz Mean-Variance Portfolio Theoy (MVP). We need to retrieve some stock price data first to have something to work with. We build a portfolio of tech companies.

# In[2]:

symbols = ['AAPL', 'MSFT', 'YHOO', 'AMZN', 'GOOG']  # our symbols
data = pd.DataFrame()  # empty DataFrame
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo')['Adj Close']


# In[3]:

data.tail()


# In[4]:

(data / data.ix[0] * 100).plot(figsize=(15, 10));


# # Portfolio Returns
# The first step in the calculation of a portfolio return is the calculation of the annualized returns of the different stocks based on the log returns for the respective time series.

# In[5]:

# vectorized calculation of the log returns
log_rets = np.log(data / data.shift(1))


# In[6]:

# annualized average log returns
rets = log_rets.mean() * 252
rets


# # We now need to represent a portfolio by (normalized) weightings for the single stocks. Let us start with an equal weighting scheme.

# In[7]:

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # equal weightings


# In mathematical terms, the portfolio return is given as
# 
# \begin{eqnarray*}
# \mu_p &=& \mathbf{E} \left( \sum_I w_i r_i \right) \\
#         &=& \sum_I w_i \mathbf{E}\left( r_i \right) \\
#         &=& \sum_I w_i \mu_i \\
#         &=& w^T \mu
# \end{eqnarray*}
# 
# where the $w_i$ are the weights for the single portfolio components and the $r_i$ are the respective returns.
# 
# We get for our specific case the following result.

# In[8]:

np.dot(weights, rets)  # portfolio return (equal weights)


# ## Portfolio Variance

# The calculation of the **portfolio variance** is a bit more involved. Let us start with the definition of the **covariance matrix** which is needed to this end.
# 
# \begin{eqnarray*}
# \Sigma = \begin{bmatrix}
#         \sigma_{1}^2 \ \sigma_{12} \ \dots \ \sigma_{1I} \\
#         \sigma_{21} \ \sigma_{2}^2 \ \dots \ \sigma_{2I} \\
#         \vdots \ \vdots \ \ddots \ \vdots \\
#         \sigma_{I1} \ \sigma_{I2} \ \dots \ \sigma_{I}^2
#     \end{bmatrix}
# \end{eqnarray*}
# 
# Here, we have the variances of the single stocks on the diagonal and the covariances between two stocks in the other places.

# In Python, this matrix is easily calculated.

# In[9]:

log_rets.cov() * 252  # annualized covariance matrix


# In[10]:

log_rets.corr()


# Being equipped with the covariance matrix, the **portfolio variance** is defined as follows.
# 
# \begin{eqnarray*}
# \sigma_p^2 &=& \mathbf{E}\left( (r - \mu)^2 \right) \\
#         &=& \sum_{i \in I}\sum_{j \in I} w_i w_j \sigma_{ij} \\
#         &=& w^T \Sigma w
# \end{eqnarray*}

# In Python, using NumPy, this is again a straightforward calculation.

# In[11]:

# portfolio variance
pvar = np.dot(weights.T, np.dot(log_rets.cov() * 252, weights))
pvar


# The **portfolio volatility** then is

# In[12]:

pvol = pvar ** 0.5
pvol


# ## Random Portfolio Compositions

# Next, let us generate a random portfolio composition and calculate the resulting portfolio return and variance.

# In[13]:

weights = np.random.random(5)  # random numbers
weights /= np.sum(weights)  # normalization to 1


# In[14]:

weights  # random portfolio composition


# In[15]:

np.dot(weights, rets)  # portfolio return (random weights)


# In[16]:

# portfolio variance (random weights)
np.dot(weights.T, np.dot(log_rets.cov() * 252, weights))


# We are now going to implement a **Monte Carlo simulation** for the portfolio weights and collect the resulting portfolio returns and volatilities.

# In[17]:

get_ipython().run_cell_magic(u'time', u'', u"prets = []\npvols = []\nfor p in xrange(10000):\n    weights = np.random.random(5)\n    weights /= np.sum(weights)\n    prets.append(np.sum(log_rets.mean() * weights) * 252)\n    pvols.append(np.sqrt(np.dot(weights.T, \n                        np.dot(log_rets.cov() * 252, weights))))\nprets = np.array(prets)\npvols = np.array(pvols)\nportfolio = pd.DataFrame({'return': prets, 'volatility': pvols})")


# The collected results allow for an **insightful visualization**. We can easily spot the area of the **minimum variance portfolio** and also see the **efficient frontier** quite well.

# In[18]:

portfolio.plot(x='volatility', y='return', kind='scatter', figsize=(15, 10));


# In[56]:

portfolio


# In[57]:

sharpe = (data[stock].returns() / np.std(data[stock].returns())) * np.sqrt(365.25)  


# In[19]:

(data / data.ix[0] * 100).plot(figsize=(15, 10));


# In[22]:

aapl = web.DataReader("AAPL", 'yahoo', '2010-1-1', '2010-12-31')


# In[25]:

aapl['Adj Close'].returns()


# In[ ]:



