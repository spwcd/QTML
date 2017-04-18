from loadData import *
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def linreg(X,Y):
    # Running the linear regression
    X = sm.add_constant(X)
    model = regression.linear_model.OLS(Y, X).fit()
    a = model.params[0]
    b = model.params[1]
    X = X[:, 1]
    colors = ("red", "green", "blue")
    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    plt.scatter(X, Y, c=colors) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=5);  # Add the regression line, colored in red
    plt.xlabel('STA Value')
    plt.ylabel('RS Value')
    return model.summary()

stock = getStock('PTT')
benchmark = getStock('SET')
r_y = stock['2016':'2017']['Close'].pct_change()[1:]
r_x = benchmark['2016':'2017']['Close'].pct_change()[1:]


linreg(r_x.values, r_y.values)

from matplotlib.finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as datetime
import numpy as np

quotes = stock['2016':'2017']
fig, ax = plt.subplots()
candlestick2_ohlc(ax,quotes['Open'],quotes['High'],quotes['Low'],quotes['Close'],width=0.6)
plt.show()

