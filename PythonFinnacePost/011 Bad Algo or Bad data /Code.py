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
    colors = ("blue")
    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    plt.scatter(X, Y, c=colors) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=5);  # Add the regression line, colored in red
    plt.xlabel('Indicator Value')
    plt.ylabel('PTT percent change')
    return model.summary()

stock = getStock('PTT')



import talib

ADX = talib.ADX(high=stock['High'].__array__(), low=stock['Low'].__array__(), close=stock['Close'].__array__())
adx = ADX[-502:]/ADX[-502]-1
adx = adx[1:501]
r_y = stock[-500:]['Close'].pct_change()[0:]
r_x = adx


linreg(r_x[1:], r_y.values[1:])
