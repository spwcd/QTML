import sys
sys.path.append('/Users/junenyjune/anaconda/lib/python3.5/site-packages')

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings; warnings.simplefilter('ignore')
from datetime import datetime
from random import randrange
from array import array
import csv

#upward
temp = 500
close = pd.Series(np.zeros(1000))
for i in range(0, 1000):
    temp = temp + ((temp/100) * (randrange(0,35)/10) ) + ((temp/100) * (randrange(-27,0)/10) )
    close[i] = temp

close.plot()
prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)
#prob.sort(reverse=True)
high = close + (close/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close + (close/100)*(0-(np.random.choice(np.arange(0,30), p=prob)/10))
open = close.shift(1)
open[0] = close[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close})

data.to_csv("up.csv")


#downward
temp = 500
close = pd.Series(np.zeros(1000))
for i in range(0, 1000):
    temp = temp + ((temp/100) * (randrange(0,27)/10) ) + ((temp/100) * (randrange(-30,0)/10) )
    close[i] = temp

close.plot()

prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)
#prob.sort(reverse=True)
high = close + (close/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close + (close/100)*(0-(np.random.choice(np.arange(0,30), p=prob)/10))
open = close.shift(1)
open[0] = close[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close})

data.to_csv("down.csv")



x = np.linspace(- np.pi, np.pi, 2000)
y = np.sin(x)

close1 = y[0:1000]

close1 = pd.Series(close1+5)
xxx = pd.Series(close1*0)
for i in range(0, 500):
    xxx[i] = ((close1[i]/100) * (randrange(0,27)/100) ) + ((close1[i]/100) * (randrange(-30,0)/100) )

for i in range(500, 1000):
    xxx[i] = ((close1[i]/100) * (randrange(0,35)/100) ) + ((close1[i]/100) * (randrange(-27,0)/100) )

close1 = close1 +xxx
plt.plot(close1)

prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)

prob1 = np.random.random(25)
prob1 /= np.sum(prob1)
prob1 = sorted(prob1,reverse=True)
#prob.sort(reverse=True)
high = close1 + (close1/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close1 + (close1/100)*(0-(np.random.choice(np.arange(0,25), p=prob1)/10))
open = close1.shift(1)
open = close1.shift(1)
open[0] = close1[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close1})
data['Close'].plot()
data['Low'].plot()
data['High'].plot()

data.to_csv("reBell.csv")



close2 = y[1000:]
close2 = pd.Series(close2+5)
xxx = pd.Series(close2*0)
for i in range(0, 500):
    xxx[i] = ((close2[i] / 100) * (randrange(0, 35) / 200)) + ((close2[i] / 100) * (randrange(-27, 0) / 200))

for i in range(500, 1000):

    xxx[i] = ((close2[i] / 100) * (randrange(0, 27) / 200)) + ((close2[i] / 100) * (randrange(-30, 0) / 200))

close2 = close2 +xxx
plt.plot(close2)

prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)

prob1 = np.random.random(20)
prob1 /= np.sum(prob1)
prob1 = sorted(prob1,reverse=True)
#prob.sort(reverse=True)
high = close2 + (close2/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close2 + (close2/100)*(0-(np.random.choice(np.arange(0,20), p=prob1)/10))
open = close2.shift(1)
open = close2.shift(1)
open[0] = close2[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close2})
data['Close'].plot()
data['Low'].plot()
data['High'].plot()

data.to_csv("Bell.csv")



# **** mix up and down ***** #
# you have to delete data in the middle (row500) 

#upward
temp = 500
close = pd.Series(np.zeros(1000))
for i in range(0, 500):
    temp = temp + ((temp/100) * (randrange(0,35)/10) ) + ((temp/100) * (randrange(-27,0)/10) )
    close[i] = temp
    
for i in range(501, 1000):
    temp = temp + ((temp/100) * (randrange(0,27)/10) ) + ((temp/100) * (randrange(-30,0)/10) )
    close[i] = temp
    
close.plot()
prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)
#prob.sort(reverse=True)
high = close + (close/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close + (close/100)*(0-(np.random.choice(np.arange(0,30), p=prob)/10))
open = close.shift(1)
open[0] = close[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close})

data.to_csv("updown.csv")   


#downward
temp = 500
close = pd.Series(np.zeros(1000))
for i in range(0, 1000):
    temp = temp + ((temp/100) * (randrange(0,27)/10) ) + ((temp/100) * (randrange(-30,0)/10) )
    close[i] = temp

close.plot()

prob = np.random.random(30)
prob /= np.sum(prob)
prob = sorted(prob,reverse=True)
#prob.sort(reverse=True)
high = close + (close/100)*np.random.choice(np.arange(0,30), p=prob)/10
low = close + (close/100)*(0-(np.random.choice(np.arange(0,30), p=prob)/10))
open = close.shift(1)
open[0] = close[0]
data = pd.DataFrame({'Open': open, 'High': high, 'Low': low, 'Close': close})

data.to_csv("down.csv")















'''
#poly
x = np.linspace(- np.pi, np.pi, 2000)
y = np.sin(x)
close = y +10
temp = close[0]
close = pd.Series(close)
xxx = pd.Series(close*0)

for i in range(0, 1000):
    xxx[i] = ((close[i] / 100) * (randrange(-300, 300) / 1000))
    print(xxx[i])

close = close + xxx
plt.plot(close)

'''