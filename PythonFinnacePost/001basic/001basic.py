import sys
sys.path.append("/home/poon/anaconda3/pkgs/quandl-3.0.1-py35_0/lib/python3.5/site-packages")
import quandl as ql
import talib as tl
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

end = datetime.now()
start = datetime(end.year - 3,end.month,end.day)
STA = ql.get('GOOG/BKK_STA',start_date=start_date, end_date = end_date)

STA.info()

STA.Close.plot(figsize=(15, 10))

STA['SMA20'] = STA['Close'].rolling(window=20).mean()
STA['SMA40'] = STA['Close'].rolling(window=40).mean()
STA.dropna(inplace=True)
STA.head()

STA[['Close', 'SMA20', 'SMA40']].plot(figsize=(15, 10))

STA['position'] = np.where(STA['SMA20'] > STA['SMA40'], 1, -1)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(STA['Close'])
axarr[0].set_title('Sharing X axis')
axarr[1].plot(STA['position'])
plt.ylim(-1.1, 1.1)


STA['benchmark(buyandhold)'] = np.log(STA['Close'] / STA['Close'].shift(1))
STA['strategy'] = STA['position'].shift(1) * STA['benchmark(buyandhold)']

STA[['benchmark(buyandhold)', 'strategy']].cumsum().apply(np.exp).tail()

STA[['benchmark(buyandhold)', 'strategy']].cumsum().apply(np.exp).plot(figsize=(15, 10))
plt.grid(True)
