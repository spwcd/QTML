import glob
import pandas as pd

path =r'/home/poon/Documents/set-archive_EOD_UPDATE' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

frame.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def getStock(name):
    stock = frame.loc[frame['Symbol'] == name]
    stock = stock.sort_values(by='Date')
    stock['Date'] = pd.to_datetime(stock['Date'], format='%Y%m%d')
    stock = stock.set_index(['Date'])
    return  stock


'''
benchmark = frame.loc[frame['Symbol'] == 'SET']
benchmark = benchmark.sort_values(by='Date')
benchmark['Date'] = pd.to_datetime(benchmark['Date'], format='%Y%m%d')
benchmark = benchmark.set_index(['Date'])
'''