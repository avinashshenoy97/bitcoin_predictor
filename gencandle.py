from matplotlib.finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as datetime
import numpy as np
import pandas


rows = 100

ipdata = pandas.read_csv("data/bitcoin_price.csv",  parse_dates=['Date'], index_col = 'Date')
ipdata = ipdata[:(rows+1)]
ipdata = ipdata[::-1]

fig, ax = plt.subplots()
candlestick2_ohlc(ax,ipdata['Open'],ipdata['High'],ipdata['Low'],ipdata['Close'],width=0.6)

xdate = ipdata.index

ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

def mydate(x,pos):
    try:
        return xdate[int(x)]
    except IndexError:
        return ''

ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

fig.autofmt_xdate()
fig.tight_layout()

plt.show()