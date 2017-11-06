import pandas, sklearn, numpy
from datetime import datetime as dt
from dateutil.parser import parse

def init():
    # Load input data
    ipdata = pandas.read_csv("data/bitcoin_dataset.csv")
    #ipdata['diff'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    #ipdata['dir'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    ipdata['next'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)

    # Convert Date field to Python Date type
    #ipdata['date'] = pandas.Series([dt(1,1,1,1,1,1)] * len(ipdata['btc_market_price']), index = ipdata.index)
    for ind in range(len(ipdata['Date'])):
        ipdata.ix[ind, 'Date'] = parse(ipdata['Date'][ind])

    # Add next day's price
    for ind in range(len(ipdata['btc_market_price']) - 1):
        ipdata.ix[ind, 'next'] = ipdata['btc_market_price'][ind+1]
        '''
        # Legacy code
        # Get difference in market price between every row and the next
        # If difference is positive, price decreased so 0; else 1
        temp = ipdata['btc_market_price'][ind] - ipdata['btc_market_price'][ind+1]
        ipdata.ix[ind, 'diff'] = temp

        if temp > 0:
            ipdata.ix[ind, 'dir'] = False
        else:
            ipdata.ix[ind, 'dir'] = True
        '''

    return ipdata.drop([len(ipdata['Date']) - 1])