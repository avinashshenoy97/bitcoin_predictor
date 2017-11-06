import pandas, sklearn, numpy as np
from datetime import datetime as dt
from dateutil.parser import parse

def init():
    # Load input data
    ipdata = pandas.read_csv("data/bitcoin_dataset.csv")

    # Drop rows with NaN
    for key in ipdata:
        try:
            ipdata = ipdata[np.isfinite(ipdata[key])]
        except:
            pass

    #ipdata['diff'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    #ipdata['dir'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    ipdata['next'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)

    # Convert Date field to Python Date type
    #ipdata['date'] = pandas.Series([dt(1,1,1,1,1,1)] * len(ipdata['btc_market_price']), index = ipdata.index)
    for ind in ipdata.index:
        ipdata.ix[ind, 'Date'] = parse(ipdata['Date'][ind])

    # Add next day's price
    for ind in ipdata.index:
        try:
            ipdata.ix[ind, 'next'] = ipdata['btc_market_price'][ind+1]
        except:
            if ind == max(ipdata.index):
                pass
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

    return ipdata.drop([max(ipdata.index)])