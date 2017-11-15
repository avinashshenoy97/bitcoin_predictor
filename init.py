'''
Loading input, preprocessing and defining functions to be used for all models.
'''

import math, pandas, sklearn, numpy as np
from datetime import datetime as dt
from dateutil.parser import parse
import matplotlib.pyplot as plt

def init():
    # Load input data
    ipdata = pandas.read_csv("data/bitcoin_dataset.csv",  parse_dates=['Date'])

    # Drop rows with NaN
    for key in ipdata:
        try:
            ipdata = ipdata[np.isfinite(ipdata[key])]
        except:
            pass

    #ipdata['diff'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    #ipdata['dir'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    ipdata['next'] = pandas.Series([0] * len(ipdata['btc_market_price']), index = ipdata.index)
    ipdata = ipdata.drop('btc_trade_volume', 1)
    # Convert Date field to Python Date type
    #ipdata['date'] = pandas.Series([dt(1,1,1,1,1,1)] * len(ipdata['btc_market_price']), index = ipdata.index)
    #for ind in ipdata.index:
    #    ipdata.ix[ind, 'Date'] = parse(ipdata['Date'][ind])

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

def accuracy(predicted, actual, margin=100):
    '''
    predicted:  iterable
    actual:     iterable
    margin:     (optional) integer

    Compares values in "predicted" with corresponding values in "actual" and if their difference is less than "margin", considers it to be correct.

    Returns:    Percentage of correct predictions

    Note:   "predicted" and "actual" must be of the same length.
    '''

    if len(predicted) != len(actual):
        raise ValueError('"predicted list" and "actual" list are of unequal lengths!')

    total = len(predicted)
    correct = 0
    for p, a in zip(predicted, actual):
        if math.fabs(p - a) < margin:
            correct += 1

    return (correct/total)

def plot_results(predicted_data, true_data, title='', xlab='', ylab=''):
    '''
    Plot predicted vs. true data using matplotlib
    '''
    plt.title(title)
    plt.plot(range(len(predicted_data)), predicted_data, label='Prediction')
    plt.plot(range(len(true_data)), true_data, label='True Data')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    return

def newPlot():
    plt.figure(plt.gcf().number+1)

def accuracyStats(l1, l2, *args):
    levels = set([100, 50, 25, 10])
    for l in args:
        levels.add(l)
    
    levels = list(levels)
    levels.sort(reverse=True)
    for l in levels:
        print("Accuracy with a margin of", str(l) + "$ : ", accuracy(l1, l2, l))