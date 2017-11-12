from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

from init import *


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


ipdata = init()
ts = pd.DataFrame(ipdata[['Date', 'next']])
ts.set_index(keys = 'Date', drop = True, inplace = True)
ts = ts['next']
test_stationarity(ts)

'''
Results of Dickey-Fuller Test:
Test Statistic                   -1.921634
p-value                           0.321953
#Lags Used                       22.000000
Number of Observations Used    1082.000000
Critical Value (1%)              -3.436408
Critical Value (5%)              -2.864215
Critical Value (10%)             -2.568194
dtype: float64
'''

# Cannot use log transformation because of 0s

#ts_log = np.log(ts)
#newPlot()
#plt.plot(ts_log, label='logclear')

ts_moving_avg_diff = ts - pd.rolling_mean(ts, 12)
ts_moving_avg_diff.dropna(inplace = True)
newPlot()
test_stationarity(ts_moving_avg_diff)

'''
Results of Dickey-Fuller Test:
Test Statistic                -5.968024e+00
p-value                        1.966452e-07
#Lags Used                     2.200000e+01
Number of Observations Used    1.071000e+03
Critical Value (1%)           -3.436470e+00
Critical Value (5%)           -2.864242e+00
Critical Value (10%)          -2.568209e+00
dtype: float64
'''

# Exponentially weighted moving average

expwighted_avg = pd.ewma(ts, halflife=12)
newPlot()
plt.plot(ts, color='blue', label='Original')
plt.plot(expwighted_avg, color='red', label='Exp')
plt.title('Exponentially weighted rolling stats')
plt.legend()
plt.show()

ts_ewma_diff = ts - expwighted_avg
newPlot()
test_stationarity(ts_ewma_diff)

# Differencing

ts_shift_diff = ts - ts.shift()
newPlot()
plt.title('Differencing')
plt.plot(ts_shift_diff)
plt.show()

newPlot()
ts_shift_diff.dropna(inplace=True)
test_stationarity(ts_shift_diff)

'''
Results of Dickey-Fuller Test:
Test Statistic                -7.121636e+00
p-value                        3.709455e-10
#Lags Used                     2.100000e+01
Number of Observations Used    1.082000e+03
Critical Value (1%)           -3.436408e+00
Critical Value (5%)           -2.864215e+00
Critical Value (10%)          -2.568194e+00
dtype: float64
'''

# Seperate seasonality from data
decomposition = seasonal_decompose(ts)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

ts_without_seasonal = residual
ts_without_seasonal.dropna(inplace=True)
newPlot()
test_stationarity(ts_without_seasonal)

'''
Results of Dickey-Fuller Test:
Test Statistic                -1.206144e+01
p-value                        2.464064e-22
#Lags Used                     2.100000e+01
Number of Observations Used    1.071000e+03
Critical Value (1%)           -3.436470e+00
Critical Value (5%)           -2.864242e+00
Critical Value (10%)          -2.568209e+00
dtype: float64
'''