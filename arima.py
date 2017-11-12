from sklearn import svm
from ainit import *
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


ipdata = init()
ts = pandas.DataFrame(ipdata[['Date', 'next']])
ts.set_index(keys = 'Date', drop = True, inplace = True)
ts = ts['next']

# Differencing
ts_shift_diff = ts - ts.shift()
ts_shift_diff.dropna(inplace=True)

# Auto Correlation
acf = acf(ts_shift_diff, nlags=20)
# Partial Auto Correlation
pacf = pacf(ts_shift_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_shift_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_shift_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_shift_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_shift_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

# ARIMA Model
data = ainit()

data = data[['Date', 'next']]
data.set_index('Date', inplace = True)

model = ARIMA(data, order = (0,1,0))
model_fit = model.fit(disp = 0)

# plot residual errors
residuals = pandas.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind = 'kde')
plt.show()

ndata = data.values
size = int(len(ndata) * 0.80)
train, test = ndata[0:size], ndata[size:len(ndata)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order = (0,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

plot_results(predictions, test)     # Plot the results
acc = accuracy(predictions, test)   # calculate accuracy

print("Accuracy for ARIMA : ", acc)