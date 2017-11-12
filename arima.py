from sklearn import svm
from ainit import *
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

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