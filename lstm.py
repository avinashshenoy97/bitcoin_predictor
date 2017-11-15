'''
Single hidden layer LSTM RNN
'''

from init import *
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(7)

# load the dataset
df = init()
dataset = pandas.DataFrame(df, columns = ['btc_market_price'])
dataset = dataset.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets, 80:20 split
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network

model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=400, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
testPredict = [x for x in testPredict]
print("Accuracy of Single Layer LSTM:")
accuracyStats(testPredict, testY[0])
plot_results(testPredict, testY[0], 'LSTM (Single) Predictions', 'Day', 'Price (in USD)')

errors = [math.fabs(x-y) for x,y in zip(testPredict, testY[0])]
print("Average error : ", numpy.average(errors))
plt.plot()
plt.title('LSTM (Single) Errors')
plt.xlabel('Day')
plt.ylabel('Price (in USD)')
plt.show()

'''
Our Results :

Accuracy of Single Layer LSTM:
Accuracy with a margin of 100$ :  0.9723502304147466
Accuracy with a margin of 50$ :  0.7603686635944701
Accuracy with a margin of 25$ :  0.4700460829493088
Accuracy with a margin of 10$ :  0.11981566820276497
Average error :  34.9432634626
'''