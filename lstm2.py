'''
Multiple hidden layer LSTM RNN
'''
from ainit import *
from init import *
import numpy, math
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

scaler = MinMaxScaler(feature_range=(0, 1))
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_data.append(scaler.fit_transform(window))
    return normalised_data

def denormalise_windows(window_data):
    denormalised_data = []
    for window in window_data:
        try:
            denormalised_data.append(scaler.inverse_transform(window[0])[0])
        except:
            denormalised_data.append(scaler.inverse_transform([[window]])[0])
    return denormalised_data

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title('LSTM Predictions')
    plt.show()


# Configuration
sequence_length = 50
layers = [1, 50, 100, 1]
epochs = 400


# Load the dataset
df = init()
data = pandas.DataFrame(df, columns = ['btc_market_price'])
data = data.values
data = data.astype('float32')

# Reshape input data for LSTM
sequence_length += 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
sequence_length -= 1

result = normalise_windows(result)
result = np.array(result)

# Split data in training and test set; 80:20 split
row = round(0.8 * result.shape[0])
train = result[:int(row), :]
np.random.shuffle(train)
trainX = train[:, :-1]
trainY = train[:, -1]
testX = result[int(row):, :-1]
testY = result[int(row):, -1]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))  

# Build RNN Model
model = Sequential()
model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layers[2],return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=layers[3]))
model.add(Activation("tanh"))
compile_start = dt.now()
model.compile(loss="mse", optimizer="rmsprop")
compile_end = dt.now()
print("Time to compile : ", (compile_end - compile_start))

# Train
train_start = dt.now()
model.fit(trainX, trainY, batch_size=512, nb_epoch=epochs, validation_split=0.05)
train_end = dt.now()
print("Time to train : ", (train_end - train_start))

# Test
predicted = predict_point_by_point(model, testX)

s = denormalise_windows(predicted)

r = [x[0] for x in testY]
r = denormalise_windows(r)

news = [x[0] for x in s]
newr = [x[0] for x in r]

# Plot prediction vs original prices
print("Accuracy stats of LSTM :")
accuracyStats(news, newr)
plot_results(news, newr)

# Plot errors
errors = [math.fabs(x-y) for x,y in zip(news, newr)]
print("Average error : ", np.average(errors))
plt.plot(errors, label='Error')
plt.title('Days Ahead Vs. Error')
plt.legend()
plt.show()


'''
Our results :

Time to compile :  0:00:00.028948
Time to train :  0:03:31.755918
Accuracy stats of LSTM :
Accuracy with a margin of 100$ :  1.0
Accuracy with a margin of 50$ :  1.0
Accuracy with a margin of 25$ :  0.985781990521327
Accuracy with a margin of 10$ :  0.8767772511848341
Average error :  4.88935209109
'''