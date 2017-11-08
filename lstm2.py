'''
Multiple hidden layer LSTM RNN
'''

from init import *
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler


def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[numpy.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[numpy.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

scaler = MinMaxScaler(feature_range=(0, 1))
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        #normalised_window = [((float(p) / (float(window[0]) + 1)) - 1) for p in window]
        #normalised_data.append(normalised_window)
        normalised_data.append(scaler.fit_transform(window))
    return normalised_data

def denormalise_windows(window_data):
    denormalised_data = []
    for window in window_data:
        #denormalised_window = [((float(p)+1) * (float(window[0]) + 1)) for p in window]
        #denormalised_data.append(denormalised_window)
        try:
            denormalised_data.append(scaler.inverse_transform(window[0])[0])
        except:
            denormalised_data.append(scaler.inverse_transform([[window]])[0])
    return denormalised_data

sequence_length = 50

# load the dataset
df = init()
data = pandas.DataFrame(df, columns = ['btc_market_price'])
data = data.values
data = data.astype('float32')

sequence_length += 1
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
sequence_length -= 1

result = normalise_windows(result)
result = np.array(result)

row = round(0.9 * result.shape[0])
train = result[:int(row), :]
np.random.shuffle(train)
trainX = train[:, :-1]
trainY = train[:, -1]
testX = result[int(row):, :-1]
testY = result[int(row):, -1]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))  

epochs = 200

model = Sequential()
layers = [1, 50, 100, 1]

model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layers[2],return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=layers[3]))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="rmsprop")

model.fit(trainX, trainY, batch_size=512, nb_epoch=epochs, validation_split=0.05)

predictions = predict_sequences_multiple(model, testX, sequence_length, 50)
#predicted = predict_sequence_full(model, testX, sequence_length)
#predicted = predict_point_by_point(model, testX)

#print(accuracy(predictions, testX))
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

plot_results_multiple(predictions, testY, 50)

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

#plot_results(predicted, testY)