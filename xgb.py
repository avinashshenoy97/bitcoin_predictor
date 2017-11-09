import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from init import *

data = init()
ndata = data.drop('Date', 1)

nrow1 = int(ndata.shape[0]*0.8)     # 80% of the data for training 
nrow2 = int(ndata.shape[0]*0.2)     # 20% of the data for testing

train_data = ndata      
test_data = ndata
train_data = ndata.sample(n = nrow1)    # training data
test_data = ndata.sample(n = nrow2)     # testing data

train_data_m = train_data.drop('next', 1)   
train_data_m = train_data_m[0:train_data.shape[0]].as_matrix()  # reshape data to be used in xgb function
test_data_m = test_data.drop('next', 1)
test_data_m = test_data_m[0:test_data.shape[0]].as_matrix()

gbm = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate=0.2).fit(train_data_m, train_data['next']) # fit the model
predictions = gbm.predict(test_data_m)  # predict the values

print("Accuracy of xgboost :", accuracy(predictions, test_data['next']) * 100)  # calculate and print accuracy
