'''
Nonlinear regression
'''

import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from init import *


# Load input data
ipdata = init()
del ipdata['Date']

X = ipdata.copy(deep=True)
del X['next']
y = ipdata['next']

# Split into training and testing set; 80:20 split
rows = round(0.8 * len(ipdata.index))
trainSet = X[:rows]
testSet = X[rows:]
trainDependant = y[:rows]
testDependant = y[rows:]

# Generate a model of polynomial features
poly = PolynomialFeatures(degree=4)

# Transform the x data for proper fitting
trainSet = poly.fit_transform(trainSet)

# Generate the regression object
model = linear_model.LinearRegression()
# Preform the actual regression
model.fit(trainSet, trainDependant)

# Transform test set
testSet_t = poly.fit_transform(testSet)

# Predict for test set
prediction = [float(x) for x in model.predict(testSet_t)]

print("Accuracy =", accuracy(list(prediction), list(testDependant)) * 100, "%")
plot_results(list(prediction), list(testDependant))