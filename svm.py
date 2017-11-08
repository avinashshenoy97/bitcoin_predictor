from sklearn import svm
from init import init, accuracy

data = init()
ndata = data.drop('Date', 1)

nrow1 = int(ndata.shape[0]*0.8)     # 80% of the data for training 
nrow2 = int(ndata.shape[0]*0.2)     # 20% of the data for testing

train_data = ndata      
test_data = ndata
train_data = ndata.sample(n = nrow1)    # training data
test_data = ndata.sample(n = nrow2)     # testing data

model1 = svm.SVR(kernel='rbf', C = 1e4, gamma = 0.3)     # radial basis function kernel, dosent work well with linear kernel
model1 = model1.fit(train_data.drop('next', 1), train_data['next'])   # fit the model
res1 = model1.predict(test_data.drop('next', 1))

model2 = svm.SVR(kernel='poly', C = 1e4, degree = 2)     # radial basis function kernel, dosent work well with linear kernel
model2 = model2.fit(train_data.drop('next', 1), train_data['next'])   # fit the model
res2 = model2.predict(test_data.drop('next', 1))

acc1 = accuracy(res1, test_data['next'])  # find the accuracy for radial kernel
acc2 = accuracy(res2, test_data['next'])  # find the accuracy for polynomial kernel

print("Accuracy for radial basis function kernel :",acc1 * 100) 
print("Accuracy for linear function kernel :",acc2 * 100)

# As the data is not linear, no convergence is obtained in case of a linear kernel

# Radial basis kernel is the best choice for Regression with SVMs