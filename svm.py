from sklearn import svm
import matplotlib.pyplot as plt
from init import *


data = init()
ndata = data.drop('Date', 1)

nrow1 = int(ndata.shape[0]*0.8)     # 80% of the data for training 
nrow2 = int(ndata.shape[0]*0.2)     # 20% of the data for testing

train_data = ndata      
test_data = ndata
train_data = ndata.sample(n = nrow1)    # training data
test_data = ndata.sample(n = nrow2)     # testing data

model1 = svm.SVR(kernel='rbf', C = 1e4, gamma = 0.3)     # radial basis function kernel
model1 = model1.fit(train_data.drop('next', 1), train_data['next'])   # fit the model
res1 = model1.predict(test_data.drop('next', 1))

#model2 = svm.SVR(kernel='poly', C = 1e3, degree = 2)     # polynomial function kernel
#model2 = model2.fit(train_data.drop('next', 1), train_data['next'])   # fit the model
#res2 = model2.predict(test_data.drop('next', 1))

print('Accuracy stats of SVM with radial kernel : ')
accuracyStats(res1, test_data['next'])
plot_results(res1, test_data['next'], 'SVM with radial kernel', 'Day', 'Price (in USD)')

errors = [math.fabs(x-y) for x,y in zip(res1, test_data['next'])]
print("Average error : ", np.average(errors))
plt.plot(errors)
plt.title('SVM (radial kernel) Errors')
plt.xlabel('Day')
plt.ylabel('Price (in USD)')
plt.show()

# As the data is not linear, no convergence is obtained in case of a linear kernel

# Radial basis kernel is the best choice for Regression with SVMs