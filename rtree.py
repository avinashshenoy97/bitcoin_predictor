'''
Regression Tree
'''

from sklearn import tree
import numpy as np
from init import *

data = init()
ndata = data.drop('Date', 1)    # Drop the date column

nrow1 = int(ndata.shape[0]*0.8)     # 80% of the data for training 
nrow2 = int(ndata.shape[0]*0.2)     # 20% of the data for testing

train_data = ndata      
test_data = ndata
train_data = ndata.sample(n = nrow1)    # training data
test_data = ndata.sample(n = nrow2)     # testing data

model = tree.DecisionTreeRegressor()    
model = model.fit(train_data.drop('next', 1), train_data['next'])   # fit the model
res = model.predict(test_data.drop('next', 1))  

print('Accuracy stats of Regression Tree Model : ')
accuracyStats(res, test_data['next'])
plot_results(res, test_data['next'], 'Regression Tree Predictions')

errors = [math.fabs(x-y) for x,y in zip(res, test_data['next'])]
print("Average error : ", np.average(errors))
plt.plot(errors)
plt.title('Regression Tree Errors')
plt.show()


'''
Our Results :
Accuracy stats of Regression Tree Model :
Accuracy with a margin of 100$ :  0.9909502262443439
Accuracy with a margin of 50$ :  0.9909502262443439
Accuracy with a margin of 25$ :  0.9819004524886877
Accuracy with a margin of 10$ :  0.9592760180995475
Average error :  4.43136073454
'''