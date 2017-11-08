from sklearn import tree
from init import init, accuracy

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

acc = accuracy(res, test_data['next'])  # find the accuracy

print("Accuracy :",acc)