'''
Using Multiple Linear Regression to predict the next day's closing values of Bitcoin's market price in USD.
'''

import pandas, sklearn, numpy
from sklearn import linear_model
from init import *


ipdata = init()

def correlation(dataset_arg, threshold, toprint = False):
    '''
    Return a copy of 'dataset_arg' without the columns that have correlation > threshold
    '''
    dataset = dataset_arg.copy(deep = True)
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    if toprint:
        print(dataset)

    return dataset

# Drop columns with correlation > 0.75
# Drops 10 columns of 25
# Columns = ['Date', 'btc_market_price', 'btc_total_bitcoins', 'btc_trade_volume', 'btc_n_orphaned_blocks', 'btc_median_confirmation_time', 'btc_cost_per_transaction_percent', 'btc_cost_per_transaction', 'btc_output_volume', 'btc_estimated_transaction_volume']
data = correlation(ipdata, 0.75)

# Predictor variables
df = data.copy(deep = True)
del df['Date']
# Target
target = pandas.DataFrame(ipdata, columns = ["next"])

X = df
y = target['next']

lm = linear_model.LinearRegression()
model = lm.fit(X, y)
predictions = lm.predict(X)

print("Accuracy with Multiple Linear Regression:", accuracy(predictions, y) * 100, "%")