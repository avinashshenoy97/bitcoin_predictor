Bitcoin Price Prediction
========================

In this repository, we've attempted to build a bitcoin price change predictor that's supposed to predict the closing price of a Bitcoin in USD one day ahead of time.

## Models

Our primary performance metric for evaluation of the models was to check how close the predicted price was to the actual price the next day. If the predicted price was within 25$ of the actual price, we considered the prediction to be accurate. We've written scripts to use several different analytic models :

| Script              | Model                            |
|:-------------------:|:--------------------------------:|
| mlr.py              | Multiple Linear Regression       |
| nonlinreg.py        | Non-Linear Regression            |
| svm.py              | SVM For Regression               |
| rtree.py            | Regression Tree                  |
| xgb.py              | XGBoost                          |
| arima.py            | ARIMA Model                      |
| lstm.py             | LSTM with Single Hidden Layer    |
| lstm2.py            | LSTM with Multiple Hidden Layers |

## LSTM

Ultimately, the LSTM with multiple hidden layers gave best results with an accuracy of about 90% considering our primary performance metric. Here's a graph depicting the predictions.

![LSTM Predictions Graph](https://github.com/avinashshenoy97/test/Resources/LSTM_Multiple_Predictions.png "LSTM Predictions")



## Data

The dataset used for this project was obtained from [Kaggle](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory) on 28th September 2017. More data has been added since then. The data we used to generate predictions and plot the graph above is present in the top-level folder `Resources` of this repository.

## Contributors 

<img src="https://github.com/aditisrinivas97.png" width="48">  [Aditi Srinivas](https://github.com/aditisrinivas97)

<img src="https://github.com/avinashshenoy97.png" width="48">  [Avinash Shenoy](https://github.com/avinashshenoy97)

### Note
This repository was created for a project that was done as a part of an undergraduate course on Data Analytics.