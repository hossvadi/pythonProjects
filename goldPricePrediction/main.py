# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:36:42 2022

@author: Hossein.JvdZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading to a Pandas DataFrame
gold_data = pd.read_csv('dataset.csv')

#show 5 first rows
#gold_data.head()

#show 5 last rows
#gold_data.tail()

#missing values
#gold_data.isnull().sum()

# getting statistical measure of data
#gold_data.describe()

#Possitive or Negative Correlation
correlation = gold_data.corr()
# construct a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, cmap='Blues')

# Correlation of GLD
#print(correlation['GLD'])

#check the distribution og gold price
sns.histplot(gold_data['GLD'], color='green')
#old implementation
# sns.distplot(gold_data['GLD'], color='green')


#splitting features and target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

#Splitting into Train and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=2)

#implement to the random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

#training the model
regressor.fit(X_train, Y_train)

#prediction on test data
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

#R square error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R square error : ", error_score)

#Compare the actual values and predicted values in a plot
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

