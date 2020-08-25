# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:15:38 2020

@author: Gaurav
"""

#Importing Libraries
import sys
import sklearn
import matplotlib
import seaborn
import pandas

print(sys.version)
print(pandas.__version__)
print(seaborn.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#Load the dataset
games = pandas.read_csv("C:\\Users\Gaurav\Desktop\Board\games.csv")

# Print name of the columns and the shapes
print(games.columns)

print(games.shape)

# Make the histogram for all the ratings in the average_rating
plt.hist(games["average_rating"])
plt.show()

# Print the all of the row games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

# Print the all of the row games with greater than zero scores
print(games[games["average_rating"] > 0].iloc[0])

# Remove the any rows without user review
games = games[games["users_rated"] > 0]

#Remove any rows with missing values
games = games.dropna(axis = 0)

# Make the histogram for all the average rating value
plt.hist(games["average_rating"])
plt.show()

# Corelation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# Get all the column from the DataFrame
columns = games.columns.tolist()

#Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on
target = "average_rating"

# Genrate traning data set
train = games.sample(frac = 0.8, random_state = 1)

# Select anything not in the traning set put in the set test
test = games.loc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)

# Import Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()

# Fit the model
LR.fit(train[columns], train[target])

# Generate prediction
predictions = LR.predict(test[columns])
print(predictions)

#compute error between prediction and actual value
mean_squared_error(predictions, test[target])

# Import the Random Forest model
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)
# Fit the data
RFR.fit(train[columns], train[target])

# Make prediction
predictions = RFR.predict(test[columns])
print(predictions)

#compute error between prediction and actual value
mean_squared_error(predictions, test[target])

# Make prediction both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

# Printing the predictions
print(rating_LR)
print(rating_RFR)





















































