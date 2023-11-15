#Lab 2
#resampling methods for model evaluation and attribute selection.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ipywidgets as widgets
import random as rnd
import pandas as pd
import math

def leave_one_out(data):
    MSEs = 0
    print(data.head(10))
    for i in range(3):
        #split into train and test, test is just one example, split into x and y
        test_x = data.iloc[i].drop(columns= 'ViolentCrimesPerPop')
        test_y = data.iloc[i][['ViolentCrimesPerPop']]
        print(test_y[0])
        #TODO : fix something with the data arrays" reshape for 2D array, now is 1D array

        train_x = data.drop(data.iloc[[i]].index).drop(columns= 'ViolentCrimesPerPop')
        train_y = data[['ViolentCrimesPerPop']].drop(data.iloc[[i]].index)

        curr_model = LinearRegression().fit(test_x,test_y)
        curr_fit = curr_model.predict(train_x) 

        #calculate the MSE (mean squared error) for test example and save it 
        curr_mse = math.pow(test_y[i] - curr_fit,2)
        MSEs += curr_mse
    

        
        
    leave_one_out_MSE = MSEs/len(data)
    return leave_one_out_MSE
        


#Download the "Communities and Crime" dataset and prepare the data so that you will be able to use them for linear regression.

# Reading the CSV file
data = pd.read_csv('communities+and+crime/communities.data', na_values='?')

# Data Manipulation
data = data.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1)
data = data.sample(frac=1, random_state=3)

# Displaying the count of null values in each column
#print(data.isnull().sum().to_numpy())

x  = data.drop(columns= 'ViolentCrimesPerPop')
y = data[['ViolentCrimesPerPop']]

#TODO: what about the null values??


#TODO: Implement the cross-validation method and the leave-one-out method.
loo_MSE = leave_one_out(data)

#TODO: Implement forward attribute selection. Fit linear regression.

#TODO: Use the attribute selection method with the implemented cross-validation to select a reasonable set of attributes for your linear model.
#which metric and the criteria

#TODO: Test your model and report the results.

#TODO: Implement the bootstrap method and apply it to the train set to generate 1000 different train sets and train 1000 different linear models.

#TODO: Use the bootstrapped results to assess the confidence intervals of the results of the linear model. (only on the selected attributes)