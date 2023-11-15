#Lab 2
#resampling methods for model evaluation and attribute selection.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ipywidgets as widgets
import random as rnd
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

def leave_one_out(data):
    MSEs = 0
    for i in range(len(data)): #in range(len(data))
        #split into train and test, test is just one example, split into x and y

        train_x = np.array(data.drop(data.iloc[[i]].index).drop(columns= 'ViolentCrimesPerPop'))
        train_y = np.array(data[['ViolentCrimesPerPop']].drop(data.iloc[[i]].index))
        train_y = train_y.reshape(-1,1)

        test_x = np.array(data.drop(columns= 'ViolentCrimesPerPop').iloc[i]).reshape(1,-1)
        test_y = np.array(data.iloc[i][['ViolentCrimesPerPop']])
        test_y = test_y.reshape(-1,1)

        #train the model
        curr_model = LinearRegression().fit(train_x,train_y)
        curr_fit = curr_model.predict(test_x) 

        #calculate the MSE (mean squared error) for test example and save it 
        curr_mse = math.pow(test_y[0] - curr_fit,2)
        MSEs += curr_mse

    leave_one_out_MSE = MSEs/len(data)
    return leave_one_out_MSE

def k_fold(data, k):
    iterations = len(data) // k
    overall_mse = 0
    for i in range(iterations):
        start_index = i*10
        end_index = start_index + 9
        #print(start_index, end_index)

        train_x = data.drop(columns= 'ViolentCrimesPerPop')
        train_x = np.array(train_x.drop(train_x.index[start_index:end_index+1]))
        train_y = data[['ViolentCrimesPerPop']]
        train_y = np.array(train_y.drop(train_y.index[start_index:end_index+1]))
    

        test_x = np.array(data.drop(columns= 'ViolentCrimesPerPop').iloc[start_index:end_index])
        test_y = np.array(data.iloc[start_index:end_index][['ViolentCrimesPerPop']])
        test_y = test_y.reshape(-1,1)

        fold_model = LinearRegression().fit(train_x,train_y)
        fold_predicts = fold_model.predict(test_x)
        fold_mse = mean_squared_error(test_y,fold_predicts)
        
        overall_mse += fold_mse
      
    #posebaj zadnjo iteracijo ker ni nujno, da je 10 še ostalo
    overall_mse = overall_mse / iterations
    return overall_mse
        


#Download the "Communities and Crime" dataset and prepare the data so that you will be able to use them for linear regression.

# Reading the CSV file
data = pd.read_csv('data/communities+and+crime/communities.data', na_values='?')

# Data Manipulation
data = data.drop(['state', 'county', 'community', 'communityname', 'fold'], axis=1)
data = data.sample(frac=1, random_state=3)

#delete the columns with a lot of null values and the remaining rows with a few null values
data = data.dropna(axis=1, thresh= 1500)
data = data.dropna(axis = 0)
#print(data.info)  [1891 rows x 101 columns]

#into x and y sets
x  = data.drop(columns= 'ViolentCrimesPerPop')
y = data[['ViolentCrimesPerPop']]

#Implement the cross-validation method and the leave-one-out method.
loo_MSE = leave_one_out(data)
#print(loo_MSE) #0.018644326733171977
k = 10
k_fold = k_fold(data,k)
#print(k_fold) #0.018313116621930674


#TODO: Implement forward attribute selection. Fit linear regression.

#TODO: Use the attribute selection method with the implemented cross-validation to select a reasonable set of attributes for your linear model.
#which metric and the criteria

#TODO: Test your model and report the results.

#TODO: Implement the bootstrap method and apply it to the train set to generate 1000 different train sets and train 1000 different linear models.

#TODO: Use the bootstrapped results to assess the confidence intervals of the results of the linear model. (only on the selected attributes)