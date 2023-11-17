#lab 3 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random as rnd
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from numpy import arange
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

##with ridge we get a lot less variance for small amount of added bias
def Ridge_function(trainX, testX, trainY, testY):
    '''
    Fit model using ridge and tries different regularization parameter for evaluating its effect
    '''
    n_alphas = 500
    alphas = np.linspace(0, 10, n_alphas)
    max_r = 0
    max_alpha = 0

    scores = list()
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(trainX, trainY)
        curr_r_score = r2_score(testY, ridge.predict(testX))
        scores.append(curr_r_score)
        if curr_r_score > max_r:
            max_r = curr_r_score
            max_alpha = a

    #print(max_alpha) # 2.224448897795591
    plt.plot(alphas, scores)
    plt.xlabel("Alpha value")
    plt.ylabel("R^2 score")
    plt.title("Ridge regression")
    plt.axvline(x = max_alpha, color = 'red', linestyle = '--') # 2.224448897795591
    plt.show()

def Lasso_function(trainX, testX, trainY, testY):
    '''
    Fit model using ridge and tries different regularization parameter for evaluating its effect
    '''
    n_alphas = 500
    alphas = np.linspace(0.01, 10, n_alphas)
    lasso = Lasso(max_iter=1000)
    max_r = 0
    max_alpha = 0
    scores = []
    rs = list()

    for a in alphas:
        lasso.set_params(alpha = a)
        lasso.fit(trainX, trainY)
        print(lasso.get_params())
        curr_r_score = mean_squared_error(testY, lasso.predict(testX))
        rs.append(curr_r_score)
        if curr_r_score > max_r:
             max_r = curr_r_score
             max_alpha = a
        scores.append(lasso.coef_)

    print(max_alpha)
    # plt.plot(alphas, scores)
    # plt.xlabel("Alpha value")
    # plt.ylabel("R^2 score")
    # plt.title("Lasso regression")
    # plt.axvline(x = max_alpha, color = 'red', linestyle = '--') #max alpha = 10
    # plt.show()

    # ax = plt.gca()
    # ax.plot(alphas, scores)
    # ax.set_xscale('log')
    # plt.axis('tight')
    # plt.xlabel('alpha')
    # plt.ylabel('Standardized Coefficients')
    # plt.title('Lasso coefficients as a function of alpha')
    # plt.show()

def my_Ridge_DG_fit(X, Y, learning_rate, iterations, penality):
     # no_of_training_examples, no_of_features         
        m, n = X.shape 
          
        # weight initialization         
        W = np.zeros(n, dtype='int64').reshape(-1,1)
        b = 0        

        # gradient descent learning 
                  
        for i in range( iterations ) :             
            #update the weights
            Y_pred = my_Ridge_DG_predict(X, W, b) 
            #print(Y_pred)
            
          
        # calculate gradients       
            dW = ( - ( 2 * (X.T).dot(Y - Y_pred ) ) + ( 2 * penality * W ) ) / m     
            #print(dW) 
            db = - 2 * np.sum(Y - Y_pred ) / m  
            
            # update weights     
            W = W - learning_rate * dW     
            b = b - learning_rate * db                   
        return W,b


def my_Ridge_DG_predict(X, W, b):
    return X.dot(W) + b 




#Download the "Communities and Crime" dataset and prepare the data so that you will be able to use them for linear regression.

# Reading the CSV file
data = pd.read_csv('communities.data', na_values='?')

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

trainX, testX, trainY, testY = train_test_split(x, y, test_size= 0.3, random_state= 42)

## Fit models using ridge and lasso regression.
## Try different values of the regularization parameter and evaluate its effect. 

#Ridge_function(trainX, testX, trainY, testY)
#Lasso_function(trainX, testX, trainY, testY)

# changing the alpha changes the slope of the predicted regression line. 
# Alpha is optimal alpha is around 2,22 with ridge and 0.05 with lasso. 


## Compare the results of the feature selection you implemented in the previous assignment
## with the attributes lasso selected. - ##features are similar


# Download the "Wine quality" dataset. Choose only the white wine data. Prepare
# your data for modeling.

dataWine = pd.read_csv('winequality-white.csv', na_values='?', sep=';')
#print(dataWine.info) - no null values, no qualitative attributes

X = dataWine.drop(columns= "quality").values
Y = dataWine[['quality']].values


X_train_w, X_test_w, Y_train_w, Y_test_w = train_test_split( X, Y, test_size = 0.3, random_state = 42) 

# Implement ridge regression with:
# • gradient descent

iterations = 100
learning_rate = 0.01
penalty = 1
mses = list()

for i in range(iterations):
    W,b = my_Ridge_DG_fit(X_train_w, Y_train_w, learning_rate, iterations, penalty)
    ridgeDG_predictions = my_Ridge_DG_predict(X_test_w,W,b)
    

# print( "Predicted values ", np.round( ridgeDG_predictions[:3], 2 ) )      
# print( "Real values      ", ridgeDG_predictions[:3] )
# print( "Trained W        ",  round(W[0][0],2))     
# print( "Trained b        ", round(b,2) )   

# Trained W         -1.9193278495634436e+262
# Trained b         -2.788072701411143e+261
    
# Visualization on test set      
# plt.scatter( X_test_w[:,0], Y_test_w, color = 'blue' )     
# plt.plot( X_test_w, ridgeDG_predictions, color = 'orange' )     
# plt.show() 


