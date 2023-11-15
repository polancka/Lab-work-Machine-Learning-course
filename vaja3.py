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


def Ridge_function(x, y):
    '''
    Fit model using ridge and tries different regularization parameter for evaluating its effect
    '''
    n_alphas = 100
    alphas = np.logspace(-10, -2, n_alphas)

    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)

    ax = plt.gca()

    #print(coefs)
    ax.plot(alphas, coefs)
    #ax.set_xscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()









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


Ridge_function(x, y)