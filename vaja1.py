#Lab work 1: 
#Write the code to fit the multiple linear regression in Python.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from myLinearRegression import myLinearRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import math

#some call functions and plots are commented out because of time consuption

#implementatioin of the linear regression fitting and predicting: 

def my_linear_regression(X, Y):
        X = np.array(X)
        Y = np.array(Y)
        #add the column of ones to calculate the intercept as well
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate the coefficients using the normal equation
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return beta

def my_predict(X, beta):
        X = np.c_[np.ones(X.shape[0]), X]
        # Make predictions using the calculated coefficients
        return X.dot(beta)


##DATA PREPARATION

#Download the dataset Auto MPG and remove the attribute cylinders and carname.
data = pd.read_csv('data/auto-mpg.csv').drop(columns= 'car name').drop(columns= 'cylinders')

# Data Sampling (shuffling the DataFrame)
data = data.sample(frac=1, random_state=0)

#remove the question marks in horsepower
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce', downcast='integer')

# Removing Null Values
data = data.dropna()

#change the categorical value to dummie value --> this provided with worse results
#data = pd.get_dummies(data, columns=['origin'])

#Split your dataset on train and test data.
 # - first try splitting half half
 # - then k-fold cross validation

split_index = len(data) // 2

train = data.iloc[:split_index]
test = data.iloc[split_index:]

#split into variables and target

x  = data.drop(columns= 'mpg')
y = data[['mpg']]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size= 0.3, random_state= 42)

##LINEAR REGRESSION

#Test your linear regression on the Auto dataset and compare it with the liear regression implemented in SciKit-learn.
ofc_reg = LinearRegression().fit(train_x, train_y)
betas = my_linear_regression(train_x, train_y)


# print(ofc_reg.intercept_)
# print(ofc_reg.coef_)
# print(betas)

#comparing on r2 score (they both yield the same svore)
scikit_score = r2_score(test_y, ofc_reg.predict(test_x))
my_score = r2_score(test_y, my_predict(test_x, betas))

#print(scikit_score)
#print(my_score)


#Diagnose your linear regression with the diagnostic plots.

# Residuals vs. Fitted Values Plot:This plot displays the residuals (the differences between observed and predicted values) on the vertical axis and the fitted 
# values (the predicted values) on the horizontal axis. It helps you check for linearity, homoscedasticity, and outliers.

#get fitted values and residuals
fitted_values = my_predict(test_x,betas)
residuals = []
print(test_y.iloc[0][0])


for i in range(len(fitted_values)):
    print(test_y.iloc[i][0] - fitted_values[i])
    residuals +=  [test_y.iloc[i][0] - fitted_values[i]]

print(residuals)


#print(fitted_values)
#print(residuals)

##FITED VS. RESIDUALS
print(len(fitted_values))
print(len(residuals))
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values Plot')
plt.show()

##QQ PLOT

sm.qqplot(np.asarray(residuals), line='45')
plt.title('Normal Q-Q Plot')
plt.show()

##COOKS DISTANCE - checks for outliers with big leverage. 

##correlation between attributes - there are some correlations, we do not like that
correlation_matrix = data.corr()

# Creating a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')
plt.show()



#Is there something you can improve like transform your target variable or remove
#some attributes/instances? 

# Origin variable could be hotcoded (tried with it, results were worse)


