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

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)


#Download the dataset Auto MPG and remove the attribute cylinders and carname.
data = pd.read_csv('auto-mpg.csv').drop(columns= 'car name').drop(columns= 'cylinders')

# pay attention to missing values (horsepower!) and categorical attributes 

# Data Sampling (shuffling the DataFrame)
data = data.sample(frac=1, random_state=0)

#remove the question marks in horsepower
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce', downcast='integer')

# Removing Null Values
data = data.dropna()

#change the categorical value to dummie value
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


#Test your linear regression on the Auto dataset and compare it with the liear regression implemented in SciKit-learn.
ofc_reg = LinearRegression().fit(train_x, train_y)
model = myLinearRegression()
betas = model.my_linear_regression(train_x, train_y)

# print(ofc_reg.intercept_)
# print(ofc_reg.coef_)
# print(betas)


#comparing on r2 score
scikit_score = r2_score(test_y, ofc_reg.predict(test_x))
my_score = r2_score(test_y, model.my_predict(test_x, betas))

#print(scikit_score)
#print(my_score)


#Diagnose your linear regression with the diagnostic plots.


# Residuals vs. Fitted Values Plot:This plot displays the residuals (the differences between observed and predicted values) on the vertical axis and the fitted 
# values (the predicted values) on the horizontal axis. It helps you check for linearity, homoscedasticity, and outliers.

#get fitted values and residuals
fitted_values = model.my_predict(test_x,betas)
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

##COOKS DISTANCE

##correlation between attributes
correlation_matrix = data.corr()

# Creating a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')
plt.show()


## 
attrs = ['displacement', 'horsepower', 'weight', 'acceleration', 'model year']
fig, ax = plt.subplots(5,5)
for i, attr1 in enumerate(attrs):
    for j, attr2 in enumerate(attrs):
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].scatter(data[attr1], data[attr2], s=0.2)

plt.show()
#Is there something you can improve like transform your target variable or remove
#some attributes/instances? Try different things and refit the linear regression.


