#KERNEL METHODS
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

## MY KERNEL REGRESSION
def kernel_regression(trainX, trainY, testX, testY): 
    linear_predictions = ()
    poly_predictions = ()
    radial_predictions = () 
    #implement three kernel functions 
        #linear kernel K(x, xi) = sum(x * xi)
        #polynomial kernel K(x,xi) = 1 + sum(x * xi)^d
        #radial basis kernel K(x,xi) = exp(-gamma * sum((x – xi^2))
    #Implement kernel matrix 
    #Compute α = K−1y to get the model coefficients in the multidimensional space.
    #Compute predictions for the test point z: y(z) = k∗α. Where k∗ is the kernel (vector) of the test points with the training points.
    return linear_predictions, poly_predictions, radial_predictions

#import and preprocess the breast-cancer data
data = pd.read_csv("bc.csv")

data['diagnosis'].replace(["B", "M"], #change the diagnosis to a numeric variable
                        [0, 1], inplace=True)

data = data.drop(columns= 'Unnamed: 32')

dataX = data.drop(columns= "diagnosis")
dataY = data["diagnosis"]
print("data imported")

#split into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3,random_state=109) # 70% training and 30% test
print("data split")
#generate the model 
#Create a svm Classifier
clf = SVC(kernel='linear') # Linear Kernel (try different kernels - linear, polynominal, radial basis)
print("Model made")
#Train the model using the training sets
clf.fit(X_train, y_train)
print("model fit")
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("predictions")
print(y_pred)



# TWEAK THE PARAMETERS : KERNEL, REGULARIZATION (C), GAMMA - WHICH VALUES??
# Kernel: The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
# Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
# Gamma: A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.

#Evaluate the model 
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


#import data for fitting (#2nd part) 
dataToFit = pd.read_csv("vaja6data.csv")
xToFit = dataToFit['x']
yToFit = dataToFit['y']

Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(xToFit, yToFit, test_size=0.3,random_state=109) # 70% training and 30% test
linear_pred, poly_pred, radial_pred = kernel_regression(Xtrain2, ytrain2, Xtest2, ytest2)




