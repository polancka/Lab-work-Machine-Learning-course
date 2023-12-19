#KERNEL METHODS
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


#import and preprocess the breast-cancer data
data = pd.read_csv("bc.csv")

data['diagnosis'].replace(["B", "M"], #change the diagnosis to a numeric variable
                        [0, 1], inplace=True)

data = data.drop(columns= 'Unnamed: 32')

dataX = data.drop(columns= "diagnosis")
dataY = data["diagnosis"]

# standardize the data
scaler = preprocessing.StandardScaler().fit(dataX)
dataX_scaled = scaler.transform(dataX)


print("data imported")

#split into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(dataX_scaled, dataY, test_size=0.3,random_state=109) # 70% training and 30% test
print("data split")
#standardize the data


#generate the model 

#Create a svm Classifier
clf = SVC(kernel='linear', C=0.5) # Linear Kernel (try different kernels - linear, polynominal (set the degree), radial basis)
print("Model made")
#Train the model using the training sets
clf.fit(X_train, y_train)
print("model fit")
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("predictions")
print(y_pred)

#Evaluate the model 
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))



# # TWEAK THE PARAMETERS : KERNEL, REGULARIZATION (C), GAMMA - WHICH VALUES??
# # Kernel: The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
# # Regularization: Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
# # Gamma: A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000],'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'degree': [2,3,4,5,6,7,8,9,10], 'kernel': ['poly']},
 ]
grid_search = GridSearchCV(clf,param_grid)
grid_search.fit(X_train, y_train)

# print best parameter after tuning 
print(grid_search.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid_search.best_estimator_) 


#--------------------------------------------------------------2nd part -----------------------------------------------------------------------------------------#
#implement three kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (np.dot(x1, x2) + 3) ** degree

def radial_basis_kernel(x1, x2, sigma=1.0):
    return np.exp(-0.5 * ((x1 - x2) / sigma)**2)

    

#kernel matrix 
def kernel_matrix(X, kernel_func):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

## MY KERNEL REGRESSION
def kernel_regression(trainX, trainY, testX, reg_lambda=1e-4): 
    K_linear = kernel_matrix(trainX, linear_kernel)
    K_poly = kernel_matrix(trainX, polynomial_kernel)
    K_radial = kernel_matrix(trainX, radial_basis_kernel)

    alpha_linear = np.linalg.inv(K_linear + reg_lambda * np.identity(len(trainX))) @ trainY 
    alpha_poly = np.linalg.inv(K_poly  + reg_lambda * np.identity(len(trainX))) @ trainY
    alpha_radial = np.linalg.inv(K_radial  + reg_lambda * np.identity(len(trainX))) @ trainY

    linear_predictions = np.zeros(len(testX))
    for i in range(len(testX)):
        kernel_vec = np.zeros(len(trainX))
        for j in range(len(trainX)):
            kernel_vec[j] = linear_kernel(testX[i], trainX[j])
        linear_predictions[i] = np.dot(kernel_vec, alpha_linear)
    

    poly_predictions = np.zeros(len(testX))
    for i in range(len(testX)):
        kernel_vec = np.zeros(len(trainX))
        for j in range(len(trainX)):
            kernel_vec[j] = polynomial_kernel(testX[i], trainX[j])
        poly_predictions[i] = np.dot(kernel_vec, alpha_poly)
    

    radial_predictions = np.zeros(len(testX))
    for i in range(len(testX)):
        kernel_vec = np.zeros(len(trainX))
        for j in range(len(trainX)):
            kernel_vec[j] = radial_basis_kernel(testX[i], trainX[j])
        radial_predictions[i] = np.dot(kernel_vec, alpha_radial)
    
    return linear_predictions, poly_predictions, radial_predictions


#import data for fitting (#2nd part) 
dataToFit = pd.read_csv("vaja6data.csv")
xToFit = np.array(dataToFit['x']).reshape(-1,1)
yToFit = np.array(dataToFit['y']).reshape(-1,1)

#standardize the data
scaler = preprocessing.StandardScaler().fit(xToFit)
xToFit_scaled = scaler.transform(xToFit)

Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(xToFit_scaled, yToFit, test_size=0.3,random_state=109) # 70% training and 30% test
#X_test_points = np.linspace(min(Xtest2), max(Xtest2), 500).reshape(-1, 1)

lin_pred, poly_pred, rad_pred = kernel_regression(Xtrain2, ytrain2, Xtest2)

# Plot the data and predictions for each kernel
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(Xtest2, ytest2, label='Data', color="blue")
plt.scatter(Xtest2, lin_pred, color='red', label='Predictions')
plt.title(f'Linear kernel')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(Xtest2, ytest2, label='Data', color="blue")
plt.scatter(Xtest2, poly_pred, color='red', label='Predictions')
plt.title(f'Polynomial kernel')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(Xtest2, ytest2, label='Data', color="blue")
plt.scatter(Xtest2, rad_pred, color='red', label='Predictions')
plt.title(f'Radial kernel')
plt.legend()

plt.show()


# Accuracy: 0.9883040935672515
# Precision: 1.0
# Recall: 0.9682539682539683
# {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
# SVC(C=1000, gamma=0.001)


