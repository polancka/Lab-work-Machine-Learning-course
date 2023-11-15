import numpy as np

class myLinearRegression:
    
    def my_linear_regression(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        #add the column of ones to calculate the intercept as well
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate the coefficients using the normal equation
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return beta

    def my_predict(self, X, beta):
        X = np.c_[np.ones(X.shape[0]), X]
        # Make predictions using the calculated coefficients
        return X.dot(beta)



# #TODO: my own function for linear regression
# def my_linear_regression(X, Y):
#     #add the column of ones to calculate the intercept as well
#     X = np.c_[np.ones(X.shape[0]), X]
#     # Calculate the coefficients using the normal equation
#     beta = np.linalg.inv(X.T @ X) @ X.T @ Y
#     return beta

# def my_predict(X, beta):
#     X = np.c_[np.ones(X.shape[0]), X]
#     # Make predictions using the calculated coefficients
#     return X @ beta
