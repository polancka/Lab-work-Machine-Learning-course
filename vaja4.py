#Regression trees
import pandas as pd
import numpy as np
from sklearn.model_selection import  KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin


#Select a criteria of your choice to stop splitting the nodes - depth
# Selected split criteria is RSS (residual sum squared), which is most commonly used for regression. 
#Criteria for ending the tree building can be depth of tree, minimal number of examples in a leaf node or a certain RSS value. 

# Define the regression tree class with RSS as the splitting criterion
class MyRegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_split
        self.tree = None
    
    def mse(self, y):
        return np.mean((y - np.mean(y))**2)

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_leaf:
            return None, None

        total_mse= self.mse(y)

        best_split_id = None
        best_split_value = None
        best_mse = float('inf')

        for i in range(n):
            partX = X[:i]
            unique_values = np.unique(partX)
            for value in unique_values:
                left_mask = partX <= value
                right_mask = ~left_mask

                if len(y[left_mask]) != 0 and len(y[right_mask]) != 0:
                    right_mse = self.mse(y[right_mask])
                    left_mse = self.mse(y[left_mask])

                    weighted_mse = (len(y[left_mask]) / m) * left_mse + (len(y[right_mask]) / m) * right_mse

                    if weighted_mse < best_mse:
                        best_mse = weighted_mse
                        best_split_id = i
                        best_split_value = value
                else: 
                    continue

        if best_mse >= total_mse:
            return None, None 
        else:
            return best_split_id, best_split_value
            
                
    def fit(self, X,y, depth):
        #izhodni pogoj rekurzija
        if len(np.unique(y) == 1) or depth == self.max_depth:
            return np.mean(y)
        
        #najdi najboljši split
        split_index, split_value = self.find_best_split(X,y)

        if split_index is not None:
            left_mask = X[:,split_index] <= split_value
            right_mask = ~left_mask

            #rekurzivni klic
            left_subtree = self.fit(X[left_mask, :], y[left_mask], depth + 1)
            right_subtree = self.fit(X[right_mask, :], y[right_mask], depth + 1)

            return (split_index, split_value, left_subtree, right_subtree)
        else:
            return np.mean(y)


    def predict_single(self, x, tree):
        if isinstance(tree, np.float64):
            return tree  # Leaf node, return the predicted value
        else:
            index, value, left_subtree, right_subtree = tree
            if x[index] <= value:
                return self.predict_single(x, left_subtree)
            else:
                return self.predict_single(x, right_subtree)
            
    def predict(self, X):
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() before predict().")

        return np.array([self.predict_single(x, self.tree) for x in X])


##---------------------------------------------------------------------------------------------------------------------------------

#Download the dataset "House price". Price is the target.
#Preprocessinf
data = pd.read_csv("House_Price.csv")

#get rid of NaN values
data = data.dropna()
X = data.drop(columns=["price", "waterbody"])

#adjust the variables
X['airport'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)
X['bus_ter'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)

Y = data['price']

#spliting the data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)


# # Build the regression tree with RSS as the splitting criterion

my_cv = MyRegressionTree(max_depth=5, min_samples_split=2)
my_cv.tree = my_cv.fit(trainX, trainY, 0)
preds = my_cv.predict(testX)
print(preds)
my_accuracay = accuracy_score(testY, preds)

# Test the regression tree using cross-validation

sl_cv = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

#Lists to store accuracy scores for each fold
sl_scores = []
my_scores = []
tree_depths = [1,2,3,4,5,6,7,8,9,10]

for train_index, test_index in kf.split(X): #loop for every fold
    # Split the data into training and testing sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    for tree_depth in tree_depths:
        scores_r2 = []
        scores_r2_scikit = []
        for train, test in kf.split(X):
            # reg_tree = RegressionTreeCustom(tree_depth)
            # reg_tree.fit(X_train, y_train)
            # y_test_pred = reg_tree.predict(X_test)
            # scores_r2.append(r2_score(y_test, y_test_pred))
            
            regressor = DecisionTreeRegressor(max_depth=tree_depth)
            regressor.fit(X_train, y_train)
            y_test_pred_scikit = regressor.predict(X_test)
            scores_r2_scikit.append(r2_score(y_test, y_test_pred_scikit))
        #score = np.average(scores_r2)
        score_scikit = np.average(scores_r2_scikit)
        print(f'Depth: {tree_depth};  R2 : {score_scikit}')

# # Print the results
# print("Custom Gradient Boosting 10-Fold Cross-Validation Scores:", my_scores)
# print("Scikit-learn Gradient Boosting 10-Fold Cross-Validation Scores:", sl_scores)
# #21.3412603, 11.79355521, 24.5071886, 24.56891826, 23.03698228
# #21.84118053

# # Compare average scores
# print("Average Custom Accuracy:", np.mean(my_scores))
# print("Average Scikit-learn Accuracy:", np.mean(sl_scores))
#




