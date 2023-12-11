#Regression trees
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin


#Select a criteria of your choice to stop splitting the nodes.
# Selected split criteria is RSS (residual sum squared), which is most commonly used for regression. 
#Criteria for ending the tree building can be depth of tree, minimal number of examples in a leaf node or a certain RSS value. 

# Define the regression tree class with RSS as the splitting criterion
class MyRegressionTree(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def rss(self, targets):
        return np.sum((targets - np.mean(targets))**2)

    def find_best_split(self, features, targets):
        m, n = features.shape
        if m <= 1:
            return None, None

        total_rss = self.rss(targets)

        best_split_idx = None
        best_split_value = None
        best_rss = float('inf')

        for i in range(n):
            if isinstance(features, pd.DataFrame):
                feature_values = features.iloc[:, i].values
            else:
                feature_values = features[:, i]
            unique_values = np.unique(feature_values)

            for value in unique_values:
                left_mask = feature_values <= value
                right_mask = ~left_mask

                if np.sum(left_mask) >= self.min_samples_leaf and np.sum(right_mask) >= self.min_samples_leaf:
                    left_rss = self.rss(targets[left_mask])
                    right_rss = self.rss(targets[right_mask])

                    weighted_rss = (np.sum(left_mask) / m) * left_rss + (np.sum(right_mask) / m) * right_rss

                    if weighted_rss < best_rss:
                        best_rss = weighted_rss
                        best_split_idx = i
                        best_split_value = value

        return best_split_idx, best_split_value

    def fit(self, features, targets, depth=0):
        if self.max_depth is not None and depth == self.max_depth:
            return np.mean(targets)

        best_split_idx, best_split_value = self.find_best_split(features, targets)

        if best_split_idx is None:
            return np.mean(targets)
        
        if isinstance(features, pd.DataFrame):
            features = features.values

        left_mask = features[:, best_split_idx] <= best_split_value
        right_mask = ~left_mask

        left_subtree = self.fit(features[left_mask], targets[left_mask], depth + 1)
        right_subtree = self.fit(features[right_mask], targets[right_mask], depth + 1)

        return (best_split_idx, best_split_value, left_subtree, right_subtree)

    def predict(self, features):
        if self.tree is None:
            raise ValueError("The tree has not been fitted yet.")

        predictions = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            node = self.tree
            while isinstance(node, tuple):
                split_idx, split_value, left_subtree, right_subtree = node
                if features[i, split_idx] <= split_value:
                    node = left_subtree
                else:
                    node = right_subtree
            predictions[i] = node  # Leaf node value

        return predictions


##---------------------------------------------------------------------------------------------------------------------------------

#Download the dataset "House price". Price is the target.
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


# Visualize your data to identify outliers
# plt.boxplot(testX)
# plt.show()


# Build the regression tree with RSS as the splitting criterion
regression_tree_rss = MyRegressionTree(max_depth=5, min_samples_leaf=10)
regression_tree_rss.tree = regression_tree_rss.fit(trainX, trainY)

# Test the regression tree using cross-validation
#predicted = regression_tree_rss.predict(testX)
#cv_scores_scratch_rss = cross_val_score(predicted, testX.values, testY.values, cv=10, scoring='neg_mean_squared_error')
#cv_rmse_scratch_rss = np.sqrt(-cv_scores_scratch_rss)

# Build a regression tree with scikit-learn for comparison
scikit_tree_rss = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
cv_scores_scikit_rss = cross_val_score(scikit_tree_rss, testX.values, testY.values, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scikit_rss = np.sqrt(-cv_scores_scikit_rss)

# Compare the cross-validation results
#print("Cross-validation RMSE (from scratch with RSS):", cv_rmse_scratch_rss)
print("Cross-validation RMSE (scikit-learn with RSS):", sum(cv_rmse_scikit_rss)/len(cv_rmse_scikit_rss))
#MSEs with scikit: [3.01503952 2.68341541 3.48119011 3.53989764 3.989915   5.00732851 3.47742382 4.92615124 2.96002243 5.2240361 ]
#Mean of MSEs with scikit : 3.8304419784368635

#MSEs with my implementation: 
#Mean of MSEs with my implementation: 
