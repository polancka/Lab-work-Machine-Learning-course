#Regression trees
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#Select a criteria of your choice to stop splitting the nodes.
# Selected split criteria is RSS (residual sum squared), which is most commonly used for regression. 
#Criteria for ending the tree building can be depth of tree, minimal number of examples in a leaf node or a certain RSS value. 

class MyRegressionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        X = np.array(X)  # Convert X to numpy array
        if depth == self.max_depth or len(np.unique(y)) == 1:
            # If maximum depth or pure node, create a leaf node
            return np.mean(y)

        # Find the best split
        split_index, split_value = self.find_best_split(X, y)

        if split_index is None:
            # If no split improves RSS, create a leaf node
            return np.mean(y)

        # Split the data
        X_left, y_left, X_right, y_right = self.split_data(X, y, split_index, split_value)

        # Recursively build the left and right subtrees
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        # Return a node representing the split
        return {'split_index': split_index,
                'split_value': split_value,
                'left': left_subtree,
                'right': right_subtree}

    def find_best_split(self, X, y):
        best_rss = float('inf')
        best_split_index = None
        best_split_value = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for value in values:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, value)
                rss = self.calculate_rss(y_left) + self.calculate_rss(y_right)

                if rss < best_rss:
                    best_rss = rss
                    best_split_index = feature
                    best_split_value = value

        return best_split_index, best_split_value

    def split_data(self, X, y, feature_index, split_value):
        left_mask = X[:, feature_index] <= split_value
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def calculate_rss(self, y):
        return np.sum((y - np.mean(y))**2)

    def predict_instance(self, instance, tree):
        if 'split_index' not in tree:
            return tree  # leaf node

        split_value = tree['split_value']
        feature_value = instance[tree['split_index']]

        if np.issubdtype(type(split_value), np.number) and np.issubdtype(type(feature_value), np.number):
            # Check if both values are numeric before making the comparison
            if feature_value <= split_value:
                return self.predict_instance(instance, tree['left'])
            else:
                return self.predict_instance(instance, tree['right'])
        else:
            # If not numeric, treat as leaf node
            return tree

    def predict(self, X, tree):
        return np.array([self.predict_instance(instance, tree) for instance in X])

##---------------------------------------------------------------------------------------------------------------------------------

#Download the dataset "House price". Price is the target.
data = pd.read_csv("House_Price.csv")
X = data.drop(columns=["price", "waterbody"])
X['airport'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)
X['bus_ter'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)
Y = data['price']

#spliting the data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)

#Build a regression tree for the selected dataset.
max_depth = 7
my_tree = MyRegressionTree(max_depth=max_depth)

# Train the regression tree
tree = my_tree.fit(trainX, trainY)

# Make predictions
y_pred = my_tree.predict(testX, tree)


#Build a regression tree with scikit-learn
regressor = DecisionTreeRegressor(trainX, trainX, random_state=0)


#  Compare the cross-validation results with those you get while building a regression
#tree with scikit-learn. Use the same cross-validation splits on both models.
#compare scores
my_score = cross_val_score(my_tree, testX, testY, cv = 10)
sc_score = cross_val_score(regressor, testX, testY, cv=10)
