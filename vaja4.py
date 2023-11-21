#Regression trees
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

class DecisionNode:
    def __init__(self, right, left, rss_value, split_attribute):
        self.left = left
        self.right = right
        self.rss_value = rss_value
        self.split_attribute = split_attribute


#Implement the regression tree algorithm from scratch. (split on a criteria of your own choosing)
def build_regression_tree(data): #save the RSS value on every split and plot them to determine optimal value for stopping the tree
    root_node = DecisionNode(None, None, 0, "")
    return root_node

#Select a criteria of your choice to stop splitting the nodes.
# Selected split criteria is RSS (residual sum squared), which is most commonly used for regression. 
#Criteria for ending the tree building can be depth of tree, minimal number of examples in a leaf node or a certain RSS value. 

#Download the dataset "House price". Price is the target.
data = pd.read_csv("House_Price.csv")
print(sum(data.isnull))
X = data.drop(columns="price")
Y = data['price']

#TODO: prepare the data - what to do with categorical variables? Shoulkd the tree be build only on train set?

#Build a regression tree for the selected dataset.
root_tree = build_regression_tree(data)

#Build a regression tree with scikit-learn
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, X, Y, cv=10)

#TODO: Test the regressing tree (both) using cross-validation (save results as you go) 


# TODO: Compare the cross-validation results with those you get while building a regression
#tree with scikit-learn. Use the same cross-validation splits on both models.