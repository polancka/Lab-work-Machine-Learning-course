#Regression trees
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# lahko droppaš variablo ki ima 4 različne (water body)

class DecisionNode:
    def __init__(self, right, left, rss_value, split_attribute):
        self.left = left
        self.right = right
        self.rss_value = rss_value
        self.split_attribute = split_attribute


#Implement the regression tree algorithm from scratch. (split on a criteria of your own choosing)
def build_regression_tree(trainX, trainY): #save the RSS value on every split and plot them to determine optimal value for stopping the tree
    root_node = DecisionNode(None, None, 0, "")
    return root_node

#Select a criteria of your choice to stop splitting the nodes.
# Selected split criteria is RSS (residual sum squared), which is most commonly used for regression. 
#Criteria for ending the tree building can be depth of tree, minimal number of examples in a leaf node or a certain RSS value. 

#Download the dataset "House price". Price is the target.
data = pd.read_csv("House_Price.csv")
X = data.drop(columns=["price", "waterbody"])
X['airport'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)
X['bus_ter'].replace(["NO", "YES"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)
print(X.info)
Y = data['price']

#TODO: prepare the data - what to do with categorical variables? Shoulkd the tree be build only on train set?

trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)
#Build a regression tree for the selected dataset.
root_tree = build_regression_tree(trainX, trainY)
my_score = cross_val_score(root_tree, testX, testY, cv = 10)


#Build a regression tree with scikit-learn
regressor = DecisionTreeRegressor(trainX, trainX, random_state=0)
sc_score = cross_val_score(regressor, testX, testY, cv=10)

#TODO: Test the regressing tree (both) using cross-validation (save results as you go) 


# TODO: Compare the cross-validation results with those you get while building a regression
#tree with scikit-learn. Use the same cross-validation splits on both models.