import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score
import collections
import math
import re

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV

#BOOSTING FOR CLASSIFICATION
#Write the code for gradient boosting of trees for solving a binary classification problem from scratch.
#have small depth for the trees

#look at sensititvty, specificity, accuracy for positive and negative class

#LOSS FUNCTION = ((y - h(x))^2 (before for linear)
#residuals are still predicted - observed
#First we build the first model: 
    #we compute the odds = log(P(Y=1)/P(y=0))
    #from the odds we calculate the probability that Y = 1 --> P(Y=1) = e^odds/1 + e ^odds
    #then calculate residuals residuals = predicted - observed (we get residuals 1)
    #Now we fit the tree to the residuals (NOT TO Y!) We take the residuals and see in which leaf they qualify. 
    # For each leaf we have to calculate the final score for the leaf
    # (final score is calculated like : newPrediction =  sum(residuals in a leaf)/sum(previousPrediciton * (1 - previousPrediction)))
    #in the first tree the "previousPrediction" is the initial probability P(Y=1)
    #calculate a new prediciton : Ypred = previousPrediction + learningRate*newPrediction (value of the leaf that the example lands in)
    #again calculate a probability P(Y=1)= e^0.97/(1+e^0.97)
    #calulacte residuals
    #fit a new tree to the residuals
    #new prediction = odds + learningRate*predictionOfThatTree + learningRate*predicitonOfTheNextTree + ....


#my implementation of gradient boosting from scratch
class MyGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.firstPred = None

    def fit(self, X, y):
        self.firstPred = math.log((np.sum(y==1)) / (np.sum(y == 0))) # Initialize first predictions (log odds) with class shares
        current_odds = np.full(y.shape, self.firstPred)

        for _ in range(self.n_estimators): #building however many trees
            #Calculate probabilities (sigmoid function)
            current_prediction_prob = self.sigmoid_function(current_odds)

            # Calculate residuals
            residuals = y - current_prediction_prob

            # Fit a weak tree to the residuals and predict
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)

            # Store the weak learner in the list of models
            self.models.append(tree)

            
            leaf_ids = tree.apply(X)

            for unique_leaf_id in np.unique(leaf_ids):
                current_value= np.reshape(current_prediction_prob, (1, -1))[0]
                mask = np.zeros_like(leaf_ids, dtype=bool)

                for leaf_id in leaf_ids:
                    if leaf_id == unique_leaf_id:
                        mask = np.logical_or(mask, leaf_ids == leaf_id)
                current_predictions_leaf = current_value[mask]
                
                denominator = sum(np.multiply(current_predictions_leaf, (1 - current_predictions_leaf)))
                
                numerator = sum(residuals[mask])
                
                current_gamma = numerator / denominator
                
                tree.tree_.value[unique_leaf_id] = current_gamma

            new_tree_predictions = [self.learning_rate * tree.predict(X)]
            current_leaf_prediction = np.reshape(new_tree_predictions, (-1, 1))
            
            flat_list = [item[0] for item in current_leaf_prediction]
        
            current_odds = current_odds + flat_list
           
            

    def sigmoid_function(self, x):
        return np.exp(x) / (1 + np.exp(x))
    
    def predict_proba(self, X):
        # Initialize gamma values as log odds
        gamma_value = np.zeros(X.shape[0], dtype=float)

        # Sum up predictions from all trees
        for tree in self.models:
            leaf_values = tree.apply(X)
            gamma_value += self.learning_rate * np.take(tree.tree_.value[:, 0, 0], leaf_values)

        # Convert gamma values to probabilities using the sigmoid function
        probabilities = self.sigmoid_function(gamma_value)

        # Return probabilities for class 1 (assuming binary classification)
        return np.vstack((1 - probabilities, probabilities)).T

    def predict(self, X, threshold=0.5):
        # Make probability predictions
        probabilities = self.predict_proba(X)

        # Convert probabilities to binary predictions based on the threshold
        binary_predictions = (probabilities[:, 1] >= threshold).astype(int)

        return binary_predictions


#Data import and preprocessing
data = pd.read_csv("wine_quality.csv")
data['alcohol_level'].replace(["low", "high"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)

data['quality'].replace([[1,2,3,4,5,6], [7,8,9,10]], #change the quality (numeric value) to categorical high/low
                       [0, 1], inplace=True)

#renaming for the sake of LGBM boost library
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

#print(data.info)
X = data.drop(columns="quality")
Y = data["quality"]

#splitting into train and test data set 
trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)

#compare the two implementations
#Sklearn 
skl_fit = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(trainX, trainY)
skl_predictions = skl_fit.predict(testX)
skl_accuracy = accuracy_score(testY, skl_predictions)
skl_f1 = f1_score(testY, skl_predictions)
print("Sklearn accuracy:", skl_accuracy)
print("Sklearn F1 score: ", skl_f1)
# Sklearn accuracy: 0.8163265306122449
# Sklearn F1 score:  0.5072992700729926


custom_boost = MyGradientBoostingClassifier()
custom_fit = custom_boost.fit(trainX, trainY)
custom_predictions = custom_boost.predict(testX)
custom_accuracy = accuracy_score(testY, custom_predictions)
custom_f1 = f1_score(testY, custom_predictions)
print("Custom accuracy",custom_accuracy)
print("Custom f1", custom_f1)
# Custom accuracy 0.7802721088435374
# Custom f1 0.6265895953757225

#Test different learning rates. What is a good learning rate : test the usual suspects [0.1, 0.01, 0.001, 0.03, 0.003, 0.06, 0.006],
#Test different numbers of trees that are built during gradient boosting and comment on the results.  [10, 50, 200, 300, 400, 1000]

param_grid_one = {
    'learning_rate': [0.1, 0.01, 0.001, 0.03, 0.003, 0.06, 0.006],
    'n_estimators': [10, 50, 200, 300, 400, 1000],
    'max_depth': [1, 3, 5]
}
learning_rates_list = [0.1, 0.01, 0.001, 0.03, 0.003, 0.06, 0.006]
n_estimators_list = [10, 50, 200, 300, 400, 1000]
accuracies = list()

for i in range(len(learning_rates_list)):
    for j in range(len(n_estimators_list)):
        curr_model = MyGradientBoostingClassifier(n_estimators= n_estimators_list[j], learning_rate=learning_rates_list[i])
        fitted = curr_model.fit(trainX, trainY)
        curr_preds = curr_model.predict(testX)
        curr_accuracy = accuracy_score(testY, curr_preds)
        accuracies.append((i,j,curr_accuracy))

# print(accuracies)
# print(max(accuracies, key = lambda x : x[2]))
        
# custom_best = MyGradientBoostingClassifier()
# custom_grid_search = GridSearchCV(custom_best, param_grid_one, cv=10, scoring='accuracy')
# custom_grid_search.fit(trainX, trainY)
# best_params_custom = custom_grid_search.best_params_
# print("Best parameters for custom: ", best_params_custom)
#Does your model overfit? If yes, try to prevent overfitting.
#The model can overfit - it can learn too much. we have to stop the learning at some point. We use stop criterion - depth of 3. s

#---------------------------------------------------------------------------------------------------------------------------------------#
#Try modelling your data also with XGBoost, LightGBM and CatBoost models (fit all three models). Compare all the models you built. Be careful to evaluate all three models on the same test dataset.
# ((catboost is good for categorical data, these three are best for medium size data))

# XGBoost, LightGB, CatBoost

# # XGBoost Model
# xgb_model = XGBClassifier()
# xgb_model.fit(trainX, trainY)
# xgb_predictions = xgb_model.predict(testX)
# xgb_accuracy = accuracy_score(testY, xgb_predictions)
# xgb_f1 = f1_score(testY, xgb_predictions)
# print("XGBoost Accuracy:", xgb_accuracy)
# print("XGBoost f1:", xgb_f1)


# # LightGBM Model
# lgb_model = LGBMClassifier()
# lgb_model.fit(trainX, trainY)
# lgb_predictions = lgb_model.predict(testX)
# lgb_accuracy = accuracy_score(testY, lgb_predictions)
# lgb_f1 = f1_score(testY, lgb_predictions)
# print("LightGBM Accuracy:", lgb_accuracy)
# print("LightGBM f1:", lgb_f1)

# # CatBoost Model
# cb_model = CatBoostClassifier()
# cb_model.fit(trainX, trainY, verbose=False)  # Setting verbose to False to avoid printing too much output
# cb_predictions = cb_model.predict(testX)
# cb_accuracy = accuracy_score(testY, cb_predictions)
# cb_f1 = f1_score(testY, cb_predictions)
# print("CatBoost Accuracy:", cb_accuracy)
# print("CatBoost f1:", cb_f1)

# #Accuracy and f1_score before hypertuning the parameters
# XGBoost Accuracy: 0.8802721088435375
# XGBoost f1: 0.705685618729097
# LightGBM Accuracy: 0.8802721088435375
# LightGBM f1: 0.7046979865771813
# CatBoost Accuracy: 0.8687074829931973
# CatBoost f1: 0.6571936056838366

##HYPER PARAMETER TUNING WITH GRIDSEARCH
# param_grid = {
#     'learning_rate': [0.1, 0.01, 0.001, 0.03, 0.003, 0.06, 0.006],
#     'n_estimators': [10, 50, 200, 300, 400, 1000],
#     'max_depth': [1, 3, 5]
# }

# xgb_best = XGBClassifier()
# grid_search_xgb = GridSearchCV(xgb_best, param_grid, cv=10, scoring='accuracy')
# grid_search_xgb.fit(trainX, trainY)
# best_params_xgb = grid_search_xgb.best_params_

# lgb_best = LGBMClassifier()
# grid_search_lgb = GridSearchCV(lgb_best, param_grid, cv=10, scoring='accuracy')
# grid_search_lgb.fit(trainX, trainY)
# best_params_lgb = grid_search_lgb.best_params_
# print("LGBM best parameters(f1): ", best_params_lgb)

# cat_best = CatBoostClassifier()
# grid_search_cat = GridSearchCV(cat_best, param_grid, cv=10, scoring='accuracy')
# grid_search_cat.fit(trainX, trainY)
# best_params_cat = grid_search_cat.best_params_

# print("XGBOOST best parameters(f1): ", best_params_xgb)
# print("LGBM best parameters(f1): ", best_params_lgb)
# print("CAT best parameters(f1): ", best_params_cat)

#ACCURACY (with first grid that did not have a lot of values)
#XGBOOST best parameters (accuracy):  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}
#LGBM best parameters(accuracy):  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}
#CAT best parameters(accuracy):  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}

#ACCURACY (expanded grid)
# XGBOOST best parameters(f1):  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}
# LGBM best parameters(f1):  {'learning_rate': 0.03, 'max_depth': 5, 'n_estimators': 1000}
# CAT best parameters(f1):  {'learning_rate': 0.06, 'max_depth': 5, 'n_estimators': 1000

#F1 (expanded grid)
# XGBOOST best parameters(f1):  {'learning_rate': 0.06, 'max_depth': 5, 'n_estimators': 1000}
# LGBM best parameters(f1):  {'learning_rate': 0.03, 'max_depth': 5, 'n_estimators': 1000}
# CAT best parameters(f1):  {'learning_rate': 0.06, 'max_depth': 5, 'n_estimators': 1000}

# # #XGBoost, LightGB, CatBoost with tuned hyperparameters
# # Rerun, readjust values of hyperparameters
# # XGBoost Model
# xgb_model = XGBClassifier(learning_rate= 0.1, max_depth= 5, n_estimators= 300)
# xgb_model.fit(trainX, trainY)
# # Predictions and evaluation
# xgb_predictions = xgb_model.predict(testX)
# xgb_accuracy = accuracy_score(testY, xgb_predictions)
# print("Tuned XGBoost Accuracy:", xgb_accuracy)


# # LightGBM Model
# lgb_model = LGBMClassifier(learning_rate= 0.03, max_depth= 5, n_estimators= 1000)
# lgb_model.fit(trainX, trainY)
# # Predictions and evaluation
# lgb_predictions = lgb_model.predict(testX)
# lgb_accuracy = accuracy_score(testY, lgb_predictions)
# print("Tuned LightGBM Accuracy:", lgb_accuracy)


# # CatBoost Model
# cb_model = CatBoostClassifier(learning_rate= 0.06, max_depth= 5, n_estimators= 1000)
# cb_model.fit(trainX, trainY, verbose=False)  # Setting verbose to False to avoid printing too much output
# # Predictions and evaluation
# cb_predictions = cb_model.predict(testX)
# cb_accuracy = accuracy_score(testY, cb_predictions)
# print("Tuned CatBoost Accuracy:", cb_accuracy)

#ACCURACY TUNED: 
#Tuned XGBoost Accuracy: 0.8850340136054422
# Tuned LightGBM Accuracy: 0.8761904761904762
# Tuned CatBoost Accuracy: 0.8755102040816326



