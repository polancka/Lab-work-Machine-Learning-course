import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier



#BOOSTING FOR CLASSIFICATION

#Write the code for gradient boosting of trees for solving a binary classification problem from scratch.
#have small depth for the trees

#LOSS FUNCTION = ((y - h(x))^2
#residuals are still predicted - observed

def gradient_boosting_fit(trainX, trainY, n_estimators, learning_rate, max_depth, random_state):
    # you can use scikit trees, they should be weak (= small depth) - even stump is a valid tree (so stick to depth 1, 2, 3)
    #we want them to underfit
    for i in range(n_estimators):
        tree = DecisionTreeClassifier()

    return 1

def gradient_boosting_predict(testX, testY):
    my_score
    return my_score

#Download the dataset on Uˇcilnica "wine-quality.csv". Target is "quality (high or low)"
data = pd.read_csv("wine_quality.csv")
#print(data.info)
X = data.drop(columns="quality")
Y = data["quality"]

trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)

# ##gradient boosting from sklearn 
# skl_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(trainX, trainY)
# skl_score = skl_boosting.score(testX, testY)
# print(skl_score)

my_boosting = gradient_boosting_fit(trainX, trainY, 100, 1.0, 1, 42)
my_score = gradient_boosting_predict(my_boosting, testX, testY)
print(my_score)


#Test different learning rates. What is a good learning rate

#Test different numbers of trees that are built during gradient boosting and comment
#on the results. Does your model overfit? If yes, try to prevent overfitting.

#Compare the cross-validation results from your implementation with the "GradientBoostingClassifier" classifier implemented in Scikit-learn.

#Try modelling your data also with XGBoost, LightGBM and CatBoost models (fit all
#three models). Compare all the models you built. Be careful to evaluate all three
#models on the same test dataset.