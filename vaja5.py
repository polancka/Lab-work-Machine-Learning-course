import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
import collections
import math

#possible exam questions
#what is regularization, why do we use it, what types do we know
#what does lasso do, what is the differnece between laso/ridge
#wh dont we use linear regression for lasso?
#what are the principles of linear regression? how can we see if the principles are met from diagnostic plots

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


def gradient_boosting_fit(trainX, trainY, testX, testY, n_estimators, learning_rate, max_depth, random_state):
    # you can use scikit trees, they should be weak (= small depth) - even stump is a valid tree (so stick to depth 1, 2, 3)
    #we want them to underfit
    
    #calculating the odds of the classes, initial "Model" is just the "average"
    np_trainY = np.array(trainY)
    nu_values = collections.Counter(np_trainY)
    nu_class_0 = nu_values.get('low')
    nu_class_1 = nu_values.get('high')
    odds = math.log(nu_class_1/nu_class_0)
    prob_odds_class_1 = math.pow(math.e, odds)/(1 + math.pow(math.e, odds))
    intial_residuals = list()

    print(trainY)
    trainY.replace(["low", "high"], #change the quality (numeric value) to categorical high/low
                       [0, 1], inplace=True)


    # for i in range(len(trainY)):
    #     res = trainY[i] - prob_odds_class_1
    #     intial_residuals.append(res)


    #inital model

    tree = DecisionTreeClassifier().fit(trainX, trainY)
    Y_predicted = tree.predict(testX)
    curr_metric = metrics.accuracy_score(testY, Y_predicted)


    return 1


# def gradient_boosting_predict(testX, testY):
#     my_score
#     return my_score

#Download the dataset on Uˇcilnica "wine-quality.csv". Target is "quality (high or low)"
data = pd.read_csv("wine_quality.csv")
print(data.info)
data['alcohol_level'].replace(["low", "high"], #change the alchocol level (categorical variable) to a numeric variabče
                        [0, 1], inplace=True)

data['quality'].replace([[1,2,3,4,5,6], [7,8,9,10]], #change the quality (numeric value) to categorical high/low
                       ["low", "high"], inplace=True)

print(data.info)
X = data.drop(columns="quality")
Y = data["quality"]

trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3, random_state= 42)

# ##gradient boosting from sklearn 
skl_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(trainX, trainY)
skl_score = skl_boosting.score(testX, testY)
print(skl_score)

my_boosting = gradient_boosting_fit(trainX, trainY, testX, testY, 100, 1.0, 1, 42)
# my_score = gradient_boosting_predict(my_boosting, testX, testY)
# print(my_score)

#compare the two implementations


#Test different learning rates. What is a good learning rate

#Test different numbers of trees that are built during gradient boosting and comment
#on the results. Does your model overfit? If yes, try to prevent overfitting. 
    # The model can overfit - it can learn too much. we have to stop the learning at some point. We use stop criterion. 

#Compare the cross-validation results from your implementation with the "GradientBoostingClassifier" classifier implemented in Scikit-learn.

#Try modelling your data also with XGBoost, LightGBM and CatBoost models (fit all
#three models). Compare all the models you built. Be careful to evaluate all three
#models on the same test dataset. 
    # catboost is good for categorical data
    #these three are best for medium size data
