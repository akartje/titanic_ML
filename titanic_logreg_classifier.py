# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:18:38 2022

@author: akart
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("C:\\Users\\akart\\Desktop\\Coding\\Data\\titanic_data\\train.csv")
df['Male'] = df['Sex'] == 'male' # Set sex and age columns to something we can work with
df['Old'] = df['Age'] > 35 # Basically a workaround for the missing values
df['Young'] = df['Age'] < 13 # VERY informative!
df['EmbS'] = df['Embarked'] == 'S' #Ended up being quite helpful!
#df['EmbQ'] = df['Embarked'] == 'Q' Didn't end up helpful
X = df[['Pclass', 'Male', 'Fare', 'SibSp', 'Parch', 'EmbS', 'Young', 'Old']].values
y = df['Survived'].values

scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train,y_train)
    scores.append(model.score(X_test, y_test))
    
print("Scores: ", scores)
print("Mean Score: ", round(np.mean(scores), 4))
final_model = LogisticRegression()
final_model.fit(X,y)
y_pred = final_model.predict(X)
print("Final accuracy: ", round(accuracy_score(y, y_pred), 4))
print("Final precision: ", round(precision_score(y, y_pred), 4))
print("Final recall: ", round(recall_score(y, y_pred), 4))
print("Final f1:", round(f1_score(y, y_pred), 4))