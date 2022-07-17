# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 22:50:24 2022

@author: akart
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd


df = pd.read_csv("C:\\Users\\akart\\Desktop\\Coding\\Data\\titanic_data\\train.csv")
df['Male'] = df['Sex'] == 'male' 
df['Old'] = df['Age'] > 40
df['Young'] = df['Age'] < 14
df['EmbS'] = df['Embarked'] == 'S' 
X = df[['Pclass', 'Male', 'Fare', 'SibSp', 'Parch', 'EmbS', 'Young', 'Old']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

nn = MLPClassifier(activation='tanh', alpha=0.00005, max_iter=10000, random_state=101)
nn.fit(X_train, y_train)
"""
param_grid = {
#    'activation': ['identity', 'logistic', 'tanh', 'relu'],
#    'alpha': [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
#    'max_iter': [3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
}

gs = GridSearchCV(nn, param_grid, scoring='f1', cv=5)
gs.fit(X_test, y_test)

print("Best params: ", gs.best_params_)
#>> Best params: 'activation': 'tanh', 'alpha': 5e-05, 'max_iter': 5500
"""
print("score: ", nn.score(X_test, y_test))