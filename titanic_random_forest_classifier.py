# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:37:37 2022

@author: akart
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def prepTitanicData (filepath):
    df = pd.read_csv(filepath)
    df['Male'] = df['Sex'] == 'male' # Set sex and age columns to something we can work with
    df['Old'] = df['Age'] > 35 # Basically a workaround for the missing values
    df['Young'] = df['Age'] < 12 # VERY informative!
    df['EmbS'] = df['Embarked'] == 'S' #Ended up being quite helpful! Only one between S, C, and Q
    #df['IsAlone'] = (df['SibSp']+df['Parch])==0 Tried this to consolidate the two vars, turned out less useful than both of them seperate
    return df

titanic_folder = "C:\\Users\\akart\\Desktop\\Coding\\Data\\titanic_data\\" #again to stop retyping!
df1 = prepTitanicData(titanic_folder+"train.csv")
feature_list = ['Pclass', 'Male', 'Fare', 'SibSp', 'Parch', 'EmbS', 'Young', 'Old'] #sick of retyping this
X = df1[feature_list].values
y = df1['Survived'].values

rf = RandomForestClassifier(n_estimators=14, random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
rf.fit(X_train, y_train)
"""
Section where I determined best num of estimators
best params: 'n_estimators'=14

from sklearn.model_selection import GridSearchCv
param_grid = {
    'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
}

gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
gs.fit(X_test, y_test)
print("Best params: ", gs.best_params_)
"""

print('Accuracy: ', rf.score(X_test, y_test))
ft_imp = pd.Series(rf.feature_importances_, index=feature_list).sort_values(
     ascending=False)
print(ft_imp)

df2 = prepTitanicData(titanic_folder+"test.csv")
X_final = df2[feature_list].values


import csv
data = ([int(df2['PassengerId'][i]), int(rf.predict([X_final[i]]))] for i in range(418))
#So funny story about that, I guess my laptop sucks and if I don't break the data into chunks like this I get overflow errors??? Didn't have this problem on the train set for some reason
#UPDATE: went to look at the test data, and at the line my program was blanking the Fare value was blank!! Set to 0 and all works now
with open(titanic_folder+'titanic_submission_2022_2_18.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['PassengerId', 'Survived'])
    writer.writerows(data)
