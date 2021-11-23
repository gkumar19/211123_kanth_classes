#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:17:21 2017

@author: kanth
"""


import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# load the iris datasets
dataset = datasets.load_iris()

dataset = pd.read_csv("/Users/kanth/Documents/Certified Data Science Program Files/Machine Learning Files/Random Forest/Diabetes.csv")

dataset.data = dataset.iloc[:,:8]
dataset.target = dataset.iloc[:,8:]
# fit a CART model to the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.data,dataset.target, test_size=0.2, random_state=0)


model = DecisionTreeClassifier()

#Random Forest
model = RandomForestClassifier()

model.fit(X_train, y_train)


#Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1,
     max_depth=1, random_state=0).fit(X_train, y_train)



model = GradientBoostingClassifier(n_estimators=100, learning_rate=1,
     max_depth=1, random_state=0, loss='exponential').fit(X_train, y_train)




print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

