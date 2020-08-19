# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:09:25 2020

@author: user
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

#Importing Data
bank = pd.read_csv("E:\\Data\\Assignments\\i made\\Logistic Regression\\bank.csv")
bank.head()
bank.columns

# to get dummies
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

# for binary class
bank['default'] = le.fit_transform(bank['default'])
bank['housing'] = le.fit_transform(bank['housing'])
bank['loan'] = le.fit_transform(bank['loan'])
bank['y'] = le.fit_transform(bank['y'])

# for more than 2 class
bank['job'] = le.fit_transform(bank['job'])
bank['marital'] = le.fit_transform(bank['marital'])
bank['education'] = le.fit_transform(bank['education'])
bank['contact'] = le.fit_transform(bank['contact'])
bank['poutcome'] = le.fit_transform(bank['poutcome'])
bank['month'] = le.fit_transform(bank['month'])

# to check whether the null values are present or not
bank.isnull().sum() # no null values
bank.describe()

# Model building 
from sklearn.linear_model import LogisticRegression
bank.shape

X = bank.iloc[:,0:16]
X.shape
Y = bank.iloc[:,16]
Y.shape

model1 = LogisticRegression()
model1.fit(X,Y)

model1.coef_ # coefficients of features 
model1.predict_proba (X) # Probability values 

# Prediction
y_pred = model1.predict(X)
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(model1.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)

# confusion matrix... 
# to calculate the accuracy.....
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/bank.shape[0]
pd.crosstab(y_pred,Y)

help(pd.crosstab)

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y, y_pred)

# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 



















