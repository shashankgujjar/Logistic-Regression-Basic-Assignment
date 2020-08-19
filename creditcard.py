# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:09:25 2020

@author: user
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

#Importing Data
CC = pd.read_csv("E:\\Data\\Assignments\\i made\\Logistic Regression\\creditcard.csv")
CC.head()
CC.columns

# data frame creation
df = pd.DataFrame(cc)

# Dropping sl numner column
cc = df.drop(['sl'], axis=1)

# to get dummies
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

# for binary class
cc['card'] = le.fit_transform(cc['card'])
cc['owner'] = le.fit_transform(cc['owner'])
cc['selfemp'] = le.fit_transform(cc['selfemp'])

# to check whether the null values are present or not
cc.isnull().sum() # no null values
cc.describe()

# Model building 
from sklearn.linear_model import LogisticRegression
cc.shape

X = cc.iloc[:,0:16]
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

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y, y_pred)

# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 



















