"""
Created on Sat May  5 15:16:18 2018

@author: KaranJaisingh
"""

# Declare imports used in program
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

# Import the datasets and separate them into feature and output matrices
dataset = pd.read_csv('ckd-dataset.csv')
X = dataset.iloc[:, 0:24].values
y = dataset.iloc[:, 24].values

# Check whether each piece of data is in a valid correct data type
def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

# Change the string values in the dataset into numeric representations
for i in range(0,399):
    if y[i] == 'ckd':
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)

for a in range(0, 399):
    if X[a][5] == 'normal':
        X[a][5] = 0
    if X[a][5] == 'abnormal':
        X[a][5] = 1
        
for a in range(0, 399):
    if X[a][6] == 'normal':
        X[a][6] = 0
    if X[a][6] == 'abnormal':
        X[a][6] = 1
        
for a in range(0, 399):
    if X[a][7] == 'notpresent':
        X[a][7] = 0
    if X[a][7] == 'present':
        X[a][7] = 1
        
for a in range(0, 399):
    if X[a][8] == 'notpresent':
        X[a][8] = 0
    if X[a][8] == 'present':
        X[a][8] = 1
        
for a in range(0, 399):
    for b in range(18, 24):
        if X[a][b] == 'yes' or X[a][b] == 'good':
            X[a][b] = 0
        if X[a][b] == 'no' or X[a][b] == 'poor':
            X[a][b] = 1
    
for a in range(0,399):
    for b in range(0, 24):
        if(isinstance(X[a][b], int)):
            X[a][b] = float(X[a][b])
        elif(isinstance(X[a][b], str)):
            if(is_float(X[a][b])):
                X[a][b] = float(X[a][b])
                
totals = [0] * 24
added = [0] * 24           
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float)):
            totals[b] += X[a][b]
            added[b] += 1
            
averages = [0] * 24          
for a in range(0, 24):
    averages[a] = totals[a] / added[a]
 
c = 0
for a in range(0, 399):
    for b in range(0, 24):
        if(isinstance(X[a][b], float) == 0):
            X[a][b] = averages[b]
            c += 1
    
# Convert all features to a Float data type
X = X.astype(float)

# Find the features that have the best correlation with the output class
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((399,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Select the feature dataset to hold just two features from the dataset
X_small = X[:, [15, 16]]
X = X_small

# Use a built-in library to separate the matrices into separate training and testing datasets for both feature and output matrices 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

import seaborn as sns
from sklearn import datasets

plt.figure(figsize=(12, 8))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.title('Logistic Regression (Dataset)')
plt.xlabel('Haemoglobin (grams)')
plt.ylabel('Packed Cell Volume')
plt.legend(loc=2, fancybox=True, framealpha=1, frameon=True);

class LogisticRegression:
    def __init__(self, alpha=0.01, iters=1000, fit_offset=True, verbose=False):
        self.alpha = alpha
        self.iters = iters
        self.fit_offset = fit_offset
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_offset:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.iters):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.alpha * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 100 == 0):
                print(f'loss: {loss} \t')
    
    def get_predicted_prob(self, X):
        if self.fit_offset:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def get_predicted_class(self, X):
        return self.get_predicted_prob(X).round()
    

model = LogisticRegression(alpha=0.1, iters=100000)

model.fit(X_train, y_train)

preds = model.get_predicted_class(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, preds)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, preds)

from sklearn.metrics import f1_score
f1_score(y_test, preds, average='binary')

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.get_predicted_class(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Haemoglobin (grams)')
plt.ylabel('Packed Cell Volume')
plt.legend()
plt.show()

"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='binary')""" 
