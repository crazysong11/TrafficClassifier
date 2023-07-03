# !/usr/bin/env/python
# -*- coding: utf-8 -*-

import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



seed = int(time.time_ns())
random.seed(seed)

np.set_printoptions(suppress=True, precision=20, threshold=10, linewidth=40)  # np禁止科学计数法显示
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # pd禁止科学计数法显示

df_all = pd.read_csv(r'result_real.csv')

param_grid = {
    'n_estimators': [600],
    'max_depth': [18],
    'max_features': [10]
}


X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values
y = y.astype('int')
X, y = shuffle(X, y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


y_pred = grid_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

import scikitplot as skplt
import matplotlib.pyplot as plt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()