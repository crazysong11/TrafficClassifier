import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier

df_all = pd.read_csv(r'result_real.csv')

X = df_all.iloc[:, :-1].values
y = df_all.iloc[:, -1].values
y = y.astype('int')
X, y = shuffle(X, y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestClassifier(n_estimators=600,
                            criterion='entropy',
                            max_depth=18,
                            bootstrap=True,
                            random_state=0, max_features=10, )
rf.fit(X_train, y_train)
rf_predictions = cross_val_predict(rf, X_train, y_train, cv=5, method='predict_proba')

xgb = XGBClassifier(booster='gbtree',
                    objective= 'reg:squarederror',
                    eval_metric='error',
                    gamma = 0.01,
                    min_child_weight= 1.5,
                    max_depth= 24,
                    subsample= 0.8,
                    colsample_bytree= 1.0,
                    tree_method= 'exact',
                    learning_rate=0.01,
                    n_estimators=300,
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
xgb.fit(X_train, y_train)
xgb_predictions = cross_val_predict(xgb, X_train, y_train, cv=5, method='predict_proba')

stacking_train = np.column_stack((rf_predictions[:, 1], xgb_predictions[:, 1]))


lr_model = LogisticRegression()
lr_model.fit(stacking_train, y_train)

rf_test_predictions = rf.predict_proba(X_test)
xgb_test_predictions = xgb.predict_proba(X_test)
stacking_test = np.column_stack((rf_test_predictions[:, 1], xgb_test_predictions[:, 1]))
ensemble_predictions = lr_model.predict(stacking_test)

accuracy = accuracy_score(y_test, ensemble_predictions)
precision = precision_score(y_test, ensemble_predictions)
recall = recall_score(y_test, ensemble_predictions)
f1 = f1_score(y_test, ensemble_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

print(confusion_matrix(y_test, ensemble_predictions))
print('\n')
print(classification_report(y_test, ensemble_predictions))

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_confusion_matrix(y_test, ensemble_predictions, normalize=True)
plt.show()