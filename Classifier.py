import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sklearn as sks
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
from sklearn import tree
data = 'car_evaluation.csv'
from sklearn.model_selection import train_test_split
df = pd.read_csv(data, header=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#ce.OrdinalEncoder(cols=['QUARTER', 'SURF_COND_DESC', 'REPORT_TYPE', 'WEATHER_DESC'])
encoder= sks.OrdinalEncoder(cols=['QUARTER', 'SURF_COND_DESC', 'REPORT_TYPE', 'WEATHER_DESC'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state =42)
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test,y_pred_en)))
y_pred_train_en = clf_en.predict(X_train)
y_pred_train_en(plt.figure(figsize=(12,8)))
tree.plot_tree(clf_en.fit(X_train, y_train))
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)
