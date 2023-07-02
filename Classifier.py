import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sklearn as sks
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
from sklearn import tree
data = 'test_ds.csv'
from sklearn.model_selection import train_test_split
df = pd.read_csv(data,sep=";")
X=df
print(set(X))
y = df['QUARTER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
encoder= ce.OrdinalEncoder(cols=['QUARTER', 'SURF_COND_DESC', 'REPORT_TYPE', 'WEATHER_DESC'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
print(set(y_train))
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test,y_pred_en)))
y_pred_train_en = clf_en.predict(X_train)
plt.rcParams['figure.figsize'] = (12, 8)
plt.plot(y_pred_train_en)
plt.show()
tree.plot_tree(clf_en.fit(X_train, y_train))
plt.show()
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)