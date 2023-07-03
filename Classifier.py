import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn as sks
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz
warnings.filterwarnings('ignore')
from sklearn import tree
#Selecting the file that contains the dataset
data = 'test_ds.csv'
from sklearn.model_selection import train_test_split
#Reading the file and using the time library in order to measure the time it takes.
#Also loading the dataset into the python dataframe
start_time = time.time()
df = pd.read_csv(data,sep=";")
end_time = time.time()
read_time = end_time - start_time 
print("Time to read file: {:.4f} seconds".format(read_time))
#Splitting the dataset onto the training set and the test set
X=df
y = df['QUARTER'].astype(str) + '_' + df["REPORT_TYPE"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#Specifying the columns that will be used for the encoding 
encoder= ce.OrdinalEncoder(cols=['QUARTER', 'SURF_COND_DESC', 'REPORT_TYPE', 'WEATHER_DESC'])
#Performing the encoiding for the training and testing sets
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
#Building the Decision Tree Classifier and measuring the time it takes
start_time = time.time()
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
#Fitting/Training the the Classifier using the training set 
clf_en.fit(X_train, y_train)
end_time = time.time()
build_time = end_time - start_time
print("Time to build the model: {:.4f} seconds".format(build_time))
#Predicting the labels for the test set and measuring the time it takes
start_time = time.time()
y_pred_en = clf_en.predict(X_test)
end_time = time.time()
use_time = end_time - start_time
print("Time to use the model: {:.4f} seconds".format(use_time))
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test,y_pred_en)))
y_pred_train_en = clf_en.predict(X_train)
#Plotting the Classifier and the encoded predictions
plt.rcParams['figure.figsize'] = (12, 8)
plt.plot(y_pred_train_en)
plt.show()
#plt.rcParams['figure.figsize'] = (40, 30)
#tree.plot_tree(clf_en.fit(X_train, y_train),filled=True)
#plt.show()
dot_data = tree.export_graphviz(clf_en, out_file=None, filled=True, feature_names=list(X_train.columns),
                                class_names=clf_en.classes_)
# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph.render("decision_tree_graphivz")
#Calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)