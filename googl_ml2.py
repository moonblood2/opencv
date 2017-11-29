from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5)
print(len(y_train))
print(len(y_test))
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
acc = clf.score(x_test,y_test)
acc2 = accuracy_score(clf.predict(x_test),y_test)
print(acc)
print(acc2)
