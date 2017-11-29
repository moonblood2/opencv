from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import os
from six.moves import cPickle as pickle

os.chdir('C:/Users/ASUS/Desktop/deeplearn')

with open('notMNIST.pickle','rb') as f:
    dataset = pickle.load(f)
x = dataset['test_dataset']
x = x.reshape(x.shape[0],28*28)
y = dataset['test_labels']

test_size = 0.2

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=test_size)

clf = neighbors.KNeighborsClassifier()
##clf = LinearRegression()
##clf = LogisticRegression()

clf.fit(x_train,y_train)

acc = clf.score(x_test,y_test)

print(acc)
