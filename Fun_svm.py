import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.metrics.pairwise import chi2_kernel
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv(url,names=names)

df.replace('?',np.nan,inplace=True)
df.dropna(inplace=True)
df.drop(['Sample code number'],1,inplace=True)

xs=np.array(df.drop(['Class'],1))
ys=np.array(df['Class'])

x_train,x_test,y_train,y_test = cross_validation.train_test_split(xs,ys,test_size=0.20)

k=chi2_kernel(x_train, gamma=0.5)
clf=svm.SVC(kernel=chi2_kernel).fit(k,y_train)

##accuracy = clf.score(x_test,y_test)
##print(accuracy)

predict = clf.predict(x_test)
print(predict)


