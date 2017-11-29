import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df=pd.read_csv(url,names=names)

df.replace('?',-99999, inplace=True)
df.drop(['Sample code number'],1,inplace=True)

x=np.array(df.drop(['Class'],1))
y=np.array(df['Class'])
test_size = 0.20

x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=test_size)

##clf = neighbors.KNeighborsClassifier()
clf=svm.SVC()

clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)
