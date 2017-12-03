import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.model_selection import train_test_split
import pdb

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']*100
plt.style.use('ggplot')

df = pd.read_csv('csv/noName - more.csv')
x = np.array(df[['salary','work','ot']])
##x = preprocessing.scale(x)
y = np.array(df['ot'])
##y = preprocessing.scale(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)

clf = svm.SVR()
clf.fit(x_train,y_train)

prediction = clf.predict(x_test)
pdb.set_trace()

pct = []

for i in range(len(y_test)):
    pct.append(int(abs(y_test[i]-prediction[i])))

print(pct)

ax = [i for i in range(max(pct))]
plt.hist(pct,ax)
plt.show()
