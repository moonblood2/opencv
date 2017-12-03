import pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']*100
plt.style.use('ggplot')

df = pd.read_csv('csv/noName - more.csv')
##df = pd.read_csv('csv/noName - faker.csv')
##df = pd.read_csv('csv/noName.csv')
x = np.array(df[['work','ot']])
x = preprocessing.scale(x)

k=8

clf = AgglomerativeClustering(n_clusters=k)
clf.fit(x)
labels = clf.labels_

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

plt.show()

clf = MeanShift()
clf.fit(x)
labels = clf.labels_

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

plt.show()


clf = KMeans(n_clusters=k)
clf.fit(x)
labels = clf.labels_

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

plt.show()

clf = DBSCAN()
clf.fit(x)
labels = clf.labels_

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

plt.show()
