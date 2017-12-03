import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']*100
plt.style.use('ggplot')

df = pd.read_csv('csv/noName - more.csv')
##df = pd.read_csv('csv/noName - faker.csv')
##df = pd.read_csv('csv/noName.csv')
x = np.array(df[['work','ot']])
x = preprocessing.scale(x)

f = np.array([[7,44]])

clf = KMeans(n_clusters=10)
clf.fit(x)
labels = clf.predict(x)

##for i in range(len(x)):
##    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

dist={}

for i in range(len(clf.cluster_centers_)):
    dist[i] = []

for i in range(len(x)):
    dist[labels[i]].append(np.linalg.norm(x[i]-clf.cluster_centers_[labels[i]]))

for i in range(len(dist)):
    dist[i] = np.mean(dist[i])

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color='')

##for i in range(len(dist)):
##    dist[i] = np.std(dist[i])
##for i in range(len(clf.cluster_centers_)):
##    plt.scatter(clf.cluster_centers_[i][0], clf.cluster_centers_[i][1], facecolors='none', edgecolors='r', s=dist[i]*1100)

for i in range(len(dist)):
    dist[i] = np.mean(dist[i])
for i in range(len(clf.cluster_centers_)):
    plt.scatter(clf.cluster_centers_[i][0], clf.cluster_centers_[i][1],marker = '^', facecolors='white', edgecolors='black', s=20)
for i in range(len(clf.cluster_centers_)):
    plt.scatter(clf.cluster_centers_[i][0], clf.cluster_centers_[i][1], facecolors='none', edgecolors='r', s=dist[i]*5000)
plt.ylabel('Main work\'s amounts')
plt.xlabel('Over Time hours')

new_data = [-0.05,2.2]
new_label=clf.predict(np.array(new_data).reshape(1, -1))
new_centroid = clf.cluster_centers_[new_label]
plt.scatter(new_data[0],new_data[1], marker = '+', color = 'black')
plt.plot([-0.05,new_centroid[0][0]],[2.2,new_centroid[0][1]])

new_data2 = [-0.5,1.7]
new_label2=clf.predict(np.array(new_data2).reshape(1, -1))
new_centroid2 = clf.cluster_centers_[new_label2]
plt.scatter(new_data2[0],new_data2[1], marker = '+', color = 'black')
plt.plot([new_data2[0],new_centroid2[0][0]],[new_data2[1],new_centroid2[0][1]])

plt.show()
