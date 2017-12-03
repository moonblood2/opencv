import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

df = pd.read_csv('csv/noName - faker.csv')
##df = pd.read_csv('csv/noName.csv')
x = np.array(df[['work','ot']])


f = np.array([[7,44]])

clf = KMeans(n_clusters=10)
clf.fit(x)
labels = clf.predict(x)

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1], color=colors[labels[i]])

##dist0=[]
##dist1=[]
##dist2=[]
##dist3=[]
##dist4=[]
##dist5=[]
##dist6=[]
##dist7=[]
##dist8=[]
##dist9=[]

dist=[]

for i in range(len(x)):
    point = x[i]
    label = labels[i]
    centroid = clf.cluster_centers_[label]

    if label == i:
        dist[i].append(np.linalg.norm(centroid-point))
'''
for i in range(len(dist)):
    dist[i] = np.mean(dist[i])
'''    

##for i in range(len(x)):
##    point = x[i]
##    label = labels[i]
##    centroid = clf.cluster_centers_[label]
##    
##    if label == 0:
##        dist0.append(np.linalg.norm(centroid-point))
##    if label == 1:
##        dist1.append(np.linalg.norm(centroid-point))
##    if label == 2:
##        dist2.append(np.linalg.norm(centroid-point))
##    if label == 3:
##        dist3.append(np.linalg.norm(centroid-point))
##    if label == 4:
##        dist4.append(np.linalg.norm(centroid-point))
##    if label == 5:
##        dist5.append(np.linalg.norm(centroid-point))
##    if label == 6:
##        dist6.append(np.linalg.norm(centroid-point))
##    if label == 7:
##        dist7.append(np.linalg.norm(centroid-point))
##    if label == 8:
##        dist8.append(np.linalg.norm(centroid-point))
##    if label == 9:
##        dist9.append(np.linalg.norm(centroid-point))
##    
##dist = []
##dist.append(np.mean(dist0))
##dist.append(np.mean(dist1))
##dist.append(np.mean(dist2))
##dist.append(np.mean(dist3))
##dist.append(np.mean(dist4))
##dist.append(np.mean(dist5))
##dist.append(np.mean(dist6))
##dist.append(np.mean(dist7))
##dist.append(np.mean(dist8))
##dist.append(np.mean(dist9))

##for i in range(len(clf.cluster_centers_)):
##    plt.scatter(clf.cluster_centers_[i][0], clf.cluster_centers_[i][1], facecolors='none', edgecolors='r', s=dist[i]*500)
##
##plt.show()
##



    
