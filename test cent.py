import numpy as np
import math
import matplotlib.pyplot as plt

data = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

k=2

centroids={}
interation=300
tol=0.001

centroids={}
for i in range(k):
    centroids[i] = data[i]
    
for i in range(interation):
    classification={}
    for i in range(k):
        classification[i] = []
        
    for feature in data:
        dist = [np.linalg.norm(feature-centroids[cent]) for cent in centroids]  
        min_dist = dist.index(min(dist))
        classification[min_dist].append(feature)

    prev_centroids = dict(centroids)

    for c in classification:
        centroids[c]=np.average(classification[c],axis=0)

    s=[np.linalg.norm(centroids[c]-prev_centroids[c]) for c in range(k)]

    if sum(s) <tol:
        break

data_2_pred = np.array([[1.1, 2.1],
              [1.6, 1.9],
              [5.1, 7 ],
              [7, 7],
              [1.1, 0.4],
              [7,10]])
result=[]
for feature in data_2_pred:
    disc=[np.linalg.norm(feature-centroids[cent]) for cent in centroids]  
    min_dist = disc.index(min(disc))
    result.append(min_dist)

print(result)

plt.scatter(centroids[0][0],centroids[0][1],color='r',s=150)
plt.scatter(centroids[1][0],centroids[1][1],color='b',s=150)

for i in range(len(result)):
    if result[i]==0:
        plt.scatter(data_2_pred[i][0],data_2_pred[i][1],color='r')
    elif result[i]==1:
        plt.scatter(data_2_pred[i][0],data_2_pred[i][1],color='b')

plt.show()
