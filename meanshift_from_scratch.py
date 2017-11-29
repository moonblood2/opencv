import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

radius = 4

# initiate initial centroid

centroids={}
for i in range(len(x)):
    centroids[i]=x[i]

# go tru every feature
while True:
    new_centroids = []

    for i in centroids:
        dist_set=[]
        for feature in x:
            dist = np.linalg.norm(feature-centroids[i])
            if dist < radius:
                dist_set.append(feature)
        new_centroids.append(np.average(dist_set,axis=0))
        
    prev_centroids = dict(centroids)
    centroids = {}

    for i in range(len(x)):
        centroids[i] = new_centroids[i]

    if np.average(list(prev_centroids.values()))== np.average(list(centroids.values())):
        break

array_centroids = np.array(list(centroids.values()))
array_centroids = np.unique(array_centroids,axis=0)

for i in range(len(centroids)):
    plt.scatter(x[i][0],x[i][1])

for i in range(len(array_centroids)):
    plt.scatter( array_centroids[i][0],array_centroids[i][1], s=150 )

plt.show()





















