import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

colors = 10*["g","r","c","b","k"]

df = pd.read_csv('csv/noName.csv')
data = np.array(df[['work','salary']])
s=np.array(df['salary'])
w=np.array(df['work'])
o=np.array(df['ot'])
c=np.array(df['credit'])

clf = KMeans(n_clusters=1)
clf.fit(data)

plt.scatter(w,o)

for centroid in clf.cluster_centers_:
    plt.scatter(centroid[0],centroid[1],marker="o", color="k", s=1, linewidths=5)

plt.show()

##
##print('kmean',clf.cluster_centers_)

