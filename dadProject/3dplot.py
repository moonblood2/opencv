import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import pandas as pd
from sklearn import preprocessing
style.use("ggplot")

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']*100

df = pd.read_csv('csv/noName - more.csv')
x = np.array(df[['salary','work','ot']])
x = preprocessing.scale(x)

##clf = MeanShift()
clf = KMeans(n_clusters=10)
clf.fit(x)
labels = clf.predict(x)

''' 3d plot '''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

''' raw data '''
##ax.scatter(x[:,0], x[:,1], x[:,2], c='black', marker='o')

''' clustered data '''
for i in range(len(x)):
    ax.scatter(x[i][0], x[i][1], x[i][2], c=colors[labels[i]], marker='o')


ax.set_xlabel('Base salary')
ax.set_ylabel('Main work\'s amounts')
ax.set_zlabel('Over Time hours')
              
plt.show()
