import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.style.use('ggplot')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']*100

df = pd.read_csv('csv/noName - more.csv')
##df = pd.read_csv('csv/noName - faker.csv')
##df = pd.read_csv('csv/noName.csv')
x = np.array(df[['work','ot']])
x = preprocessing.scale(x)

f = np.array([[7,44]])

clf = KMeans(n_clusters=10)
clf.fit(x)
labels = clf.predict(x)

plt.scatter(x[:,0],x[:,1], c='black')
plt.ylabel('Main work\'s amounts')
plt.xlabel('Over Time hours')
plt.show()
