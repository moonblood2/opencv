import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

##centers1 = [[10,10]]
##centers2 = [[6,6]]
##
##X1, _= make_blobs(n_samples=100,centers=centers1,random_state=0,cluster_std=0.5)
##X2, _= make_blobs(n_samples=200,centers=centers2,random_state=0,cluster_std=2.0)
##X = np.concatenate((X1,X2),axis=0)
##
##ms = MeanShift()
##ms.fit(X)
##colors = 10*['r','g','b','c','k','y','m']
##labels = ms.labels_
####ax = plt.figure().add_subplot(111, projection = '3d')
##for i in range(len(X)):
##    plt.scatter(X[i][0],X[i][1], c=colors[labels[i]])
##plt.show()


url ='https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls'

df = pd.read_excel(url)
orig_df = pd.DataFrame.copy(df)
df.drop(['name','body'],1,inplace=True)
##df.drop(['pclass','survived','age','sibsp','parch','ticket','fare','cabin','boat','home.dest'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            unique_elements = set(df[column])
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X= preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
centers = clf.cluster_centers_

orig_df['cluster_group']=np.nan

for i in range(len(df)):
    orig_df['cluster_group'].iloc[i] = labels[i]

survival_rate={}
for i in range(len(centers)):
    temp_df = orig_df[(orig_df['cluster_group']==float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate[i] = len(survival_cluster)/len(temp_df)

print(survival_rate)
    
