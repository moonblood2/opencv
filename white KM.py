import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation, neighbors, svm
from sklearn.cluster import KMeans


url ='https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls'

df = pd.read_excel(url)
df.drop(['name','body'],1,inplace=True)
##df.drop(['pclass','survived','age','sibsp','parch','ticket','fare','cabin','boat','home.dest'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)


def convert_to_number(data):
    columns = data.columns.values

    for column in columns:
        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            #create uniqe set
            u_set = set(data[column])
            #set id to unique data
            s=0
            u_list={}
            for i in u_set:
                u_list[i]=s
                s+=1
                
            s=0
            for i in range(len(data[column])):
##                print(data[column][s])
##                print('to')
                data[column][s]=u_list[data[column][s]]
##                print(data[column][s])
                s+=1

    return data

            
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

##xs=np.array(df.drop('survived',1).astype(float))
##ys=np.array(df['survived'])

#x_train,x_test,y_train,y_test = cross_validation.train_test_split(xs,ys,test_size=0.2)

##clf = svm.SVC()
##clf = neighbors.KNeighborsClassifier()

X = np.array(df.drop(['survived'], 1).astype(float))
X= preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
    
