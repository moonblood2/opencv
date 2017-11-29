# Load libraries
import pandas,quandl,math
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

def compat(data):
    data=math.ceil(data*100)
    return data


dataset=quandl.get('WIKI/GOOGL')

dataset=dataset[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

dataset['HL_PCT'] = (dataset['Adj. High']-dataset['Adj. Low'])/dataset['Adj. Close']*100.0
dataset['PCT_change'] = (dataset['Adj. Close']-dataset['Adj. Open'])/dataset['Adj. Close']*100.0

dataset = dataset[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
##
##dataset['HL_PCT']=dataset['HL_PCT'].apply(compat)
##dataset['PCT_change']=dataset['PCT_change'].apply(compat)
##dataset['Adj. Close']=dataset['Adj. Close'].apply(compat)

forecast_col = 'Adj. Close'
dataset.fillna(value=-99999, inplace=True)
forecast_out= int(len(dataset)*0.01)

dataset['label']=dataset[forecast_col].shift(-forecast_out)

x_forecast = np.array(dataset.drop(['label'], 1))
x_forecast = x_forecast[-forecast_out:]
x_forecast = preprocessing.scale(x_forecast)

y_check = np.array(dataset['Adj. Close'])
y_check = y_check[-forecast_out:]

dataset.dropna(inplace=True)

x = np.array(dataset.drop(['label'], 1))
x = preprocessing.scale(x)
y = np.array(dataset['label'])


##
##array=dataset.values
##
##x=array[:,0:3]
##y=array[:,3]
##
#constant
validation_size = 0.20
seed = 7

#spliting dataset
x_train,x_validation,y_train,y_validation = model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)

clf = svm.SVR()
clf.fit(x_train,y_train)
accuracy = clf.score(x_validation,y_validation)
print("SVM %s" %accuracy)

clf = LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_validation,y_validation)
print("LR %s" %accuracy)


forecast = clf.predict(x_forecast)
print(forecast)


##scoring = 'accuracy'
##
##models=[]
##
##models.append(('LR',LogisticRegression()))
##models.append(('LDA',LinearDiscriminantAnalysis()))
##models.append(('KNN',KNeighborsClassifier()))
##models.append(('CART',DecisionTreeClassifier()))
##models.append(('NB',GaussianNB()))
##models.append(('SVM',SVC()))
##models.append(('CLF',LinearRegression()))
##
##results=[]
##names=[]
##
##for name, model in models:
##    kfold = model_selection.KFold(n_splits=10, random_state=seed)
##    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
##    results.append(cv_results)
##    names.append(name)
##    print("%s: %f (%f)" %(name, cv_results.mean(), cv_results.std()))

