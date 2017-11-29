# Load libraries
import pandas,numpy,quandl
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

url ='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(url, names=names)


array=dataset.values
x=array[:,0:4]
y=array[:,4]
validation_size=0.20
seed=7
x_train,x_validation,y_train,y_validation = model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)

scoring = 'accuracy'

models=[]

models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results=[]
names=[]
    
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" %(name, cv_results.mean(), cv_results.std()))


