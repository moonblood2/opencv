from sklearn import tree, neighbors, svm
import numpy as np
from random import shuffle

x=[]
l=[]
test_x=[]
test_l=[]

for i in range(10000):
    rnd = np.random.randint(0, high=2)
    if rnd == 0:
        y=[]
        y.append(np.random.randint(100,high=200))
        y.append(np.random.randint(10,high=25))
        x.append(y)
        l.append(1)
    else:
        y=[]
        y.append(np.random.randint(150,high=250))
        y.append(np.random.randint(15,high=35))
        x.append(y)
        l.append(2)

test_x = x[-100:]
test_l = l[-100:]

sum_acc=[]
for i in range(50):
##    clf = tree.DecisionTreeClassifier()
##    clf = neighbors.KNeighborsClassifier()
    clf = svm.SVC()
    clf.fit(x[:900],l[:900])
    acc=clf.score(test_x,test_l)
    sum_acc.append(acc)
print(np.mean(sum_acc))

