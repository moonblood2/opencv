import matplotlib.pyplot as plt
import numpy as np
import csv

##x=np.arange(10)
##y=x*10
##
##x2=np.arange(10)
##y2=x2*5
##
##plt.plot(x,y,label='1')
##plt.plot(x2,y2, label = '2')
##plt.bar(x,y, label='1bar', color ='r')
##plt.xlabel('x')
##plt.ylabel('y')
##plt.legend()
##plt.title('my\ngraph')
##plt.show()
x=np.arange(20)
x = x.reshape(-1,2)
with open('text.txt','wt') as f:
    for i in x:
        f.write(str(i[0])+','+str(i[1])+'\n')

with open('text.txt','r') as f:
    data = csv.reader(f, delimiter=',')
    for i in data:
