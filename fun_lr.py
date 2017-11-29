import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('ggplot')
xs = np.array([1,2,3,4,5], dtype=np.float64)
ys = np.array([5,4,6,5,6], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation='pos'):
    val = 1
    ys=[]
    xs=[]
    for i in range(hm):
        ys.append(val+random.randrange(-variance,variance))
        if correlation == 'pos':
            val += step
        elif correlation =='neg':
            val -= step
    for i in range(hm):
        xs.append(i)
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def my_LinearRegression(x,y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    xy_bar = np.mean(x*y)
    xx_bar = np.mean(x*x)
    m = (x_bar*y_bar-xy_bar)/(x_bar*x_bar-xx_bar)
    b = y_bar-m*x_bar
    return m,b

def squared_error(y_orig,y_line):
    return np.sum((y_orig-y_line)**2)

def r_squared(y_orig,y_line):
    y_mean_line = [np.mean(y_orig) for y in y_orig]
    squared_err_y_mean = squared_error(y_orig,y_mean_line)
    square_err_y_reqr = squared_error(y_orig,y_line)
    return 1-square_err_y_reqr/squared_err_y_mean



xs,ys=create_dataset(40,40,2,'no')
m,b = my_LinearRegression(xs,ys)
regr_line = [(m*x+b) for x in xs]
x_predict = 8
y_predict = x_predict*m+b
    
r_sqr = r_squared(ys,regr_line)
print(r_sqr)
plt.scatter(xs,ys)
plt.plot(xs,regr_line)
plt.scatter(x_predict,y_predict,s=40,color='g')
plt.show()
