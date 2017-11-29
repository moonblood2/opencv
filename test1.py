import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

x=[]
c=0
stat=[]
rng = 1
for i in range(1000):
    rnd = np.random.randint(0,high=100)
    if rnd <= rng:
        c+=1
        rng-=1
        x.append(c)
    elif rnd > rng and c>0:
        c-=1
        rng+=1
        x.append(c)
print(max(x))
##fit = stats.norm.pdf(x, np.mean(x), np.std(x))
##plt.plot(x,fit,'-o')
plt.plot(x)
##plt.hist(x,normed=True)
plt.show()  
