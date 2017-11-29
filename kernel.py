from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
X = [[0, 1,0.9], [1, 0,0.2], [.2, .8,0.3], [.7, .3,.3]]
y = [0, 1, 0, 1]
K = chi2_kernel(X, gamma=.5)
svm = SVC(kernel='precomputed').fit(K, y)

print(svm.predict(K))
