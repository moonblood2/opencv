import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

class cluster_master:
    def __ini__(self,k=2,tolerance=0.001, max_inter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_inter = max_inter

    def cenfinder(data):

        self.centroids={}

        for i in range(self.k):
            self.centroids[i]=data[i]

        self.classification_set=[]
        for featureset in data:
            self.dist=[]
            for centroid in range(self.k):
                self.dist.append(np.linalg.norm(self.centroids[centroid]-featureset[data]))
            self.classification_set.append(self.dist.index(min(self.dist)))
