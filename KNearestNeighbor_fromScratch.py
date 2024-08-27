# Knn algoritm without using scikit learn

import numpy as np
from collections import Counter # To count occurrences and find the most common elements

def euclideanDistance(a,b):
    return np.sqrt(np.sum((a-b)**2))

class Knn:
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
    def _predict(self,x):
        #computing distances
        distances = [euclideanDistance(x,x_train) for x_train in self.X_train]
        #extracting k nearest neighbors
        k_nearest = np.argsort(distances)[:self.k]
        #ectracting the labels of those neighbors
        k_nearest_labels = [self.Y_train[i] for i in k_nearest]
        #determining the most common label
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]

np.random.seed() #or np.random.seed(42) for reproducibility
X_train = np.random.rand(1000, 10)  # 1000 samples, 10 features
y_train = np.random.randint(0, 2, 1000)  # Binary labels (0 or 1) for 1000 samples

X_test = np.random.rand(200, 10)  # 200 test samples, 10 features
y_test = np.random.randint(0, 2, 200)  # Binary labels (0 or 1) for 200 samples


knn = Knn(k=3)
knn.fit(X_train, y_train)


predictions = knn.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy of the predictions: {accuracy:.2f}")
