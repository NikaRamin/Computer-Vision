import numpy as np

'''
This is not good because the time complexity of train function is O(1) and the prediction function is O(n).
It's Ok if training takes time but we don't want the prediction function to be slow
'''


class nearestNeighbor:
    def __init__(self):
        # Initialize the nearestNeighbor class; no parameters required for initialization.
        pass
    
    def train(self, X, Y):
        """
        Train the nearest neighbor classifier with training data and labels.
        
        Parameters:
        X (numpy.ndarray): Training data, where each row represents a sample.
        Y (numpy.ndarray): Labels corresponding to the training data.
        """
        self.Xtr = X  # Store the training data
        self.Ytr = Y  # Store the training labels
    
    def predict(self, X):
        """
        Predict the labels for the given test data using the nearest neighbor classifier.
        
        Parameters:
        X (numpy.ndarray): Test data, where each row represents a sample for which we want to predict the label.
        
        Returns:
        numpy.ndarray: Predicted labels for the test data.
        """
        testCases = X.shape[0]  # Number of test samples
        labelPredict = np.zeros(testCases, dtype=self.Ytr.dtype)  # Array to store predicted labels
        
        for i in range(testCases):
            # Calculate the distance between the i-th test sample and all training samples
            # Using L1 norm (Manhattan distance) here
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            
            # Find the index of the training sample with the smallest distance to the i-th test sample
            min_index = np.argmin(distances)
            
            # Assign the label of the closest training sample to the i-th test sample
            labelPredict[i] = self.Ytr[min_index]
        
        return labelPredict
