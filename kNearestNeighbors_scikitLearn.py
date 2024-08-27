
from sklearn.datasets import load_iris #classic dataset for classification
from sklearn.model_selection import train_test_split #splits the dataset into training and testing sets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score # calculates the accuracy of predictions

iris = load_iris()
x  , y = iris.data,iris.target 
"""
Splits the data into training (70%) and testing (30%) sets.
random_state=42 ensures reproducibility.
"""
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# initialize kNN model with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
#Traning model using the traning data
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test) #Uses the trained model to predict labels for the test data
"""
Compares the predicted labels (y_pred)
with the actual labels (y_test) to calculate the accuracy.
"""
accuracy = accuracy_score(y_test,y_pred,normalize=True)
print(f"Accuracy: {accuracy:.2f}")