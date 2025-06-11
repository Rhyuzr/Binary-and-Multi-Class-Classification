import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = pd.read_csv('iris.csv')

# Map the target classes to integers
iris['variety'] = iris['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

# Separate each class
setosa = iris[iris['variety'] == 0]
versicolor = iris[iris['variety'] == 1]
virginica = iris[iris['variety'] == 2]

# Define a function to split data manually into 70% train and 30% test
def split_data(class_data):
    split_index = int(0.7 * len(class_data))
    train_data = class_data[:split_index]
    test_data = class_data[split_index:]
    return train_data, test_data

# Split each class into training and test sets
setosa_train, setosa_test = split_data(setosa)
versicolor_train, versicolor_test = split_data(versicolor)
virginica_train, virginica_test = split_data(virginica)

# Combine the training and test data from each class
train_data = pd.concat([setosa_train, versicolor_train, virginica_train])
test_data = pd.concat([setosa_test, versicolor_test, virginica_test])

# Separate features and labels
X_train = train_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y_train = train_data['variety']
X_test = test_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y_test = test_data['variety']

# Define and train the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the multi-class classifier using KNN:", accuracy)

# Plot the confusion matrix using matplotlib
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title(f"Confusion Matrix of KNN Classifier (Accuracy: {accuracy:.2f})")

# Add text annotations
class_names = ['Setosa', 'Versicolor', 'Virginica']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Annotate each cell with the respective count
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

plt.show()
