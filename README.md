# Iris Dataset Classification Project

This repository contains two Python scripts for classifying the Iris dataset using different machine learning approaches: a multi-class classifier with K-Nearest Neighbors (KNN) and a binary classifier with Linear Regression.

## Files

### 1. `knn.py`
- **Description**: Implements a multi-class classifier using the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.
- **Features**:
  - Loads the Iris dataset and maps the target classes to integers.
  - Splits the dataset into 70% training and 30% test data for each class.
  - Trains a KNN classifier with `k=5` neighbors.
  - Evaluates the model using accuracy and a confusion matrix.
  - Visualizes the confusion matrix with annotations for better interpretation.

### 2. `binary_classifier.py`
- **Description**: Implements a binary classifier using Linear Regression to classify Iris flowers into two user-selected species.
- **Features**:
  - Allows the user to choose which class to exclude (Setosa, Versicolor, or Virginica).
  - Filters the dataset to keep only the selected classes and converts labels to binary values.
  - Splits the data into 70% training and 30% test sets.
  - Trains a Linear Regression model and converts predictions to binary values using a threshold of 0.5.
  - Evaluates the model using accuracy and plots actual vs. predicted values for visualization.

## Dataset
The scripts use the `iris.csv` dataset, which contains the following features:
- `sepal.length`
- `sepal.width`
- `petal.length`
- `petal.width`
- `variety` (target variable: Setosa, Versicolor, or Virginica)

## Requirements
To run the scripts, ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

You can install the required libraries using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib
