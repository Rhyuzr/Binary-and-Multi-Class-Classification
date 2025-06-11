import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load the Iris dataset
iris = pd.read_csv('iris.csv')

# User input to choose which class to exclude
s = 'setosa'
vi = 'virginica'
ve = 'versicolor'

a = ''
while a not in ['s', 've', 'vi']:
    a = input("Choose which one you want to drop (s for setosa, ve for versicolor, or vi for virginica): ").lower()

# Define classes to keep based on user input
if a == 's':
    f1 = 'Versicolor'
    f2 = 'Virginica'
elif a == 've':
    f1 = 'Setosa'
    f2 = 'Virginica'
else:
    f1 = 'Setosa'
    f2 = 'Versicolor'

# Filter data to keep only the selected classes
filtered_data = [(sl, sw, pl, pw, variety) for sl, sw, pl, pw, variety in zip(
    iris['sepal.length'], iris['sepal.width'], iris['petal.length'], iris['petal.width'], iris['variety']
) if variety in [f1, f2]]

# Unzip the filtered data to separate lists
sl, sw, pl, pw, variety = zip(*filtered_data)

# Convert lists back to the appropriate format
sl = list(sl)
sw = list(sw)
pl = list(pl)
pw = list(pw)
variety = list(variety)

# Convert variety to binary values based on the chosen classes
for i in range(len(variety)):
    if variety[i] == f1:
        variety[i] = 0
    elif variety[i] == f2:
        variety[i] = 1

# Manually split data into 70% of each class for training, and 30% for testing
setosa_data = [(sl[i], sw[i], pl[i], pw[i], variety[i]) for i in range(len(variety)) if variety[i] == 0]
versicolor_data = [(sl[i], sw[i], pl[i], pw[i], variety[i]) for i in range(len(variety)) if variety[i] == 1]

# Split setosa and versicolor data
split_setosa = int(0.7 * len(setosa_data))
split_versicolor = int(0.7 * len(versicolor_data))

setosa_train = setosa_data[:split_setosa]
setosa_test = setosa_data[split_setosa:]

versicolor_train = versicolor_data[:split_versicolor]
versicolor_test = versicolor_data[split_versicolor:]

# Combine training and testing data
train_data = setosa_train + versicolor_train
test_data = setosa_test + versicolor_test

# Unzip the combined data into separate features and labels
sl_train, sw_train, pl_train, pw_train, variety_train = zip(*train_data)
sl_test, sw_test, pl_test, pw_test, variety_test = zip(*test_data)

# Convert lists back to the appropriate format
X_train = pd.DataFrame({
    'sl': sl_train,
    'sw': sw_train,
    'pl': pl_train,
    'pw': pw_train
})

y_train = pd.DataFrame(variety_train, columns=['variety'])

X_test = pd.DataFrame({
    'sl': sl_test,
    'sw': sw_test,
    'pl': pl_test,
    'pw': pw_test
})

y_test = pd.DataFrame(variety_test, columns=['variety'])

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions
y_pred = lin_reg.predict(X_test)

# Convert predictions to binary (0 or 1) using a threshold of 0.5
y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy of the binary classifier using linear regression:", accuracy)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual values')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', marker='x', label='Predicted values')
plt.xlabel("Sample index")
plt.ylabel("Class (0 or 1)")
plt.title(f"Comparison of Actual vs Predicted Classes (Accuracy: {accuracy:.2f})")
plt.legend()
plt.show()
