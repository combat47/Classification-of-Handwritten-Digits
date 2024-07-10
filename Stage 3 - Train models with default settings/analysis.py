import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Use the first 6000 rows of the dataset
x = x[:6000]
y = y[:6000]

# Reshape the features array to 2D array with n rows and m columns
# n = number of images, m = number of pixels in each image (28*28 = 784)
x_flattened = x.reshape(x.shape[0], -1)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_flattened, y, test_size=0.3, random_state=40)

# Function to fit, predict and evaluate the model
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(target_test, y_pred)
    print(f'Model: {model}\nAccuracy: {score:.4f}\n')
    return score

# List of models to evaluate
models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=40),
    LogisticRegression(random_state=40, max_iter=1000),
    RandomForestClassifier(random_state=40)
]

# Dictionary to store the accuracies
accuracies = {}

# Evaluate each model
for model in models:
    score = fit_predict_eval(model, x_train, x_test, y_train, y_test)
    model_name = model.__class__.__name__
    accuracies[model_name] = score

# Find the best model
best_model = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model]

# Print the best model and its accuracy
print(f'The answer to the question: {best_model} - {best_accuracy:.3f}')
