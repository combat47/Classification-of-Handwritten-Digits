import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

# Load the MNIST dataset
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Use the first 6000 rows of the dataset
x = x[:6000]
y = y[:6000]

# Reshape the features array to 2D array with n rows and m columns
x_flattened = x.reshape(x.shape[0], -1)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_flattened, y, test_size=0.3, random_state=40)

# Normalize the data
normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)

# Function to fit, predict and evaluate the model
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(target_test, y_pred)
    return score

# List of models to evaluate
models = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=40),
    LogisticRegression(random_state=40, max_iter=1000),
    RandomForestClassifier(random_state=40)
]

# Evaluate each model with normalized data
accuracies = {}
for model in models:
    score = fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
    model_name = model.__class__.__name__
    accuracies[model_name] = score
    print(f'Model: {model_name}()\nAccuracy: {score:.4f}\n')

# Determine if normalization had a positive impact by comparing to unnormalized results
unnormalized_accuracies = {
    model.__class__.__name__: fit_predict_eval(model, x_train, x_test, y_train, y_test)
    for model in models
}
positive_impact = any(accuracies[model] > unnormalized_accuracies[model] for model in accuracies)

# Find the best two models
sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
best_two_models = sorted_accuracies[:2]

# Answer the questions
print(f'The answer to the 1st question: {"yes" if positive_impact else "no"}')
print(f'The answer to the 2nd question: {best_two_models[0][0]}-{best_two_models[0][1]:.3f}, {best_two_models[1][0]}-{best_two_models[1][1]:.3f}')
