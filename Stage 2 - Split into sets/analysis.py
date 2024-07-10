import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Print the shapes of the new datasets
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Print the proportions of samples per class in the training set
train_class_proportions = pd.Series(y_train).value_counts(normalize=True)
print("Proportion of samples per class in train set:")
print(train_class_proportions)
