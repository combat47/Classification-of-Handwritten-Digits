import tensorflow as tf
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Reshape the features array to 2D array with n rows and m columns
# n = number of images, m = number of pixels in each image (28*28 = 784)
x_train_flattened = x_train.reshape(x_train.shape[0], -1)

# Print the information about the dataset
print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {x_train_flattened.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {x_train_flattened.min()}, max: {x_train_flattened.max()}")
