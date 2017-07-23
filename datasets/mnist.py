import sys
import os
import numpy as np
from keras.datasets import mnist

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train/255.0, X_test/255.0

    # Keep some validation data
    X_train, X_val = X_train[:-5000], X_train[-5000:]
    y_train, y_val = y_train[:-5000], y_train[-5000:]

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_val = X_val.reshape(X_val.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    return np.float32(X_train), y_train, np.float32(X_val), y_val, np.float32(X_test), y_test
