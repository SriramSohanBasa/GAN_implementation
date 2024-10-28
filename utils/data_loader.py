import numpy as np
from tensorflow.keras.datasets import cifar10

def load_data():
    # Load CIFAR-10 data
    (x_train, _), (_, _) = cifar10.load_data()

    # Normalize the images to [-1, 1]
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5  # Rescale to [-1, 1]

    return x_train