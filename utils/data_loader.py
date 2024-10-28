
from tensorflow.keras.datasets import cifar10

def load_real_samples():
    (trainX, _), (_, _) = cifar10.load_data()
    # Convert from unsigned ints to floats
    X = trainX.astype('float32')
    # Scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X