from tensorflow.keras import layers, models

def build_generator(noise_dim):
    model = models.Sequential(name='Generator')
    
    # Fully connected layer to reshape noise vector
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 256)))  # Reshape to 8x8x256

    # Transposed Convolutions to upscale to 32x32
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'))

    return model