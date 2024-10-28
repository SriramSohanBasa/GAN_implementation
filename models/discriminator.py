from tensorflow.keras import layers, models

def build_discriminator(img_shape):
    model = models.Sequential(name='Discriminator')

    # Convolutional Layers
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))  # Output probability [0,1]

    return model