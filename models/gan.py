
from tensorflow.keras import models
from config import LEARNING_RATE, BETA_1
from tensorflow.keras.optimizers import Adam

def build_gan(generator, discriminator):
    # Ensure the discriminator is not trainable when compiling the GAN
    discriminator.trainable = False

    # Build the GAN model
    gan_input = generator.input
    gan_output = discriminator(generator.output)
    gan_model = models.Model(gan_input, gan_output)
    
    # The discriminator remains uncompiled here; it will be compiled separately
    return gan_model