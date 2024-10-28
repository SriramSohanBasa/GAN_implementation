from models import build_generator, build_discriminator
from utils import load_data, train, generate_and_show_images
from config import IMG_SHAPE, NOISE_DIM

def main():
    # Load data
    x_train = load_data()

    # Build models
    generator = build_generator(NOISE_DIM)
    discriminator = build_discriminator(IMG_SHAPE)

    # Build and compile the GAN model
    discriminator.trainable = False  # Ensure discriminator is not trainable in GAN
    from tensorflow.keras import Input, Model
    gan_input = Input(shape=(NOISE_DIM,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)

    # Train the GAN
    train(generator, discriminator, gan, x_train)

    # Generate and display images after training
    generate_and_show_images(generator, NOISE_DIM)

if __name__ == '__main__':
    main()