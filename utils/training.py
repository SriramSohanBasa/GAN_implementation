import numpy as np
from tqdm import tqdm
from tensorflow.keras import optimizers
from config import NOISE_DIM, LEARNING_RATE, EPOCHS, BATCH_SIZE, SAMPLE_INTERVAL
from .image_utils import save_images

def train(generator, discriminator, gan, x_train):
    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizers.Adam(LEARNING_RATE),
                          metrics=['accuracy'])

    # Compile the GAN model
    discriminator.trainable = False  # Freeze discriminator weights during generator training
    gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(LEARNING_RATE))

    # Labels for real and fake images
    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    for epoch in tqdm(range(EPOCHS)):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of real images
        idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
        real_imgs = x_train[idx]

        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        # Generate noise
        noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))

        # Train the generator (trying to fool the discriminator)
        g_loss = gan.train_on_batch(noise, real)

        # Print the progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Save generated image samples
        if (epoch + 1) % SAMPLE_INTERVAL == 0:
            save_images(generator, epoch + 1, NOISE_DIM)
            
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    print("Models saved: generator_model.h5 and discriminator_model.h5")