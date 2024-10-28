import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from config import GENERATED_IMAGES_DIR

def save_images(generator: Model, epoch: int, noise_dim: int, num_examples: int = 25):
    if not os.path.exists(GENERATED_IMAGES_DIR):
        os.makedirs(GENERATED_IMAGES_DIR)

    noise = np.random.normal(0, 1, (num_examples, noise_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images to [0,1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set up the grid
    grid_row = grid_col = int(np.sqrt(num_examples))
    fig, axs = plt.subplots(grid_row, grid_col, figsize=(grid_col, grid_row))
    count = 0
    for i in range(grid_row):
        for j in range(grid_col):
            axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig(os.path.join(GENERATED_IMAGES_DIR, f"generated_images_epoch_{epoch}.png"))
    plt.close()

def generate_and_show_images(generator: Model, noise_dim: int, num_examples: int = 16):
    noise = np.random.normal(0, 1, (num_examples, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0,1]

    grid_row = grid_col = int(np.sqrt(num_examples))
    fig, axs = plt.subplots(grid_row, grid_col, figsize=(grid_col, grid_row))
    count = 0
    for i in range(grid_row):
        for j in range(grid_col):
            axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.show()