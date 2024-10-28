# utils/image_utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
from config import GENERATED_IMAGES_DIR

def save_images(generator, epoch, latent_dim, n_examples=64):
    """Generate and save images."""
    # Generate random latent points
    latent_points = np.random.normal(0, 1, (n_examples, latent_dim))
    # Generate images
    X = generator(latent_points).cpu().detach().numpy()
    X = (X + 1) / 2.0  # Scale images from [-1,1] to [0,1]
    
    # Setup plot dimensions
    n = int(np.sqrt(n_examples))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(np.transpose(X[idx], (1, 2, 0)))
            axes[i, j].axis('off')
            idx += 1
    plt.tight_layout()

    # Save the generated plot
    if not os.path.exists(GENERATED_IMAGES_DIR):
        os.makedirs(GENERATED_IMAGES_DIR)
    plt.savefig(os.path.join(GENERATED_IMAGES_DIR, f'generated_plot_epoch{epoch}.png'))
    plt.close()

def generate_and_show_images(generator, latent_dim, n_examples=25):
    # Generate images
    latent_points = np.random.normal(0, 1, (n_examples, latent_dim))
    X = generator.predict(latent_points)
    X = (X + 1) / 2.0  # Scale from [-1,1] to [0,1]

    # Plot images
    n = int(np.sqrt(n_examples))
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    idx = 0
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(X[idx])
            axes[i, j].axis('off')
            idx += 1
    plt.tight_layout()
    plt.show()