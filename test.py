import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Generator
from config import LATENT_DIM

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the generator model
generator = Generator(nz=LATENT_DIM, ngf=64, nc=3).to(device)
generator.load_state_dict(torch.load("models/generator_final.pth", map_location=device))
generator.eval()  # Set to evaluation mode

# Function to generate and display images
def generate_images(generator, num_images=8, latent_dim=LATENT_DIM):
    # Generate random noise
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    
    # Generate images from noise
    with torch.no_grad():  # Disable gradient calculation
        generated_images = generator(noise).cpu()
    
    # Convert generated images to numpy array and scale to [0, 1]
    generated_images = (generated_images + 1) / 2.0
    
    # Display the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axes[i].imshow(generated_images[i].permute(1, 2, 0).numpy())
        axes[i].axis('off')
    plt.show()

# Generate and display images
generate_images(generator, num_images=8)