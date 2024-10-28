# training.py

import torch
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os

def train(generator, discriminator, device, dataloader, optimizerG, optimizerD, criterion, epochs, latent_dim, save_interval=2):
    """
    Trains the Generator and Discriminator models.

    Args:
        generator (torch.nn.Module): The Generator model.
        discriminator (torch.nn.Module): The Discriminator model.
        device (torch.device): The device to run the training on.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizerG (torch.optim.Optimizer): Optimizer for the Generator.
        optimizerD (torch.optim.Optimizer): Optimizer for the Discriminator.
        criterion (torch.nn.Module): Loss function.
        epochs (int): Number of training epochs.
        latent_dim (int): Dimension of the latent vector.
        save_interval (int, optional): Interval (in epochs) to save models and images. Defaults to 2.
    """

    # Ensure the directories for saving models and images exist
    model_dir = 'models'
    image_dir = 'generated_images'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Watch the models for logging gradients and parameters
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    # Labels for real and fake data
    real_label = 0.9  # Label smoothing for real labels
    fake_label = 0.1  # Label smoothing for fake labels

    # Fixed noise for consistent image generation across epochs
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    print("Starting Training Loop...")
    # Start training loop
    for epoch in range(1, epochs + 1):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Discriminator
            ############################
            discriminator.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            output_real = discriminator(real_images).view(-1)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Generate fake images and compute Discriminator loss on fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels_tensor = torch.full((batch_size,), fake_label, device=device)
            output_fake = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, fake_labels_tensor)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update Generator
            ############################
            generator.zero_grad()
            # Generator tries to fool Discriminator by using real labels for fake images
            fake_labels_tensor.fill_(real_label)
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, fake_labels_tensor)
            errG.backward()
            optimizerG.step()

            # Log losses every 50 batches
            if i % 50 == 0:
                print(f'[Epoch {epoch}/{epochs}][Batch {i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f}')
                wandb.log({"Generator Loss": errG.item(), "Discriminator Loss": errD.item()})

        # Save and log generated images every `save_interval` epochs and at the end of training
        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_grid = make_grid(fake, padding=2, normalize=True)
                wandb.log({"Generated Images": [wandb.Image(img_grid, caption=f"Epoch {epoch}")]})

                # Save generated images locally
                plt.figure(figsize=(8,8))
                plt.imshow(np.transpose(img_grid, (1, 2, 0)))
                plt.axis("off")
                plt.title(f"Epoch {epoch}")
                image_path = os.path.join(image_dir, f"epoch_{epoch}.png")
                plt.savefig(image_path)
                plt.close()

            # Save model checkpoints
            save_path_gen = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
            save_path_disc = os.path.join(model_dir, f"discriminator_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), save_path_gen)
            torch.save(discriminator.state_dict(), save_path_disc)
            print(f"Saved Generator and Discriminator models at epoch {epoch}")

    # Save the final models after training completion
    final_gen_path = os.path.join(model_dir, "generator_final.pth")
    final_disc_path = os.path.join(model_dir, "discriminator_final.pth")
    torch.save(generator.state_dict(), final_gen_path)
    torch.save(discriminator.state_dict(), final_disc_path)
    print("Training complete. Final models saved.")