import torch
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def train(generator, discriminator, device, dataloader, optimizerG, optimizerD, criterion, epochs, latent_dim, save_interval=2):
    # Watch the models for logging gradients and parameters
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    # Labels for real and fake data
    real_label = 0.9
    fake_label = 0.1

    # Start training loop
    for epoch in range(1, epochs + 1):
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator with real images
            discriminator.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            output_real = discriminator(real_images).view(-1)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train Discriminator with fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels = torch.full((batch_size,), fake_label, device=device)
            output_fake = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, fake_labels)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator to fool Discriminator
            generator.zero_grad()
            fake_labels.fill_(real_label)  # Use real label for generator loss
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, fake_labels)
            errG.backward()
            optimizerG.step()

            # Log losses every 50 batches
            if i % 50 == 0:
                print(f'[Epoch {epoch}/{epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f}')
                wandb.log({"Generator Loss": errG.item(), "Discriminator Loss": errD.item()})

        # Save and log generated images every `save_interval` epochs
        if epoch % save_interval == 0:
            with torch.no_grad():
                fake_images = generator(noise).detach().cpu()
                img_grid = make_grid(fake_images, padding=2, normalize=True)
                wandb.log({"Generated Images": [wandb.Image(img_grid, caption=f"Epoch {epoch}")]})
                plt.imshow(np.transpose(img_grid, (1, 2, 0)))
                plt.axis("off")
                #plt.show()