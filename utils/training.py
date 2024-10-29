import torch
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models import inception_v3
import torch.nn.functional as F

def train(generator, discriminator, device, dataloader, optimizerG, optimizerD, criterion, epochs, latent_dim, start_epoch=1, save_interval=2):
    model_dir = 'models'
    image_dir = 'generated_images'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    real_label = 0.9
    fake_label = 0.1
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    for epoch in range(start_epoch, epochs + 1):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_labels = torch.full((batch_size,), real_label, device=device)
            output_real = discriminator(real_images).view(-1)
            errD_real = criterion(output_real, real_labels)
            errD_real.backward()
            D_x = output_real.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels_tensor = torch.full((batch_size,), fake_label, device=device)
            output_fake = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output_fake, fake_labels_tensor)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            generator.zero_grad()
            fake_labels_tensor.fill_(real_label)
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, fake_labels_tensor)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[Epoch {epoch}/{epochs}][Batch {i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f}')
                wandb.log({"Generator Loss": errG.item(), "Discriminator Loss": errD.item()})

        if epoch % save_interval == 0 or epoch == epochs:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_grid = make_grid(fake, padding=2, normalize=True)
                wandb.log({"Generated Images": [wandb.Image(img_grid, caption=f"Epoch {epoch}")]})

                plt.figure(figsize=(8,8))
                plt.imshow(np.transpose(img_grid, (1, 2, 0)))
                plt.axis("off")
                plt.title(f"Epoch {epoch}")
                image_path = os.path.join(image_dir, f"epoch_{epoch}.png")
                plt.savefig(image_path)
                plt.close()

            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch}.pth"))

        # Calculate Inception Score every 10 epochs
        if epoch % 10 == 0:
            inception_score = calculate_inception_score(generator, latent_dim, device)
            wandb.log({"Inception Score": inception_score})
            print(f"Inception Score at epoch {epoch}: {inception_score}")

    torch.save(generator.state_dict(), os.path.join(model_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator_final.pth"))

def calculate_inception_score(generator, latent_dim, device, n_samples=1000, batch_size=32, splits=10):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    scores = []

    for _ in range(n_samples // batch_size):
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake_images = generator(noise)
            fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)
            preds = F.softmax(inception_model(fake_images), dim=1)
        
        # Detach to avoid gradient tracking before converting to numpy
        scores.append(preds.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    
    # Inception Score calculation
    split_scores = []
    for k in range(splits):
        part = scores[k * (n_samples // splits): (k + 1) * (n_samples // splits), :]
        py = np.mean(part, axis=0)
        scores_per_image = part * np.log(part / py[np.newaxis, :])
        split_scores.append(np.exp(np.mean(np.sum(scores_per_image, axis=1))))

    inception_score = np.mean(split_scores)
    inception_std = np.std(split_scores)
    return inception_score, inception_std