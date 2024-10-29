import torch
import wandb
import random
import numpy as np
from config import BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, BETA1, NUM_EPOCHS, LATENT_DIM, NC
from models import Generator, Discriminator
from utils.training import train, calculate_inception_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os

# Initialize WandB for logging
wandb.init(project="dcgan")

# Set manual seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Dataloader configuration
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
generator = Generator(nz=LATENT_DIM, ngf=64, nc=NC).to(device)
discriminator = Discriminator().to(device)

# Load pretrained weights (checkpoint)
start_epoch = 1
if os.path.exists("models/generator_final.pth") and os.path.exists("models/discriminator_final.pth"):
    try:
        generator.load_state_dict(torch.load("models/generator_final.pth", map_location=device))
        discriminator.load_state_dict(torch.load("models/discriminator_final.pth", map_location=device))
        print("Loaded pretrained generator and discriminator models. Resuming from epoch 51.")
        start_epoch = 51
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")

# Weight initialization for any new layers (if applicable)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss and optimizer configuration
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Training the model with Inception Score calculation every 10 epochs
train(generator, discriminator, device, dataloader, optimizerG, optimizerD, criterion, NUM_EPOCHS, LATENT_DIM, start_epoch)

# Final Inception Score calculation at the end of training
print("Calculating Inception Score...")
inception_score = calculate_inception_score(generator, LATENT_DIM, device)
wandb.log({"Inception Score": inception_score})
print(f"Inception Score: {inception_score}")




# [Epoch 100/100][Batch 350/391] Loss_D: 1.1157 Loss_G: 0.9856 D(x): 0.4983
# Inception Score at epoch 100: (4.391009, 0.26442027)
# Calculating Inception Score...
# Inception Score: (4.541724, 0.44131875)
# wandb: 🚀 View run faithful-frog-13 at: https://wandb.ai/basa-s-northeastern-university/dcgan/runs/abcbpusr
# wandb: Find logs at: wandb/run-20241028_214153-abcbpusr/logs