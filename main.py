import torch
import wandb
import random
import numpy as np
from config import BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, BETA1, NUM_EPOCHS, LATENT_DIM, NC
from models import Generator, Discriminator
from utils.training import train
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

wandb.init(project="dcgan")

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(nz=100, ngf=64, nc=3).to(device)
discriminator = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

train(generator, discriminator, device, dataloader, optimizerG, optimizerD, criterion, NUM_EPOCHS, LATENT_DIM)




#https://wandb.ai/basa-s-northeastern-university/dcgan/runs/dh6toqh3
#wandb/run-20241028_152228-dh6toqh3/logs
