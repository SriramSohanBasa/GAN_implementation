# GAN Image Generation Project

This project implements a Generative Adversarial Network (GAN) using TensorFlow and Keras to generate realistic images from random noise. The model is trained on the CIFAR-10 dataset, and both the generator and discriminator networks use convolutional layers.

## Project Structure

- `main.py`: Entry point of the application.
- `config.py`: Configuration parameters and hyperparameters.
- `models/`: Contains the generator and discriminator models.
- `utils/`: Utility functions for data loading, image saving, and training.

## How to Run

1. Install the required packages:
   ```bash
   pip install -r requirements.txt