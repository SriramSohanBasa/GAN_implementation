# Hyperparameters and configuration
BATCH_SIZE = 128
IMAGE_SIZE = 32
LEARNING_RATE = 0.0002
BETA1 = 0.5
NUM_EPOCHS = 100  # Increased epochs for more training time
LATENT_DIM = 100
NC = 3  # Number of channels (RGB)

# Directories for storing generated images and models
GENERATED_IMAGES_DIR = 'generated_images'
MODEL_DIR = 'models'

# Parameters for regularization and experimentation
LABEL_SMOOTHING_REAL = 0.9
LABEL_SMOOTHING_FAKE = 0.1
SAVE_INTERVAL = 2  # Save images every 10 epochs