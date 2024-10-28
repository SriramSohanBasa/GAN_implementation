import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from config import NOISE_DIM

class ActivationLogger(Callback):
    def __init__(self, model: Model, layer_names: list, log_dir: str):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

        # Prepare the model to output the activations of the specified layers
        self.outputs = [self.model.get_layer(name).output for name in self.layer_names]
        self.activation_model = Model(inputs=self.model.input, outputs=self.outputs)

    def on_epoch_end(self, epoch, logs=None):
        # Generate input data
        noise = np.random.normal(0, 1, (1, NOISE_DIM))
        activations = self.activation_model.predict(noise)

        # Log activations as images
        with self.summary_writer.as_default():
            for name, activation in zip(self.layer_names, activations):
                # Add batch and channel dimensions if necessary
                if len(activation.shape) == 3:
                    activation = np.expand_dims(activation, axis=3)
                tf.summary.image(f'Activation_{name}', activation, step=epoch)