"""Original code generated with chatgpt-4
"""

import numpy as np
from tensorflow import keras

# Let's say each of your images is 224x224 and you have RGB color channels.
input_shape = (224, 224, 3)

# Load pre-trained ResNet without the top (include_top=False).
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=input_shape)

# The base model contains batchnorm layers. We want to keep them in inference mode when we unfreeze the base model for fine-tuning, so we make sure that the base_model is running in inference mode here.
x = base_model(inputs, training=False)

# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)

# A Dense layer to output 4 coordinates (x_min, y_min, x_max, y_max)
outputs = keras.layers.Dense(4)(x)

# Build the model
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# Assume you have two numpy arrays: "images" for your image data and "coords" for your coordinates
# images = np.array([...])  # shape should be (num_samples, 224, 224, 3)
# coords = np.array([...])  # shape should be (num_samples, 4)
# model.fit(images, coords, epochs=10, batch_size=32)
