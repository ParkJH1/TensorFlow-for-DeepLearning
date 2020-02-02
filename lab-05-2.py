import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [0], [0], [1]])

model.fit(x_data, y_data, epochs=1000, batch_size=4)
