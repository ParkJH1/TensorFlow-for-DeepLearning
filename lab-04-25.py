import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(3, 3, 1), name='img'))
model.add(tf.keras.layers.Conv2D(6, 2, activation='relu'))
model.summary()

x_data = np.array([[[[1], [2], [3]],
                   [[2], [1], [3]],
                   [[3], [2], [1]]]])

res = model.predict(x_data)


# model.compile(optimizer=tf.keras.optimizers.Adamax(0.0005),
#               loss=tf.keras.losses.sparse_categorical_crossentropy,
#               metrics=[tf.keras.metrics.sparse_categorical_accuracy])

# history = model.fit(x_train, y_train,
#                     batch_size=16,
#                     epochs=10,
#                     validation_data=(x_val, y_val),
#                     callbacks=[tensorboard])

# pred_x = np.array()
# print(model.predict(x_val))
