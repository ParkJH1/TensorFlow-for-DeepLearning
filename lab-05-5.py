import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(2,), name='img'))
model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(0.01),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.mean_squared_error])


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [0], [0], [1]])

# Reserve 10,000 samples for validation
x_val = x_train[:]
y_val = y_train[:]

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='summary')

history = model.fit(x_train, y_train,
                    batch_size=4,
                    epochs=1000,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard])

print(model.predict(x_train))
