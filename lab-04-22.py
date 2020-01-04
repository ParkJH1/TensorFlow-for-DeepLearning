import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


inputs = tf.keras.Input(shape=(28, 28, 1), name='img')
x = tf.keras.layers.Conv2D(6, 5, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(16, 5, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(84, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs, name='model')
# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adamax(0.0005),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype(np.float32) / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype(np.float32) / 255

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='summary')

history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=10,
                    validation_data=(x_val, y_val),
                    callbacks=[tensorboard])

pred_x = np.array()
print(model.predict(x_val))
