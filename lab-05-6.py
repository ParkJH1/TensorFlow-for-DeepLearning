import tensorflow as tf
import numpy as np
import mnist

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(32, 32, 1), name='img'))
model.add(tf.keras.layers.Conv2D(6, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))
model.add(tf.keras.layers.Conv2D(16, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.005),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

N = 10
x_data, y_data, _ = mnist.load(N, dimension=2)
x_data = np.pad(x_data, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=(0))

history = model.fit(x_data, y_data, batch_size=N, epochs=1000)

print(history.history)

pred_y = model.predict(x_data)
print(pred_y)
print(np.argmax(pred_y, axis=1))
