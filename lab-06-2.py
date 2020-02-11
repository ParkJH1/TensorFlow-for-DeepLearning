import tensorflow as tf
import numpy as np
import mnist

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28, 28), name='img'))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

N = 10
x_data, y_data, _ = mnist.load(N, dimension=2)
x_data = np.reshape(x_data, [-1, 28, 28])

history = model.fit(x_data, y_data, batch_size=N, epochs=1000)

print(history.history)

pred_y = model.predict(x_data)
print(pred_y)
print(np.argmax(pred_y, axis=1))
print(y_data.flatten())
