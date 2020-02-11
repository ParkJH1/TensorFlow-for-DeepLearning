import tensorflow as tf
import numpy as np
import mnist

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28, 28), name='img'))
model.add(tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(1, activation=None)))
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[])

N = 10
x_data, y_data, _ = mnist.load(N, dimension=2)
print(x_data.shape)
x_data = np.reshape(x_data, [-1, 28, 28])

history = model.fit(x_data, y_data, batch_size=N, epochs=1000)

print(history.history)

pred_y = model.predict(x_data)
pred_y = np.round(pred_y)
print(pred_y.flatten())
print(y_data.flatten())
