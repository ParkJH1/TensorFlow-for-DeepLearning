from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import mnist

N = 9
x_data, y_data, _ = mnist.load(N)

model = keras.Sequential()
model.add(keras.Input(shape=(784,)))
model.add(keras.layers.Dense(100, activation=keras.activations.sigmoid))
model.add(keras.layers.Dense(1))
model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=[])

history = model.fit(x_data, y_data, epochs=1000, batch_size=N)
print(history.history)

pred_y = np.round(model.predict(x_data))
print(pred_y)
