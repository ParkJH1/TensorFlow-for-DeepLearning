from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(2, activation=keras.activations.sigmoid))
model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=[])

history = model.fit(x_data, y_data, epochs=10000, batch_size=4)
print(history.history)

pred_y = np.round(model.predict(x_data))
print(pred_y)
