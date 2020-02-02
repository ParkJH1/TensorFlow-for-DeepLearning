from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 데이터 N개 랜덤생성
N = 100
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = a * x_data + b + noise

model = keras.Sequential()
model.add(keras.Input(shape=(1,)))
model.add(keras.layers.Dense(1))
model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=[])

history = model.fit(x_data, y_data, batch_size=N, epochs=5000)
print(history.history)

pred_y = model.predict(x_data)
print(pred_y)
plt.figure(0)
plt.scatter(x_data, y_data)
plt.plot(x_data, pred_y, '-r')
plt.show()

#
# inputs = keras.Input(shape=(2,), name='digits')
# outputs = keras.layers.Dense(1, activation='sigmoid', name='predictions')(inputs)
#
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype(np.float32) / 255
# x_test = x_test.reshape(10000, 784).astype(np.float32) / 255
#
# y_train = y_train.astype(np.float32)
# y_test = y_test.astype(np.float32)
#
# print(x_train)
#
# # Reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]
#
# x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_train = np.array([[0], [0], [0], [1]])
#
# x_test = x_val = x_train
# y_test = y_val = y_train
#
# # Specify the training configuration (optimizer, loss, metrics)
# model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
#               # Loss function to minimize
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               # List of metrics to monitor
#               metrics=[keras.metrics.SparseCategoricalAccuracy()])
#
# # Train the model by slicing the data into "batches"
# # of size "batch_size", and repeatedly iterating over
# # the entire dataset for a given number of "epochs"
# print('# Fit model on training data')
# history = model.fit(x_train, y_train,
#                     batch_size=1,
#                     epochs=10,
#                     # We pass some validation for
#                     # monitoring validation loss and metrics
#                     # at the end of each epoch
#                     validation_data=(x_val, y_val))
#
# # The returned "history" object holds a record
# # of the loss values and metric values during training
# print('\nhistory dict:', history.history)
#
# # Evaluate the model on the test data using `evaluate`
# print('\n# Evaluate on test data')
# results = model.evaluate(x_test, y_test, batch_size=128)
# print('test loss, test acc:', results)
#
# # Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print('\n# Generate predictions for 3 samples')
# predictions = model.predict(x_test[:3])
# print('predictions shape:', predictions.shape)
#
