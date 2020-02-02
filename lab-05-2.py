from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 데이터 N개 생성
N = 100

x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2, 1))

x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=0.1*np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2, 1))

x_data = np.vstack([x_zeros, x_ones])
y_data = np.concatenate([y_zeros, y_ones])

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
model.summary()

model.compile(optimizer=keras.optimizers.SGD(0.01),
              loss=keras.losses.binary_crossentropy,
              metrics=[])

history = model.fit(x_data, y_data, batch_size=N, epochs=1000)
print(history.history)

plt.figure(0)
x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data[:, 0], s=40, cmap=plt.cm.Spectral)
plt.show()
