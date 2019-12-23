import numpy as np
import matplotlib.pyplot as plt

N = 100
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = np.reshape(a * x_data + b + noise, (-1))

plt.figure(0)
plt.scatter(x_data, y_data)
plt.show()
