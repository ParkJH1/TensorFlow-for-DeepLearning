import numpy as np
import matplotlib.pyplot as plt

# 생성할 데이터 개수
N = 100

# y = ax + b 직선을 중심으로 가우스 잡음 생성
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = np.reshape(a * x_data + b + noise, (-1))

plt.figure(0)
plt.scatter(x_data, y_data)
plt.show()
