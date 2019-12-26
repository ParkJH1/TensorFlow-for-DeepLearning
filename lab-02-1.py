import numpy as np
import matplotlib.pyplot as plt

# 선형 회귀 예시 데이터
# 생성할 데이터 개수
N = 100

# y = ax + b 직선을 중심으로 가우스 잡음 생성
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = a * x_data + b + noise

plt.figure(0)
plt.scatter(x_data, y_data)
plt.show()
