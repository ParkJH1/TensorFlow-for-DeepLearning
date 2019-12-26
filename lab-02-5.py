import numpy as np
import matplotlib.pyplot as plt

# 생성할 데이터 개수
N = 100

# 0번 그룹 데이터셋 생성
# (-1, -1)을 중심으로 가우스 잡음 생성
x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2,))

# 1번 그룹 데이터셋 생성
# (1, 1)을 중심으로 가우스 잡음 생성
x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=0.1*np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2,))

x_data = np.vstack([x_zeros, x_ones])
y_data = np.concatenate([y_zeros, y_ones])

plt.figure(0)
plt.scatter(x_data[:, 0], x_data[:, 1])
plt.show()
