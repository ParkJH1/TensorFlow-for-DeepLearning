import numpy as np
import matplotlib.pyplot as plt

# 로지스틱 회귀 예시 데이터
# 생성할 데이터 개수
N = 100

# 0번 그룹 데이터셋 생성
# (-1, -1)을 중심으로 가우스 잡음 생성
x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2, 1))

# 1번 그룹 데이터셋 생성
# (1, 1)을 중심으로 가우스 잡음 생성
x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=0.1*np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2, 1))

x_data = np.vstack([x_zeros, x_ones])
y_data = np.concatenate([y_zeros, y_ones])

plt.figure(0)
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color='r')
plt.scatter(x_ones[:, 0], x_ones[:, 1], color='b')
plt.show()
