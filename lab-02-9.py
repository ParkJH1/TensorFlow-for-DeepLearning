import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt

tf = tf_new.compat.v1
tf.set_random_seed(777)

# 시그모이드 함수
x = np.linspace(-10, 10, 100)

g = tf.Graph()
with g.as_default() as graph:
    y = tf.sigmoid(x)
    with tf.Session() as sess:
        y = sess.run(y)
        plt.figure(0)
        plt.plot(x, y, '-')
        plt.show()
