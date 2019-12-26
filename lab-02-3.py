import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt
tf = tf_new.compat.v1
tf.set_random_seed(777)

# 데이터 N개 랜덤생성
N = 100
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = a * x_data + b + noise

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [N, 1])
    y = tf.placeholder(tf.float32, [N, 1])
    W = tf.Variable(tf.random_normal([1, 1]))
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.square(y - hypothesis))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    n_steps = 10000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))
        print(sess.run(y, feed_dict={y: y_data}))
        print(sess.run(hypothesis, feed_dict={x: x_data}))
        pred_y = sess.run(hypothesis, feed_dict={x: x_data})
        plt.figure(0)
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred_y, '-r')
        plt.show()
