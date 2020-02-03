import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt
import mnist

tf = tf_new.compat.v1
tf.set_random_seed(777)

x_data, y_data, _ = mnist.load(9)

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.random_normal([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    L1 = tf.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([100, 1]))
    b2 = tf.Variable(tf.zeros([1]))
    hypothesis = tf.matmul(L1, W2) + b2

    loss = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    n_steps = 10000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        pred_y = sess.run(tf.round(hypothesis), feed_dict={x: x_data})
        print(pred_y)

        correct_prediction = tf.equal(pred_y, y_data)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))
