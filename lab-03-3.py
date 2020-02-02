import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt
import mnist

tf = tf_new.compat.v1
tf.set_random_seed(777)

x_data, y_data, y_onehot = mnist.load(9)

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    W1 = tf.Variable(tf.random_normal([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    L1 = tf.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal([100, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    L2 = tf.matmul(L1, W2) + b2

    hypothesis = tf.nn.softmax(L2)

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=L2, labels=y_onehot)
    loss = tf.reduce_sum(entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    n_steps = 10000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_onehot})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        print(sess.run(hypothesis, feed_dict={x: x_data}))
        pred_y = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x: x_data})
        print(pred_y)

        correct_prediction = tf.equal(pred_y, y_data.flatten())
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data}) * 100))
