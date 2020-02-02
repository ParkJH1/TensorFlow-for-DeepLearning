import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt
import mnist

tf = tf_new.compat.v1
tf.set_random_seed(777)

x_data, y_data, y_onehot = mnist.load(1000, dimension=2)
x_data = np.pad(x_data, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant', constant_values=(0))

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y = tf.placeholder(tf.int32, [None, 10])

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
    b1 = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, W1, [1, 1, 1, 1], 'SAME')
    L1 = tf.nn.relu(conv1) + b1
    L1 = tf.nn.max_pool(L1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
    b2 = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(L1, W2, [1, 1, 1, 1], 'SAME')
    L2 = tf.nn.relu(conv2) + b2
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    L2_shape = L2.get_shape().as_list()
    L2_reshape = tf.reshape(L2, [-1, L2_shape[1] * L2_shape[2] * L2_shape[3]])
    L2_shape = L2_reshape.get_shape().as_list()
    W3 = tf.Variable(tf.truncated_normal([L2_shape[1], 120]))
    b3 = tf.Variable(tf.zeros([120]))
    L3 = tf.nn.relu(tf.matmul(L2_reshape, W3) + b3)

    W4 = tf.Variable(tf.truncated_normal([120, 84]))
    b4 = tf.Variable(tf.zeros([84]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.Variable(tf.truncated_normal([84, 10]))
    b5 = tf.Variable(tf.zeros([10]))
    L5 = tf.matmul(L4, W5) + b5

    hypothesis = tf.nn.softmax(L5)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L5, labels=y_onehot))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
    train = optimizer.minimize(loss)

    n_steps = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_onehot})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        print(sess.run(hypothesis, feed_dict={x: x_data}))
        pred_y = sess.run(tf.arg_max(hypothesis, 1), feed_dict={x: x_data})
        print(pred_y)

        correct_prediction = tf.equal(pred_y, y_data.flatten())
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy %s%%" % (sess.run(accuracy) * 100))

        # correct_prediction = tf.equal(pred_y, y_data)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))
