import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt

from six.moves import urllib
import os
import gzip

tf = tf_new.compat.v1
tf.set_random_seed(777)


def mnist_load(num_images=100):
    if num_images > 60000:
        num_images = 60000
    elif num_images < 1:
        num_images = 1
    filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    filepaths = []
    for filename in filenames:
        filepath = os.path.join('data', filename)
        filepaths.append(filepath)
        if not os.path.exists(filepath):
            print('Downloading...', filename)
            filepath, _ = urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + filenames[i], filepath)
            size = os.stat(filepath).st_size
            print('Successfully downloaded', filename, size, 'bytes.')
        else:
            print('Already downloaded', filename)

    print('Extracting...', filenames[0])
    with gzip.open(filepaths[0]) as bytestream:
        size = os.stat(filepaths[0]).st_size
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(-1, 28, 28, 1)
        print('Successfully Extracted', filenames[0])

    print('Extracting...', filenames[1])
    with gzip.open(filepaths[1]) as bytestream:
        size = os.stat(filepaths[1]).st_size
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
        labels = np.eye(10)[labels]
        print('Successfully Extracted', filenames[1])

    return images, labels


x_data, y_data = mnist_load(9)

g = tf.Graph()
with g.as_default() as graph:
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.int32, [None, 10])

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
    b1 = tf.Variable(tf.zeros([32]))
    conv1 = tf.nn.conv2d(x, W1, [1, 1, 1, 1], 'SAME')
    L1 = tf.nn.relu(conv1) + b1
    L1 = tf.nn.max_pool(L1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
    b2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    conv2 = tf.nn.conv2d(L1, W2, [1, 1, 1, 1], 'SAME')
    L2 = tf.nn.relu(conv2) + b2
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    L2_shape = L2.get_shape().as_list()
    L2_reshape = tf.reshape(L2, [-1, L2_shape[1] * L2_shape[2] * L2_shape[3]])
    L2_shape = L2_reshape.get_shape().as_list()
    W3 = tf.Variable(tf.truncated_normal([L2_shape[1], 512]))
    b3 = tf.Variable(tf.zeros([512]))
    L3_logits = tf.matmul(L2_reshape, W3) + b3
    L3 = tf.nn.relu(L3_logits)

    W4 = tf.Variable(tf.truncated_normal([512, 10]))
    b4 = tf.Variable(tf.zeros([10]))
    hypothesis = tf.matmul(L3, W4) + b4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=hypothesis))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    train = optimizer.minimize(loss)

    n_steps = 200
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, l = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        print(sess.run(hypothesis, feed_dict={x: x_data}))
        pred_y = sess.run(tf.arg_max(hypothesis, 1), feed_dict={x: x_data})
        print(pred_y)

        # correct_prediction = tf.equal(tf.arg_max(pred_y, 1), tf.arg_max(y_data, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))

        # correct_prediction = tf.equal(pred_y, y_data)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))
