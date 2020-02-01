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
        images = images.reshape(-1, 784)
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


x_data, y_data = mnist_load(10000)

g = tf.Graph()
with g.as_default() as graph:
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.random_normal([784, 100]), name='weight1')
        b1 = tf.Variable(tf.zeros([100]), name='bias1')
        W2 = tf.Variable(tf.random_normal([100, 10]), name='weight2')
        b2 = tf.Variable(tf.zeros([10]), name='bias2')
    with tf.name_scope('prediction'):
        L1 = tf.matmul(x, W1) + b1
        L1 = tf.sigmoid(L1)
        L2 = tf.matmul(L1, W2) + b2
        hypothesis = tf.nn.softmax(L2)
    with tf.name_scope('loss'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=L2, labels=y)
        loss = tf.reduce_sum(entropy)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
    # 학습과정 시각화를 위해서 로그파일 생성
    train_writer = tf.summary.FileWriter('./summary', tf.get_default_graph())

    n_steps = 1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            _, summary, l = sess.run([train, merged, loss], feed_dict={x: x_data, y: y_data})
            train_writer.add_summary(summary, i)
            if i % 100 == 0:
                print('step %d, loss: %f' % (i, l))

        pred_y = sess.run(tf.round(hypothesis), feed_dict={x: x_data})
        print(pred_y)

        correct_prediction = tf.equal(pred_y, y_data)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy %s%%" % (sess.run(accuracy, feed_dict={x: x_data, y: y_data}) * 100))
