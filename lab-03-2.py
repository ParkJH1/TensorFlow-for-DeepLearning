import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt

tf = tf_new.compat.v1
tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

g = tf.Graph()
with g.as_default() as graph:
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.float32, [None, 1])
    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
        b1 = tf.Variable(tf.zeros([2]), name='bias1')
        W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
        b2 = tf.Variable(tf.zeros([1]), name='bias2')
    with tf.name_scope('prediction'):
        L1 = tf.matmul(x, W1) + b1
        L1 = tf.sigmoid(L1)
        L2 = tf.matmul(L1, W2) + b2
        hypothesis = tf.sigmoid(L2)
    with tf.name_scope('loss'):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=L2, labels=y)
        loss = tf.reduce_sum(entropy)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
    # 학습과정 시각화를 위해서 로그파일 생성
    train_writer = tf.summary.FileWriter('./summary', tf.get_default_graph())

    n_steps = 10000
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

        plt.figure(0)
        x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
        y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = sess.run(tf.round(hypothesis), feed_dict={x: np.c_[xx.ravel(), yy.ravel()]})
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data[:, 0], s=40, cmap=plt.cm.Spectral)
        plt.show()
