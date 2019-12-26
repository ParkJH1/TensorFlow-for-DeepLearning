import tensorflow as tf_new
import numpy as np
import matplotlib.pyplot as plt

tf = tf_new.compat.v1
tf.set_random_seed(777)

# 데이터 N개 생성
N = 100

x_zeros = np.random.multivariate_normal(mean=np.array((-1, -1)), cov=0.1 * np.eye(2), size=(N // 2,))
y_zeros = np.zeros((N // 2, 1))

x_ones = np.random.multivariate_normal(mean=np.array((1, 1)), cov=0.1*np.eye(2), size=(N // 2,))
y_ones = np.ones((N // 2, 1))

x_data = np.vstack([x_zeros, x_ones])
y_data = np.concatenate([y_zeros, y_ones])

g = tf.Graph()
with g.as_default() as graph:
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.float32, [None, 1])
    with tf.name_scope('weights'):
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.zeros([1]), name='bias')
    with tf.name_scope('prediction'):
        y_logit = tf.matmul(x, W) + b
        hypothesis = tf.sigmoid(y_logit)
    with tf.name_scope('loss'):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
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
