import tensorflow as tf_new
import numpy as np
tf = tf_new.compat.v1
tf.set_random_seed(777)

N = 100
a = 5
b = 2
x_data = np.random.rand(N, 1)
noise = np.random.normal(scale=0.1, size=(N, 1))
y_data = a * x_data + b + noise

g = tf.Graph()
with g.as_default() as graph:
    # 텐서보드 시각화를 위해서 네임스페이스 추가
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, [N, 1])
        y = tf.placeholder(tf.float32, [N, 1])
    with tf.name_scope('weights'):
        W = tf.Variable(tf.random_normal([1, 1]), name='weight')
        b = tf.Variable(tf.zeros([1]), name='bias')
    with tf.name_scope('prediction'):
        hypothesis = tf.matmul(x, W) + b
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - hypothesis))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
            print('step %d, loss: %f' % (i, l))
            # 학습내용 기록
            train_writer.add_summary(summary, i)
