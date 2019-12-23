import tensorflow as tf_new
tf = tf_new.compat.v1

g = tf.Graph()
with g.as_default() as graph:
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    c = x1 + x2   # c = tf.add(a, b)
    sess = tf.Session()
    print(sess.run(c, feed_dict={x1: 3, x2: 4.5}))
    print(sess.run(c, feed_dict={x1: [1, 3], x2: [2, 4]}))
