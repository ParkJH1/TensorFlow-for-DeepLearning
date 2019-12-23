import tensorflow as tf_new
tf = tf_new.compat.v1
tf.set_random_seed(777)

g = tf.Graph()
with g.as_default() as graph:
    a = tf.Variable(tf.random_normal([10]))
    b = tf.Variable(tf.zeros([10]))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
