import tensorflow as tf_new
tf = tf_new.compat.v1

g = tf.Graph()
with g.as_default() as graph:
    hello = tf.constant("Hello TensorFlow!")
    sess = tf.Session()
    print(sess.run(hello))
