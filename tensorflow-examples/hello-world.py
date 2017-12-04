import tensorflow as tf

hello = tf.constant("Hello world!")
sess = tf.Session()

# You can run a session more than once.
print(sess.run(hello))
print(sess.run(hello))
