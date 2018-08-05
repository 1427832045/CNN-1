import tensorflow as tf
a = tf.Variable([[1.,1.],[1.,1.]])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.reduce_sum(a)))
print(sess.run(tf.nn.softmax(a)))
sess.close()