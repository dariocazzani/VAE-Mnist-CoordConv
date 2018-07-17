import tensorflow as tf
import numpy as np
from coordConv import CoordConv, CoordDeconv

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
x = tf.placeholder(tf.float32, [None, 6, 6, 1], name="input")

coordDeconv1 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=4, kernel_size=1, strides=2, padding='valid')
conv1 = tf.nn.relu(coordDeconv1(x))
coordDeconv2 = CoordDeconv(int(conv1.get_shape()[1]), int(conv1.get_shape()[2]), False, filters=4, kernel_size=1, strides=2, padding='valid')
deconv1 = tf.nn.relu(coordDeconv2(conv1))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

feed_dict = {x: np.random.randn(1, 6, 6, 1)}
deconv1_out = sess.run(deconv1, feed_dict=feed_dict)

print(deconv1_out[...,-1])
print(deconv1_out[...,-2])
tf.InteractiveSession.close(sess)
