# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", validation_size=10000, one_hot=True)

import tensorflow as tf
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

model_path = "saved_models/"
model_name = model_path + 'model'

from coordConv import CoordConv, CoordDeconv

class Network(object):
	# Create model
	def __init__(self):
		self.image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')

		self.z_mu, self.z_logvar = self.encoder(self.image)
		self.z = self.sample_z(self.z_mu, self.z_logvar)
		self.reconstructions = self.decoder(self.z)
		tf.summary.image('reconstructions', self.reconstructions, 20)

		self.merged = tf.summary.merge_all()

		self.loss = self.compute_loss()

	def sample_z(self, mu, logvar):
		eps = tf.random_normal(shape=tf.shape(mu))
		return mu + tf.exp(logvar / 2) * eps

	def encoder(self, x):
		coordConv1 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
		x = coordConv1(x)

		coordConv2 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
		x = coordConv2(x)

		coordConv3 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
		x = coordConv3(x)

		x = tf.layers.flatten(x)
		z_mu = tf.layers.dense(x, units=32, name='z_mu')
		z_logvar = tf.layers.dense(x, units=32, name='z_logvar')

		return z_mu, z_logvar

	def decoder(self, z):
		x = tf.layers.dense(z, 768, activation=None)
		x = tf.reshape(x, [-1, 1, 1, 768])

		x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
		# x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
		# x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2, strides=2, padding='valid', activation=tf.nn.sigmoid)
		coordDeconv2 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
		x = coordDeconv2.call(x)

		coordDeconv3 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=1, kernel_size=2, strides=2, padding='valid', activation=tf.nn.sigmoid)
		x = coordDeconv3.call(x)

		return x

	def compute_loss(self):
		logits_flat = tf.layers.flatten(self.reconstructions)
		labels_flat = tf.layers.flatten(self.image)
		reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
		vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
		return vae_loss

def train_vae():
	sess = tf.InteractiveSession()

	global_step = tf.Variable(0, name='global_step', trainable=False)

	writer = tf.summary.FileWriter('logdir')

	batch_size = 128
	learning_rate = 1E-3

	network = Network()
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(network.loss, global_step=global_step)
	tf.global_variables_initializer().run()

	saver = tf.train.Saver(max_to_keep=1)
	step = global_step.eval()

	try:
		saver.restore(sess, tf.train.latest_checkpoint(model_path))
		print("Model restored from: {}".format(model_path))
	except:
		print("Could not restore saved model")

	try:
		while True:
			batch_x, _ = mnist.train.next_batch(batch_size)
			batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
			feed_dict = {network.image: batch_x}
			_, loss_value, summary, rec = sess.run([train_op, network.loss, network.merged, network.reconstructions],
								feed_dict={network.image: batch_x})
			writer.add_summary(summary, step)

			if np.isnan(loss_value):
				raise ValueError('Loss value is NaN')
			if step % 100 == 0 and step > 0:

				batch_x_val, _ = mnist.validation.next_batch(10000)
				batch_x_val = np.reshape(batch_x_val, [-1, 28, 28, 1])
				feed_dict = {network.image: batch_x_val}
				val_loss_value = sess.run(network.loss, feed_dict=feed_dict)
				save_path = saver.save(sess, model_name, global_step=global_step)

				print ('step {}: training   loss {:.6f}'.format(step, loss_value))
				print ('step {}: validation loss {:.6f}'.format(step, val_loss_value))
			step+=1

	except (KeyboardInterrupt, SystemExit):
		print("Manual Interrupt")

	except Exception as e:
		print("Exception: {}".format(e))

if __name__ == '__main__':
	train_vae()
