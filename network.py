import tensorflow as tf
from coordConv import CoordConv, CoordDeconv

LEARNING_RATE = 1E-3

class Network(object):
	# Create model
	def __init__(self, coordConv=True):
		self.useCoordConv = coordConv

		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.latent_vec_size = 32
		self.image = tf.placeholder(tf.float32, [None, 28, 28, 1], name='image')
		self.latent_vector = tf.placeholder(tf.float32, [None, self.latent_vec_size])
		tf.summary.image('Originals', self.image, 10)

		self.z_mu, self.z_logvar = self.encoder(self.image)
		self.z = self.sample_z(self.z_mu, self.z_logvar)

		self.reconstructions = self.decoder(self.z)
		self.generated_images = self.decoder(self.latent_vector)
		tf.summary.image('Reconstructions', self.reconstructions, 10)

		self.loss = self.compute_loss()
		tf.summary.scalar('Loss', self.loss)

		optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
		gradients = optimizer.compute_gradients(loss=self.loss)

		# for gradient, variable in gradients:
		# 	tf.summary.histogram("gradients/" + variable.name, gradient)
		# 	tf.summary.histogram("variables/" + variable.name, variable)

		self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

		self.merged = tf.summary.merge_all()

	def sample_z(self, mu, logvar):
		eps = tf.random_normal(shape=tf.shape(mu))
		return mu + tf.exp(logvar / 2) * eps

	def encoder(self, x):
		if self.useCoordConv:
			coordConv1 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
			x = coordConv1(x)
			coordConv2 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
			x = coordConv2(x)
			coordConv3 = CoordConv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
			x = coordConv3(x)
		else:
			x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
			x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
			x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

		x = tf.layers.flatten(x)
		z_mu = tf.layers.dense(x, units=self.latent_vec_size, name='z_mu')
		z_logvar = tf.layers.dense(x, units=self.latent_vec_size, name='z_logvar')

		return z_mu, z_logvar

	def decoder(self, z):
		x = tf.layers.dense(z, 768, activation=None)
		x = tf.reshape(x, [-1, 1, 1, 768])

		if self.useCoordConv:
			coordDeconv1 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
			x = coordDeconv1(x)
			coordDeconv2 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
			x = coordDeconv2(x)
			coordDeconv3 = CoordDeconv(int(x.get_shape()[1]), int(x.get_shape()[2]), False, filters=1, kernel_size=2, strides=2, padding='valid', activation=tf.nn.sigmoid)
			x = coordDeconv3(x)
		else:
			x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
			x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
			x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=2, strides=2, padding='valid', activation=tf.nn.sigmoid)

		return x

	def compute_loss(self):
		logits_flat = tf.layers.flatten(self.reconstructions)
		labels_flat = tf.layers.flatten(self.image)
		reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
		vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
		return vae_loss
