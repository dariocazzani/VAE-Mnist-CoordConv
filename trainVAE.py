# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", validation_size=10000, one_hot=True)

import tensorflow as tf
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--useCoordConv', dest='coordConv', type=str,
					required=True, help='Use coordConv layers or not')
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

from network import Network

TOT_STEPS = 10000
BATCH_SIZE = 128

def train_vae(coordConv):
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('logdir')

		if coordConv:
			model_path = "saved_models_coordConv/"
			model_name = model_path + 'model'
		else:
			model_path = "saved_models_standard/"
			model_name = model_path + 'model'

		network = Network(coordConv=coordConv)
		tf.global_variables_initializer().run()

		saver = tf.train.Saver(max_to_keep=1)

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
			print("Model restored from: {}".format(model_path))
		except:
			print("Could not restore saved model")

		step = network.global_step.eval()

		try:
			while(step <= TOT_STEPS):
				batch_x, _ = mnist.train.next_batch(BATCH_SIZE)
				batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
				feed_dict = {network.image: batch_x}
				_, loss_value, summary, rec = sess.run([network.train_op, network.loss, network.merged, network.reconstructions],
									feed_dict={network.image: batch_x})
				writer.add_summary(summary, step)

				if np.isnan(loss_value):
					raise ValueError('Loss value is NaN')

				if step % 100 == 0 and step > 0:
					batch_x_val, _ = mnist.validation.next_batch(10000)
					batch_x_val = np.reshape(batch_x_val, [-1, 28, 28, 1])
					feed_dict = {network.image: batch_x_val}
					val_loss_value = sess.run(network.loss, feed_dict=feed_dict)
					save_path = saver.save(sess, model_name, global_step=step)

					print ('step {}: training   loss {:.6f}'.format(step, loss_value))
					print ('step {}: validation loss {:.6f}'.format(step, val_loss_value))
				step+=1

		except (KeyboardInterrupt, SystemExit):
			print("Manual Interrupt")

		except Exception as e:
			print("Exception: {}".format(e))
		finally:
			print("Model was saved here: {}".format(model_name))

if __name__ == '__main__':
	args = parser.parse_args()
	train_vae(coordConv=str2bool(args.coordConv))
