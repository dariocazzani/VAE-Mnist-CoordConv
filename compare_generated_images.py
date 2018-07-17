import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", validation_size=10000, one_hot=True)

from network import Network

def load_vae(coordConv):
    if coordConv:
        model_path = "saved_models_coordConv/"
    else:
        model_path = "saved_models_standard"

    graph_vae = tf.Graph()
    with graph_vae.as_default():

        network = Network(coordConv=coordConv)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network

if __name__ == '__main__':
    sess_coordConv, network_coordConv = load_vae(True)
    sess_standard, network_standard = load_vae(False)

    try:
        while True:
            image = np.reshape(mnist.test.images[random.randint(0, 5000), :], [1, 28, 28, 1])
            feed_dict={network_coordConv.image: image}
            gen_images_c = sess_coordConv.run(network_coordConv.reconstructions, feed_dict=feed_dict)
            feed_dict={network_standard.image: image}
            gen_images_s = sess_standard.run(network_standard.reconstructions, feed_dict=feed_dict)

            gen_images_c = np.reshape(np.squeeze(gen_images_c), [28, 28])
            gen_images_s = np.reshape(np.squeeze(gen_images_s), [28, 28])

            plt.imshow(np.hstack((np.squeeze(image), gen_images_c, gen_images_s)), cmap='Greys')
            plt.show()
    except (KeyboardInterrupt, SystemExit):
        print("quitting...")
