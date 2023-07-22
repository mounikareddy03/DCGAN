import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)
X = tf.placeholder(tf.float32 , [None , 784])
Z = tf.placeholder(tf.float32 , [None , 150])

#DISCRIMINATOR

D_weights = tf.Variable(tf.random_normal([784,150] , stddev = 0.001))
D_bias = tf.Variable(tf.random_normal([150] , stddev = 0.001))

D_weights2 = tf.Variable(tf.random_normal([150,1] , stddev = 0.001))
D_bias2 = tf.Variable(tf.random_normal([1] , stddev = 0.001))

theta_D = [D_weights , D_weights2 , D_bias , D_bias2]

def discriminator(x):
	D_layer1 = tf.nn.relu(tf.add(tf.matmul(x,D_weights), D_bias))
	D_layer2 = tf.add(tf.matmul(D_layer1 , D_weights2), D_bias2) 
	D_prob = tf.nn.sigmoid(D_layer2)
	
	return D_prob,D_layer2	


#GENERATOR

G_weights = tf.Variable(tf.random_normal([150,300] , stddev = 0.001))
G_bias = tf.Variable(tf.random_normal([300] , stddev = 0.001))

G_weights2 =tf.Variable(tf.random_normal([300 , 784] , stddev = 0.001))
G_bias2 = tf.Variable(tf.random_normal([784] , stddev = 0.001))

theta_G = [G_weights , G_bias , G_weights2 , G_bias2]

def generator(z):
	G_layer1 = tf.nn.relu(tf.add(tf.matmul(z,G_weights), G_bias))
	G_layer2 = tf.add(tf.matmul(G_layer1,G_weights2), G_bias2)
	G_prob = tf.nn.sigmoid(G_layer2)

	return G_prob

Fake = generator(Z)
D_real , D_logit_real = discriminator(X)
D_fake , D_logit_fake = discriminator(Fake)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_compute = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_compute = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

mb_size = 150
Z_dim = 150


def plot(samples):
	figr = plt.figure(figsize = (4,4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace = 0.05 , hspace = 0.05)

	for i, sample in enumerate(samples):
		a = plt.subplot(gs[i])
		plt.axis('off')
		a.set_xticklabels([])
		a.set_yticklabels([])
		a.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap = 'gray_r')

	return figr


i = 0

if not os.path.exists('out/'):
	os.makedirs('out/')


init = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(init)

	for it in range(200000):

		if it % 10000 == 0:
			samples = session.run(Fake, feed_dict = {Z : sample_Z(16,Z_dim)})
			fig = plot(samples)
			plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
			i+=1
			plt.close(fig)

		X_mb , _ = mnist.train.next_batch(mb_size)

		data1 = {X : X_mb, Z : sample_Z(mb_size , Z_dim)}
		data2 = {Z : sample_Z(mb_size , Z_dim)}

		_, D_loss_curr = session.run([D_compute , D_loss], feed_dict = data1)
		_, G_loss_curr = session.run([G_compute , G_loss], feed_dict = data2)

		

	