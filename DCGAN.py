import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")
batch = 64
X_shape = tf.placeholder(tf.float32 , [None , 784])
X = tf.reshape(X_shape , [batch,28,28,1])
Z = tf.placeholder(tf.float32 , [None , 150])


kernel = [3,3,1,32]
kernel2 = [3,3,32,64]
kernel3 = [3,3,64,128]
kernel4 = [4*4*128,1]

stride = [1,2,2,1]

filter1 = tf.Variable(tf.random_normal(kernel , stddev = 0.005))
filter2 = tf.Variable(tf.random_normal(kernel2 , stddev = 0.005))
filter3 = tf.Variable(tf.random_normal(kernel3 , stddev = 0.005))
filter4 = tf.Variable(tf.random_normal(kernel4 , stddev = 0.005))

theta_D = [filter1,filter2,filter3,filter4]



def leakyrelu(tnsr, alpha = 0.15):
	return tf.maximum(alpha*tnsr , tnsr)


def batch_norm(tnsr):

	 mean , variance = tf.nn.moments(tnsr , axes = [0,1,2])

	 alpha = tf.Variable(tf.constant(0.0 , shape = [tnsr.get_shape().as_list()[-1]]))
	 beta = tf.Variable(tf.constant(1.0 , shape = [tnsr.get_shape().as_list()[-1]]))

	 norm = tf.nn.batch_normalization(tnsr , mean , variance , alpha , beta , 1e-3)
	

	 return norm


def discriminator(x):

	l1 = leakyrelu(batch_norm(tf.nn.conv2d(x , filter1 , strides = stride , padding = 'SAME')))
	
	l2 = leakyrelu(batch_norm(tf.nn.conv2d(l1 , filter2 , strides = stride , padding = 'SAME')))
	
	l3 = leakyrelu(batch_norm(tf.nn.conv2d(l2 , filter3 , strides = stride , padding = 'SAME')))
	
	l4_shaping = tf.reshape(l3 , [-1 , 4 * 4 * 128])
	
	l4 = leakyrelu(tf.matmul(l4_shaping , filter4))
	
	return tf.nn.sigmoid(l4) , l4


shape = Z.get_shape().as_list()
weight = tf.Variable(tf.random_normal([shape[-1] , 512], stddev = 0.005))
bias = tf.Variable(tf.random_normal([512] , stddev = 0.005))
filter_1 = tf.Variable(tf.random_normal([3,3,16,32] , stddev = 0.005))
filter_2 = tf.Variable(tf.random_normal([3,3,8,16] , stddev = 0.005))
filter_3 = tf.Variable(tf.random_normal([3,3,1,8] , stddev = 0.005))

theta_G = [weight , bias , filter_1 , filter_2 , filter_3]

def generator(z):

	l1 = tf.add(tf.matmul(z , weight) , bias)

	l2_shaping = tf.reshape(l1 , [batch , 4 , 4 , 32]) 
	
	l2_deconv = tf.nn.conv2d_transpose(l2_shaping , filter_1 ,output_shape = [batch , 7 , 7 , 16] ,strides = [1 , 2, 2, 1], padding = 'SAME')
	
	l2 = leakyrelu(batch_norm(l2_deconv))
	
	l3_deconv = tf.nn.conv2d_transpose(l2, filter_2 , output_shape = [batch , 14 , 14 , 8] , strides = [1 , 2, 2, 1], padding = 'SAME')
	
	l3 = leakyrelu(batch_norm(l3_deconv))
	
	l4 = tf.nn.conv2d_transpose(l3 , filter_3 , output_shape = [batch , 28 , 28 , 1] , strides = [1 , 2, 2, 1], padding = 'SAME')
	
	return tf.nn.sigmoid(l4)

G = generator(Z)
D , D_Logits = discriminator(X)
D_fake , D_fake_Logits = discriminator(G)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Logits, labels=tf.ones_like(D_Logits)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_Logits, labels=tf.zeros_like(D_fake_Logits)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_Logits, labels=tf.ones_like(D_fake)))

D_compute = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_compute = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)


init = tf.initialize_all_variables()

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


Z_dim = 150

def plot(samples):
	figr = plt.figure(figsize = (4,4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace = 0.05 , hspace = 0.05)
	for i, sample in enumerate(samples):
		if i <= 15 :
			a = plt.subplot(gs[i])
			plt.axis('off')
			a.set_xticklabels([])
			a.set_yticklabels([])
			a.set_aspect('equal')
			plt.imshow(sample.reshape(28,28), cmap = 'gray_r')

		else :
			break

	return figr


i = 0

if not os.path.exists('output/'):
	os.makedirs('output/')




with tf.Session() as session:
	session.run(init)

	for it in range(150000):
		
		if it % 10000 == 0:
			samples = session.run(G, feed_dict = {Z : sample_Z(64,Z_dim)})
			fig = plot(samples)
			plt.savefig('output/{}.png'.format(str(i).zfill(3)), bbox_inches = 'tight')
			i+=1
			plt.close(fig)

		X_mb , _ = mnist.train.next_batch(batch)

		data1 = {X_shape : X_mb, Z : sample_Z(batch , Z_dim)}
		data2 = {Z : sample_Z(batch , Z_dim)}

		_, D_loss_curr = session.run([D_compute , D_loss], feed_dict = data1)
		_, G_loss_curr = session.run([G_compute , G_loss], feed_dict = data2)
