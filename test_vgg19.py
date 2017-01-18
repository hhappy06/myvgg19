import numpy as np
import tensorflow as tf
import vgg19
import utils
import time

import skimage.io as io

MODEL_PARAMETERS_PATH = './model/vgg19.npy'
IMAGE1_PATH = './data/tiger.jpeg'
IMAGE2_PATH = './data/puzzle.jpeg'

ori_image1, test_image1 = utils.load_image(IMAGE1_PATH)
ori_image2, test_image2 = utils.load_image(IMAGE2_PATH)

test_image1 = test_image1.reshape((1, 224, 224, 3))
test_image2 = test_image2.reshape((1, 224, 224, 3))

batch = np.concatenate((test_image1, test_image2), 0)

# label 
img1_true_result = np.array([1 if i == 292 else 0 for i in xrange(1000)])
img2_true_result = np.array([1 if i == 611 else 0 for i in xrange(1000)])

img1_true_result = img1_true_result.reshape((1, 1000))
img2_true_result = img2_true_result.reshape((1, 1000))

label = np.concatenate((img1_true_result, img2_true_result), 0)

with tf.Session() as session:
	# construct vgg 19 network
	print 'Constructing the VGG19 network'
	start_time = time.time()
	vgg = vgg19.VGG19([224, 224, 3], MODEL_PARAMETERS_PATH)
	print 'time used %d'%(time.time() - start_time)

	# initialize paramerter
	print '\nInitializing all variables'
	start_time = time.time()
	session.run(tf.global_variables_initializer())
	print 'time used %d'%(time.time() - start_time)

	# detection sample
	print '\nTesting detection of VGG19 network using two images (tiger and puzzle)'
	start_time = time.time()
	prob = session.run(vgg.get_predict_op(), feed_dict = {
		vgg.get_input_tensor(): batch,
		vgg.get_trainable_tensor(): False
		})
	print 'time used %d, testing result:'%(time.time() - start_time)
	utils.print_prob(prob[0], './synset.txt')
	utils.print_prob(prob[1], './synset.txt')

	# training network 
	print '\nTesting training of VGG19 network using two images (tiger and puzzle)'
	start_time = time.time()
	_, loss = session.run([vgg.get_optimization_op(), vgg.get_loss_tensor()], feed_dict = {
		vgg.get_input_tensor(): batch,
		vgg.get_input_real_label_tensor(): label,
		vgg.get_trainable_tensor(): True
		})
	print 'time used %d, loss %f'%(time.time() - start_time, loss)

	print 'All testings done!'


