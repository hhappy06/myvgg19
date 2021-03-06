import os
import numpy as np
import tensorflow as tf

_VGG19_IMAGE_MEAN = [103.939, 116.779, 123.68]
_WEIGHT_INDEX = 0
_BIAS_INDEX = 1
_REGULAR_FACTOR = 1.0e-4
_LEARNING_RATE = 1.0e-4

_VGG19_NETWORK = {
	'conv1_1': [3, 64],
	'conv1_2': [3, 64],
	'conv2_1': [3, 128],
	'conv2_2': [3, 128],
	'conv3_1': [3, 256],
	'conv3_2': [3, 256],
	'conv3_3': [3, 256],
	'conv3_4': [3, 256],
	'conv4_1': [3, 512],
	'conv4_2': [3, 512],
	'conv4_3': [3, 512],
	'conv4_4': [3, 512],
	'conv5_1': [3, 512],
	'conv5_2': [3, 512],
	'conv5_3': [3, 512],
	'conv5_4': [3, 512],
	'fc6': [4096],
	'fc7': [4096],
	'fc8': [1000],
}

_CONV_KERNEL_STRIDES = [1, 1, 1, 1]
_MAX_POOL_KSIZE = [1, 2, 2, 1]
_MAX_POOL_STRIDES = [1, 2, 2, 1]

class VGG19:
	def __init__(self, input_images_size = [224, 224, 3], initialized_parameter_file = None):
		if initialized_parameter_file and os.path.exists(initialized_parameter_file):
			self.initialized_parameter_dict = np.load(initialized_parameter_file, encoding = 'latin1').item()
		else:
			self.initialized_parameter_dict = None

		self.variable_dict = {}
		self.input, self.predict, self.is_trainable, self.input_real_label, self.loss, self.opt = self._build_vgg19_network(input_images_size)
		self.initialized_parameter_dict = None

	def get_input_tensor(self):
		return self.input

	def get_predict_op(self):
		return self.predict

	def get_loss_tensor(self):
		return self.loss

	def get_optimization_op(self):
		return self.opt

	def get_input_real_label_tensor(self):
		return self.input_real_label

	def get_trainable_tensor(self):
		return self.is_trainable

	def _build_vgg19_network(self, input_images_size):
		# input_images is a placeholder with [None, height, width, nchannels]
		input_dimension = [None]
		input_dimension.extend(input_images_size)
		input_images = tf.placeholder(tf.float32, input_dimension)
		r, g, b = tf.split(3, 3, input_images)
		# check the image size
		assert r.get_shape().as_list()[1:] == [224, 224, 1]
		assert g.get_shape().as_list()[1:] == [224, 224, 1]
		assert b.get_shape().as_list()[1:] == [224, 224 ,1]
		whiten_images = tf.concat(3, [
			b - _VGG19_IMAGE_MEAN[0],
			g - _VGG19_IMAGE_MEAN[1],
			r - _VGG19_IMAGE_MEAN[2]])
		# check concated image size
		assert whiten_images.get_shape().as_list()[1:] == [224, 224, 3]

		# construct VGG19 network -- convolution layer
		conv1_1 = self._construct_conv_layer(whiten_images, 'conv1_1')
		conv1_2 = self._construct_conv_layer(conv1_1, 'conv1_2')
		pool1 = self._max_pool(conv1_2, 'pool1')

		conv2_1 = self._construct_conv_layer(pool1, 'conv2_1')
		conv2_2 = self._construct_conv_layer(conv2_1, 'conv2_2')
		pool2 = self._max_pool(conv2_2, 'pool2')

		conv3_1 = self._construct_conv_layer(pool2, 'conv3_1')
		conv3_2 = self._construct_conv_layer(conv3_1, 'conv3_2')
		conv3_3 = self._construct_conv_layer(conv3_2, 'conv3_3')
		conv3_4 = self._construct_conv_layer(conv3_3, 'conv3_4')
		pool3 = self._max_pool(conv3_4, 'pool3')

		conv4_1 = self._construct_conv_layer(pool3, 'conv4_1')
		conv4_2 = self._construct_conv_layer(conv4_1, 'conv4_2')
		conv4_3 = self._construct_conv_layer(conv4_2, 'conv4_3')
		conv4_4 = self._construct_conv_layer(conv4_3, 'conv4_4')
		pool4 = self._max_pool(conv4_4, 'pool4')

		conv5_1 = self._construct_conv_layer(pool4, 'conv5_1')
		conv5_2 = self._construct_conv_layer(conv5_1, 'conv5_2')
		conv5_3 = self._construct_conv_layer(conv5_2, 'conv5_3')
		conv5_4 = self._construct_conv_layer(conv5_3, 'conv5_4')
		pool5 = self._max_pool(conv5_4, 'pool5')

		# construct VGG19 network -- full connection layer
		tensor_trainable = tf.placeholder(tf.bool)
		fc6 = self._construct_full_connection_layer(pool5, 'fc6', active = True)
		fc6 = tf.cond(tensor_trainable, lambda: tf.nn.dropout(fc6, 0.5), lambda: fc6)

		fc7 = self._construct_full_connection_layer(fc6, 'fc7', active = True)
		fc7 = tf.cond(tensor_trainable, lambda: tf.nn.dropout(fc7, 0.5), lambda: fc7)

		fc8 = self._construct_full_connection_layer(fc7, 'fc8', active = False)

		# prediction op
		predict = tf.nn.softmax(fc8, name = 'predict')

		output_dimension = predict.get_shape().as_list()[-1]
		real_label = tf.placeholder(tf.float32, [None, output_dimension])
		# optimization op
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc8, real_label))
		opt = tf.train.AdamOptimizer(_LEARNING_RATE).minimize(loss)

		return input_images, predict, tensor_trainable, real_label, loss, opt

	def _construct_conv_layer(self, input_layer, layer_name):
		assert layer_name in _VGG19_NETWORK
		conv_config = _VGG19_NETWORK[layer_name]

		with tf.variable_scope(layer_name):
			if self.initialized_parameter_dict and layer_name in self.initialized_parameter_dict:
				init_weight = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX])
				init_bias = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_BIAS_INDEX])
				# print 'conv initialize from model'
			else:
				init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.001, dtype = tf.float32)
				init_bias = tf.zeros_initializer([conv_config[1]], dtype = tf.float32)

			filter_shape = [conv_config[0], conv_config[0], input_layer.get_shape()[3], conv_config[1]]
			weight = tf.get_variable(
				name = layer_name + '_weight',
				shape = filter_shape,
				initializer = init_weight,
				regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR))
			bias = tf.get_variable(
				name = layer_name + '_bias',
				shape = [conv_config[1]],
				initializer = init_bias,
				regularizer = None)
			# weight = tf.constant(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX], name="filter")
			# bias = tf.constant(self.initialized_parameter_dict[layer_name][_BIAS_INDEX], name="biases")

			self.variable_dict[layer_name] = [weight, bias]

			conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES, padding = 'SAME')
			active = tf.nn.relu(tf.nn.bias_add(conv, bias))

			return active

	def _construct_full_connection_layer(self, input_layer, layer_name, active = True):
		assert layer_name in _VGG19_NETWORK
		fc_config = _VGG19_NETWORK[layer_name]

		input_dimension = 1
		for dim in input_layer.get_shape().as_list()[1:]:
			input_dimension *= dim

		with tf.variable_scope(layer_name):
			if self.initialized_parameter_dict and layer_name in self.initialized_parameter_dict:
				init_weight = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX])
				init_bias = tf.constant_initializer(self.initialized_parameter_dict[layer_name][_BIAS_INDEX])
			else:
				init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.001, dtype = tf.float32)
				init_bias = tf.zeros_initializer([fc_config[0]], dtype = tf.float32)
			weight = tf.get_variable(
				name = layer_name + '_weight',
				shape = [input_dimension, fc_config[0]],
				initializer = init_weight,
				regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR))
			bias = tf.get_variable(
				name = layer_name + '_bias',
				shape = [fc_config[0]],
				initializer = init_bias,
				regularizer = None)

			# weight = tf.constant(self.initialized_parameter_dict[layer_name][_WEIGHT_INDEX], name="filter")
			# bias = tf.constant(self.initialized_parameter_dict[layer_name][_BIAS_INDEX], name="biases")

			self.variable_dict[layer_name] = [weight, bias]

			reshape_input = tf.reshape(input_layer, [-1, input_dimension])
			if active:
				return tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape_input, weight), bias))
			return tf.nn.bias_add(tf.matmul(reshape_input, weight), bias)

	def _max_pool(self, input_layer, name):
		return tf.nn.max_pool(input_layer, ksize = _MAX_POOL_KSIZE, strides = _MAX_POOL_STRIDES, padding = 'SAME', name = name)

