import os
from abc import abstractmethod

import tensorflow as tf
import numpy as np

def glorot(shape, name=None, trainable=True):
	"""
	Create a random variable follow by 
		Glorot & Bengio (AISTATS 2010)
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
		trainable: Boolean
			which indicate the created weights are trainable or not
	Returns:
		Tensor
	"""
	# scale = np.sqrt(6.0/(shape[-1]+shape[-2]))
	scale = np.sqrt(6.0/(shape[0]+shape[1]))
	kernel = tf.random.uniform(
					shape, minval=-scale, maxval=scale, 
					dtype=tf.float32, seed=1)
	return tf.Variable(kernel, name=name, trainable=trainable)

def zeros(shape, name=None, trainable=True):
	"""
	Create a random variable what have all
		elements equal 0
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
		trainable: Boolean
			which indicate the created weights are trainable or not
	Returns:
		Tensor
	"""
	kernel = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(kernel, name=name, trainable=trainable)

def ones(shape, name=None, trainable=True):
	"""
	Create a random variable what have all
		elements equal 1
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
		trainable: Boolean
			which indicate the created weights are trainable or not
	Returns:
		Tensor
	"""
	kernel = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(kernel, name=name, trainable=trainable)

_NAME2ID = {}

def get_layer_uid(name=""):
	"""
	Assign layer to a unique IDs
	"""
	if name not in _NAME2ID:
		_NAME2ID[name] = 1
		return 1
	else:
		_NAME2ID[name] += 1
		return _NAME2ID[name]

def leak_relu(x):
	"""
	Perform leak_relu function
	params:
		x: Tensor Object
	returns:
		y: Tensor Object
	"""
	return tf.maximum(x*0.2, x)

def linear(x):
	"""
	perform linear function

	params:
		x: Tensor Object
	returns
		x: Tensor Object
	"""
	return x

class Layer(object):
	"""
	Abtract Class, which all Layers class inherit  
	"""
	def __init__(self, **kwargs):
		allowed_kwargs = {"name"}
		
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, "Invalid Keyword Argument: {}".format(kwarg)
		name = kwargs.get("name")

		if not name:
			layer_name = self.__class__.__name__.lower()
			name = "{}_{}".format(layer_name, get_layer_uid(layer_name))	

		self._name = name

		self._vars = {}

	def _call(self, inputs):
		"""
		Perform layer operations

		Params:
			inputs: Tensorflow instance
		Returns:
			Tensorflow instance
		"""
		return inputs

	def __call__(self, inputs):
		"""
		Perform layer's function

		Params:
			inputs: Tensorflow instance
		Returns:
			outputs: Tensorflow instance
		"""
		with tf.name_scope(self._name):
			outputs = self._call(inputs)
			return outputs

class Dense(Layer):
	"""
	Feed-forward layer class
	"""
	def __init__(self, 
				input_dim,
				output_dim,
				act, 
				bias,
				**kwargs):
		"""
		Initial method

		Params:
			input_dim: Input's dimention
			output_dim: Output's dimention
			act: Activation function
			bias: Boolean, which indicates using bias or not
			sparse_inputs: Boolean,
				which indecates input is a sparse matric or a dense matric
		Returns:
			None			 
		"""
		super(Dense, self).__init__(**kwargs)
		self._input_dim = input_dim
		self._output_dim = output_dim

		self._act = act
		self._bias = bias

		# declare layer's paramters
		with tf.compat.v1.variable_scope("{}_vars".format(self._name)):
			self._vars['weights'] = glorot([self._input_dim, self._output_dim],
											name="weights")
			if self._bias:
				self._vars["bias"] = zeros([self._output_dim], name="bias")

	def _call(self, inputs):
		"""
		Perform feed-forward operation
		Y = activation_function (W*X + b)

		Params:
			inputs: 2-D tensor instance
		Returns:
			outputs: 2-D tensor instance 
		"""
		x = inputs
		outputs = tf.matmul(x, self._vars['weights'])
		if self._bias:
			outputs += self._vars["bias"]

		outputs = self._act(outputs)
		return outputs
