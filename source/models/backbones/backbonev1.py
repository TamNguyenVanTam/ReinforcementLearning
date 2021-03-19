"""
A Simple Network Architecture only Using Dense Layer
Authors: TamNV
====================================================
"""
import tensorflow as tf
from layers.layer import Dense

def lrelu(x):
	"""
	Perform Leak ReLU
	"""
	return tf.maximum(x*0.2, x)

def linear(x):
	"""
	Perform Liear Function
	"""
	return x

class TSDV1:
	"""
	Declare a fisrt version using only fully connected layers 
	"""
	def __init__(self, in_dims, out_dims, name):
		"""
		Initial Method
		Params:	+ in_dims: Integer
		Params: + out_dims: Integer	
		Params: + name: String
		"""
		self._name = name 
		self._in_dims = in_dims
		self._out_dims = out_dims

		self.create_network()

	def create_network(self):
		"""
		Create Deep Learning Network
		"""

		layer1 = Dense(input_dim=self._in_dims,
					output_dim=32,
					act=lrelu,
					bias=True)
			
		layer2 = Dense(input_dim=32,
					output_dim=32,
					act=lrelu,
					bias=True)

		layer3 = Dense(input_dim=32,
					output_dim=self._out_dims,
					act=linear,
					bias=True)

		self._layers = [layer1, layer2, layer3]
		# Get Trainble Variable
		self._train_vars = []
		for layer in self._layers:
			for key in layer._vars.keys():
				self._train_vars.append(layer._vars[key])

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

	def _call(self, inputs):
		"""
		Perform Inference Phase
		+ Params: inputs: Tensor Object
		+ Returns: outputs: Tensor Object
		"""

		x = inputs
		for layer in self._layers:
			x = layer(x)
		outputs = x
		return outputs
