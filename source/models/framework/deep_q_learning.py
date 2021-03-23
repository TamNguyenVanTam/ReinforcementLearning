"""
Defining Deep Q Learning Framework
Authors: TamNV
===============================
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DeepQLearning:
	
	def __init__(self,
				num_obser_dims,
				num_action_states,
				backbone):

		self._num_obser_dims = num_obser_dims
		self._num_action_states = num_action_states

		self._backbone = backbone

		self._states = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self._num_obser_dims))

		self._next_states = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self._num_obser_dims))
		
		self._actions = tf.compat.v1.placeholder(dtype=tf.int32,
							shape=(None, ))  # One-Hot-Vector Convention 

		self._rewards = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
		self._dones = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ))

		self._gamma = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
		self._lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)

	def _define_model(self):
		"""
		Create Agent Model
		"""
		self._agent = self._backbone(in_dims=self._num_obser_dims,
									out_dims=self._num_action_states,
									name="q_learning_agent")

		# Get Agent's Trainable Variables
		self._agent_variables = self._agent._train_vars
		print("The Number of Variables for Agent: {}".format(len(self._agent_variables)))


	def inference(self):
		"""
		Perfome Model Inference 

		"""
		one_hot_actions = tf.one_hot(self._actions, self._num_action_states)
		self._outputs = self._agent(self._states)

		q_action_states = tf.reduce_sum(one_hot_actions * self._outputs , axis=-1)
		
		# Using Bellman Equation to Caculate Ground-Truth
		q_next_states = self._agent(self._next_states)
		q_max_next_actions = tf.reduce_max(q_next_states, axis=-1)

		grouth_truth = (q_max_next_actions * (1.0 - self._dones)) * self._gamma + self._rewards

		self._loss = tf.losses.mean_squared_error(grouth_truth, q_action_states)

		self._opt = tf.compat.v1.train.AdamOptimizer(self._lr).minimize(loss=self._loss, var_list=self._agent_variables)

# if __name__ == "__main__":
# 	DeepQLearning()