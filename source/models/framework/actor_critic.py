"""
Defining Actor Critic Framework
Authors: TamNV
===============================
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ActorCritic:
	"""
	Actor Critic Framework
	"""
	def __init__(self,
				num_obser_dim,
				num_action_dim,
				act_backbone,
				cri_backbone):
		"""
		Initial Method
		+ Params: num_obser_dim: Integer
		+ Params: num_action_dim: Integer
		+ act_backbone: Class Name
		+ cri_backbone: Class Name
		+ Returns: None
		"""
		self._num_obser_dim = num_obser_dim
		self._num_action_dim = num_action_dim
		self._act_backbone = act_backbone
		self._cri_backbone = cri_backbone

		self._states = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self._num_obser_dim))

		self._next_states = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self._num_obser_dim))
		
		self._actions = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self._num_action_dim))  # One-Hot-Vector Convention 

		self._rewards = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))

		self._gamma = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)

		self._cri_lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
		self._act_lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)

	def init_actor_critic(self):
		"""
		Create An Actor and a Critic
		"""
		self._actor = self._act_backbone(in_dims=self._num_obser_dim,
										out_dims=self._num_action_dim,
										name="actor")

		self._critic = self._cri_backbone(in_dims=self._num_obser_dim,
										out_dims=1,
										name="critic")
		"""
		Get Actor's trainable variables and Critic's trainable variables
		"""
		self._act_variables = self._actor._train_vars
		self._cri_variables = self._critic._train_vars

		print("The Number of Variables for Actor:  {}".format(len(self._act_variables)))
		print("The Number of Variables for Critic: {}".format(len(self._cri_variables)))

	def inference(self):
		
		# Perform Critic Phase
	
		gt = self._critic(self._next_states) * self._gamma + self._rewards
		td_error = gt - self._critic(self._states)
		self._cri_loss = tf.reduce_mean(td_error ** 2)
		
		# Perform Actor Phase
		act_probs = self._actor(self._states)
		self._act_loss = tf.reduce_mean(-td_error * tf.nn.log_softmax(act_probs, axis=-1))

		self._act_op = tf.compat.v1.train.AdamOptimizer(self._act_lr).minimize(self._act_loss, var_list=self._act_variables)
		self._cri_op = tf.compat.v1.train.AdamOptimizer(self._cri_lr).minimize(self._cri_loss, var_list=self._cri_variables)

		self._losses = [self._act_loss, self._cri_loss]
		self._ops = [self._act_op, self._cri_op]

