"""
Action Selection Critiria is Implemeneted here
Authors: TamNV
==============================================
"""
import tensorflow as tf
import numpy as np

def sel_action_actor_critic(env, model, obs, sess, eps):
	"""
	This Function Supports for Selection Policy

	+ Params: env: environment instance
	+ Params: obs: Numpy Array
 	+ Params: model: deep learning model instance
 	+ Params: sess: Session running instance
 	+ Params: eps: Float
 	+ Returns: action
	"""
	if np.random.random() < eps:
		action = env.action_space.sample()
	else:
		if len(obs.shape) == 1:
			obs = np.expand_dims(obs, axis=0)
		if len(obs.shape) != 2:
			raise Exception("Observation Dimension must be 2 but {}".\
				format(len(obs.shape)))

		# Run inference to sellect new actions
		action = sess.run(model._act_probs, feed_dict={model._states:obs})
		action = np.argmax(action, axis=-1)[0]

	return action

