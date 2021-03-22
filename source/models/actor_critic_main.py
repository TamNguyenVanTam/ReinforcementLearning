"""
Main File for Actor and Critic
"""
import os
import json
import tensorflow as tf
import numpy as np
import gym

from framework.actor_critic import ActorCritic
from framework.utils import sel_action_actor_critic
from backbones.backbonev1 import TSDV1
from base.memory import Memory

from utils import load_json_file, save_checkpoint, load_checkpoint

import argparse
parser = argparse.ArgumentParser(description='Arguments for Actor and Critic Project')
parser.add_argument("--config_file", dest="config_file", help="for Actor and Critic",\
		default="actor_critic_config.json")
parser.add_argument("--phase", dest="phase", help="Dicide for Training or Testing", default="train")
args = parser.parse_args()

if __name__ == "__main__":
	# Load Config File
	config = load_json_file(args.config_file)
	
	if not os.path.exists(config["log_dir"]):
		os.makedirs(config["log_dir"])
	"""
	Define Environment Varible which simulates the considered environment
	"""
	env =  gym.make(config["env_name"])
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.n

	print("Number Actions: {} and Number Dimensions of Observation {}".\
		format(num_actions, num_states))

	# Define Memory
	container = Memory(config["buffer_size"])

	# Need To Declare an number of Obser and Action from env variable 
	policy = ActorCritic(num_obser_dim=num_states,
						num_action_dim=num_actions,
						act_backbone=TSDV1,
						cri_backbone=TSDV1)

	policy.init_actor_critic()
	policy.inference()

	# Create Section for Running

	os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_idx"]
	gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
	sess.run(tf.global_variables_initializer())
	
	exp_rewards_log = {}


	"""
	Perform Training Phase	
	"""
	epsilon = config["max_eps"]
	
	for episode in range(config["num_episodes"]):
		state = env.reset()
		done = False
		
		if (episode + 1) % 50 == 0:
			epsilon *= 0.99
		epsilon = max(epsilon, config["min_eps"])
		
		while not done:

			action = sel_action_actor_critic(env, policy, state, sess, epsilon)
			next_state, reward, done, info = env.step(action)
			
			if done:
				reward = -50.0

			container.insert_samples({'s': [state],
									'a':[action],
									'ns':[next_state],
									'r':[reward]})

			state = next_state


		for e in range(config["num_epoch_per_episode"]):
			batch_data = container.sel_samples(config["batch_size"])

			states = np.array(batch_data["s"])
			actions = np.array(batch_data["a"])
			next_states = np.array(batch_data["ns"])
			rewards = np.array(batch_data["r"])

			one_hot_actions = np.zeros((actions.shape[0], num_actions))
			for idx in range(actions.shape[0]):
				one_hot_actions[idx, actions[idx]] = 1.0
			
			[_, _, act_loss, cri_loss] = sess.run([policy._act_op, policy._cri_op,
												policy._act_loss, policy._cri_loss], 
												feed_dict={policy._states: states,
															policy._next_states:next_states,
															policy._actions: one_hot_actions,
															policy._rewards: rewards,
															policy._gamma: config["gamma"],
															policy._cri_lr: config["cri_lr"],
															policy._act_lr: config["act_lr"]})
		
		# print("Episode {:5f} - Number Size for Container {} - epsilon {:.3f}".format(episode, container._cur_size, epsilon))

		if (episode +1) % 50 == 0:
			# Perform Evaluatation
			print("Episode {:5d} Actor Loss: {:.5f} - Critic Loss {:.5f}".format(episode+1, act_loss, cri_loss))
			
			average_reward = []

			for game in range(10):
				state = env.reset()
				done = False
				
				total_reward = 0
				while not done:
					action = sel_action_actor_critic(env, policy, state, sess, -1.0)
					next_state, reward, done, info = env.step(action)
					
					state = next_state
					total_reward += reward

				average_reward.append(total_reward)

			average_reward = np.mean(average_reward)
			exp_rewards_log["episode_{:8d}".format(episode+1)] = average_reward

			log_file = os.path.join(config["log_dir"], config["log_file"])
		
			with open(log_file, "w") as f:
				json.dump(exp_rewards_log, f, indent=4, sort_keys=True)
