"""
Main File for Actor and Critic
"""

import tensorflow as tf
import numpy as np

from framework.actor_critic import ActorCritic
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

	# Need To Declare an number of Obser and Action from env variable 
	policy = ActorCritic(num_obser_dim=4,
						num_action_dim=4,
						act_backbone=TSDV1,
						cri_backbone=TSDV1)

	policy.init_actor_critic()
	policy.inference()

	# Define Memory
	container = Memory(config["buffer_size"])

	"""
	Define Environment Varible which simulates the considered environment

	"""
	env = .....

	"""
	Perform Training Phase	
	"""
	for episode in range(config["num_episodes"]):
		ob = env.reset()
		done = False

		while not done:
			a = sel_act()


