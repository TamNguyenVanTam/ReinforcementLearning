"""
Main File for Deep Q Learning Model
"""
import os
import json
import tensorflow as tf
import numpy as np
import gym

from framework.deep_q_learning import DeepQLearning
from framework.utils import sel_action_deep_q_learning
from backbones.backbonev1 import TSDV1
from base.memory import Memory

from utils import load_json_file

import argparse
parser = argparse.ArgumentParser(description='Arguments for Deep Q Learning')
parser.add_argument("--config_file", dest="config_file", help="for Deep Q Learning",\
        default="actor_critic_config.json")
parser.add_argument("--phase", dest="phase", help="Dicide for Training or Testing", default="train")
args = parser.parse_args()

if __name__ == "__main__":
    # Load Config File
    config = load_json_file(args.config_file)
    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
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
    policy = DeepQLearning(num_states, num_actions, TSDV1)
    policy._define_model()
    policy.inference()
    print("Build Agent Successfully !")

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_idx"]
    gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())

    exp_rewards_log, best_reward = {}, -np.Inf
    """
    Perform Training Phase  
    """
    epsilon = config["max_eps"]
    for episode in range(config["num_episodes"]):
        state = env.reset()
        done = False
        if (episode + 1) % 50 == 0:
            epsilon *= 0.999
        epsilon = max(epsilon, config["min_eps"])
        while not done:
            action = sel_action_deep_q_learning(env, policy, state, sess, epsilon)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = -10
            container.insert_samples({'s': [state],
                                    'a':[action],
                                    'ns':[next_state],
                                    'r':[float(reward)],
                                    "d":[float(done)]})
            state = next_state

        batch_data = container.sel_samples(config["batch_size"])
        states = np.array(batch_data["s"])
        actions = np.array(batch_data["a"])
        next_states = np.array(batch_data["ns"])
        rewards = np.array(batch_data["r"])
        dones = np.array(batch_data["d"])
            
        [_, loss] = sess.run([policy._opt, policy._loss],
                            feed_dict={
                                policy._states: states,
                                policy._next_states:next_states,
                                policy._actions: actions,
                                policy._rewards: rewards,
                                policy._dones: dones,
                                policy._gamma: 0.9,
                                policy._lr: 1e-3})

        if (episode +1) % 50 == 0:          
            average_reward = []
            for game in range(10):
                state = env.reset()

                done = False
                total_reward = 0
                while not done:
                    action = sel_action_deep_q_learning(env, policy, state, sess, -np.Inf)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    total_reward += reward
                average_reward.append(total_reward)
                
            average_reward = np.mean(average_reward)
            exp_rewards_log["episode_{:8d}".format(episode+1)] = average_reward
            # Perform Evaluatation
            print("Episode {:5d}  Loss: {:.5f} - Expectation Reward {:.5f} - Epsilon {:.5f}".format(episode+1, loss, average_reward, epsilon))