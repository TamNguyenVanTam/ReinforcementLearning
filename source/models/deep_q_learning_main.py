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
        default="deep_q_learning_config.json")
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
    saver = tf.train.Saver()
    """
    Perform Training Phase  
    """
    epsilon = config["max_eps"]
    dis_lr = 1.0
    for episode in range(config["num_episodes"]):

        state = env.reset()
        done = False
        if (episode + 1) % 50 == 0:
            epsilon *= 0.99

        if (episode + 1) % 1000 == 0:
            dis_lr *= 10.0

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
                                policy._gamma: config["gamma"],
                                policy._lr: config["lr"]/dis_lr})

        if (episode +1) % 50 == 0:          
            average_reward, updated = [], False
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

            
            log_file = os.path.join(config["log_dir"], config["log_file"])
        
            with open(log_file, "w") as f:
                json.dump(exp_rewards_log, f, indent=4, sort_keys=True)

            if average_reward > best_reward:
                updated = True   
                best_reward = average_reward
                save_path = os.path.join(config["checkpoint_dir"], config["checkpoint"])
                saver.save(sess, save_path)
            
            # Perform Evaluatation
            print("Episode {:5d}  Loss: {:.5f} - Expectation Reward {:.5f} - Epsilon {:.5f} - Updated : {}".\
                format(episode+1, loss, average_reward, epsilon, updated))