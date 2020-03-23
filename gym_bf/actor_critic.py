import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import gym_bf
from gym.envs.registration import register 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def make_env(n=10, m=2, space_seed=0):
  # create environment
  id = "BF"+str(n)+"_"+str(m)+"_"+str(space_seed)+"-v0"
  try :
    register(id=id,entry_point='gym_bf.envs:BFEnv',kwargs = {"space_seed":space_seed,"n":n, "m":m})
  except :
    print("Environment with id = "+id+" already registered.Continuing with that environment.")
  env=gym.make(id)
  env.seed(0)
  return env

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, n_x, n_y, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=(n_x), name="state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            # self.state = tf.placeholder(tf.int32, shape=(), name="state")

            self.n_x = n_x
            self.n_y = n_y

            n_hidden = 64
            
            self.state = self.state - 0.5
            # Experiments with varying model architecture

            # hidden = tf.contrib.layers.fully_connected(
            #     inputs=tf.expand_dims(self.state, 0),
            #     num_outputs=n_hidden,
            #     activation_fn=tf.nn.relu,
            #     weights_initializer=tf.zeros_initializer)
            # hidden = tf.contrib.layers.fully_connected(
            #     inputs=hidden,
            #     num_outputs=n_hidden,
            #     activation_fn=tf.nn.relu,
            #     weights_initializer=tf.zeros_initializer)
            # filt1 = tf.Variable(np.array([3, 1, 3]))
            # n_chan = 4
            # hidden = tf.nn.conv1d(
            #     input=tf.reshape(self.state, (1, n_x, 1)),
            #     filters=tf.Variable(tf.random_normal([3, 1, n_chan]), dtype=tf.float32),
            #     stride=1,
            #     padding="VALID"
            # )
            # hidden = tf.nn.relu(hidden)
            # hidden = tf.nn.conv1d(
            #     input=hidden,
            #     filters=tf.Variable(tf.random_normal([3, n_chan, 1]), dtype=tf.float32),
            #     stride=1,
            #     padding="VALID"
            # )
            # hidden = tf.nn.relu(hidden)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=n_y,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, n_x, n_y, learning_rate=0.01, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=(n_x), name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            # self.state = tf.placeholder(tf.int32, shape=(), name="state")

            self.n_x = n_x

            n_hidden = 64
            n_hidden2 = 256
            # self.output_layer = tf.contrib.layers.fully_connected(
            #     inputs=tf.expand_dims(state_one_hot, 0),
            #     num_outputs=n_y,
            #     activation_fn=None,
            #     weights_initializer=tf.zeros_initializer)
            self.state = self.state - 0.5
            hidden = tf.contrib.layers.fully_connected(
                inputs=tf.reshape(self.state, (1, n_x)),
                num_outputs=n_hidden,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.zeros_initializer)
            hidden = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=n_hidden2,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.zeros_initializer)
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hidden,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            # self.output_layer = tf.contrib.layers.fully_connected(
            #     inputs=tf.expand_dims(self.state, 0),
            #     num_outputs=1,
            #     activation_fn=None,
            #     weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())        
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        A stats dict with two numpy arrays for episode_lengths and episode_rewards.
    """

    stats = {
        'episode_lengths' : np.zeros(num_episodes),
        'episode_rewards' : np.zeros(num_episodes)
    }
    
    test_render_file = open("./ac_31_2.txt","w")
    for i_episode in range(num_episodes):
        # Reset env
        state = env.reset()
        
        episode = []
        
        for t in itertools.count():
            
            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Update statistics
            stats['episode_rewards'][i_episode] += reward
            stats['episode_lengths'][i_episode] = t
            
            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            
            # Update the value estimator
            estimator_value.update(state, td_target)
            
            # Update the policy estimator using td error
            estimator_policy.update(state, td_error, action)
            
            # Print out step for debugging 
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats['episode_rewards'][i_episode - 1]), end="")
            render_string = env.render(mode='ansi')+"\n"
            test_render_file.write(render_string)

            if done:
                break
                
            state = next_state
    test_render_file.close()
    return stats

env = make_env(31,2,space_seed=12)
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(n_x=env.observation_space.shape[0], n_y=env.action_space.n)
value_estimator = ValueEstimator(n_x=env.observation_space.shape[0], n_y=env.action_space.n)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = actor_critic(env, policy_estimator, value_estimator, 25000)
    
    # plt.plot(np.arange(len(self.cost_history)), [c/(i+1) for i,c in enumerate(np.cumsum(self.cost_history))])
    plt.plot(savgol_filter(stats['episode_rewards'], 15, 3))
    plt.savefig('episode_rewards.png')
    plt.clf()
    # plt.plot(np.arange(len(self.reward_history)), [c/(i+1) for i,c in enumerate(np.cumsum(self.reward_history))])
    # plt.plot(stats.episode_rewards)
    # plt.savefig('episode_rewards.png')