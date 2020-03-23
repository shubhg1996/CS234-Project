import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from scipy.signal import savgol_filter


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.cost_history = []
        self.reward_history = []

        self.returns = []

        self.sess = tf.Session()

        init = tf.initialize_all_variables()
        self.sess.run(init)

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s-0.5)
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        # action = np.zeros(self.n_y)
        # action[a] = 1
        self.episode_actions.append(a)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = observation[:, np.newaxis]

        # Run forward propagation to get softmax probabilities
        action, logits = self.sess.run([self.sampled_action, self.logits], feed_dict = {self.X: observation.T})
        # print(logits)
        # print("prob_weights: ", prob_weights)
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        # action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        # action = np.argmax(np.random.multinomial(1, prob_weights[0]))
        
        # deterministic policy
        # action = np.argmax(prob_weights.ravel())
        return action[0]

    def learn(self):
        # Discount and normalize episode reward
        # discounted_episode_rewards_norm = self.discount_and_norm_rewards()
        # for i in range(len(self.episode_observations)):
        #     print("obs: ", self.episode_observations[i])
        #     print("act: ", self.episode_actions[i])
        #     print("ret: ", self.returns[i])
        # print(list(zip(self.episode_observations, self.episode_actions, self.episode_rewards)))
        # Train on episode
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
             self.X: np.vstack(self.episode_observations),
             self.Y: np.array(self.episode_actions),
             self.discounted_episode_rewards_norm: self.returns
        })
        # print(np.argmax(grad[0][0], axis=1)==np.argmax(grad[0][1], axis=1))
        # exit(0)
        # print(loss)
        self.cost_history.append(loss)
        # if len(self.reward_history)<1:
        # else:
        #     max_rew = max(sum(self.episode_rewards), self.reward_history[-1])
        #     self.reward_history.append(max_rew)
        # Reset the episode data
        self.episode_observations, self.episode_actions  = [], []

        # # Save checkpoint
        # if self.save_path is not None:
        #     save_path = self.saver.save(self.sess, self.save_path)
        #     print("Model saved in file: %s" % save_path)
        self.returns = []
        

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        # discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        # discounted_episode_rewards /= np.std(discounted_episode_rewards)+0.000001
        self.reward_history.append(sum(self.episode_rewards))
        self.episode_rewards = []
        self.returns.extend(discounted_episode_rewards)
        #return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_x), name="X")
        self.Y = tf.placeholder(tf.int32, shape=(None), name="Y")
        self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, (None), name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        # W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer())
        # b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer())
        # W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer())
        # b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer())
        # W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer())
        # b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer())
        # W1 = weight_variable([self.n_x, units_layer_1])
        # b1 = bias_variable([units_layer_1])
        # a1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)

        # Wo = weight_variable([units_layer_1, self.n_y])
        # bo = bias_variable([self.n_y])
        # logits = tf.matmul(a1, Wo) + bo
        # self.outputs_softmax = tf.nn.softmax(logits)

        # h1 = tf.layers.dense(inputs=self.X, units=units_layer_1, activation=tf.nn.relu)
        # h1 = tf.layers.dense(inputs=h1, units=units_layer_1, activation=tf.nn.relu)
        # h1 = tf.layers.dense(inputs=h1, units=units_layer_1, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=self.X, units=self.n_y, activation=None)
        self.sampled_action = tf.squeeze(tf.multinomial(self.logits, 1), axis=1)
        # self.sampled_action = tf.argmax(logits, axis=1)
        # Forward prop
        # with tf.name_scope('layer_1'):
        #     Z1 = tf.add(tf.matmul(W1,self.X), b1)
        #     A1 = tf.nn.relu(Z1)
        # with tf.name_scope('layer_2'):
        #     Z2 = tf.add(tf.matmul(W2, A1), b2)
        #     A2 = tf.nn.relu(Z2)
        # with tf.name_scope('layer_3'):
        #     Z3 = tf.add(tf.matmul(W3, A2), b3)
        #     A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        # logits = tf.transpose(Z3)
        # labels = tf.transpose(self.Y)
        # self.outputs_softmax = tf.nn.softmax(logits, name='A3')


        logprob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
        # self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y)
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = -tf.reduce_mean(logprob * self.discounted_episode_rewards_norm)
        # self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_norm) #+ 0.01*sum(reg_losses)  # reward guided loss
        # log_prob = tf.log(tf.diag_part(tf.matmul(self.outputs_softmax, tf.transpose(self.Y))))# fix this so it doesn't need diag
        # log_prob = tf.reshape(log_prob, (1,-1))
        # loss = tf.matmul(log_prob, self.discounted_episode_rewards_norm)
        # self.loss = -tf.reshape(loss, [-1])

        optim = tf.train.AdamOptimizer(self.lr)
        # self.grad = optim.compute_gradients(self.loss, self.logits)
        self.train_op = optim.minimize(self.loss)

    def plot_cost(self):
        # print(self.cost_history)
        import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(self.cost_history)), [c/(i+1) for i,c in enumerate(np.cumsum(self.cost_history))])
        plt.plot(np.arange(len(self.cost_history)), savgol_filter(self.cost_history, 15, 3))
        plt.ylabel('Loss')
        plt.xlabel('Training Steps')
        plt.savefig('policy_grad_loss.png')
        plt.clf()
        # plt.plot(np.arange(len(self.reward_history)), [c/(i+1) for i,c in enumerate(np.cumsum(self.reward_history))])
        plt.plot(np.arange(len(self.reward_history)), savgol_filter(self.reward_history, 15, 3))
        plt.ylabel('Rewards')
        plt.xlabel('Training Steps')
        plt.savefig('policy_grad_reward.png')