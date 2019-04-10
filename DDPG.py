"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
from AC_env import Maze
import time


#####################  hyper parameters  ####################
#USE_MEMORY = False
MEMORY_CAPACITY = 20000

MAX_EPISODES = 200000
MEMORY_FULL = False
EXPLORE_EP_STEPS = 250
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

SHOW_EPISODE = 1000
SHOW_REWARD = -80
origins = []
for i in range(9):
    for j in range(9):
        origins.append([i, j])
env = Maze(origins,
    [[8, 2], [4, 4]])

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.a, {self.S: s})  # get probabilities for all actions
        return probs
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int

    def learn(self):
        if self.pointer >= MEMORY_CAPACITY:
            MEMORY_FULL = True
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        #elif self.pointer >= 2000:
        #    indices = np.random.choice(self.pointer, size=BATCH_SIZE)
        else:
            return
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return tf.layers.dense(net, self.a_dim, activation=tf.nn.softmax, name='act_prob', trainable=trainable)

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

# env = gym.make(ENV_NAME)
# env = env.unwrapped
# env.seed(1)



# s_dim = env.observation_space.shape[0]
# a_dim = env.action_space.shape[0]
# a_bound = env.action_space.high

s_dim = env.n_features
a_dim = env.n_actions

#ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg = DDPG(a_dim, s_dim)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    # Decrease max episode steps when starting to learn
    if MEMORY_FULL:
        EXPLORE_EP_STEPS = MAX_EP_STEPS
    for j in range(EXPLORE_EP_STEPS):
        if RENDER:
            env.render()
        #env.render()

        # Add exploration noise
        act_probs = ddpg.choose_action(s).flatten()
        a = np.random.choice(act_probs.size, p=act_probs)  # return a int
        # a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, act_probs, r, s_)

        ddpg.learn()
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
        #    ddpg.learn()

        s = s_
        ep_reward += r
        if j >= MAX_EP_STEPS-1 or done:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if i > SHOW_EPISODE and ep_reward >= SHOW_REWARD: RENDER = True
            break

print('Running time: ', time.time() - t1)

# """
# Deep Deterministic Policy Gradient (DDPG) for sound source localization.
#
# Author: Junjie Wang
# Date: March 26, 2019
#
# """
# import tensorflow as tf
# import numpy as np
# import time
# from AC_env import Maze
#
#
# class Actor(object):
#     def __init__(self, sess, s_dim, a_dim, lr):
#         self.sess = sess
#         self.s_dim = s_dim
#         self.a_dim = a_dim
#         self.lr = lr
#         self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
#         self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
#
#         with tf.variable_scope('Actor'):
#             # For prediction
#             self.a = self._build_net(S, self.a_dim)
#
#             # For critic
#             self.a_ = self._build_net(S_, self.a_dim)
#
#
#     def _build_net(self, input, output_dim):
#         init_w = tf.random_normal_initializer(0., 0.3)
#         init_b = tf.constant_initializer(0.1)
#         l1 = tf.layers.dense(input, 30, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b)
#         return tf.layers.dense(l1, output_dim, activation=tf.nn.softmax, kernel_initializer=init_w, bias_initializer=init_b)
#
#     def choose_action(self, s):
#         s = s[np.newaxis, :]  # single state
#         return self.sess.run(self.a, feed_dict={self.S: s})[0]  # single action
#
#     def learn(self):
#
