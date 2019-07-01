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
import matplotlib.pyplot as plt


#####################  hyper parameters  ####################
SEED = 1

#USE_MEMORY = False
MEMORY_CAPACITY = 80000

MAX_EPISODES = 40000
MEMORY_FULL = False
EXPLORE_EP_STEPS = 200
MAX_EP_STEPS = 200
LR_A = 0.0005    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 128
val = 1       # exploration factor

RENDER = False
#ENV_NAME = 'Pendulum-v0'

SHOW_EPISODE = 3000
SHOW_REWARD = -80
origins = []
from AC_env import MAZE_W
from AC_env import MAZE_H
for i in range(MAZE_W):
    for j in range(MAZE_H):
        origins.append([i, j])


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

    def seed(self, s):
        np.random.seed(s)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.a, {self.S: s})  # get probabilities for all actions
        return probs

    def learn(self):
        global MEMORY_FULL, MEMORY_CAPACITY
        if self.pointer >= MEMORY_CAPACITY:
            if MEMORY_FULL == False:
                print("Start learning.")
            MEMORY_FULL = True
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
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

env = Maze(origins,
    origins)
env.seed(SEED)

s_dim = env.n_features
a_dim = env.n_actions

ddpg = DDPG(a_dim, s_dim)
ddpg.seed(SEED)

#gap = 0.0
#y_reward = []

for i in range(MAX_EPISODES):
    if i >= SHOW_EPISODE: RENDER = True
    if RENDER:
        time.sleep(0.5)
    s, min_steps = env.reset()
    ep_reward = 0

    # Decrease max episode steps when starting to learn
    if MEMORY_FULL:
        EXPLORE_EP_STEPS = MAX_EP_STEPS
    for j in range(1, EXPLORE_EP_STEPS+1):
        if RENDER:
            env.render()
            time.sleep(0.15)

        # Choose action and get reward
        act_probs = ddpg.choose_action(s).flatten()

        # Add exploration
        if RENDER:
            val = 1
        if np.random.uniform() < val:
            a = np.random.choice(act_probs.size)               # Randomly choose one
        else:
            a = np.random.choice(act_probs.size, p=act_probs)  # return a int
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, act_probs, r, s_)
        ddpg.learn()

        s = s_
        ep_reward += r
        if j >= MAX_EP_STEPS or done:
            if RENDER:
                env.render()
                time.sleep(0.5)

            if i >= SHOW_EPISODE:
                # Compute gap between the result and the optimal path
                #print("Gap = %f, gap + %d" %(gap, (j-min_steps)))
                assert(j >= min_steps)
                #gap += (j - min_steps)

            print('Episode:', i, ' Reward: %.3f' % ep_reward)
            #y_reward.append(ep_reward)

            break

#avg_gap = gap / (MAX_EPISODES - SHOW_EPISODE)
#print("Average gap is %f" %(avg_gap))
#print("%f" %(avg_gap))

#import pickle
#with open('figures/r_dis', 'wb') as f:
#    pickle.dump(y_reward, f)


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
