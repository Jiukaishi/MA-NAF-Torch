
import tensorflow as tf
import numpy as np
import gym
import time
###TD3 for lower trust learning, example 1 DDPG###
###2020/5/11###
###########
##Undo##
##未加入memory正则化##
##target action方差调参##
###########
#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
Noise_clip = 0.5
RENDER = False
ENV_NAME = "Walker2d-v1"

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim+s_dim+a_dim+1), dtype=np.float32)
        # 1(the last dimension) for reward
        self.pointer = 0
        self.sess = tf.Session()

        self.std = 0.1
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.noise = tf.placeholder(tf.float32, [None, a_dim], 'noise')
        self.generate_sample_from_outside_buffer = False
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_plus_noise = tf.add(self.a, tf.clip_by_value(self.noise, -a_bound,a_bound))
            #self.a_plus_noise = tf.add(self.a, tf.clip_by_value(tf.random_normal(shape=[BATCH_SIZE, a_dim],mean=0.0,stddev=self.std,dtype=tf.float32)), -a_bound, a_bound)
            self.a_with_noise = tf.clip_by_value(self.a_plus_noise, -a_bound,a_bound)

            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):

            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
            q1 = self._build_c(self.S, self.a, scope='eval_1', trainable=True)
            q_1 = self._build_c(self.S_, a_, scope='target_1', trainable=False)
        the_target = tf.minimum(q_, q_1)
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.ce1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_1')
        self.ct1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_1')

        # soft replace, combine old para with new para according to coefficient TAU
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params + self.ct1_params, self.ae_params + self.ce_params + self.ce1_params )]
        q_target = self.R + GAMMA * the_target
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        td_error_1 = tf.losses.mean_squared_error(labels=q_target, predictions=q1)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        self.ctrain_1 = tf.train.AdamOptimizer(LR_C).minimize(td_error_1, var_list=self.ce1_params)
        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, current_time):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # indice: which memory you want to use
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        var = (self.std)*np.ones((BATCH_SIZE, self.a_dim))
        noise =np.random.normal(0, var)
        noise = np.clip(noise, -Noise_clip, Noise_clip)
        #print(noise)
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.noise: noise})
        self.sess.run(self.ctrain_1, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.noise: noise})
        #noise = np.rand()
        if (current_time+1) % 3 == 0:
            self.sess.run(self.atrain, {self.S: bs})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net_1 = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_2 = tf.layers.dense(net_1, units=30, use_bias=True, activation=tf.nn.relu,
                                       trainable=trainable)  # 3:    60神经元&relu
            net_3 = tf.layers.dense(net_2, units=40, use_bias=True, activation=tf.nn.relu,
                                       trainable=trainable)  # 4:    40神经元&relu
            a = tf.layers.dense(net_3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(inputs=net_1, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            net_3 = tf.layers.dense(inputs=net_2, units=100, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            net_4 = tf.layers.dense(inputs=net_3, units=60, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                      bias_initializer=tf.constant_initializer(0.1),
                                      trainable=trainable)
            return tf.layers.dense(net_4, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print("action bound is", a_bound, env.action_space.high.shape, env.action_space.low)
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for episode in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if ddpg.generate_sample_from_outside_buffer == False:
            if RENDER:
                env.render()

        # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var,a.shape), -1, 1)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r , s_)
            if var >= 0.1:
                var *= .9999  # decay the action randomness
            ddpg.learn(j)
            if done:
                break

            # if ddpg.pointer > MEMORY_CAPACITY:
            #     if var >= 0.1:
            #         var *= .9999  # decay the action randomness
            #     ddpg.learn(j)
        # else:
        #     var *= .9999  # decay the action randomness
        #     ddpg.learn(j)

            s = s_
            ep_reward += r
        # if j == MAX_EP_STEPS-1:
        #     print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
        #     # if ep_reward > -300:RENDER = True
        #     break
    print(episode, ep_reward)
#############test###########
    # if episode % 50 == 0:
    #   total_reward = 0
    #   for i in range(10):
    #     this_reward = 0
    #     state = env.reset()
    #     for j in range(MAX_EP_STEPS):
    #       env.render()
    #       action = ddpg.choose_action(state) # direct action for test
    #       state, reward, done,_ = env.step(action)
    #       total_reward += reward
    #       this_reward += reward
    #       if done:
    #         break
    #     print("this episode reward:", this_reward)
    #   ave_reward = total_reward/10
      #print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
#print('Running time: ', time.time() - t1)