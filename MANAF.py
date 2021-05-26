import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import ctypes
from network import MA_Brain
from Buffer import FIFO_Buffer
import gym
class MANAF_Trainer():
    #We assume that states have been processed into Multi-Agent version
    def __init__(self,  single_action_space, single_state_space, joint_a_dim, joint_s_dim, torpos,\
                 global_s_index, agent_s_index, agent_a_index, lr=0.001, tau=0.01):
        '''
        :param env:
        :param single_action_space: [agent1_adim, agent2_adim, ... ]
        :param single_state_space: [agent1_sdim, agent2_sdim, ... ]
        :param joint_a_dim:
        :param joint_s_dim:
        :param torpos:
        :param lr:
        :param tau:
        '''

        self.torpos = torpos
        self.joint_a_dim = joint_a_dim
        self.joint_s_dim = joint_s_dim
        self.MAX_EPISODES = 1000
        self.MAX_EP_STEPS = 200
        self.single_action_space = single_action_space
        self.single_state_space = single_state_space
        self.LR = lr  # learning rate for actor
        self.GAMMA = 0.995  # reward discount
        self.TAU = tau  # soft replacement
        self.MEMORY_CAPACITY = 50000
        self.BATCH_SIZE = 100
        self.RENDER = False
        self.device = 'cuda'
        self.action_space = None#env.action_space
        self.state_space = None #env.observation_space
        #print('action_bound', self.action_space)
        self.s_dim = 12
        self.a_dim = 4
        self.global_s_index, self.agent_s_index, self.agent_a_index = global_s_index, agent_s_index, agent_a_index
        self.train_timer = 0
        self.buffer = FIFO_Buffer(self.s_dim, self.a_dim, self.MEMORY_CAPACITY)
        self.brain = MA_Brain(self.single_action_space, self.single_state_space, self.joint_a_dim,self.joint_s_dim,\
                              self.torpos, self.LR, self.TAU)

    def store_memory(self, transitions, s, r):
        """
        :param transitions: [s, a, r, s_, done] (narray)

        :return: current length of the buffer
        """
        self.buffer.store_transitions(transitions, s, r)
        return self.buffer.get_length()
    def obs_to_joints(self, obs):
        result = []
        for partial_topo in self.torpos:
            joint_s_index = self.agent_s_index[partial_topo[0]] + self.agent_s_index[partial_topo[1]] + self.global_s_index
            s = obs[joint_s_index]
            result.append(s)
        return result
    def parse_transitions(self, bs, ba, bs_, br, bdone):
        '''
        :param bs:
        :param ba:
        :param global_index: the indexes of the global information, a list e.g. [4,5]
        :param agent_index: the indexes of each agent's information, a list with lists e.g.[[1],[2],[3]]
        :return: a list with every torpo's states, actions, next states and reward, dones (a dict). {state_batch}
        '''
        global_s_index, agent_s_index, agent_a_index = self.global_s_index, self.agent_s_index, self.agent_a_index
        dics = []
        for partial_topo in self.torpos:
            joint_s_index = agent_s_index[partial_topo[0]] + agent_s_index[partial_topo[1]] + global_s_index
            joint_a_index = agent_a_index[partial_topo[0]] + agent_a_index[partial_topo[1]]
            s = bs[:,joint_s_index]
            a =  ba[:, joint_a_index]
            s_ = bs_[:,joint_s_index]
            r = br
            done =  bdone
            s = torch.FloatTensor(s).to(self.device)
            a = torch.FloatTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            s_ = torch.FloatTensor(s_).to(self.device)
            done = torch.FloatTensor(done).to(self.device)
            dic = {'states': s, 'actions': a,\
                   'next_states': s_, 'reward':r, 'dones': done,}
            dics.append(dic)
        return dics

    def sample_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        total_candidates = self.buffer.get_length()
        indices = np.random.choice(total_candidates, size=batch_size)
        # indice: which memory you want to use
        scale, offset = self.buffer.s_regular.get()
        # scale[-1] = 1
        # offset[-1] = 0
        r_scale, r_offset = self.buffer.r_regular.get()
        bt = self.buffer.memory[indices, :]
        bs = (bt[:, :self.s_dim] - offset)*scale
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        br = (bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]-r_offset)*r_scale
        bs_ = (bt[:,self.s_dim + self.a_dim + 1 : 2*self.s_dim + self.a_dim + 1] - offset)*scale
        bdone = bt[:, 2*self.s_dim + self.a_dim + 1: 2*self.s_dim + self.a_dim + 2]
        # TODO: set params
        dics = self.parse_transitions(bs, ba, bs_, br, bdone)
        return dics

    def update(self, minimal_buffer_size=1000):
        if self.buffer.get_length() < minimal_buffer_size:
            return False
        batch_data = self.sample_data()
        losses = self.brain.update(batch_data)

        return losses


if __name__ == '__main__':
    # env = gym.make('Walker2d-v1')
    # env = env.unwrapped
    # env.seed(1)
    result = []
    #TODO: set agent params
    CUR_PATH = os.path.dirname(__file__)
    dllPath = os.path.join(CUR_PATH, "manta.dll")
    # print(dllPath)
    pdll = 'D:/仿生鱼材料/仿生鱼/biomanta/x64/Debug/manta.dll'
    pdll = ctypes.WinDLL(pdll)
    manta_step = pdll.manta_step
    state_act_time = np.zeros(17, dtype='float64')

    manta_step.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]

    global_s_index = [0,1,2,3,4,5,6,7,8,9,10,11]
    agent_s_index = \
        [[],\
        [],\
        [],\
        [],\
        [],\
        []]
    agent_a_index = [[0,2],[1,3]]
    torpos =[[0, 1]]
    single_action_space = [2,2]
    single_state_space = []
    joint_s_dim = [12]
    joint_a_dim = [4]
    agent = MANAF_Trainer( single_action_space, single_state_space, joint_a_dim, joint_s_dim, torpos,\
                          global_s_index, agent_s_index, agent_a_index)

    ep = 0
    while True:
        done = False
        s = np.zeros(12,dtype='float64')
        s[1] = -5
        s[6] = 0.1
        total_reward = 0
        ep += 1
        step = 0
        time = 0
        state_act_time = np.zeros(17,dtype='float64')
        while not done and step<10000:
            scale, offset = agent.buffer.s_regular.get()
            regular_s = (s-offset)*scale
            joint_s = agent.obs_to_joints(regular_s)
            act = agent.brain.select_action(joint_s,sum(single_action_space))
            noise = np.random.normal(0, 0.2, act.shape)
            act = np.clip(act + noise, -1, 1)
            state_act_time[1:5] = act
            state_act_time[5:17] = s
            state_act_time[0] = time
            manta_step(state_act_time)

            s_, reward, done = state_act_time[0:12], state_act_time[12:13],\
            state_act_time[13:14]
            #防止出现nan
            # if not (reward<10000 and reward >-10000):
            #     break
            total_reward += reward
            #print(reward)
            temp = np.hstack((s, act, reward, s_, done))
            #temp =np.hstack((s, act, reward, s_, done)).reshape(-1,l)
            agent.store_memory(temp.reshape(-1, len(temp)), s, reward)
            s = s_
            agent.update()
            step+=1
            time+=0.001

        #agent.exploration.reset()
        print('epoch number: ',ep, 'step:', step, total_reward)
        # if ep % 5 == 0:
        #     test_time = 0
        #     total_reward = 0
        #     step_ = 0
        #     while test_time < 5:
        #         done = False
        #         s = env.reset()
        #         test_time+=1
        #         while not done and step_<5000:
        #             scale, offset = agent.buffer.s_regular.get()
        #             regular_s = (s - offset) * scale
        #             joint_s = agent.obs_to_joints(regular_s)
        #             act = agent.brain.select_action(joint_s, sum(single_action_space))
        #             act = np.clip(act, -1, 1)
        #             s_, reward, done, _ = env.step(act)
        #             if not (reward < 10000 and reward > -10000):
        #                 break
        #             total_reward += reward
        #             s = s_
        #             step_ += 1
        #     print('test avg reward:', total_reward/5, 'avg_epoch:',  step_/5)
        #     result.append(total_reward/5)
        #     #TODO: save model
        #     #torch.save(agent_a_.state_dict(), '\parameter.pkl')
