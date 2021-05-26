import numpy as np
from Running_mean_std import RunningMeanStd
class FIFO_Buffer():
    def __init__(self, s_dim, a_dim, MEMORY_CAPACITY):
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.s_dim = s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1+1), dtype=np.float32)
        self.pointer = 0
        self.s_regular = RunningMeanStd(s_dim)
        self.r_regular = RunningMeanStd(1)


    def store_transitions(self, transitions, s, r):
        s = s.reshape([-1, self.s_dim])
        r = r.reshape([-1,1])
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        if index + len(transitions) >= self.MEMORY_CAPACITY:
            end = self.MEMORY_CAPACITY
        else:
            end = index + len(transitions)
        self.memory[index:end, :] = transitions[:end - index, :]
        self.pointer += len(transitions)
        self.s_regular.update(s)
        self.r_regular.update(r)

    def get_length(self):
        if self.pointer <= self.MEMORY_CAPACITY:
            return self.pointer
        else: return self.MEMORY_CAPACITY
    def parse_data(self, state_list, reward_list, action_list):
        transitions = np.zeros((reward_list.shape[0] * (reward_list.shape[1] - 1), self.state_space.shape[0] * 2 +
                                self.action_space.shape[0] + 1), dtype=np.float32)
        # TODO:reward维度未必是1
        i = 0
        for n in range(reward_list.shape[0]):
            for t in range(reward_list.shape[1] - 1):
                transition = np.hstack((state_list[n][t], action_list[n][t], reward_list[n][t], state_list[n][t + 1]))
                # 这里可能出错
                transitions[i] = transition
                i += 1
        return transitions