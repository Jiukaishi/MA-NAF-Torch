import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class Build_q(nn.Module):
    def __init__(self, states_dim, actions_dim):
        """
        :param states_dim: 输入应该是两个agent共同状态的维度
        :param actions_dim: 输入应该是两个agent共同动作的维度
        """
        super(Build_q, self).__init__()
        self.device = 'cuda'
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        #self.bn1 = nn.BatchNorm1d(states_dim)
        self.fc1 = nn.Linear(states_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.V = nn.Linear(32, 1)
        self.mu = nn.Linear(32, actions_dim)
        self.L = nn.Linear(32,actions_dim**2)
        self.lower_tril_musk = torch.autograd.Variable(\
            torch.tril(torch.ones(actions_dim,actions_dim),diagonal=-1).unsqueeze(0)\
            ).to(self.device)
        #diagonal=-1 下三角矩阵对角线上元素为0
        self.diag_mask = torch.autograd.Variable(torch.diag(torch.diag(
            torch.ones(actions_dim, actions_dim))).unsqueeze(0)).to(self.device)


    def forward(self, state, action):
        """
        :param state: state是两个agent的联合动作
        :param action: action是两个agent的联合动作
        :return:
        """
        # state_action = torch.cat([state, action], 1)
       # state = self.bn1(state)
        s = F.relu(self.fc1(state))
        s = F.relu(self.fc2(s))
        V = self.V(s)
        mu = F.tanh(self.mu(s))
        L = self.L(s).view(-1, self.actions_dim, self.actions_dim)
        L = L * self.lower_tril_musk.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
        P = torch.bmm(L, L.transpose(1, 2))
        A, Q = None, None
        if action is not None:
            a_minus_mu = (action - mu).unsqueeze(2).cuda()
            A =  -0.5*torch.bmm(\
                torch.bmm(a_minus_mu.transpose(1,2),P),a_minus_mu)[:,:,0]
            Q = A+V
        return V, mu, P, A, Q

class MA_Brain(nn.Module):
    def __init__(self, single_action_space, single_state_space, joint_a_dim, joint_s_dim, torpos, lr, tau):
        """
        :param single_action_space:  一个列表，每个元素 对应着 每个agent的动作维度
        :param single_state_space:  一个列表，每个元素 对应着 每个agent的状态维度
        :param torpos:  一个列表，每个元素 对应着 不同链接对应的两个agent的序号
        :param joint_action_bound: 一个列表，每个元素是一个小列表，是[[low_bound, up_bound],[]]
        """
        super(MA_Brain, self).__init__()
        self.LR = lr
        self.Tau = tau
        self.GAMMA = 0.999
        self.device = 'cuda'
        self.obs_dims = single_state_space
        self.act_dims = single_action_space
        self.torpos = torpos
        self.joint_a_dim = joint_a_dim
        self.joint_s_dim = joint_s_dim

        self.qs= [Build_q(self.joint_s_dim[i], self.joint_a_dim[i]).to(self.device) for i in range(len(torpos))]
        self.target_qs = [Build_q(self.joint_s_dim[i], self.joint_a_dim[i]).to(self.device) for i in range(len(torpos))]
        #define optimizers
        self.policy_optimizers =\
            [torch.optim.Adam(self.qs[i].parameters(), lr=self.LR) for i in range(len(torpos))]
        #keep the target network in consist with the network
        for i in range(len(torpos)):
            self.qs[i].load_state_dict(self.target_qs[i].state_dict())

    #TODO:编写训练过程，注意维度
    def select_action(self, batch_joint_states, total_action_dims):
        """
        :param batch_joint_states: [torpo_number * joint states]
        :param total_action_dims: the sum of every agent's actions
        :return:
        """
      #  print(len(self.torpos))
        P_values = [np.zeros((self.act_dims[self.torpos[i][0]]+self.act_dims[self.torpos[i][1]],\
                             self.act_dims[self.torpos[i][0]]+self.act_dims[self.torpos[i][1]])\
                             )for i in range(len(self.torpos))]
        Mu_values = [np.zeros((self.act_dims[self.torpos[i][0]]+self.act_dims[self.torpos[i][1]]))\
                     for i in range(len(self.torpos))]
        J = np.zeros([total_action_dims, total_action_dims])

        for i in range(len(self.qs)):
            model = self.qs[i]
            with torch.no_grad():
                input = torch.tensor(batch_joint_states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                _, mu, P, _, _ = model(input, None)
            mu = mu.detach().cpu().numpy()
            if len(P.shape)==3:
                P=P.reshape(-1,P.shape[-1])
            P = P.detach().cpu().numpy()
            P_values[i] = P
            Mu_values[i] = mu


        self.act_dim_starts = []
        start = 0
        for act_dim in self.act_dims:
            self.act_dim_starts.append(start)
            start += act_dim

        for P, partial_topo in zip(P_values, self.torpos):
            partial_row_start = 0
            for current_agent_num in partial_topo:
                partial_col_start = 0
                row_block_length = self.act_dims[current_agent_num]
                row_start = self.act_dim_starts[current_agent_num]

                for other_agent_num in partial_topo:
                    col_start = self.act_dim_starts[other_agent_num]
                    col_block_length = self.act_dims[other_agent_num]
                    # tempa = J[row_start:row_start + row_block_length, col_start:col_start + col_block_length]
                    #tempb = P[partial_row_start: partial_row_start + row_block_length,
                    #partial_col_start: partial_col_start + col_block_length]
                    J[row_start:row_start + row_block_length, col_start:col_start + col_block_length] += \
                        P[partial_row_start: partial_row_start + row_block_length,
                        partial_col_start: partial_col_start + col_block_length]

                    partial_col_start += col_block_length

                partial_row_start += row_block_length

        U = np.zeros(total_action_dims)

        for P, MU, partial_topo in zip(P_values, Mu_values, self.torpos):
            partial_start = 0
            #TODO:
            M = np.dot(P, MU.reshape([-1, 1])).reshape(-1)
            for agent_num in partial_topo:
                U[self.act_dim_starts[agent_num]: self.act_dim_starts[agent_num] + self.act_dims[agent_num]] += \
                    M[partial_start: partial_start + self.act_dims[agent_num]]
                partial_start += self.act_dims[agent_num]

        x = np.linalg.solve(J, U)
        return x

    def update(self, batch_datas):
        """
        :param batch_datas: a list with every torpo's states, actions, next states and reward, dones (a dict). {state_batch}
        :return:
        """
        details = []
        for i in range(len(self.torpos)):
            s = torch.tensor(batch_datas[i]['states'].reshape(-1,self.joint_s_dim[i]), dtype=torch.float32).to(self.device)
            a = torch.tensor(batch_datas[i]['actions'].reshape(-1, self.joint_a_dim[i]), dtype=torch.float32).to(self.device)
            s_ = torch.tensor(batch_datas[i]['next_states'].reshape(-1,self.joint_s_dim[i]), dtype=torch.float32).to(self.device)
            r = torch.tensor(batch_datas[i]['reward'].reshape(-1, 1), dtype=torch.float32).to(self.device).to(self.device)
            done = torch.tensor(batch_datas[i]['dones'].reshape(-1, 1), dtype=torch.float32).to(self.device).to(self.device)
            model = self.qs[i]
            _, _, _, _, q = model(s,a)
            V_, _, _, _, _ = self.target_qs[i](s_, None)
            target_q = r + self.GAMMA* (1-done) * V_
            loss = F.mse_loss(target_q, q)
            optimizer = self.policy_optimizers[i]
            optimizer.zero_grad()
            loss.backward()

            #TODO：判断这一句是否需要
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()
            for target_param, param in zip(self.target_qs[i].parameters(), model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.Tau) + param.data * self.Tau)
            details.append(loss.item())
        return details



