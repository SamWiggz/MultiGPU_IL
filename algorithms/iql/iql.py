import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable

import sys
sys.path.append('../')

from utils import hard_update, soft_update, gumbel_softmax, onehot_from_logits, OUNoise
import Config
import time

class QNetwork(nn.Module):
    def __init__(self, num_in_pol, num_out_pol):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_in_pol, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], num_out_pol)
        )

        # Initialize the weights and bias of BatchNorm1d
        self.model[0].weight.data.fill_(1)
        self.model[0].bias.data.fill_(0)
    
    def forward(self, state):
        return self.model(state)

class IQLAgent(object):
    def __init__(self, num_in_pol, num_out_pol, discrete_action=True):
        self.q_network = QNetwork(num_in_pol, num_out_pol)
        self.target_q_network = QNetwork(num_in_pol, num_out_pol)

        hard_update(self.target_q_network, self.q_network)

        # for param in self.policy_nn.state_dict().values():
        #     param.share_memory_()
        # for param in self.target_policy_nn.state_dict().values():
        #     param.share_memory_()

        self.models = {
            'q_network': self.q_network,
            'target_q_network': self.target_q_network
        }

        self.optim = torch.optim.Adam(params=self.q_network.parameters(), lr = Config.lr)

        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.q_network(obs)
        #print(torch.argmax(action, dim=1))
        # if self.discrete_action:
        #     if explore:
        #         action = gumbel_softmax(action, hard=True)
        #     else:
        #         action = onehot_from_logits(action)
        # else:  # continuous action
        #     if explore:
        #         action += Variable(Tensor(self.exploration.noise()),
        #                            requires_grad=False)
        #     action = action.clamp(-1, 1)
        #print(action)
        #return torch.argmax(action, dim=1)
        return action

class IQL(object):
    def __init__(self, agent_init_params,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, rank=0):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to q network
                num_out_pol (int): Output dimensions to q network
        """
        self.num_agents = Config.n_agents
        self.agents = [IQLAgent(discrete_action=discrete_action,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = Config.gamma
        self.tau = Config.tau
        self.lr = Config.lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.MSELoss = torch.nn.MSELoss()
        self.qnet_dev = 'cpu'  # device for policies
        self.trgt_qnet_dev = 'cpu'  # device for target policies
        self.rank = rank

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, device, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        ##
        ## Update Q Network
        ##
        with torch.no_grad():
            target_max, _ = curr_agent.target_q_network(next_obs).max(dim=1)
            td_target = rews.flatten() + self.gamma * target_max * (1 - dones.flatten())
        old_val = curr_agent.q_network(obs).gather(1, acs.long()).squeeze()
        #print(td_target)
        #print(old_val)
        loss = F.mse_loss(td_target, old_val)
        
        curr_agent.optim.zero_grad()
        loss.backward()
        curr_agent.optim.step()

    def update_all_targets(self, a):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        with torch.no_grad():
            soft_update(self.agents[a].target_q_network, self.agents[a].q_network, self.tau)
            self.niter += 1

    def prep_training(self, start_idx, end_idx, device='gpu'):
        for a in range(start_idx, end_idx):
            self.agents[a].q_network.train()
            self.agents[a].target_q_network.train()
        fn = lambda x: x.to(device)
        for a in range(start_idx, end_idx):
            self.agents[a].q_network = fn(self.agents[a].q_network)
        for a in range(start_idx, end_idx):
            self.agents[a].target_q_network = fn(self.agents[a].target_q_network)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.q_network.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.qnet_dev == device:
            for a in self.agents:
                a.q_network = fn(a.q_network)
            self.qnet_dev = device

    @classmethod
    def init_from_env(cls, env,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=128, rank = 0):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        print(env.observation_space.shape)
        print(env.action_space)
        for i in range(Config.n_agents):
            num_in_pol = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
            if isinstance(env.action_space, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(env.action_space)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'rank': rank}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    