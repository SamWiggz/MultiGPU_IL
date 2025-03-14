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

class PolicyNN(nn.Module):
    def __init__(self, num_in_pol, num_out_pol):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_in_pol),
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
    
class CriticNN(nn.Module):
    def __init__(self, num_in_critic):
        super(CriticNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_in_critic),
            nn.Linear(num_in_critic, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], 1)
        )

        # Initialize the weights and bias of BatchNorm1d
        self.model[0].weight.data.fill_(1)
        self.model[0].bias.data.fill_(0)
    
    def forward(self, state_action):
        return self.model(state_action)

class DDPGAgent(object):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, discrete_action=True):
        self.policy_nn = PolicyNN(num_in_pol, num_out_pol)
        self.target_policy_nn = PolicyNN(num_in_pol, num_out_pol)
        self.critic_nn = CriticNN(num_in_critic)
        self.target_critic_nn = CriticNN(num_in_critic)

        hard_update(self.target_policy_nn, self.policy_nn)
        hard_update(self.target_critic_nn, self.critic_nn)

        self.policy_optim = torch.optim.Adam(params=self.policy_nn.parameters(), lr = Config.lr)
        self.critic_optim = torch.optim.Adam(params=self.critic_nn.parameters(), lr = Config.lr)

        #if Config.gpu_process == 0:
            #self.policy_nn, self.policy_optim = ipex.optimize(self.policy_nn, optimizer=self.policy_optim, dtype=torch.bfloat16)
            #self.critic_nn, self.critic_optim = ipex.optimize(self.critic_nn, optimizer=self.critic_optim, dtype=torch.bfloat16)
            #self.policy_nn, self.policy_optim = ipex.optimize(self.policy_nn, optimizer=self.policy_optim, dtype=torch.float32)
            #self.critic_nn, self.critic_optim = ipex.optimize(self.critic_nn, optimizer=self.critic_optim, dtype=torch.float32)

        self.models = {
            'policy_nn': self.policy_nn,
            'target_policy_nn': self.target_policy_nn,
            'critic_nn': self.critic_nn,
            'target_critic_nn': self.target_critic_nn
        }

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
        #print("in here")
        action = self.policy_nn(obs)
        #print("did action")
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        #print(action)
        #return torch.argmax(action, dim=1)
        return action

    def get_params(self):
            return {'policy_nn': self.policy_nn.state_dict(),
                    'critic_nn': self.critic_nn.state_dict(),
                    'target_policy_nn': self.target_policy_nn.state_dict(),
                    'target_critic_nn': self.target_critic_nn.state_dict(),
                    'policy_optim': self.policy_optim.state_dict(),
                    'critic_optim': self.critic_optim.state_dict()}
    
    def load_params(self, params):
            self.policy_nn.load_state_dict(params['policy_nn'])
            self.critic_nn.load_state_dict(params['critic_nn'])
            self.target_policy_nn.load_state_dict(params['target_policy_nn'])
            self.target_critic_nn.load_state_dict(params['target_critic_nn'])
            self.policy_optim.load_state_dict(params['policy_optim'])
            self.critic_optim.load_state_dict(params['critic_optim'])

class IDDPG(object):
    def __init__(self, agent_init_params,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, rank=0):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
        """
        self.num_agents = Config.n_agents
        self.agents = [DDPGAgent(discrete_action=discrete_action,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = Config.gamma
        self.tau = Config.tau
        self.lr = Config.lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.MSELoss = torch.nn.MSELoss()
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.rank = rank

    @property
    def policies(self):
        return [a.policy_nn for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy_nn for a in self.agents]

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
        ## Update Critic
        ##
        # FW Target Policy
        if self.discrete_action:
            trgt_vf_in = torch.cat((next_obs,
                                    onehot_from_logits(
                                        curr_agent.target_policy_nn(
                                            next_obs))),
                                    dim=1)
        else:
            trgt_vf_in = torch.cat((next_obs,
                                    curr_agent.target_policy_nn(next_obs)),
                                    dim=1)
        target_value = (rews.view(-1, 1) + self.gamma *
                        curr_agent.target_critic_nn(trgt_vf_in) *
                        (1 - dones.view(-1, 1)))

        vf_in = torch.cat((obs, acs), dim=1)
        actual_value = curr_agent.critic_nn(vf_in)
        vf_loss = self.MSELoss(actual_value, target_value.detach())

        curr_agent.critic_optim.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic_nn.parameters(), 0.5)
        curr_agent.critic_optim.step()

        ##
        ## Update Policy
        ##  
        if self.discrete_action:
            curr_pol_out = curr_agent.policy_nn(obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, device=device, hard=True) 
        else:
            curr_pol_out = curr_agent.policy_nn(obs)
            curr_pol_vf_in = curr_pol_out
        vf_in = torch.cat((obs, curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic_nn(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3

        curr_agent.policy_optim.zero_grad()
        pol_loss.backward()
        #print(pol_loss)

        torch.nn.utils.clip_grad_norm_(curr_agent.policy_nn.parameters(), 0.5)
        curr_agent.policy_optim.step()
    
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)


    def update_all_targets(self, a):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        with torch.no_grad():
            soft_update(self.agents[a].target_critic_nn, self.agents[a].critic_nn, self.tau)
            soft_update(self.agents[a].target_policy_nn, self.agents[a].policy_nn, self.tau)
            self.niter += 1

    def prep_training(self, start_idx, end_idx, device='gpu'):
        for a in range(start_idx, end_idx):
            self.agents[a].policy_nn.train()
            self.agents[a].critic_nn.train()
            self.agents[a].target_policy_nn.train()
            self.agents[a].target_critic_nn.train()
        fn = lambda x: x.to(device)
        for a in range(start_idx, end_idx):
            self.agents[a].policy_nn = fn(self.agents[a].policy_nn)
        for a in range(start_idx, end_idx):
            self.agents[a].critic = fn(self.agents[a].critic_nn)
        for a in range(start_idx, end_idx):
            self.agents[a].target_policy_nn = fn(self.agents[a].target_policy_nn)
        for a in range(start_idx, end_idx):
            self.agents[a].target_critic_nn = fn(self.agents[a].target_critic_nn)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy_nn.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy_nn = fn(a.policy_nn)
            self.pol_dev = device

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
            num_in_critic = num_in_pol + get_shape(env.action_space)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'rank': rank}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    
    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
    