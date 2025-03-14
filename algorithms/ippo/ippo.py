import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

import sys
sys.path.append('../')

import Config
import time

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic):
        super(PPOAgent, self).__init__()
        self.policy = nn.Sequential(
            layer_init(nn.Linear(num_in_pol, Config.hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(Config.hidden_sizes[1], num_out_pol))
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_in_pol, Config.hidden_sizes[0])),
            nn.ReLU(),
            layer_init(nn.Linear(Config.hidden_sizes[1], Config.hidden_sizes[1])),
            nn.ReLU(),
            layer_init(nn.Linear(Config.hidden_sizes[1], 1))
        )

        self.models = {
            'policy': self.policy,
            'critic': self.critic,
        }

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr = Config.lr, eps=1e-5)
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.policy(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

class IPPO(object):
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
        self.agents = [PPOAgent(**params).share_memory()
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = Config.gamma
        self.gae_lambda = Config.gae_lambda
        self.tau = Config.tau
        self.lr = Config.lr
        self.discrete_action = discrete_action
        self.niter = 0
        self.MSELoss = torch.nn.MSELoss()
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.rank = rank

    def get_action_and_value(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.get_action_and_value(obs) for a, obs in zip(self.agents,
                                                                 observations)]

    def get_value(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.get_value(obs) for a, obs in zip(self.agents,
                                                        observations)]

    def update(self, sample, agent_i, clipfracs, device, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
        """
        obs, acs, rews, next_obs, dones, values, log_probs, adv, = sample
        returns = values + adv
        curr_agent = self.agents[agent_i]

        _, newlogprob, entropy, newvalue = curr_agent.get_action_and_value(obs, acs)
        logratio = newlogprob - log_probs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > Config.clip_coef).float().mean().item()]

        mb_advantages = adv
        if Config.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.clip_coef, 1 + Config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if Config.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -Config.clip_coef,
                Config.clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - Config.ent_coef * entropy_loss + v_loss * Config.vf_coef

        curr_agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(curr_agent.parameters(), Config.max_grad_norm)
        curr_agent.optimizer.step()


    def prep_training(self, start_idx, end_idx, device='cpu'):
        for a in range(start_idx, end_idx):
            self.agents[a].policy.train()
            self.agents[a].critic.train()
        fn = lambda x: x.to(device)
        for a in range(start_idx, end_idx):
            self.agents[a].policy = fn(self.agents[a].policy)
        for a in range(start_idx, end_idx):
            self.agents[a].critic = fn(self.agents[a].critic)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
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
    