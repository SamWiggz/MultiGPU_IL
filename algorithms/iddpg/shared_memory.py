import numpy as np
import torch
from torch.autograd import Variable
from torch import Tensor
import multiprocessing.shared_memory as shm
import Config


from pogema import pogema_v0, GridConfig
import os
from gym.spaces import Box

import sys
sys.path.append('../')
from env_wrappers import SubprocVecEnv, DummyVecEnv

def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            grid_config = GridConfig(num_agents=Config.n_agents,  # number of agents
                         size=Config.mapsize, # size of the grid
                         density=0.4,  # obstacle density
                         seed=(seed + rank * 1000),  # set to None for random 
                                  # obstacles, agents and targets 
                                  # positions at each reset
                         max_episode_steps=Config.episode_length,  # horizon
                         obs_radius=3,  # defines field of view
                         )
            env = pogema_v0(grid_config=grid_config)
            # np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def cleanupSharedMemory(shared_memory):
    """
    Cleanup Shared Memory Blocks
    """
    for sm in shared_memory.values():
        if isinstance(sm, shm.SharedMemory):
            sm.close()
            sm.unlink()

def create_shared_memory_for_agents(agents):
    total_size = sum(param.numel() * param.element_size() for agent in agents for model in agent.models.values() for param in model.parameters())
    
    shared_mem = shm.SharedMemory(create=True, size=total_size)
    offset = 0
    shared_memory_blocks = {}
    
    for agent_id, agent in enumerate(agents):
        shared_memory_blocks[agent_id] = {}
        for model_name, model in agent.models.items():
            shared_memory_blocks[agent_id][model_name] = {}
            for name, param in model.named_parameters():
                size = param.numel() * param.element_size()
                shared_memory_blocks[agent_id][model_name][name] = (shared_mem.name, offset, size)
                offset += size  # Move the offset
    
    return shared_memory_blocks, shared_mem


def copy_agents_to_shared_memory(agents, shared_memory_blocks, shared_mem, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(agents)
    
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bfloat16: np.float16  # Approximate handling for bfloat16
    }
    
    buffer = np.ndarray((shared_mem.size,), dtype=np.uint8, buffer=shared_mem.buf)
    
    for agent_id in range(start_idx, end_idx):
        agent = agents[agent_id]
        for model_name, model in agent.models.items():
            for name, param in model.named_parameters():
                shm_name, offset, size = shared_memory_blocks[agent_id][model_name][name]
                np_dtype = dtype_map.get(param.dtype)
                
                if np_dtype is None:
                    raise TypeError(f"Unsupported torch dtype: {param.dtype}")
                
                np_array = np.frombuffer(buffer, dtype=np_dtype, count=param.numel(), offset=offset).reshape(param.shape)
                np_array[:] = param.detach().cpu().numpy()


def load_agents_from_shared_memory(agents, shared_memory_blocks, shared_mem, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(agents)
    
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bfloat16: np.float16  # Approximate handling for bfloat16
    }
    
    buffer = np.ndarray((shared_mem.size,), dtype=np.uint8, buffer=shared_mem.buf)
    
    for agent_id in range(start_idx, end_idx):
        agent = agents[agent_id]
        for model_name, model in agent.models.items():
            for name, param in model.named_parameters():
                shm_name, offset, size = shared_memory_blocks[agent_id][model_name][name]
                np_dtype = dtype_map.get(param.dtype)
                
                if np_dtype is None:
                    raise TypeError(f"Unsupported torch dtype: {param.dtype}")
                
                np_array = np.frombuffer(buffer, dtype=np_dtype, count=param.numel(), offset=offset).reshape(param.shape)
                param.data.copy_(torch.from_numpy(np_array))

def initSharedMemory(algorithm):
    """
    Initialize Shared Memory Blocks and Inter-process Parameters

    Parameters:
        obs_shm (shared_memory): Shared Memory Block for Observations
        ac_shm (shared_memory): Shared Memory Block for Actions
        rew_shm (shared_memory): Shared Memory Block for Rewards
        next_obs_shm (shared_memory): Shared Memory Block for Next Observations
        done_shm (shared_memory): Shared Memory Block for Done Signals

        curr_i (shared_memory): Shared Memory Block for Current Index
        filled_i (shared_memory): Shared Memory Block for Filled Index

        fin_dc (shared_memory): Shared Memory Block for Data Collection Completion Signal
        fin_mu (shared_memory): Shared Memory Block for Model Update Completion Signal
        fin_done (shared_memory): Shared Memory Block for Completion Signal
        t (shared_memory): Shared Memory Block for Time
        ep (shared_memory): Shared Memory Block for Episodes

    Returns:
        shared_memory_params (dict): Dictionary of Shared Memory Blocks
    """
    ### Initialize Dummy Environment to setup Replay Buffer ###
    grid_config = GridConfig(num_agents=Config.n_agents,  # number of agents
                size=Config.mapsize, # size of the grid
                density=0.4,  # obstacle density
                seed=1,  # set to None for random 
                        # obstacles, agents and targets 
                        # positions at each reset
                max_episode_steps=Config.episode_length,  # horizon
                obs_radius=3,  # defines field of view
                )
    env = pogema_v0(grid_config=grid_config)

    # Create shared memory blocks with unique names
    obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    ac_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    rew_shm = shm.SharedMemory(create=True, size=Config.buffer_length * Config.n_agents * np.dtype(np.float64).itemsize)
    next_obs_shm = shm.SharedMemory(create=True, size=Config.buffer_length * sum(env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)) * np.dtype(np.float64).itemsize)
    done_shm = shm.SharedMemory(create=True, size=Config.buffer_length * Config.n_agents * np.dtype(np.float64).itemsize)

    ### Close Dummy Environment ###
    env.close()

    curr_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    filled_i = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    fin_dc = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    fin_mu = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    fin_done = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    t = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)
    ep = shm.SharedMemory(create=True, size=np.dtype(np.int32).itemsize)

    # Initialize shared memory blocks
    np.ndarray((1,), dtype=np.int32, buffer=curr_i.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=filled_i.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=fin_dc.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=fin_mu.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=fin_done.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=t.buf)[0] = 0
    np.ndarray((1,), dtype=np.int32, buffer=ep.buf)[0] = 0

    # Create shared memory for each agent's models
    shared_memory_blocks, shared_mem = create_shared_memory_for_agents(algorithm.agents)
    copy_agents_to_shared_memory(algorithm.agents, shared_memory_blocks, shared_mem)

    return {
        'obs_shm': obs_shm,
        'ac_shm': ac_shm,
        'rew_shm': rew_shm,
        'next_obs_shm': next_obs_shm,
        'done_shm': done_shm,
        'curr_i': curr_i,
        'filled_i': filled_i,
        'fin_dc': fin_dc,
        'fin_mu': fin_mu,
        'fin_done': fin_done,
        't': t,
        'ep': ep,
        'shared_memory_blocks': shared_memory_blocks,
        'shared_mem': shared_mem
    }

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts using shared memory
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims,obs_shm, ac_shm, rew_shm, next_obs_shm, done_shm, curr_i, filled_i):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of observation dimensions for each agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.filled_i = np.ndarray((1,), dtype=np.int32, buffer=filled_i.buf) # index of first empty location in buffer (last index when full)
        self.curr_i = np.ndarray((1,), dtype=np.int32, buffer=curr_i.buf)  # current index to write to (overwrite oldest data)


        # Create shared memory for each buffer
        self.obs_shm = obs_shm
        self.ac_shm = ac_shm
        self.rew_shm = rew_shm
        self.next_obs_shm = next_obs_shm
        self.done_shm = done_shm

        # Create numpy arrays backed by shared memory
        self.obs_buffs = [np.ndarray((max_steps, odim), dtype=np.float32, buffer=self.obs_shm.buf) for odim in obs_dims]
        self.ac_buffs = [np.ndarray((max_steps, adim), dtype=np.float32, buffer=self.ac_shm.buf) for adim in ac_dims]
        self.rew_buffs = np.ndarray((max_steps, num_agents), dtype=np.float32, buffer=self.rew_shm.buf)
        self.next_obs_buffs = [np.ndarray((max_steps, odim), dtype=np.float32, buffer=self.next_obs_shm.buf) for odim in obs_dims]
        self.done_buffs = np.ndarray((max_steps, num_agents), dtype=np.float32, buffer=self.done_shm.buf)

        np.random.seed(Config.seed)

    def __len__(self):
        return self.filled_i[0]

    def push(self, obs, actions, rewards, next_obs, dones):
        observations = np.array(obs, dtype=object)
        next_observations = np.array(next_obs, dtype=object)
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i[0] + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i[0]  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[:, agent_i] = np.roll(self.rew_buffs[:, agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[:, agent_i] = np.roll(self.done_buffs[:, agent_i], rollover)
            self.curr_i[0] = 0
            self.filled_i[0] = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = np.vstack(observations[:, agent_i])
            self.ac_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = actions[agent_i]
            self.rew_buffs[self.curr_i[0]:self.curr_i[0] + nentries, agent_i] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i[0]:self.curr_i[0] + nentries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[self.curr_i[0]:self.curr_i[0] + nentries, agent_i] = dones[:, agent_i]
        self.curr_i[0] += nentries
        if self.filled_i[0] < self.max_steps:
            self.filled_i[0] += nentries
        if self.curr_i[0] == self.max_steps:
            self.curr_i[0] = 0

    def sample(self, a_i, N, device='cpu', norm_rews=False, rank = 0):
        np.random.seed((rank+1) * 100)
        inds = np.random.choice(np.arange(self.filled_i[0]), size=N, replace=False)
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = cast(self.rew_buffs[inds,a_i])
        return (cast(self.obs_buffs[a_i][inds]),
                cast(self.ac_buffs[a_i][inds]),
                ret_rews,
                cast(self.next_obs_buffs[a_i][inds]),
                cast(self.done_buffs[inds,a_i]))

    def get_average_rewards(self, N):
        if self.filled_i[0] == self.max_steps:
            inds = np.arange(self.curr_i[0] - N, self.curr_i[0])  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i[0] - N), self.curr_i[0])
        return [self.rew_buffs[inds, i].mean() for i in range(self.num_agents)]