import argparse
import torch
import time
import os
import sys
sys.path.append('../')

import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from pogema import pogema_v0, GridConfig
from iql import IQL
import Config

import torch.multiprocessing as mp
import torch.distributed as dist
import multiprocessing.shared_memory as shm
from multiprocessing import Process, Pipe, Queue, Condition

from balancer import Agent_Load_Balancer
import gc
import psutil

from shared_memory import make_parallel_env, initSharedMemory, copy_agents_to_shared_memory, load_agents_from_shared_memory, cleanupSharedMemory, ReplayBuffer

def gpu_mapping(rank):
    gpu_assignment = Config.gpu_assignment  # Example: [4, 10]
    gpu_list = []
    
    for gpu_id, count in enumerate(gpu_assignment):
        gpu_list.extend([gpu_id] * count)  # Expands GPU allocation
    
    return gpu_list[rank % len(gpu_list)]  # Cycles through the list

def get_device(rank):
    return torch.device(f"cuda:{gpu_mapping(rank)}")

def main_actors(iql, shared_memory_params):
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed_all(Config.seed)
    np.random.seed(Config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(Config.n_training_threads)

    env = make_parallel_env(Config.n_rollout_threads, Config.seed)

    replay_buffer = ReplayBuffer(Config.buffer_length, Config.n_agents,
                            [env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)],
                            [1 if isinstance(env.action_space, Box) else 1 for i in range(Config.n_agents)],
                            shared_memory_params['obs_shm'], shared_memory_params['ac_shm'], shared_memory_params['rew_shm'], 
                            shared_memory_params['next_obs_shm'], shared_memory_params['done_shm'], shared_memory_params['curr_i'], shared_memory_params['filled_i'])

    t = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['t'].buf)
    fin_dc = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_dc'].buf)
    fin_mu = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_mu'].buf)
    fin_done = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_done'].buf)
    ep = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['ep'].buf)

    # Start Data Collection
    train_start = time.perf_counter()
    dc_tot = 0
    dc_count = 0
    mu_time = 0
    mu_count = 0
    first_mu_time = True
    first_dc_time = True
    iql.prep_rollouts(device='cpu')
    for ep_i in range(0, Config.n_episodes, Config.n_rollout_threads):
        ep[0] = ep_i + Config.n_rollout_threads
        ###
        # Data Collection
        ###
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                ep_i + 1 + Config.n_rollout_threads,
                                Config.n_episodes))
        obs = env.reset()
        obs = [[array.ravel() for array in obs] for i in range(Config.n_rollout_threads)]

        explr_pct_remaining = max(0, Config.n_exploration_eps - ep_i) / Config.n_exploration_eps
        iql.scale_noise(Config.final_noise_scale + (Config.init_noise_scale - Config.final_noise_scale) * explr_pct_remaining)
        iql.reset_noise()
        for et_i in range(Config.episode_length):
            ###
            # Data Collection
            ###
            dc_clock = time.perf_counter()
            torch_obs = [torch.stack([torch.tensor(arr[i], dtype=torch.float32) for arr in obs]) for i in range(iql.num_agents)]
            with torch.no_grad():
                torch_agent_actions = iql.step(torch_obs, explore=True)
            actions = [
                    tensor.argmax(dim=1).tolist()  # Take argmax along the second dimension for each tensor in the list
                    for tensor in torch_agent_actions
                ]
            actions = [[ac[i] for ac in actions] for i in range(Config.n_rollout_threads)] # rearrange actions to be per environment
            next_obs, rewards, terminated, truncations, info = env.step(actions)
            next_obs = [[array.ravel() for array in next_obs] for i in range(Config.n_rollout_threads)]
            replay_buffer.push(obs, actions, rewards, next_obs, terminated)
            obs = next_obs
            t[0] += Config.n_rollout_threads

            if not first_dc_time:
                dc_tot += time.perf_counter() - dc_clock

            #print((t % Config.steps_per_update) < Config.n_rollout_threads)
            if (len(replay_buffer) >= Config.batch_size and
                (t[0] % Config.steps_per_update) < Config.n_rollout_threads):
                mu_start = time.perf_counter()
                if not first_mu_time:
                    mu_count += 1
                if not first_dc_time:
                    dc_count += 1
                first_dc_time = False
                fin_dc[0] = 1 # Finished DC
                while (1):
                    if fin_mu[0] == sum(Config.gpu_assignment):
                        #print("continue dc")
                        if not first_mu_time:
                            mu_time += time.perf_counter() - mu_start
                        else :
                            first_mu_time = False
                        fin_dc[0] = 0
                        load_agents_from_shared_memory(iql.agents, shared_memory_params['shared_memory_blocks'], shared_memory_params['shared_mem'])
                        break
        if dc_count != 0 and  mu_count % (Config.n_rollout_threads * 4) == 0:
            print(f"Avg Data Collection Time: {dc_tot / dc_count:.3f}s")
            print(f"Avg Model Update Time: {mu_time / mu_count:.3f}s")
            print(f"Avg Iteration Time: {(dc_tot / dc_count + mu_time / mu_count):.3f}s")
    env.close()
    
    fin_done[0] = 1   
    print(f"Avg Data Collection Time: {dc_tot / dc_count:.3f}s")
    print(f"Avg Model Update Time: {mu_time / mu_count:.3f}s")
    print(f"Avg Iteration Time: {(dc_tot / dc_count + mu_time / mu_count):.3f}s")
    print(f"Total End-to-End Training Time: {(time.perf_counter() - train_start):.3f}s")

def main_learners(rank, iql, world_size, shared_memory_params, lock):
    torch.cuda.set_device(get_device(rank))
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    t = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['t'].buf)
    fin_dc = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_dc'].buf)
    fin_mu = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_mu'].buf)
    fin_done = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['fin_done'].buf)
    ep = np.ndarray((1,), dtype=np.int32, buffer=shared_memory_params['ep'].buf)

    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed_all(Config.seed)
    np.random.seed(Config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # set training threads
    torch.set_num_threads(Config.n_training_threads)

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
    
    replay_buffer = ReplayBuffer(Config.buffer_length, Config.n_agents,
                            [env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2] for i in range(Config.n_agents)],
                            [1 if isinstance(env.action_space, Box) else 1 for i in range(Config.n_agents)],
                            shared_memory_params['obs_shm'], shared_memory_params['ac_shm'], shared_memory_params['rew_shm'], 
                            shared_memory_params['next_obs_shm'], shared_memory_params['done_shm'], shared_memory_params['curr_i'], shared_memory_params['filled_i'])

    # close dummy environment
    env.close()

    balancer = Agent_Load_Balancer(rank, world_size, get_device(rank))
    start_idx, end_idx = balancer.get_agent_indices()
    previous_start_idx, previous_end_idx = start_idx, end_idx
                
    train_start = time.perf_counter()
    mu_tot = 0
    mu_count = 0
    exit_outer_loop = False
    while(1):
        while(1):
            if(fin_dc[0]):
                break
            if(fin_done):
                exit_outer_loop = True  # Signal to exit the outer loop
                break
        if exit_outer_loop:
            break
            
        ###
        # Model Update
        ### 
        if (len(replay_buffer) >= Config.batch_size and
            (t[0] % Config.steps_per_update) < Config.n_rollout_threads):
            iql.prep_training(start_idx, end_idx, device=get_device(rank))
            dist.barrier()
            
            mu_clock = time.perf_counter() 
            for a_i in range(start_idx, end_idx):
                for u_i in range(Config.n_updates):
                    sample = replay_buffer.sample(a_i, Config.batch_size,
                                                device=get_device(rank))         
                    iql.update(sample, a_i, device=get_device(rank))
                iql.update_all_targets(a_i)

            mu_tot = time.perf_counter() - mu_clock
            mu_count += 1
            #print("Rank: ", rank, torch.cuda.get_device_name(get_device(rank)), "Model Update Time: ", mu_tot,"s")
            copy_agents_to_shared_memory(iql.agents, shared_memory_params['shared_memory_blocks'], shared_memory_params['shared_mem'], start_idx, end_idx)
            dist.barrier()

            balancer.update(mu_tot)
            start_idx, end_idx = balancer.get_agent_indices()
            dist.barrier()

            # Check if indices have changed
            if start_idx != previous_start_idx or end_idx != previous_end_idx:
                load_agents_from_shared_memory(iql.agents, shared_memory_params['shared_memory_blocks'], shared_memory_params['shared_mem'], start_idx, end_idx)
                # Update the previous indices to the current ones
                previous_start_idx, previous_end_idx = start_idx, end_idx

            with lock:
                fin_mu[0] += 1
            dist.barrier()
            while(1):
                if(fin_dc[0] == 0):
                    #if rank == 0:
                    with lock:
                        fin_mu[0] -= 1
                    break
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = Config.port
    world_size_learners = sum(Config.gpu_assignment)

    lock = mp.Lock()

    # dummy env to get size observation and action dimensions
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

    iql = IQL.init_from_env(env,
                                  tau=Config.tau,
                                  lr=Config.lr,
                                  hidden_dim=Config.hidden_sizes)
    env.close()

    # Initialize shared memory
    shared_memory_params = initSharedMemory(iql)

    # spawn actors 
    ap = ctx.Process(target = main_actors,
             args=(iql, shared_memory_params))

    learner_processes = [ctx.Process(target = main_learners,
             args=(i, iql, world_size_learners, shared_memory_params, lock))
              for i in range(world_size_learners)]
  
    ap.start()    
    for lp in learner_processes:
        lp.start()

    for lp in learner_processes:
        lp.join()    
    ap.join()

    cleanupSharedMemory(shared_memory_params)