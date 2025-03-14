####
#### Model Definition ####
####

# IQL
hidden_sizes = [64, 64]
batch_size = 8192
gamma = 0.99
tau = 1
lr = 2.5e-4
adam_eps = 1e-8

####
#### Replay Buffer ####
####
buffer_length = int(1e6)

####
#### Environment ####
####
n_agents = 1024
mapsize = 75 
n_episodes = 100 
n_rollout_threads = 32
n_training_threads = 1
n_updates = 4
steps_per_update = 128
episode_length = 128
discrete_action = True
init_noise_scale = 0.3
final_noise_scale = 0.0
seed = 1
n_exploration_eps = 100 #17

####
#### DDP ####
####
port = '11112'

gpu_assignment = [1, 1]  ### [1 learner GPU0, 1 learner GPU1]