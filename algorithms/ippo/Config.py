####
#### Model Definition ####
####

# IPPO
hidden_sizes = [64, 64]
batch_size = 8192
gamma = 0.99
gae_lambda = 0.95
tau = 0.01
lr = 0.01
norm_adv = True
clip_coef = 0.2
clip_vloss = True
max_grad_norm = 0.5
ent_coef = 0.01 
vf_coef = 0.5
adam_eps = 1e-8

####
#### Replay Buffer ####
####
buffer_length = batch_size

####
#### Environment ####
####
n_agents = 1024
mapsize = 75 #21
n_episodes = 10000
n_rollout_threads = 64
n_training_threads = 1
n_updates = 8
n_minibatches = 4
steps_per_update = batch_size
episode_length = 128
discrete_action = True
init_noise_scale = 0.3
final_noise_scale = 0.0
seed = 1
n_exploration_eps = 10000

####
#### DDP ####
####
port = '11112'

gpu_assignment = [1, 1]  ### [1 learner on GPU0, 1 learner on GPU1]
