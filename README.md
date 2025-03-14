# Accelerating Independent Multi-Agent Reinforcement Learning on Multi-GPU Platforms

* Note: This is a submission release that is not finalized

## Overview

## 1. Environment Setup
Preliminary: Have [conda](https://www.anaconda.com/download/success) installed

1. Clone this repository:
```
git clone 
```
2. Extract the provided environment file using conda:
```
conda env create --file=environment.yml
```
3. Activate conda environment:
```
conda activate multiGPU_IL
```

## 2. Running an Example
We provide three state-of-the-art independent learning MARL algorithms:
1. [Independent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971) (IDDPG)
2. [Independent Q-Learning](https://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf) (IQL)
3. [Independent Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347) (IPPO)

Our implementations use the popular [Pogema](https://github.com/CognitiveAISystems/pogema) environment simulation, which can scale to large scenarios with many agents.

---
1. Go to the desired algorithm's directory. Example for IDDPG:
```
cd algorithms/iddpg
```
2. Edit the Config.py file to adjust desired hyperparameters.

Some Important hyperparameters:
- `n_agents`: number of agents
- `map_size`: map size of Pogema environment
- `n_episodes`: number of episodes for the experiment
- `n_rollout_threads`: max number of possible rollout simulations
- `hidden_sizes`: hidden dimension size for agent neural networks
- `batch_size`: batch size of experiences that will be sampled by learner(s)
- `percentage_search`: percentage of the entire search space that ARC will test
3. Run your Multi-GPU independent learning experiment!
```
python main.py
```
