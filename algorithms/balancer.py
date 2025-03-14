import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
import Config
from collections import defaultdict

class Agent_Load_Balancer:
    def __init__(self, rank, world_size, device) :
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.n_agents = Config.n_agents

        # Step 1: Get total number of GPUs (world_size is the number of learners)
        self.num_gpus = world_size

        # Step 2: Initial even split of GPU ratio
        self.gpu_ratios = {i: 1.0 / self.num_gpus for i in range(self.num_gpus)}
        self.gpu_ratio = self.gpu_ratios[self.rank]

        # manual_gpu_ratios = {0: 0.1, 1: 0.9}

        # # Default to equal distribution if not manually specified
        # self.gpu_ratios = manual_gpu_ratios if manual_gpu_ratios else {i: 1.0 / self.num_gpus for i in range(self.num_gpus)}

        # # Assign the ratio based on the process rank
        # self.gpu_ratio = self.gpu_ratios.get(self.rank, 0)

        # Step 3: Track iteration times dynamically
        self.updated_gpu_ratios = self.gpu_ratios.copy()

        self.device_name = torch.cuda.get_device_name(self.device)
        print(f"Learner {rank} ({self.device_name}) - Initial GPU Ratio: {self.gpu_ratio:.4f}")

    def get_agent_indices(self):
        """
        Compute agent indices for each GPU learner based on dynamic load balancing.

        Returns:
            start_idx (int): Start index of assigned agents.
            end_idx (int): End index of assigned agents.
        """
        # Step 1: Compute agent allocation per GPU using updated gpu_ratios
        agents_per_gpu = {rank: int(round(self.n_agents * self.updated_gpu_ratios[rank])) for rank in range(self.world_size)}

        # Step 2: Adjust to ensure the total number of agents is correct
        total_assigned_agents = sum(agents_per_gpu.values())
        agent_diff = self.n_agents - total_assigned_agents  # Find missing/excess agents

        # Distribute the difference by adjusting the highest-ratio GPUs first
        if agent_diff != 0:
            sorted_ranks = sorted(self.updated_gpu_ratios.keys(), key=lambda r: self.updated_gpu_ratios[r], reverse=True)
            for rank in sorted_ranks:
                if agent_diff == 0:
                    break
                agents_per_gpu[rank] += 1 if agent_diff > 0 else -1
                agent_diff += -1 if agent_diff > 0 else 1

        # Step 3: Compute start and end indices for each rank
        start_idx = sum(agents_per_gpu[i] for i in range(self.rank))  # Sum previous agents
        end_idx = start_idx + agents_per_gpu[self.rank]

        print("Rank: ", self.rank, "Start: ", start_idx, "End: ", end_idx)
        return start_idx, end_idx

    def update(self, iter, base_threshold=0.05, adjustment_rate=1):
        """
        Adjust GPU ratios dynamically to equalize iteration times over several iterations.
        - base_threshold is a fraction of the max iteration time used as the stopping criterion.
        - adjustment_rate controls the speed of the ratio change.
        """
        print(f"Learner {self.rank} ({self.device_name}) - Iteration Time: {iter:.4f}")

        # Step 1: Gather iteration times from all GPUs
        local_time = torch.tensor([iter], dtype=torch.float32, device=self.device)
        global_times = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(self.world_size)]
        
        dist.barrier()
        dist.all_gather(global_times, local_time)

        # Convert gathered results to dictionary
        avg_times = {i: global_times[i].item() for i in range(self.world_size)}

        # Step 2: Calculate the difference between iteration times
        time_diff = max(avg_times.values()) - min(avg_times.values())

        max_time = max(avg_times.values())
        ratio_threshold = base_threshold * max_time

        # Step 3: If the iteration times are still not balanced, adjust the GPU ratios
        if time_diff > ratio_threshold:
            print(f"Iteration times are not balanced yet (Time Diff: {time_diff:.4f}, Threshold: {ratio_threshold:.4f}). Adjusting ratios.")            
            # Step 4: Adjust the ratios such that faster devices get higher GPU ratios
            total_time = sum(avg_times.values())
            
            # Calculate adjustment factor based on iteration times
            adjustments = defaultdict(float)
            for rank in avg_times:
                # A learner with faster iteration time should get a higher ratio
                time_factor = (total_time / self.world_size - avg_times[rank]) / total_time
                adjustments[rank] = time_factor * adjustment_rate

            # Apply adjustments to the GPU ratios
            for rank in avg_times:
                self.updated_gpu_ratios[rank] += adjustments[rank]

            # Ensure the total ratio is still 1
            total_ratio = sum(self.updated_gpu_ratios.values())
            self.updated_gpu_ratios = {i: self.updated_gpu_ratios[i] / total_ratio for i in avg_times}
            
        else:
            print(f"Iteration times are balanced (Time Diff: {time_diff:.4f}). No adjustment needed.")

        # Assign the new ratio to the current learner
        self.gpu_ratio = self.updated_gpu_ratios[self.rank]

        print(f"Learner {self.rank} ({self.device_name}) - Updated GPU Ratio: {self.gpu_ratio:.4f}")
