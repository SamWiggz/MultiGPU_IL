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
        self.consecutive_balanced = 0
        self.toggle_threshold=5

        self.start_idx = 0
        self.end_idx = 0

        self.num_gpus = world_size

        # Initial even split of GPU ratio
        self.gpu_ratios = {i: 1.0 / self.num_gpus for i in range(self.num_gpus)}
        self.gpu_ratio = self.gpu_ratios[self.rank]
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
        # Recalculate agent indices if not balanced
        if self.consecutive_balanced != self.toggle_threshold:
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
            self.start_idx = sum(agents_per_gpu[i] for i in range(self.rank))  # Sum previous agents
            self.end_idx = self.start_idx + agents_per_gpu[self.rank]

        print("Rank: ", self.rank, "Start: ", self.start_idx, "End: ", self.end_idx)
        return self.start_idx, self.end_idx

    def update(self, iter, base_threshold=0.05, adjustment_rate=1):
        """
        Adjust GPU ratios dynamically to equalize iteration times over several iterations.
        - base_threshold is a fraction of the max iteration time used as the stopping criterion.
        - adjustment_rate controls the speed of the ratio change.
        """
        print(f"Learner {self.rank} ({self.device_name}) - Iteration Time: {iter:.4f}")

        ## Stop updating if balanced for a certain number of iterations (toggle_threshold)
        if self.consecutive_balanced == self.toggle_threshold:
            print("Balancer Toggled Off")
            return

        # Gather iteration times from all GPUs
        local_time = torch.tensor([iter], dtype=torch.float32, device=self.device)
        global_times = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(self.world_size)]
        
        dist.barrier()
        dist.all_gather(global_times, local_time)
        times = {i: global_times[i].item() for i in range(self.world_size)}

        # Calculate the difference between iteration times
        time_diff = max(times.values()) - min(times.values())

        max_time = max(times.values())
        ratio_threshold = base_threshold * max_time

        # If the iteration times are still not balanced, adjust the GPU ratios
        if time_diff > ratio_threshold and self.consecutive_balanced < self.toggle_threshold:
            print(f"Iteration times are not balanced yet (Time Diff: {time_diff:.4f}, Threshold: {ratio_threshold:.4f}). Adjusting ratios.")            
            total_time = sum(times.values())
            
            adjustments = defaultdict(float)
            for rank in times:
                # A learner with faster iteration time should get a higher ratio
                time_factor = (total_time / self.world_size - times[rank]) / total_time
                adjustments[rank] = time_factor * adjustment_rate

            # Apply adjustments to the GPU ratios
            for rank in times:
                self.updated_gpu_ratios[rank] += adjustments[rank]

            total_ratio = sum(self.updated_gpu_ratios.values())
            self.updated_gpu_ratios = {i: self.updated_gpu_ratios[i] / total_ratio for i in times}
            self.consecutive_balanced = 0
            
        elif self.consecutive_balanced < self.toggle_threshold:
            print(f"Iteration times are balanced (Time Diff: {time_diff:.4f}). No adjustment needed.")
            self.consecutive_balanced += 1

        # Assign the new ratio to the current learner
        self.gpu_ratio = self.updated_gpu_ratios[self.rank]

        print(f"Learner {self.rank} ({self.device_name}) - Updated GPU Ratio: {self.gpu_ratio:.4f}")
