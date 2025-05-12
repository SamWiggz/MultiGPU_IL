import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
import Config
from collections import defaultdict

class Agent_Load_Balancer:
    def __init__(self, rank, world_size, device, debug=False):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.debug = debug
        self.n_agents = Config.n_agents
        self.consecutive_balanced = 0
        self.toggle_threshold = 5

        self.start_idx = 0
        self.end_idx = 0

        self.num_gpus = world_size

        # Initial even split of GPU ratio
        self.gpu_ratios = {i: 1.0 / self.num_gpus for i in range(self.num_gpus)}
        self.gpu_ratio = self.gpu_ratios[self.rank]
        self.updated_gpu_ratios = self.gpu_ratios.copy()

        self.device_name = torch.cuda.get_device_name(self.device)
        all_device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        same_device = all(name == self.device_name for name in all_device_names)

        assignment_counts = Config.gpu_assignment
        same_learner_count = len(set(assignment_counts)) == 1

        self.homogeneous = same_device and same_learner_count

        if self.debug:
            print(f"Learner {rank} ({self.device_name}) - Initial GPU Ratio: {self.gpu_ratio:.4f}")
            if self.homogeneous:
                print(f"Learner {rank} - Homogeneous GPU setup detected. Load balancing will be disabled.")

    def get_agent_indices(self):
        """
        Compute agent indices for each GPU learner based on dynamic load balancing.

        Returns:
            start_idx (int): Start index of assigned agents.
            end_idx (int): End index of assigned agents.
        """
        if self.consecutive_balanced != self.toggle_threshold:
            agents_per_gpu = {rank: int(round(self.n_agents * self.updated_gpu_ratios[rank])) for rank in range(self.world_size)}

            total_assigned_agents = sum(agents_per_gpu.values())
            agent_diff = self.n_agents - total_assigned_agents

            if agent_diff != 0:
                sorted_ranks = sorted(self.updated_gpu_ratios.keys(), key=lambda r: self.updated_gpu_ratios[r], reverse=True)
                for rank in sorted_ranks:
                    if agent_diff == 0:
                        break
                    agents_per_gpu[rank] += 1 if agent_diff > 0 else -1
                    agent_diff += -1 if agent_diff > 0 else 1

            self.start_idx = sum(agents_per_gpu[i] for i in range(self.rank))
            self.end_idx = self.start_idx + agents_per_gpu[self.rank]

        if self.debug:
            print("Rank: ", self.rank, "Start: ", self.start_idx, "End: ", self.end_idx)
        return self.start_idx, self.end_idx

    def update(self, iter, base_threshold=0.05, adjustment_rate=1):
        """
        Adjust GPU ratios dynamically to equalize iteration times over several iterations.
        - base_threshold is a fraction of the max iteration time used as the stopping criterion.
        - adjustment_rate controls the speed of the ratio change.
        """
        if self.homogeneous:
            return

        if self.debug:
            print(f"Learner {self.rank} ({self.device_name}) - Iteration Time: {iter:.4f}")

        if self.consecutive_balanced == self.toggle_threshold:
            if self.debug:
                print("Balancer Toggled Off")
            return

        local_time = torch.tensor([iter], dtype=torch.float32, device=self.device)
        global_times = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(self.world_size)]
        
        dist.barrier()
        dist.all_gather(global_times, local_time)
        times = {i: global_times[i].item() for i in range(self.world_size)}

        time_diff = max(times.values()) - min(times.values())
        max_time = max(times.values())
        ratio_threshold = base_threshold * max_time

        if time_diff > ratio_threshold and self.consecutive_balanced < self.toggle_threshold:
            if self.debug:
                print(f"Iteration times are not balanced yet (Time Diff: {time_diff:.4f}, Threshold: {ratio_threshold:.4f}). Adjusting ratios.")            
            total_time = sum(times.values())
            adjustments = defaultdict(float)

            for rank in times:
                time_factor = (total_time / self.world_size - times[rank]) / total_time
                adjustments[rank] = time_factor * adjustment_rate

            for rank in times:
                self.updated_gpu_ratios[rank] += adjustments[rank]

            total_ratio = sum(self.updated_gpu_ratios.values())
            self.updated_gpu_ratios = {i: self.updated_gpu_ratios[i] / total_ratio for i in times}
            self.consecutive_balanced = 0
            
        elif self.consecutive_balanced < self.toggle_threshold:
            if self.debug:
                print(f"Iteration times are balanced (Time Diff: {time_diff:.4f}). No adjustment needed.")
            self.consecutive_balanced += 1

        self.gpu_ratio = self.updated_gpu_ratios[self.rank]

        if self.debug:
            print(f"Learner {self.rank} ({self.device_name}) - Updated GPU Ratio: {self.gpu_ratio:.4f}")
