"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

np.random.seed(1)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            #print(data)
            ob, reward, done, truncated, info = env.step(data)
            if all(done):
                ob, info = env.reset()
            remote.send((ob, reward, done, truncated, info))
        elif cmd == 'reset':
            ob, info = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob, info = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        # elif cmd == 'get_agent_types':
        #     if all([hasattr(a, 'adversary') for a in env.agents]):
        #         remote.send(['adversary' if a.adversary else 'agent' for a in
        #                      env.agents])
        #     else:
        #         remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        #self.remotes[0].send(('get_agent_types', None))
        #self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        #print(actions)
        for remote, action in zip(self.remotes, actions):
            #print(action)
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, trunc, infos = zip(*results)
        return np.array(obs[0]), np.array(rews, dtype=object), np.array(dones, dtype=object), np.array(trunc, dtype=object), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        ret = ([remote.recv() for remote in self.remotes])
        return np.array(ret[0])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        # if all([hasattr(a, 'adversary') for a in env.agents]):
        #     self.agent_types = ['adversary' if a.adversary else 'agent' for a in
        #                         env.agents]
        # else: 
        #self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        #print(len(results[0]))
        results = np.array(results,dtype=object)
        #obs, rews, dones, infos = map(np.array, zip(*results))
        obs, rews, dones, truncated, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs[0]), np.array(rews), np.array(dones), np.array(truncated), infos

    def reset(self):       
        obs, infos = self.envs[0].reset()
        #print(obs)
        return np.array(obs)

    def close(self):
        return